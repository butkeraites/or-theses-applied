"""Adds the data behind the article's figures to results.json (key "figures").

Rebuilds day 0 with the same seed and recomputes, for the day-0 routes stored in
results.json: the exact day-length distribution of each route, and the optimal unloading
thresholds along the thesis plan's longest route.

Usage (after septic_routes.py): uv run --with numpy --with scipy python figures_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import septic_routes as sr

HERE = Path(__file__).resolve().parent


def main():
    res = json.loads((HERE / "results.json").read_text())
    rng = np.random.default_rng(sr.SEED)
    day = sr.Day(rng, 0)
    assert np.allclose([p[0] for p in day.pts], [p[0] for p in res["day0"]["points"]])
    fig = {"day0_routes": {}}
    for key, optimal in (("avg|dtd", False), ("thesis|opt", True), ("pad|dtd", False)):
        routes = [r["route"] for r in res["per_day"][0]["plans"][key]["routes"]]
        rows = []
        for r in routes:
            _, dec = sr.stoch_policy(day, r, optimal=optimal)
            dist = sr.duration_dist(day, r, dec)
            cdf = np.cumsum(dist)
            rows.append(
                {
                    "route": r,
                    "p_over": float(dist[sr.SHIFT + 1 :].sum()),
                    "q05": int(np.searchsorted(cdf, 0.05)),
                    "q50": int(np.searchsorted(cdf, 0.50)),
                    "q95": int(np.searchsorted(cdf, 0.95)),
                    # pmf in 10-minute bins, 200..800 min
                    "pmf10": [float(dist[t : t + 10].sum()) for t in range(200, 800, 10)],
                }
            )
        fig["day0_routes"][key] = rows

    # Optimal unloading thresholds along the thesis plan's longest (most jobs) route:
    # after job i, unload first if the gallons left in the tank are below the threshold.
    th = max(fig["day0_routes"]["thesis|opt"], key=lambda r: (len(r["route"]), r["q95"]))
    r = th["route"]
    _, dec = sr.stoch_policy(day, r, optimal=True)
    thresholds = []
    for i in range(len(r) - 1):
        d = dec[i]
        # largest remaining capacity at which the policy still unloads first
        unl = [q for q in range(sr.Q + 1) if d[q]]
        thresholds.append(
            {
                "after_job": i + 1,
                "next_job": r[i + 1],
                "unload_if_left_below_gal": (max(unl) + 1) * sr.UNIT if unl else 0,
                "next_mean_gal": day.mean_units[r[i + 1]] * sr.UNIT,
                "min_to_yard_from_here": int(day.t[r[i], 0]),
                "min_yard_to_next": int(day.t[0, r[i + 1]]),
                "min_direct": int(day.t[r[i], r[i + 1]]),
            }
        )
    fig["thresholds_route"] = r
    fig["thresholds"] = thresholds
    res["figures"] = fig
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(thresholds, indent=1))


if __name__ == "__main__":
    main()
