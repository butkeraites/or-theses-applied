"""Follow-ups to fair_shifts.py, prompted by Bart van Rossum's review (2026-11-09).

1. Offline benchmark: with perfect information about all 26 weeks, how even could the
   season have ended? Same weekly lines as the online runs; only the assignment of lines to
   servers is chosen, jointly over the season (CP-SAT, exact).
2. What an "even" schedule looks like: re-solve the week of Sep 7, 2026 (cheapest vs most
   even among equally cheap), keep the shifts, and dump one night's coverage per hour.

Usage: uv run --with ortools python extras.py  -> extras.json
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ortools.sat.python import cp_model

import fair_shifts as fs

HERE = Path(__file__).resolve().parent


def offline(weeks, key, objective):
    """Assign each week's lines to servers knowing every week in advance.
    objective 'final': minimise the season-end range; 'worst': minimise the largest range
    over all 26 weeks."""
    n, T = fs.N_WORKERS, len(weeks)
    m = cp_model.CpModel()
    x = {}
    cum = []
    for t, w in enumerate(weeks):
        lines = w[key]["sour"]
        for i in range(n):
            m.AddExactlyOne(x.setdefault((t, i, p), m.NewBoolVar("")) for p in range(n))
        for p in range(n):
            m.AddExactlyOne(x.setdefault((t, i, p), m.NewBoolVar("")) for i in range(n))
        row = []
        for p in range(n):
            v = m.NewIntVar(0, 2000, "")
            prev = cum[-1][p] if cum else 0
            m.Add(v == prev + sum(lines[i] * x[t, i, p] for i in range(n)))
            row.append(v)
        cum.append(row)
    ranges = []
    for t in range(T):
        hi, lo = m.NewIntVar(0, 2000, ""), m.NewIntVar(0, 2000, "")
        m.AddMaxEquality(hi, cum[t])
        m.AddMinEquality(lo, cum[t])
        r = m.NewIntVar(0, 2000, "")
        m.Add(r == hi - lo)
        ranges.append(r)
    if objective == "final":
        m.Minimize(ranges[-1])
    else:
        z = m.NewIntVar(0, 2000, "")
        m.AddMaxEquality(z, ranges)
        m.Minimize(z)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 120
    s.parameters.num_workers = 8
    st = s.Solve(m)
    return {"status": s.StatusName(st), "objective": int(s.ObjectiveValue()),
            "bound": int(s.BestObjectiveBound()),
            "trajectory": [int(s.Value(r)) for r in ranges],
            "final_cum": [int(s.Value(v)) for v in cum[-1]]}


def week_with_shifts(days, games, alpha=None, base_cost=None, base_sour=None):
    m, hours, sour = fs.build(days, games, alpha, base_cost, base_sour)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = fs.TIME_LIMIT
    s.parameters.num_workers = 8
    st = s.Solve(m)
    S = fs.shifts()
    proto = m.Proto()
    shifts = {}  # (worker, day) -> (start, length)
    for idx, var in enumerate(proto.variables):
        if var.name.startswith("x") and s.Value(m.GetBoolVarFromProtoIndex(idx)):
            w, d, k = map(int, var.name[1:].split("_"))
            shifts[(w, d)] = S[k]
    return {"status": s.StatusName(st), "hours": int(s.Value(hours)),
            "sour": [int(s.Value(v)) for v in sour], "shifts": shifts}


def main():
    res = json.loads((HERE / "results.json").read_text())
    weeks = res["weeks"]
    out = {"offline": {}}
    for key in ("cost_opt", "fair_0.0"):
        for obj in ("final", "worst"):
            r = offline(weeks, key, obj)
            out["offline"][f"{key}/{obj}"] = r
            print(key, obj, r["status"], r["objective"], "bound", r["bound"], flush=True)

    games = {g["date"]: g["local_hour"] for g in fs.braves_home_games()}
    t = 23
    days = [fs.SEASON_START + dt.timedelta(days=7 * t + d) for d in range(7)]
    assert weeks[t]["week"] == days[0].isoformat()
    cheap = week_with_shifts(days, games)
    even = week_with_shifts(days, games, alpha=0.0, base_cost=cheap["hours"],
                            base_sour=sum(cheap["sour"]))
    print("week", days[0], "cheap", cheap["hours"], sorted(cheap["sour"]),
          "| stored", sorted(weeks[t]["cost_opt"]["sour"]))
    print("week", days[0], "even ", even["hours"], sorted(even["sour"]),
          "| stored", sorted(weeks[t]["fair_0.0"]["sour"]))
    nights = {}
    for d, day in enumerate(days):
        req = fs.demand(day, games.get(day.isoformat()))
        night = {"date": day.isoformat(), "game_hour": games.get(day.isoformat()),
                 "demand": [req[h] for h in fs.HOURS]}
        for name, sol in (("cheap", cheap), ("even", even)):
            night[name] = sorted(
                [{"worker": w, "start": s0, "len": L, "week_sour": sol["sour"][w]}
                 for (w, dd), (s0, L) in sol["shifts"].items() if dd == d],
                key=lambda r: (r["start"], r["len"]))
            night[name + "_coverage"] = [sum(1 for r in night[name]
                                             if r["start"] <= h < r["start"] + r["len"])
                                         for h in fs.HOURS]
        nights[day.isoformat()] = night
    out["week"] = {"start": days[0].isoformat(),
                   "cheap": {"hours": cheap["hours"], "sour": cheap["sour"]},
                   "even": {"hours": even["hours"], "sour": even["sour"]},
                   "nights": nights}
    (HERE / "extras.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
