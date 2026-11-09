"""Adds a `headline` block to results.json: the derived numbers quoted in the article, computed
from the per-setting results so every number in the post can be traced to this file.

Run after pharmacy_dual_sourcing.py:  python3 headline.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "results.json")
r = json.load(open(P))
S = r["settings"]
summ = r["summary"]
NAMES = ["BSP", "PAR", "DIP", "DCL", "EPL"]


def tot(view, group="seen", lead=None):
    recs = [
        v
        for v in S.values()
        if v["setting"]["seen"] == (group == "seen") and (lead is None or v["setting"]["lead"] == lead)
    ]
    return {n: sum(v[view][n]["cost_per_day"] for v in recs) for n in NAMES if n in recs[0][view]}


def pct(a, b):  # how much lower a is than b, in percent
    return round(100 * (1 - a / b), 1)


h = {}
st, rs, rr = tot("stationary"), tot("replay_static"), tot("replay_rolling")
h["n_settings_seen"] = summ["seen"]["n_settings"]
h["n_settings_unseen"] = summ["unseen"]["n_settings"]
h["stationary"] = {
    "rush_rule_DIP_vs_BSP_pct": pct(st["DIP"], st["BSP"]),
    "DCL_worse_than_DIP_pct": -pct(st["DCL"], st["DIP"]),
    "EPL_worse_than_DIP_pct": -pct(st["EPL"], st["DIP"]),
    "DCL_beats_DIP_by_1pct_settings": summ["seen"]["stationary"]["beats_DIP_by_1pct"]["DCL"],
    "DCL_loses_to_DIP_by_1pct_settings": summ["seen"]["stationary"]["loses_to_DIP_by_1pct"]["DCL"],
    "EPL_beats_DIP_by_1pct_settings": summ["seen"]["stationary"]["beats_DIP_by_1pct"]["EPL"],
    "EPL_loses_to_DIP_by_1pct_settings": summ["seen"]["stationary"]["loses_to_DIP_by_1pct"]["EPL"],
    "DCL_median_vs_DIP_pct": round(100 * summ["seen"]["stationary"]["median_saving_vs_DIP"]["DCL"], 1),
    "fill_rate_DIP": round(100 * summ["seen"]["stationary"]["mean_fill_rate"]["DIP"], 1),
}
h["replay_static"] = {
    "PAR_vs_BSP_pct": pct(rs["PAR"], rs["BSP"]),
    "DIP_vs_BSP_pct": pct(rs["DIP"], rs["BSP"]),
    "DCL_vs_BSP_pct": pct(rs["DCL"], rs["BSP"]),
    "EPL_vs_BSP_pct": pct(rs["EPL"], rs["BSP"]),
}
h["replay_rolling"] = {
    "PAR_vs_BSP_pct": pct(rr["PAR"], rr["BSP"]),
    "DIP_vs_BSP_pct": pct(rr["DIP"], rr["BSP"]),
    "EPL_vs_BSP_pct": pct(rr["EPL"], rr["BSP"]),
    "EPL_vs_PAR_pct": pct(rr["EPL"], rr["PAR"]),
    "DCL_frozen_vs_BSP_pct": pct(rr["DCL"], rr["BSP"]),
    "rush_orders_per_30d_PAR": round(summ["seen"]["replay_rolling"]["mean_rush_orders_per_30d"]["PAR"], 2),
    "fill_rate_BSP": round(100 * summ["seen"]["replay_rolling"]["mean_fill_rate"]["BSP"], 1),
    "fill_rate_PAR": round(100 * summ["seen"]["replay_rolling"]["mean_fill_rate"]["PAR"], 1),
}
h["replanning"] = {
    "BSP_rolling_vs_static_pct": pct(rr["BSP"], rs["BSP"]),
    "PAR_rolling_vs_static_pct": pct(rr["PAR"], rs["PAR"]),
    "EPL_rolling_vs_static_pct": pct(rr["EPL"], rs["EPL"]),
    "PAR_rolling_vs_BSP_static_pct": pct(rr["PAR"], rs["BSP"]),
}
h["by_lead"] = {}
for L in r["meta"]["grid"]["train_leads"]:
    a, b, c = tot("stationary", lead=L), tot("replay_static", lead=L), tot("replay_rolling", lead=L)
    h["by_lead"][str(L)] = {
        "stationary_DIP_vs_BSP_pct": pct(a["DIP"], a["BSP"]),
        "static_PAR_vs_BSP_pct": pct(b["PAR"], b["BSP"]),
        "rolling_PAR_vs_BSP_pct": pct(c["PAR"], c["BSP"]),
        "stationary_DCL_vs_DIP_pct": pct(a["DCL"], a["DIP"]),
    }
ut_st, ut_rr = tot("stationary", "unseen"), tot("replay_rolling", "unseen")
h["unseen"] = {
    "stationary_EPL_vs_DIP_pct": pct(ut_st["EPL"], ut_st["DIP"]),
    "rolling_EPL_vs_BSP_pct": pct(ut_rr["EPL"], ut_rr["BSP"]),
    "rolling_PAR_vs_BSP_pct": pct(ut_rr["PAR"], ut_rr["BSP"]),
}
cats = r["meta"]["categories"]
h["demand_shift_pct"] = {c: round(100 * (v["mu_test"] / v["mu_train"] - 1), 1) for c, v in cats.items()}
h["totals_cost_per_day_seen"] = {"stationary": st, "replay_static": rs, "replay_rolling": rr}
ex = S["R06|60|5"]
h["example_R06_60_5"] = {
    v: {n: {"cost_per_day": round(ex[v][n]["cost_per_day"], 2), "fill_rate": round(100 * ex[v][n]["fill_rate"], 1),
            "rush_orders_per_30d": round(ex[v][n]["rush_orders_per_30d"], 2)}
        for n in ex[v]}
    for v in ["stationary", "replay_static", "replay_rolling"]
}
h["epl_training_secs"] = round(sum(x["secs"] for x in r["meta"]["epl_hist"]), 0)
h["dcl_training_secs_total"] = round(sum(sum(x["secs"] for x in v["dcl_hist"]) for v in S.values() if "dcl_hist" in v), 0)
h["index_vs_frozen_BSP"] = {
    "frozen": {n: round(100 * rs[n] / rs["BSP"], 1) for n in rs},
    "replanned": {n: round(100 * rr[n] / rs["BSP"], 1) for n in rr},
}
h["stationary_pct_above_DIP"] = {n: round(100 * (st[n] / st["DIP"] - 1), 1) for n in st}
R1 = os.path.join(HERE, "results_run1_textbook_start.json")
if os.path.exists(R1):  # earlier run: DCL/EPL started from the untuned textbook rule
    t1 = json.load(open(R1))["summary"]["seen"]["stationary"]["total_cost_per_day"]
    h["first_run_textbook_start_stationary_pct_above_DIP"] = {
        n: round(100 * (t1[n] / t1["DIP"] - 1), 1) for n in t1
    }
r["headline"] = h
json.dump(r, open(P, "w"), indent=1)
print(json.dumps(h, indent=1))
