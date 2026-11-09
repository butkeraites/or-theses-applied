"""When a pricing model says "infinite profit": bilevel pricing for a food distributor.

Applies the unboundedness analysis of Barbara Rodrigues's PhD thesis (Linear Bilevel
Optimization and Applications in the Grid Integration of Energy Storage, Edinburgh 2025;
Chapter 3, joint work with M. Carvalho, N. Sugishita and M. F. Anjos) to a setting the
authors did not study: a broadline food distributor in metro Atlanta setting its weekly
case prices (leader) while each restaurant account decides where to buy (follower):
delivered from the distributor, or hauled from a cash-and-carry store at shelf price plus
its own time and mileage, up to what its vehicle can carry.

Bilevel model (optimistic), for one "book" of N restaurants and K categories:

  leader:   max_p  sum_i sum_k (p_k - w_k) x_ik
  follower i: min  sum_k p_k x_ik + (q_k + h_i) z_ik [+ r_k u_ik]
              s.t. x_ik + z_ik [+ u_ik] >= d_ik          (dual lambda_ik)
                   sum_k z_ik <= V_i                      (dual mu_i)
                   z_ik = 0 for items the cash-and-carry does not stock
                   x, z, u >= 0
  [u = a competing distributor's delivered quote r_k; only in the "fixed" model]

Prices sit in the follower's objective, so I use the standard KKT reformulation: under
complementarity, strong duality gives sum_k p_k x_ik = sum_k lambda_ik d_ik - mu_i V_i
- sum_k (q_k+h_i) z_ik - sum_k r_k u_ik, which makes the leader objective linear. The
single-level problem is then a linear program with complementarity constraints (LPCC).

Unboundedness certificate (my adaptation of the thesis's model (U), Theorem 3.9): find a
point v in the LPCC and a direction D, -1 <= D <= 1, such that every point v + tD (t >= 0)
stays feasible with the SAME complementarity pattern (for each pair, one side and its
direction component are both zero) and the leader's profit grows along D. Profit growth
> 0 <=> the bilevel problem is unbounded. In the thesis the follower's dual vertex is
held fixed; here prices move the follower's dual too, so the dual variables get direction
components as well. Complementarity is modelled with indicator constraints (no big-M).

Compared against:
  * HPR (high-point relaxation: the leader also chooses what restaurants buy),
  * a per-item sanity check (is there an item some restaurant can get only from us?),
  * a structure-specific per-restaurant LP ("can this restaurant get by without us?"),
    which is exact for THIS model and serves as ground truth,
  * "cap and solve": bound prices at cap x cost and solve the bilevel model anyway.
Then, on the fixed model (competitor quote added): bilevel prices vs cost-plus 20% and
vs matching the competitor.

Data: BLS average retail prices, South urban (12-month mean, Sep 2025-Sep 2026);
BLS OEWS May 2025 median hourly wage, Atlanta MSA, SOC 35-1012; IRS 2026 business mileage
rate (76 cents a mile, Jul-Dec). Everything else (accounts, demand, vehicles, distances, markups, the specialty
item) is synthetic and labelled as such in README.md.

Usage: uv run --with gurobipy --with numpy python captive_pricing.py  -> results.json
(gurobipy ships a free size-limited license, enough for these models.)
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

HERE = Path(__file__).resolve().parent
SEED = 20261126
N_BOOKS = int(__import__("os").environ.get("N_BOOKS", 200))
N_REST = 8
EPS = 1e-6
CAPS = (1.5, 2.0, 3.0, 5.0, 10.0)

# ---- real inputs ---------------------------------------------------------------------
BLS_SERIES = {  # series id -> (name, unit, units per case)
    "APU0300FF1101": ("Chicken breast", "lb", 40),
    "APU0300703112": ("Ground beef", "lb", 40),
    "APU0300708111": ("Eggs", "doz", 15),
    "APU0300710212": ("Cheddar", "lb", 20),
    "APU0300701312": ("Rice", "lb", 50),
    "APU0300712311": ("Tomatoes", "lb", 25),
}
OEWS_SERIES = "OEUM001206000000035101208"  # Atlanta MSA, 35-1012, hourly median
# $/mile, IRS business standard mileage rate in effect Jul 1-Dec 31, 2026 (IR-2026-29;
# Jan 1-Jun 30 it was 72.5 cents, IR-2025-128). Source:
# https://www.irs.gov/tax-professionals/standard-mileage-rates
IRS_MILEAGE_2026 = 0.76

# ---- synthetic assumptions (stated in README) ----------------------------------------
CC_DISCOUNT = 0.85  # cash-and-carry shelf = 85% of supermarket average retail
DIST_COST = 0.80  # distributor's landed cost = 80% of cash-and-carry shelf
COMP_PREMIUM = 1.04  # competing distributor delivered quote = 104% of shelf
SPECIALTY = {"name": "House specialty", "w": 48.0}  # not stocked at cash-and-carry
BASE_CASES = [4, 3, 2, 2, 1, 2]  # weekly cases of a mid-size kitchen, per BLS item
COST_PLUS = 1.20


def bls_post(series):
    req = urllib.request.Request(
        "https://api.bls.gov/publicAPI/v1/timeseries/data/",
        data=json.dumps({"seriesid": series, "startyear": "2025", "endyear": "2026"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=60))


def load_inputs():
    cache = HERE / "inputs.json"
    if cache.exists():
        return json.loads(cache.read_text())
    raw = bls_post(list(BLS_SERIES))
    prices = {}
    for s in raw["Results"]["series"]:
        pts = sorted(
            ((d["year"], d["period"], float(d["value"])) for d in s["data"] if d["value"] != "-"),
            reverse=True,
        )[:12]
        prices[s["seriesID"]] = {
            "mean_12m": round(sum(v for *_, v in pts) / len(pts), 4),
            "window": f"{pts[-1][1]}-{pts[-1][0]} to {pts[0][1]}-{pts[0][0]}",
        }
    wage = bls_post([OEWS_SERIES])["Results"]["series"][0]["data"][0]
    inp = {
        "bls_prices": prices,
        "oews_wage": {"series": OEWS_SERIES, "year": wage["year"], "value": float(wage["value"])},
        "irs_mileage": IRS_MILEAGE_2026,
    }
    cache.write_text(json.dumps(inp, indent=1))
    return inp


def make_items(inp):
    items = []
    for sid, (name, unit, per_case) in BLS_SERIES.items():
        retail = inp["bls_prices"][sid]["mean_12m"]
        q = round(retail * per_case * CC_DISCOUNT, 2)
        items.append(
            {
                "name": name,
                "series": sid,
                "case": f"{per_case} {unit}",
                "retail_unit": retail,
                "q": q,
                "w": round(q * DIST_COST, 2),
                "r": round(q * COMP_PREMIUM, 2),
            }
        )
    w = SPECIALTY["w"]
    items.append(
        {"name": SPECIALTY["name"], "series": None, "case": "1 case", "q": None, "w": w,
         "r": round(w / DIST_COST * COMP_PREMIUM, 2)}
    )
    return items


def make_book(rng, items, wage, mileage):
    """One distributor 'book': N restaurant accounts (synthetic)."""
    K = len(items)
    rests = []
    for _ in range(N_REST):
        size = rng.uniform(0.6, 2.0)
        d = [
            int(round(b * size)) if rng.random() < 0.85 else 0 for b in BASE_CASES
        ]
        d.append(int(rng.integers(1, 3)) if rng.random() < 0.025 else 0)  # specialty
        vehicle = rng.choice(["car", "SUV", "van"], p=[0.3, 0.4, 0.3])
        per_trip = {"car": 10, "SUV": 18, "van": 40}[vehicle]
        trips = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
        miles = rng.uniform(4, 20)  # one way to the nearest cash-and-carry
        hours = rng.uniform(1.5, 2.5)  # door to door, including shopping
        h = (2 * miles * mileage + hours * wage) / per_trip
        rests.append(
            {"d": d, "V": per_trip * trips, "h": round(h, 2), "vehicle": str(vehicle),
             "trips": trips, "miles": round(miles, 1)}
        )
    assert len(rests[0]["d"]) == K
    return rests


# ---- the LPCC ------------------------------------------------------------------------
def env():
    e = gp.Env(empty=True)
    e.setParam("OutputFlag", 0)
    e.start()
    return e


ENV = None


def build(items, rests, competitor, direction=False, cap=None, extra_bounds=None):
    """Single-level KKT reformulation. Returns (model, point vars, direction vars, objs)."""
    m = gp.Model(env=ENV)
    m.Params.Threads = 4
    m.Params.TimeLimit = 120
    K = len(items)
    cc = [k for k in range(K) if items[k]["q"] is not None]
    P = {}  # point variables, by name
    D = {}  # direction variables
    pairs = []  # (name_a, name_b) complementary nonnegative pairs

    def var(name, lb=0.0, ub=GRB.INFINITY):
        P[name] = m.addVar(lb=lb, ub=ub, name=name)
        if direction:
            D[name] = m.addVar(lb=0.0 if lb == 0.0 else -1.0, ub=1.0, name="D" + name)
        return P[name]

    for k in range(K):
        var(f"p{k}", ub=cap * items[k]["w"] if cap else GRB.INFINITY)

    lin = []  # linear equalities as dicts {name: coef}, rhs (point); direction rhs 0
    for i, R in enumerate(rests):
        for k in range(K):
            var(f"x{i}_{k}")
            if k in cc:
                var(f"z{i}_{k}")
            if competitor:
                var(f"u{i}_{k}")
            var(f"l{i}_{k}")
            var(f"s{i}_{k}")
        var(f"mu{i}")
        var(f"t{i}")
        for k in range(K):
            # demand slack: s = x + z + u - d
            e = {f"s{i}_{k}": 1.0, f"x{i}_{k}": -1.0}
            if k in cc:
                e[f"z{i}_{k}"] = -1.0
            if competitor:
                e[f"u{i}_{k}"] = -1.0
            lin.append((e, -R["d"][k]))
            pairs.append((f"l{i}_{k}", f"s{i}_{k}"))
            # reduced cost of x: p_k - lambda_ik >= 0
            var(f"rx{i}_{k}")
            lin.append(({f"rx{i}_{k}": 1.0, f"p{k}": -1.0, f"l{i}_{k}": 1.0}, 0.0))
            pairs.append((f"x{i}_{k}", f"rx{i}_{k}"))
            if k in cc:  # reduced cost of z: q_k + h_i - lambda_ik + mu_i >= 0
                var(f"rz{i}_{k}")
                lin.append(
                    ({f"rz{i}_{k}": 1.0, f"l{i}_{k}": 1.0, f"mu{i}": -1.0},
                     items[k]["q"] + R["h"])
                )
                pairs.append((f"z{i}_{k}", f"rz{i}_{k}"))
            if competitor:  # reduced cost of u: r_k - lambda_ik >= 0
                var(f"ru{i}_{k}")
                lin.append(({f"ru{i}_{k}": 1.0, f"l{i}_{k}": 1.0}, items[k]["r"]))
                pairs.append((f"u{i}_{k}", f"ru{i}_{k}"))
        # capacity slack: t = V - sum z
        e = {f"t{i}": 1.0}
        e.update({f"z{i}_{k}": 1.0 for k in cc})
        lin.append((e, R["V"]))
        pairs.append((f"mu{i}", f"t{i}"))

    for e, rhs in lin:
        m.addConstr(gp.quicksum(c * P[n] for n, c in e.items()) == rhs)
        if direction:
            m.addConstr(gp.quicksum(c * D[n] for n, c in e.items()) == 0)
    for a, b in pairs:
        y = m.addVar(vtype=GRB.BINARY)
        m.addGenConstrIndicator(y, True, P[a] == 0)
        m.addGenConstrIndicator(y, False, P[b] == 0)
        if direction:
            m.addGenConstrIndicator(y, True, D[a] == 0)
            m.addGenConstrIndicator(y, False, D[b] == 0)

    def profit(V):  # leader profit, linearised via strong duality
        terms = []
        for i, R in enumerate(rests):
            for k in range(K):
                terms.append(R["d"][k] * V[f"l{i}_{k}"])
                terms.append(-items[k]["w"] * V[f"x{i}_{k}"])
                if k in cc:
                    terms.append(-(items[k]["q"] + R["h"]) * V[f"z{i}_{k}"])
                if competitor:
                    terms.append(-items[k]["r"] * V[f"u{i}_{k}"])
            terms.append(-R["V"] * V[f"mu{i}"])
        return gp.quicksum(terms)

    return m, P, D, profit


def certificate(items, rests, competitor):
    t0 = time.perf_counter()
    m, P, D, profit = build(items, rests, competitor, direction=True)
    m.setObjective(profit(D), GRB.MAXIMIZE)
    m.optimize()
    dt = time.perf_counter() - t0
    if m.Status == GRB.INFEASIBLE:
        return {"status": "infeasible", "secs": dt}
    assert m.Status == GRB.OPTIMAL, m.Status
    unb = m.ObjVal > EPS
    out = {"status": "unbounded" if unb else "bounded", "gain_per_step": m.ObjVal, "secs": dt}
    if unb:
        K = len(items)
        out["dp"] = [D[f"p{k}"].X for k in range(K)]
        out["p"] = [P[f"p{k}"].X for k in range(K)]
        # captive restaurants: their dual price (willingness to pay) rises along the ray
        cap = []
        for i in range(len(rests)):
            dl = [D[f"l{i}_{k}"].X for k in range(K)]
            if sum(dl[k] * rests[i]["d"][k] for k in range(K)) > EPS:
                cap.append(i)
        out["captive"] = cap
        out["point"] = {n: v.X for n, v in P.items()}
        out["dir"] = {n: v.X for n, v in D.items()}
    return out


def solve_bilevel(items, rests, competitor, cap=None):
    m, P, _, profit = build(items, rests, competitor, cap=cap)
    m.setObjective(profit(P), GRB.MAXIMIZE)
    m.Params.MIPGap = 1e-6
    m.optimize()
    assert m.Status == GRB.OPTIMAL, m.Status
    K = len(items)
    return {"profit": m.ObjVal, "p": [P[f"p{k}"].X for k in range(K)], "gap": m.MIPGap}


# ---- follower response at given prices (independent check) --------------------------
def follower(items, R, p, competitor, tiebreak="optimistic"):
    m = gp.Model(env=ENV)
    K = len(items)
    cc = [k for k in range(K) if items[k]["q"] is not None]
    x = m.addVars(K, ub=R["d"])  # nobody buys more than the kitchen needs
    z = m.addVars(cc)
    u = m.addVars(K) if competitor else None
    for k in range(K):
        m.addConstr(
            x[k] + (z[k] if k in cc else 0) + (u[k] if competitor else 0) >= R["d"][k]
        )
    m.addConstr(gp.quicksum(z[k] for k in cc) <= R["V"])
    cost = gp.quicksum(p[k] * x[k] for k in range(K)) + gp.quicksum(
        (items[k]["q"] + R["h"]) * z[k] for k in cc
    )
    if competitor:
        cost += gp.quicksum(items[k]["r"] * u[k] for k in range(K))
    m.setObjective(cost, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None
    best = m.ObjVal
    if tiebreak:  # among cost-optimal plans, the best (optimistic) or worst for the leader
        m.addConstr(cost <= best + 1e-6 * max(1.0, abs(best)))
        m.setObjective(gp.quicksum((p[k] - items[k]["w"]) * x[k] for k in range(K)),
                       GRB.MAXIMIZE if tiebreak == "optimistic" else GRB.MINIMIZE)
        m.optimize()
    xs = [x[k].X for k in range(K)]
    return {
        "cost": best,
        "x": xs,
        "profit": sum((p[k] - items[k]["w"]) * xs[k] for k in range(K)),
        "dist_cases": sum(xs),
        "cases": sum(R["d"]),
    }


def evaluate(items, rests, p, competitor, tiebreak="optimistic"):
    res = [follower(items, R, p, competitor, tiebreak) for R in rests]
    if any(r is None for r in res):
        return None
    return {
        "profit": sum(r["profit"] for r in res),
        "rest_cost": sum(r["cost"] for r in res),
        "share": sum(r["dist_cases"] for r in res) / sum(r["cases"] for r in res),
        "accounts_buying": sum(1 for r in res if r["dist_cases"] > EPS),
    }


def can_leave(items, R, competitor):
    """Ground truth for this model: can the restaurant meet demand without us?"""
    m = gp.Model(env=ENV)
    K = len(items)
    cc = [k for k in range(K) if items[k]["q"] is not None]
    z = m.addVars(cc)
    u = m.addVars(K) if competitor else None
    for k in range(K):
        m.addConstr((z[k] if k in cc else 0) + (u[k] if competitor else 0) >= R["d"][k])
    m.addConstr(gp.quicksum(z[k] for k in cc) <= R["V"])
    m.optimize()
    return m.Status == GRB.OPTIMAL


def per_item_flag(items, rests, competitor):
    """Sanity check a practitioner might run: an item someone needs that only we sell."""
    if competitor:
        return False
    return any(
        R["d"][k] > 0 and items[k]["q"] is None for R in rests for k in range(len(items))
    )


def walk_ray(items, rests, cert, steps=(1, 10, 100)):
    """Independent check: move prices along the certificate and re-solve each follower."""
    K = len(items)
    out = []
    for t in steps:
        p = [cert["p"][k] + t * cert["dp"][k] for k in range(K)]
        ev = evaluate(items, rests, p, competitor=False)
        out.append({"t": t, "profit": ev["profit"] if ev else None})
    return out


def main():
    global ENV
    ENV = env()
    inp = load_inputs()
    items = make_items(inp)
    wage = inp["oews_wage"]["value"]
    rng = np.random.default_rng(SEED)
    books = [make_book(rng, items, wage, IRS_MILEAGE_2026) for _ in range(N_BOOKS)]

    rows = []
    t_all = time.perf_counter()
    for b, rests in enumerate(books):
        row = {"book": b}
        # ---- broken model: no competitor quote ----
        truth = any(not can_leave(items, R, False) for R in rests)
        row["truth_unbounded"] = truth
        row["cause"] = (
            "bounded" if not truth
            else "specialty" if per_item_flag(items, rests, False)
            else "capacity"
        )
        row["per_item_flag"] = per_item_flag(items, rests, False)
        row["hpr_unbounded"] = any(sum(R["d"]) > 0 for R in rests)  # x = d, p -> inf
        t0 = time.perf_counter()
        _ = [can_leave(items, R, False) for R in rests]
        row["truth_secs"] = time.perf_counter() - t0
        c = certificate(items, rests, competitor=False)
        row["cert"] = c["status"]
        row["cert_secs"] = c["secs"]
        if c["status"] == "unbounded":
            row["captive_found"] = c["captive"]
            row["captive_true"] = [
                i for i, R in enumerate(rests) if not can_leave(items, R, False)
            ]
            row["ray"] = walk_ray(items, rests, c)
            row["cert_dp"] = c["dp"]
            row["cert_p"] = c["p"]
        caps = {str(cp): solve_bilevel(items, rests, False, cap=cp) for cp in CAPS}
        row["cap_profit"] = {cp: v["profit"] for cp, v in caps.items()}
        row["cap_p"] = {cp: v["p"] for cp, v in caps.items()}
        # ---- fixed model: competitor quote available ----
        cf = certificate(items, rests, competitor=True)
        row["fixed_cert"] = cf["status"]
        row["fixed_cert_secs"] = cf["secs"]
        opt = solve_bilevel(items, rests, True)
        row["bilevel_p"] = opt["p"]
        ev_opt = evaluate(items, rests, opt["p"], True)
        ev_cp = evaluate(items, rests, [COST_PLUS * it["w"] for it in items], True)
        ev_match = evaluate(items, rests, [it["r"] for it in items], True)
        ev_pess = evaluate(items, rests, opt["p"], True, tiebreak="pessimistic")
        ev_cush = evaluate(items, rests, [0.99 * v for v in opt["p"]], True,
                           tiebreak="pessimistic")
        row["fixed"] = {"bilevel": ev_opt, "cost_plus": ev_cp, "match": ev_match,
                        "bilevel_pessimistic": ev_pess, "bilevel_cushion_1pct": ev_cush,
                        "bilevel_model_profit": opt["profit"]}
        rows.append(row)
        if b % 20 == 0:
            print(f"book {b}: {row['cause']}, cert={row['cert']}, "
                  f"profit bilevel={ev_opt['profit']:.0f} cp={ev_cp['profit']:.0f} "
                  f"match={ev_match['profit']:.0f}", flush=True)

    # ---- summary ----
    def mean(xs):
        return float(np.mean(xs)) if xs else None

    causes = {c: sum(r["cause"] == c for r in rows) for c in ("bounded", "specialty", "capacity")}
    unb = [r for r in rows if r["truth_unbounded"]]
    agree = sum((r["cert"] == "unbounded") == r["truth_unbounded"] for r in rows)
    captive_agree = sum(set(r["captive_found"]) == set(r["captive_true"]) for r in unb
                        if r["cert"] == "unbounded")
    per_item_caught = sum(r["per_item_flag"] for r in unb)
    ray_ok = sum(
        all(a["profit"] is not None for a in r["ray"])
        and r["ray"][0]["profit"] < r["ray"][1]["profit"] < r["ray"][2]["profit"]
        for r in unb
    )
    cap_growth = {
        grp: {
            str(cp): mean([r["cap_profit"][str(cp)] for r in rows if r["cause"] in grp_set])
            for cp in CAPS
        }
        for grp, grp_set in (("unbounded", {"specialty", "capacity"}), ("bounded", {"bounded"}))
    }
    fx = {
        k: {
            "profit": mean([r["fixed"][k]["profit"] for r in rows]),
            "rest_cost": mean([r["fixed"][k]["rest_cost"] for r in rows]),
            "share": mean([r["fixed"][k]["share"] for r in rows]),
            "accounts_buying": mean([r["fixed"][k]["accounts_buying"] for r in rows]),
        }
        for k in ("bilevel", "cost_plus", "match", "bilevel_pessimistic",
                  "bilevel_cushion_1pct")
    }
    beats_cp = sum(
        r["fixed"]["bilevel"]["profit"] > r["fixed"]["cost_plus"]["profit"] + 0.01 for r in rows
    )
    cush_beats_cp = sum(
        r["fixed"]["bilevel_cushion_1pct"]["profit"] > r["fixed"]["cost_plus"]["profit"] + 0.01
        for r in rows
    )
    beats_match = sum(
        r["fixed"]["bilevel"]["profit"] > r["fixed"]["match"]["profit"] + 0.01 for r in rows
    )
    ties_match = sum(
        abs(r["fixed"]["bilevel"]["profit"] - r["fixed"]["match"]["profit"]) <= 0.01
        for r in rows
    )
    model_vs_eval = max(
        abs(r["fixed"]["bilevel_model_profit"] - r["fixed"]["bilevel"]["profit"]) for r in rows
    )
    w = [it["w"] for it in items]

    def at_cap(r, cp):
        return any(abs(p - cp * w[k]) < 1e-3 for k, p in enumerate(r["cap_p"][str(cp)]))

    priced_at_cap = {
        cause: {str(cp): sum(at_cap(r, cp) for r in rows if r["cause"] == cause) for cp in CAPS}
        for cause in ("bounded", "specialty", "capacity")
    }
    summary = {
        "books": N_BOOKS,
        "restaurants_per_book": N_REST,
        "causes": causes,
        "cert_agrees_with_truth": agree,
        "cert_captive_matches_truth": captive_agree,
        "unbounded_books": len(unb),
        "per_item_check_caught": per_item_caught,
        "hpr_unbounded_books": sum(r["hpr_unbounded"] for r in rows),
        "ray_profit_increasing": ray_ok,
        "cert_secs_mean": mean([r["cert_secs"] for r in rows]),
        "cert_secs_max": max(r["cert_secs"] for r in rows),
        "truth_secs_mean": mean([r["truth_secs"] for r in rows]),
        "cap_profit_mean": cap_growth,
        "books_priced_at_cap": priced_at_cap,
        "fixed_cert": {s: sum(r["fixed_cert"] == s for r in rows)
                       for s in ("bounded", "unbounded", "infeasible")},
        "fixed_cert_secs_mean": mean([r["fixed_cert_secs"] for r in rows]),
        "fixed": fx,
        "bilevel_beats_match": beats_match,
        "bilevel_beats_cost_plus": beats_cp,
        "cushion_beats_cost_plus": cush_beats_cp,
        "bilevel_ties_match": ties_match,
        "max_abs_model_vs_follower_profit": model_vs_eval,
        "runtime_secs": time.perf_counter() - t_all,
    }
    out = {"inputs": inp, "items": items, "assumptions": {
        "cc_discount": CC_DISCOUNT, "dist_cost": DIST_COST, "comp_premium": COMP_PREMIUM,
        "cost_plus": COST_PLUS, "base_cases": BASE_CASES, "seed": SEED},
        "summary": summary, "books": rows, "example_books": {}}
    # keep two example books in full for the figures
    for cause in ("capacity", "specialty", "bounded"):
        r = next((r for r in rows if r["cause"] == cause), None)
        if r is not None:
            out["example_books"][cause] = {"book": r["book"], "restaurants": books[r["book"]]}
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
