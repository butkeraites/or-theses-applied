"""Where should a growing café brand open in Cobb County?  Bakker's decision-space analysis
of the capacitated facility location problem (CFLP), applied to small-business siting.

Thesis: H. Bakker, "On the interplay between data and decisions in discrete location
problems", KIT 2024 (EURO Doctoral Dissertation Award 2025 finalist). Ideas used:
  * Ch. 1, eq. (1.1)-(1.5): the profit-maximizing CFLP.
  * Ch. 2.2: k best alternative solutions via no-good cuts (2.7); persistence C(i,S)
    (Def. 2.5); kernel persistence KER(S) (Def. 2.4) vs a random counterpart E(KER) (2.9).
  * Ch. 2.3: pseudo-reduced costs prc% (Def. 2.6); dependence sets Dep(i)/Rep(i) (2.25-2.26);
    average dependence density Dep_avg (Def. 2.13); weakly connected components.
  * Ch. 5.2.1: the greedy ADD heuristic (Algorithm 3), here with exact LP allocation.
  * Ch. 7 (simplified): deterministic optimum per demand scenario (S_det), a two-stage
    stochastic program (sample average), and the expected-value solution, compared
    out of sample.
Everything below the data layer is my application, not the thesis's.

Run:  uv run --with numpy --with scipy python siting.py      (writes results.json)
"""

from __future__ import annotations

import csv
import json
import math
import os
import time

import numpy as np
from scipy.optimize import LinearConstraint, milp, linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
RNG = np.random.default_rng(20261124)

# ---------------------------------------------------------------- assumptions (synthetic)
MARGIN = 5.0  # $ contribution margin per visit
PENALTY = 0.75  # $ per visit per driving mile (stand-in for lower visit frequency far away)
RATE = 0.10  # addressable visits per household per week, at the county median income
INC_ELASTICITY = 0.5  # visits scale with (income / county median) ** 0.5
FORMATS = {
    # weekly capacity (visits) and base weekly fixed cost ($: rent, base labor, overhead)
    "cafe": dict(Q=4000, F=8500, label="Full café (4,000 visits/wk)"),
    "counter": dict(Q=1200, F=2600, label="Small counter (1,200 visits/wk)"),
}
K_BEST = 20
N_SCEN_TRAIN = 30
N_SCEN_TEST = 300
PERTURB = 0.20  # thesis Ch. 7 uses a perturbation level p = 20%


# ---------------------------------------------------------------- data
def load():
    with open(os.path.join(D, "tracts.csv")) as f:
        tr = list(csv.DictReader(f))
    with open(os.path.join(D, "candidates.csv")) as f:
        ca = list(csv.DictReader(f))
    mm = json.load(open(os.path.join(D, "drive_miles.json")))
    assert mm["tracts"] == [t["geoid"] for t in tr] and mm["sites"] == [c["id"] for c in ca]
    miles = np.array(mm["miles"]).T  # sites x tracts
    hh = np.array([float(t["households"]) for t in tr])
    hh_se = np.array([float(t["households_moe"]) for t in tr]) / 1.645
    inc = np.array([float(t["income"]) for t in tr])
    inc_se = np.array([float(t["income_moe"]) for t in tr]) / 1.645
    return tr, ca, miles, hh, hh_se, inc, inc_se


def demand(hh, inc, med):
    return RATE * hh * (inc / med) ** INC_ELASTICITY


class Instance:
    def __init__(self, miles, Dem, Q, F):
        self.m, self.n = miles.shape
        unit = MARGIN - PENALTY * miles
        ii, jj = np.nonzero(unit > 0)
        self.ai, self.aj, self.unit = ii, jj, unit[ii, jj]
        self.A = len(ii)
        self.Q = np.asarray(Q, float)
        self.F = np.asarray(F, float)
        self.set_demand(Dem)
        # static constraint blocks
        a = np.arange(self.A)
        self.Acust = coo_matrix((np.ones(self.A), (jj, a)), shape=(self.n, self.A)).tocsr()
        self.Alink = coo_matrix((np.ones(self.A), (a, ii)), shape=(self.A, self.m)).tocsr()

    def set_demand(self, Dem):
        self.Dem = np.asarray(Dem, float)
        self.w = self.unit * self.Dem[self.aj]  # profit of serving the whole arc


def solve(inst, forbid=(), nogoods=(), fix_open=None, time_limit=600):
    """Profit-max CFLP (1.1)-(1.5) + x_ij <= y_i. Returns (profit, open set, x)."""
    m, A = inst.m, inst.A
    c = np.concatenate([inst.F, -inst.w])
    # capacity: sum_j D_j x_ij - Q_i y_i <= 0
    Acap = coo_matrix(
        (np.concatenate([inst.Dem[inst.aj], -inst.Q]),
         (np.concatenate([inst.ai, np.arange(m)]), np.concatenate([m + np.arange(A), np.arange(m)]))),
        shape=(m, m + A),
    ).tocsr()
    from scipy.sparse import hstack, identity

    Acust = hstack([csr_matrix((inst.n, m)), inst.Acust]).tocsr()
    Alink = hstack([-inst.Alink, identity(A, format="csr")]).tocsr()
    rows = [Acap, Acust, Alink]
    lo = [np.full(m, -np.inf), np.full(inst.n, -np.inf), np.full(A, -np.inf)]
    hi = [np.zeros(m), np.ones(inst.n), np.zeros(A)]
    for O in nogoods:  # eq. (2.7)
        coef = np.zeros(m + A)
        Ol = list(O)
        coef[:m] = -1.0
        coef[Ol] = 1.0
        rows.append(csr_matrix(coef))
        lo.append(np.array([-np.inf]))
        hi.append(np.array([len(Ol) - 1.0]))
    ub = np.ones(m + A)
    lb = np.zeros(m + A)
    for i in forbid:
        ub[i] = 0
    if fix_open is not None:
        ub[:m] = 0
        for i in fix_open:
            lb[i] = ub[i] = 1
    integ = np.concatenate([np.ones(m), np.zeros(A)])
    res = milp(
        c,
        constraints=LinearConstraint(vstack(rows).tocsr(), np.concatenate(lo), np.concatenate(hi)),
        integrality=integ if fix_open is None else np.zeros(m + A),
        bounds=(lb, ub),
        options=dict(time_limit=time_limit, mip_rel_gap=1e-6),
    )
    assert res.status == 0, res.message
    y = res.x[:m]
    return -res.fun, frozenset(int(i) for i in np.nonzero(y > 0.5)[0]), res.x[m:]


def plan_detail(inst, O, sites):
    """Per-site weekly visits served and contribution (margin after distance penalty and
    fixed cost) for a fixed plan, with optimal allocation."""
    z, _, x = solve(inst, fix_open=O)
    out = {}
    for i in sorted(O):
        a = inst.ai == i
        visits = float((inst.Dem[inst.aj[a]] * x[a]).sum())
        contrib = float((inst.w[a] * x[a]).sum() - inst.F[i])
        out[sites[i]] = dict(visits=round(visits, 1), utilization=round(visits / inst.Q[i], 3),
                             contribution=round(contrib, 1))
    return out


def evaluate(inst, O):
    """Profit of a fixed set of open sites with optimal allocation (an LP)."""
    if not O:
        return 0.0
    return solve(inst, fix_open=O)[0]


# ---------------------------------------------------------------- thesis measures
def kernel_persistence(S, m):
    """KER(S), Def. 2.4 via eq. (2.6): sum of open decisions / (|S| * |I^R(S)|)."""
    IR = set().union(*S)
    return sum(len(s) for s in S) / (len(S) * len(IR))


def expected_kernel_persistence(S, m, draws=20000):
    """Random counterpart (eq. 2.8-2.9) by Monte Carlo, as in the thesis's Appendix A."""
    n = round(np.mean([len(s) for s in S]))
    tot = 0.0
    for _ in range(draws):
        IR = set()
        for _ in range(len(S)):
            IR.update(RNG.choice(m, size=n, replace=False).tolist())
        tot += len(IR)
    return (n * len(S)) / (len(S) * (tot / draws))


def dependence(inst, z_star, O_star):
    """prc%(i), Dep(i), Rep(i) for every optimal facility (Defs 2.6, 2.9-2.13)."""
    prc, dep, rep = {}, {}, {}
    for i in sorted(O_star):
        z_i, O_i, _ = solve(inst, forbid=[i])
        prc[i] = (z_star - z_i) / abs(z_star)
        dep[i] = sorted(set(O_star) - set(O_i) - {i})
        rep[i] = sorted(set(O_i) - set(O_star))
    n = len(O_star)
    edges = sum(len(v) for v in dep.values())
    dep_avg = edges / (n * (n - 1)) if n > 1 else 0.0
    # weakly connected components of G_Dep
    parent = {i: i for i in O_star}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, ds in dep.items():
        for j in ds:
            parent[find(i)] = find(j)
    comps = {}
    for i in O_star:
        comps.setdefault(find(i), []).append(i)
    W = len(comps)
    psi = (n / (n - 1)) * (W / n - 1 / n) if n > 1 else 1.0
    independent = [i for i in O_star if not any(i in v for v in dep.values())]
    return dict(prc=prc, dep=dep, rep=rep, dep_avg=dep_avg, components=sorted(
        (sorted(v) for v in comps.values()), key=len, reverse=True), psi=psi,
        independent=sorted(independent))


# ---------------------------------------------------------------- baselines
def add_greedy(inst):
    """ADD (thesis Algorithm 3; Jacobsen 1983), with exact LP allocation per evaluation."""
    O, z = set(), 0.0
    order = []
    while True:
        best, bi = z, None
        for i in range(inst.m):
            if i in O:
                continue
            zi = evaluate(inst, O | {i})
            if zi > best + 1e-6:
                best, bi = zi, i
        if bi is None:
            return z, frozenset(O), order
        O.add(bi)
        order.append(bi)
        z = best


def trade_area_rule(inst, miles, hh, n_open, radius=3.0, spacing=2.0):
    """What a practitioner without a solver does: rank sites by households within a
    3-mile drive, take the top ones, skipping sites within 2 miles of one already taken."""
    score = (miles <= radius).astype(float) @ hh
    order = np.argsort(-score)
    picked = []
    for i in order:
        if all(miles_site(i, k) > spacing for k in picked):
            picked.append(int(i))
        if len(picked) == n_open:
            break
    return frozenset(picked), score


_SITE_XY = None


def miles_site(a, b):
    (la1, lo1), (la2, lo2) = _SITE_XY[a], _SITE_XY[b]
    p = math.pi / 180
    h = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin(
        (lo2 - lo1) * p / 2
    ) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h)) * 1.3  # straight line x 1.3 circuity


# ---------------------------------------------------------------- uncertainty (Ch. 7)
def scenarios(hh, hh_se, inc, inc_se, med, n):
    out = []
    for _ in range(n):
        h = np.maximum(0, RNG.normal(hh, hh_se))
        x = np.maximum(10000, RNG.normal(inc, inc_se))
        p = RNG.uniform(1 - PERTURB, 1 + PERTURB, size=len(hh))
        out.append(demand(h, x, med) * p)
    return out


def solve_sp(inst, scen, time_limit=600):
    """Two-stage stochastic CFLP, sample average: y first stage, x per scenario."""
    from scipy.sparse import hstack, identity, block_diag

    m, A, S = inst.m, inst.A, len(scen)
    c = [inst.F]
    capblocks, custblocks, linkblocks = [], [], []
    for D_s in scen:
        w = inst.unit * D_s[inst.aj]
        c.append(-w / S)
        capblocks.append(coo_matrix((D_s[inst.aj], (inst.ai, np.arange(A))), shape=(m, A)))
        custblocks.append(inst.Acust)
        linkblocks.append(identity(A, format="csr"))
    c = np.concatenate(c)
    Ycap = vstack([coo_matrix((-inst.Q, (np.arange(m), np.arange(m))), shape=(m, m))] * S)
    Cap = hstack([Ycap, block_diag(capblocks)])
    Cust = hstack([csr_matrix((inst.n * S, m)), block_diag(custblocks)])
    Link = hstack([vstack([-inst.Alink] * S), block_diag(linkblocks)])
    M = vstack([Cap, Cust, Link]).tocsr()
    hi = np.concatenate([np.zeros(m * S), np.ones(inst.n * S), np.zeros(A * S)])
    res = milp(
        c,
        constraints=LinearConstraint(M, -np.inf, hi),
        integrality=np.concatenate([np.ones(m), np.zeros(A * S)]),
        bounds=(0, 1),
        options=dict(time_limit=time_limit, mip_rel_gap=1e-4),
    )
    assert res.x is not None, res.message
    return frozenset(int(i) for i in np.nonzero(res.x[:m] > 0.5)[0]), res.status == 0, res.mip_gap


def oos(inst, O, test):
    vals = []
    for D_s in test:
        inst.set_demand(D_s)
        vals.append(evaluate(inst, O))
    return np.array(vals)


# ---------------------------------------------------------------- main
def run_format(key, fmt, miles, hh, hh_se, inc, inc_se, med, sites, test, train):
    t0 = time.time()
    # weekly fixed cost: higher rent where local incomes are higher (synthetic)
    near = miles.T <= 2.5  # tracts x sites
    local_inc = np.array([
        np.average(inc[near[:, i]], weights=hh[near[:, i]]) if near[:, i].any() else med
        for i in range(miles.shape[0])
    ])
    rank = np.argsort(np.argsort(local_inc)) / (len(local_inc) - 1)
    F = fmt["F"] * (0.85 + 0.30 * rank)
    Dem = demand(hh, inc, med)
    inst = Instance(miles, Dem, np.full(miles.shape[0], fmt["Q"]), F)
    out = dict(label=fmt["label"], Q=fmt["Q"], F_base=fmt["F"],
               F_min=float(F.min()), F_max=float(F.max()),
               capacity_to_demand=float(fmt["Q"] * inst.m / Dem.sum()))

    # 1) optimum + k best alternative solutions (Sec. 2.2)
    S, Z = [], []
    for k in range(K_BEST):
        z, O, x = solve(inst, nogoods=S)
        S.append(O)
        Z.append(z)
        if k == 0:
            x_star = x
    z_star, O_star = Z[0], S[0]
    served = np.zeros(inst.n)
    np.add.at(served, inst.aj, x_star)
    out["optimal"] = dict(profit=z_star, sites=sorted(sites[i] for i in O_star), n_open=len(O_star),
                          visits_served=float((served * Dem).sum()),
                          demand_total=float(Dem.sum()))
    out["k_best"] = [dict(rank=k + 1, profit=Z[k], gap_pct=100 * (z_star - Z[k]) / z_star,
                          sites=sorted(sites[i] for i in S[k])) for k in range(K_BEST)]
    pers = {sites[i]: sum(i in s for s in S) / len(S) for i in range(inst.m)}
    out["persistence_kbest"] = {k: v for k, v in pers.items() if v > 0}
    for kk in (5, 10, 20):
        out[f"KER_{kk}"] = kernel_persistence(S[:kk], inst.m)
        out[f"E_KER_{kk}"] = expected_kernel_persistence(S[:kk], inst.m)
    out["core_sites"] = sorted(k for k, v in pers.items() if v == 1.0)
    out["gap_20th_pct"] = out["k_best"][-1]["gap_pct"]

    print(key, "k-best done", f"{time.time() - t0:.0f}s", flush=True)
    # 2) dependence structure of the optimum (Sec. 2.3)
    dp = dependence(inst, z_star, O_star)
    out["prc_pct"] = {sites[i]: 100 * v for i, v in dp["prc"].items()}
    out["dep"] = {sites[i]: [sites[j] for j in v] for i, v in dp["dep"].items()}
    out["rep"] = {sites[i]: [sites[j] for j in v] for i, v in dp["rep"].items()}
    out["dep_avg"] = dp["dep_avg"]
    out["psi"] = dp["psi"]
    out["components"] = [[sites[i] for i in c] for c in dp["components"]]
    out["independent"] = [sites[i] for i in dp["independent"]]

    # 3) baselines
    z_add, O_add, order = add_greedy(inst)
    out["add"] = dict(profit=z_add, gap_pct=100 * (z_star - z_add) / z_star,
                      sites=sorted(sites[i] for i in O_add), order=[sites[i] for i in order],
                      overlap=len(O_add & O_star) / min(len(O_add), len(O_star)),
                      rank_among_kbest=next((k + 1 for k, s in enumerate(S) if s == O_add), None))
    O_ta, score = trade_area_rule(inst, miles, hh, len(O_star))
    z_ta = evaluate(inst, O_ta)
    out["trade_area"] = dict(profit=z_ta, gap_pct=100 * (z_star - z_ta) / z_star,
                             sites=sorted(sites[i] for i in O_ta),
                             overlap=len(O_ta & O_star) / len(O_star))
    out["detail_optimal"] = plan_detail(inst, O_star, sites)
    out["detail_trade_area"] = plan_detail(inst, O_ta, sites)
    out["detail_add"] = plan_detail(inst, O_add, sites)
    out["standalone_profit"] = {sites[i]: evaluate(inst, {i}) for i in range(inst.m)}
    out["households_3mi"] = {sites[i]: float(score[i]) for i in range(inst.m)}
    # "second store" question: the single best site to open first, by each method
    best_single = max(range(inst.m), key=lambda i: evaluate(inst, {i}))
    out["best_single_site"] = sites[best_single]
    out["best_single_in_optimum"] = best_single in O_star
    out["add_first_pick"] = sites[order[0]] if order else None

    print(key, "baselines done", f"{time.time() - t0:.0f}s", flush=True)
    # 4) demand uncertainty (Ch. 7, simplified)
    S_det = []
    for D_s in train:
        inst.set_demand(D_s)
        S_det.append(solve(inst)[1])
    inst.set_demand(Dem)
    out["persistence_scenarios"] = {
        sites[i]: sum(i in s for s in S_det) / len(S_det) for i in range(inst.m)
        if any(i in s for s in S_det)
    }
    out["KER_scenarios"] = kernel_persistence(S_det, inst.m)
    out["E_KER_scenarios"] = expected_kernel_persistence(S_det, inst.m)
    print(key, "scenarios done", f"{time.time() - t0:.0f}s", flush=True)
    O_sp, sp_opt, sp_gap = solve_sp(inst, train)
    print(key, "SP done", f"{time.time() - t0:.0f}s", flush=True)
    out["sp"] = dict(sites=sorted(sites[i] for i in O_sp), proven_optimal=bool(sp_opt),
                     mip_gap=float(sp_gap) if sp_gap is not None else None,
                     same_as_ev=O_sp == O_star,
                     jaccard_with_ev=len(O_sp & O_star) / len(O_sp | O_star))
    ev_vals = oos(inst, O_star, test)
    sp_vals = oos(inst, O_sp, test)
    add_vals = oos(inst, O_add, test)
    ta_vals = oos(inst, O_ta, test)
    inst.set_demand(Dem)
    out["out_of_sample"] = {
        name: dict(mean=float(v.mean()), p05=float(np.percentile(v, 5)), min=float(v.min()))
        for name, v in (("ev", ev_vals), ("sp", sp_vals), ("add", add_vals), ("trade_area", ta_vals))
    }
    out["avus_pct"] = 100 * (sp_vals.mean() - ev_vals.mean()) / abs(ev_vals.mean())
    out["sp_beats_ev_share"] = float((sp_vals > ev_vals + 1e-6).mean())
    out["add_vs_opt_oos_mean_pct"] = 100 * (ev_vals.mean() - add_vals.mean()) / ev_vals.mean()
    out["runtime_s"] = time.time() - t0
    print(key, f"{out['runtime_s']:.0f}s", out["optimal"]["n_open"], "open;",
          f"ADD gap {out['add']['gap_pct']:.2f}%", f"TA gap {out['trade_area']['gap_pct']:.2f}%",
          f"KER20 {out['KER_20']:.2f} vs {out['E_KER_20']:.2f}", f"Dep_avg {out['dep_avg']:.3f}",
          f"AVUS {out['avus_pct']:.2f}%")
    return out


def main():
    global _SITE_XY
    tr, ca, miles, hh, hh_se, inc, inc_se = load()
    _SITE_XY = [(float(c["lat"]), float(c["lon"])) for c in ca]
    sites = [c["id"] for c in ca]
    med = float(np.median(inc))
    train = scenarios(hh, hh_se, inc, inc_se, med, N_SCEN_TRAIN)
    test = scenarios(hh, hh_se, inc, inc_se, med, N_SCEN_TEST)
    res = dict(
        data=dict(
            n_tracts=len(tr), n_sites=len(ca), households=int(hh.sum()),
            median_tract_income=med,
            hh_rel_se_median=float(np.median(hh_se / np.maximum(hh, 1))),
            inc_rel_se_median=float(np.median(inc_se / inc)),
            anchors={a: sum(c["anchor"] == a for c in ca) for a in ("Publix", "Kroger")},
        ),
        assumptions=dict(margin=MARGIN, penalty_per_mile=PENALTY, rate=RATE,
                         inc_elasticity=INC_ELASTICITY, max_drive_miles=MARGIN / PENALTY,
                         k_best=K_BEST, n_train=N_SCEN_TRAIN, n_test=N_SCEN_TEST,
                         perturb=PERTURB),
        formats={k: run_format(k, f, miles, hh, hh_se, inc, inc_se, med, sites, test, train)
                 for k, f in FORMATS.items()},
        geo=dict(
            sites=[dict(id=c["id"], anchor=c["anchor"], lat=float(c["lat"]), lon=float(c["lon"]),
                        street=c["street"], city=c["city"]) for c in ca],
            tracts=[dict(lat=float(t["lat"]), lon=float(t["lon"]), hh=int(t["households"]),
                         income=int(t["income"])) for t in tr],
        ),
    )
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
