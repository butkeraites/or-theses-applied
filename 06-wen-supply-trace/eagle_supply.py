"""Thesis series #6 -- Tao Wen, PhD thesis (University of Manchester, 2025), Chapter 6:
"Early identification of information propagation sources through generalized Bayesian
inference in large-scale social networks" (EAGLE). Under review as a paper (thesis Sec. 1.6).

Applied to a food-safety recall: place a few inspection points ("observers") in a real
supply chain so a bad lot is caught early, then infer which stage started it from the first
few positive inspections.

Data: Willems (2008), MSOM 10(1), 38 real anonymised supply chains (unofficial public copy,
see README). Featured: chain 12, SIC 2043 Cereal Breakfast Foods.

What follows the thesis (Sec. 6.3), and what I changed, is listed in README.md.

Run:  uv run --with networkx --with numpy --with scipy --with pandas --with openpyxl \
          python eagle_supply.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

import willems_data

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 20261201
RNG = np.random.default_rng(SEED)

FEATURED = 12
T_MAX = 40  # propagation steps (each step = one shipping cycle)
P_LO, P_HI = 0.1, 0.9  # heterogeneous SI: each link's pass-on probability ~ U(P_LO, P_HI)
XIS = [0.10, 0.20]  # observer ratio xi = |O| / |N|
KS = [2, 3, 4]  # infer the source from the first k positive inspections
HEADLINE = {"xi": 0.10, "k": 3}  # fixed before looking at results
REALIZ_FEATURED = 400
REALIZ_ALL = 100

# EAGLE parameters. The thesis gives kappa = 4, tau = 5, b in [-5, 5], upsilon = 2%;
# it does not state |M|, iterations, mutation rate, the l = 0 penalty or phi.
KAPPA, TAU, BL, BU, UPSILON = 4, 5, -5.0, 5.0, 0.02
N_MOTHS, N_ITERS, MUT_P, PENALTY, PHI = 20, 60, 0.1, 1, 0.05
MAX_FOCAL = 400


# ------------------------------------------------------------------------------ data
def load_chain(x, cid):
    s = x["stages in chains"]
    r = x["rels between stages"]
    ss = s[s.chainId == cid]
    G = nx.DiGraph()
    G.add_nodes_from(dict.fromkeys(ss.Stage_Name))  # chain 36 lists each stage twice
    G.add_edges_from(r[r.chainId == cid][["sourceStage", "destinationStage"]].itertuples(index=False))
    cls = dict(zip(ss.Stage_Name, ss.stageClassification))
    return G, cls


class Chain:
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        idx = {n: i for i, n in enumerate(self.nodes)}
        self.succ = [[idx[m] for m in G.successors(n)] for n in self.nodes]
        self.pred = [[idx[m] for m in G.predecessors(n)] for n in self.nodes]
        self.indeg = np.array([len(p) for p in self.pred])
        self.outdeg = np.array([len(s) for s in self.succ])
        A = nx.to_scipy_sparse_array(G, nodelist=self.nodes, format="csr")
        BIG = self.N  # stands in for "unreachable" in directed distances
        Dd = shortest_path(A, directed=True, unweighted=True)
        Dd[np.isinf(Dd)] = BIG
        self.Dd = Dd  # Dd[v, w]: hops from v downstream to w
        Du = shortest_path(A, directed=False, unweighted=True)
        Du[np.isinf(Du)] = BIG
        self.Du = Du  # undirected hops, for the error distance (Eq. 6.26)
        self.BIG = BIG
        self.sources = [i for i in range(self.N) if self.outdeg[i] > 0]
        # k-bar in Eq. 6.3: average number of stages one stage can be watched by
        self.kbar = max(1, int(round(self.outdeg.mean())))


# ---------------------------------------------------------- 6.3.1 observer deployment
def omega(l, kbar):
    if l == 0:
        return -PENALTY
    if l < kbar:
        return math.exp(-(kbar - l) / 2)
    return 1.0


def fitness(ch: Chain, O: set) -> float:
    """Eq. 6.3-6.4. A candidate is watched by observers among its one-hop downstream
    neighbours (an inspection point checks what it receives, and from whom)."""
    f = 0.0
    for c in range(ch.N):
        if c in O:
            continue
        l = sum(1 for w in ch.succ[c] if w in O)
        f += omega(l, ch.kbar)
    return f


def deploy_random(ch, m):
    return set(RNG.choice(ch.N, size=m, replace=False).tolist())


def deploy_degree(ch, m):
    # degree-based (thesis's comparison): stages that receive from the most suppliers
    order = sorted(range(ch.N), key=lambda i: (-ch.indeg[i], RNG.random()))
    return set(order[:m])


def deploy_greedy(ch, m):
    # textbook OR baseline (not in the thesis): add the stage that raises F the most.
    # Incremental: adding i raises l for each upstream neighbour of i and removes i itself.
    O = set()
    l = np.zeros(ch.N, int)
    for _ in range(m):
        best, bv = None, -1e18
        for i in range(ch.N):
            if i in O:
                continue
            g = -omega(l[i], ch.kbar)
            for c in ch.pred[i]:
                if c not in O:
                    g += omega(l[c] + 1, ch.kbar) - omega(l[c], ch.kbar)
            if g > bv:
                best, bv = i, g
        O.add(best)
        for c in ch.pred[best]:
            l[c] += 1
    return O


def deploy_mfo(ch, m):
    """Discrete moth-flame optimisation (Sec. 6.3.1), simplified as stated in the README."""
    area_n = min(ch.N, int(round(KAPPA * m)))
    area = sorted(range(ch.N), key=lambda i: (-ch.indeg[i], RNG.random()))[:area_n]
    area_arr = np.array(area)

    def repair(moth):
        seen, out = set(), []
        for v in moth:
            if v in seen:
                cand = [a for a in area if a not in seen and a not in moth]
                if not cand:
                    cand = [a for a in range(ch.N) if a not in seen]
                v = cand[RNG.integers(len(cand))]
            seen.add(v)
            out.append(v)
        return out

    pop = [sorted(area, key=lambda i: -ch.indeg[i])[:m]]  # top-|O| degree moth
    while len(pop) < N_MOTHS:
        score = RNG.random(len(area)) * ch.indeg[area_arr]
        pop.append([area[j] for j in np.argsort(-score)[:m]])
    pop = [repair(p) for p in pop]
    fit = [fitness(ch, set(p)) for p in pop]
    best, best_f = list(pop[int(np.argmax(fit))]), max(fit)
    for _ in range(N_ITERS):
        order = np.argsort(fit)[::-1]
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]
        M1 = list(pop[0])
        new = [M1]
        for i in range(1, len(pop)):
            mi = list(pop[i])
            x = 1 / (1 + np.exp(-(BL + (BU - BL) * RNG.random(m))))  # Eq. 6.7
            gamma = RNG.random()
            p = (x[0] > gamma) & (x > gamma)  # Eq. 6.8
            if i <= TAU:  # two-way exchange with the temporary best
                cand_best = list(M1)
                for j in np.nonzero(p)[0]:
                    mi[j], cand_best[j] = cand_best[j], mi[j]
                cand_best = repair(cand_best)
                fb = fitness(ch, set(cand_best))
                if fb > fitness(ch, set(M1)):
                    M1 = cand_best
                    new[0] = M1
                if RNG.random() < MUT_P:  # mutation for high performers
                    j = RNG.integers(m)
                    mi[j] = area[RNG.integers(len(area))]
            else:  # one-way replacement from the temporary best
                for j in np.nonzero(p)[0]:
                    mi[j] = M1[j]
            new.append(repair(mi))
        pop = new
        fit = [fitness(ch, set(p)) for p in pop]
        if max(fit) > best_f:
            best_f = max(fit)
            best = list(pop[int(np.argmax(fit))])
    return set(best)


# --------------------------------------------------------- propagation (downstream HSI)
def propagate(ch: Chain, s: int):
    """Heterogeneous SI, downstream only. Each link has its own pass-on probability; every
    step, each contaminated stage tries each clean downstream neighbour. Returns the step
    each stage was contaminated and by whom (the sender an inspection would record)."""
    p = {}
    t_inf = np.full(ch.N, -1)
    sender = np.full(ch.N, -1)
    t_inf[s] = 0
    active = [s]
    infected = {s}
    for t in range(1, T_MAX + 1):
        newly = {}
        for v in infected:
            for w in ch.succ[v]:
                if t_inf[w] >= 0 or w in newly:
                    continue
                if (v, w) not in p:
                    p[(v, w)] = RNG.uniform(P_LO, P_HI)
                if RNG.random() < p[(v, w)]:
                    newly[w] = v
        for w, v in newly.items():
            t_inf[w] = t
            sender[w] = v
        infected |= set(newly)
        if not newly and all(
            all(t_inf[w] >= 0 for w in ch.succ[v]) for v in infected
        ):
            break
    del active
    return t_inf, sender


# ----------------------------------------------------- 6.3.2 information collection
def candidates(ch, O, Opos):
    """Eq. 6.9 with directed distances (unreachable = |N|)."""
    Oneg = [o for o in O if o not in Opos]
    C = []
    for v in range(ch.N):
        if v in O:
            continue
        dp = ch.Dd[v, Opos].mean()
        if dp >= ch.BIG:
            continue
        if not Oneg or dp < ch.Dd[v, Oneg].mean():
            C.append(v)
    return C


def pairwise(ch, C, obs_i, obs_j):
    (oi, ti, si), (oj, tj, sj) = obs_i, obs_j
    di = ch.Dd[C, si]
    dj = ch.Dd[C, sj]
    if ti != tj:
        eps = 1 - (di - dj) / (ti - tj)
    else:
        eps = di - dj
    return np.exp(-np.abs(eps))  # Eq. 6.11


def group(ch, C, obs, leave):
    rest = [o for k, o in enumerate(obs) if k != leave]
    T = np.array([o[1] for o in rest], float)
    out = np.zeros(len(C))
    if T.std() == 0:
        return out
    for a, v in enumerate(C):
        D = np.array([ch.Dd[v, o[2]] for o in rest], float)
        if D.std() == 0:
            continue
        out[a] = max(0.0, float(np.corrcoef(T, D)[0, 1]))  # Eq. 6.13, negatives -> 0
    return out


# --------------------------------------------- 6.3.3 generalized Bayesian inference
def belief_piece(P, psi):
    """Eq. 6.17-6.18: cluster candidates with similar likelihood (within PHI) and give each
    cluster psi x its mean likelihood (rescaled so the piece's masses sum to psi)."""
    order = np.argsort(-P)
    clusters = []
    cur, ref = [], None
    for a in order:
        if ref is None or abs(P[a] - ref) <= PHI:
            if ref is None:
                ref = P[a]
            cur.append(a)
        else:
            clusters.append(cur)
            cur, ref = [a], P[a]
    if cur:
        clusters.append(cur)
    w = np.array([P[c].mean() for c in clusters])
    if w.sum() <= 0:
        return {}, 1.0
    w = psi * w / w.sum()
    m = {}
    for c, wc in zip(clusters, w):
        if wc > 0:
            mask = 0
            for a in c:
                mask |= 1 << int(a)
            m[mask] = m.get(mask, 0) + float(wc)
    return m, 1 - psi


def er_combine(pieces):
    """Eq. 6.21-6.23, the recursive ER rule; low-belief focal sets pruned (upsilon)."""
    m, mP = pieces[0][0].copy(), pieces[0][1]
    for ml, psi_l_comp in pieces[1:]:
        psi_l = 1 - psi_l_comp
        mhat = {}
        for th, v in m.items():
            mhat[th] = mhat.get(th, 0) + (1 - psi_l) * v
        for th, v in ml.items():
            mhat[th] = mhat.get(th, 0) + mP * v
        for B, vb in m.items():
            for Cs, vc in ml.items():
                inter = B & Cs
                if inter:
                    mhat[inter] = mhat.get(inter, 0) + vb * vc
        mPhat = (1 - psi_l) * mP
        tot = sum(mhat.values()) + mPhat
        if tot <= 0:
            continue
        m = {k: v / tot for k, v in mhat.items()}
        mP = mPhat / tot
        if len(m) > MAX_FOCAL:  # prune the weakest until they hold <= upsilon
            items = sorted(m.items(), key=lambda kv: kv[1])
            drop, acc = [], 0.0
            for k, v in items:
                if acc + v > UPSILON or len(m) - len(drop) <= MAX_FOCAL // 2:
                    break
                drop.append(k)
                acc += v
            for k in drop:
                del m[k]
    s = sum(m.values())
    return {k: v / s for k, v in m.items()} if s > 0 else {}  # Eq. 6.24


def eagle_infer(ch, O, obs):
    """obs: list of (observer, time, sender), time-ordered. Returns candidate list and the
    pignistic belief eta (Eq. 6.25) for each."""
    Opos = [o[0] for o in obs]
    C = candidates(ch, O, Opos)
    if not C:
        return [], np.array([])
    kO = {o[0]: max(1, ch.indeg[o[0]]) for o in obs}
    kmax = max(kO.values())
    ksum = sum(kO.values())
    pieces = []
    n = len(obs)
    ring = [(a, (a + 1) % n) for a in range(n)] if n > 2 else [(0, 1)]
    for a, b in ring:
        P = pairwise(ch, C, obs[a], obs[b])
        psi = (kO[obs[a][0]] + kO[obs[b][0]]) / (2 * kmax)  # Eq. 6.19
        pieces.append(belief_piece(P, psi))
    if n >= 4:
        for a in range(n):
            P = group(ch, C, obs, a)
            psi = (1 - kO[obs[a][0]] / ksum) / 2  # Eq. 6.20
            pieces.append(belief_piece(P, psi))
    pieces = [p for p in pieces if p[0]]
    if not pieces:
        return C, np.full(len(C), 1 / len(C))
    m = er_combine(pieces)
    eta = np.zeros(len(C))
    for th, v in m.items():
        members = [a for a in range(len(C)) if th >> a & 1]
        for a in members:
            eta[a] += v / len(members)
    return C, eta


# ---------------------------------------------------------------- baselines
def earliest_sender(ch, O, obs):
    """Practitioner's trace-back: the supplier(s) that shipped the first positive lot(s)."""
    t0 = obs[0][1]
    return sorted({o[2] for o in obs if o[1] == t0})


def common_upstream(ch, O, obs):
    """Distance-based: the stage(s) that can reach every flagged sender, closest on average
    (a 'common supplier' search). Ties are all returned."""
    S = sorted({o[2] for o in obs})
    best, bv = [], None
    for v in range(ch.N):
        if v in O:
            continue
        d = ch.Dd[v, S]
        if (d >= ch.BIG).any():
            continue
        val = d.mean()
        if bv is None or val < bv - 1e-12:
            best, bv = [v], val
        elif abs(val - bv) <= 1e-12:
            best.append(v)
    return best


def first_site_suppliers(ch, O, obs):
    """No lot records: all direct suppliers of the first positive site(s)."""
    t0 = obs[0][1]
    return sorted({p for o in obs if o[1] == t0 for p in ch.pred[o[0]] if p not in O})


def score_set(ch, est, s):
    if not est:
        return {"hit1": 0.0, "f": 0.0, "err": float("nan"), "size": 0}
    hit = s in est
    prec = (1 / len(est)) if hit else 0.0
    f = 2 * prec * 1.0 / (prec + 1.0) if hit else 0.0  # Eq. 6.27, beta = 1, recall 1 or 0
    err = float(np.mean([ch.Du[e, s] for e in est]))  # Eq. 6.26
    return {"hit1": float(hit and len(est) == 1), "f": f, "err": err, "size": len(est)}


def eval_eagle(ch, C, eta, s):
    if len(C) == 0:
        return {"hit1": 0.0, "top3": 0.0, "f": 0.0, "err": float("nan"), "size": 0,
                "in_C": 0.0}
    mx = eta.max()
    est = [C[a] for a in range(len(C)) if eta[a] >= mx - 1e-12]
    r = score_set(ch, est, s)
    order = [C[a] for a in np.argsort(-eta, kind="stable")]
    r["top3"] = float(s in order[:3])
    r["in_C"] = float(s in C)
    return r


def eval_set_top3(est, s):
    return float(s in est and len(est) <= 3)


# ---------------------------------------------------------------- experiment
METHODS = ["EAGLE", "Earliest sender", "Common upstream", "Random candidate",
           "EAGLE (no lot records)", "First site's suppliers", "Common upstream (no lot records)"]

def run_chain(ch: Chain, realiz: int, detail=False):
    out = {"stages": ch.N, "links": ch.G.number_of_edges(), "kbar": ch.kbar,
           "n_possible_sources": len(ch.sources), "by_xi": {}}
    for xi in XIS:
        m = max(2, int(round(xi * ch.N)))
        deps = {
            "Random": deploy_random(ch, m),
            "Most suppliers (degree)": deploy_degree(ch, m),
            "Greedy coverage": deploy_greedy(ch, m),
            "EAGLE (moth-flame)": deploy_mfo(ch, m),
        }
        res = {"observers": m, "deploy": {}, "infer": {str(k): {} for k in KS}}
        for name, O in deps.items():
            res["deploy"][name] = {
                "fitness": fitness(ch, O),
                "sources_watched_share": float(np.mean(
                    [any(w in O for w in ch.succ[c]) for c in ch.sources if c not in O]))}
        delays = {name: [] for name in deps}
        never = {name: 0 for name in deps}
        acc = {str(k): {m: [] for m in METHODS} for k in KS}
        reached = {str(k): 0 for k in KS}
        trials = 0
        examples = []
        for r in range(realiz):
            s = int(RNG.choice(ch.sources))
            t_inf, sender = propagate(ch, s)
            # detection delay (Fig. 6.4): first step any observer is contaminated;
            # an inspection point at the source itself catches it at step 0
            for name, O in deps.items():
                if s in O:
                    delays[name].append(0)
                    continue
                ts = [t_inf[o] for o in O if t_inf[o] >= 0]
                if ts:
                    delays[name].append(min(ts))
                else:
                    never[name] += 1
            # inference on the EAGLE deployment, non-trivial case s not in O
            O = deps["EAGLE (moth-flame)"]
            if s in O:
                continue
            trials += 1
            exp = sorted([(o, int(t_inf[o]), int(sender[o])) for o in O if t_inf[o] >= 0],
                         key=lambda z: (z[1], RNG.random()))
            for k in KS:
                if len(exp) < k:
                    continue
                tk = exp[k - 1][1]
                obs = [z for z in exp if z[1] <= tk]  # first k, plus ties at that step
                reached[str(k)] += 1
                C, eta = eagle_infer(ch, O, obs)
                e = eval_eagle(ch, C, eta, s)
                acc[str(k)]["EAGLE"].append(e)
                for bname, fn in (("Earliest sender", earliest_sender),
                                  ("Common upstream", common_upstream)):
                    est = fn(ch, O, obs)
                    sc = score_set(ch, est, s)
                    sc["top3"] = eval_set_top3(est, s)
                    acc[str(k)][bname].append(sc)
                # naive: one random member of the candidate set
                if C:
                    g = int(C[RNG.integers(len(C))])
                    sc = score_set(ch, [g], s)
                    sc["top3"] = float(s in [C[i] for i in RNG.choice(
                        len(C), size=min(3, len(C)), replace=False)])
                else:
                    sc = {"hit1": 0.0, "f": 0.0, "err": float("nan"), "size": 0, "top3": 0.0}
                acc[str(k)]["Random candidate"].append(sc)
                # variant: inspections record WHEN, not from whom (no lot records)
                obs_nl = [(o, t, o) for o, t, _ in obs]
                C2, eta2 = eagle_infer(ch, O, obs_nl)
                acc[str(k)]["EAGLE (no lot records)"].append(eval_eagle(ch, C2, eta2, s))
                for bname, fn in (("First site's suppliers", first_site_suppliers),
                                  ("Common upstream (no lot records)", common_upstream)):
                    est = fn(ch, O, obs_nl)
                    sc = score_set(ch, est, s)
                    sc["top3"] = eval_set_top3(est, s)
                    acc[str(k)][bname].append(sc)
                if detail and k == HEADLINE["k"] and len(examples) < 1 and len(obs) >= 3 \
                        and e["hit1"] == 1.0:
                    examples.append({"source": ch.nodes[s], "obs": [
                        [ch.nodes[o], t, ch.nodes[snd]] for o, t, snd in obs],
                        "eagle_top": [[ch.nodes[C[a]], float(eta[a])] for a in
                                      np.argsort(-eta)[:5]],
                        "earliest_sender": [ch.nodes[v] for v in earliest_sender(ch, O, obs)],
                        "common_upstream": [ch.nodes[v] for v in common_upstream(ch, O, obs)]})
        for name in deps:
            d = delays[name]
            res["deploy"][name].update({
                "mean_delay": float(np.mean(d)) if d else float("nan"),
                "never_detected_share": never[name] / realiz,
                "caught_at_source_share": float(np.mean([s == 0 for s in d])) * len(d) / realiz
                if d else 0.0})
        for k in KS:
            res["infer"][str(k)]["runs_reaching_k"] = reached[str(k)]
            res["infer"][str(k)]["nontrivial_runs"] = trials
            for meth, lst in acc[str(k)].items():
                if not lst:
                    continue
                res["infer"][str(k)][meth] = {
                    key: float(np.nanmean([z[key] for z in lst]))
                    for key in ("hit1", "top3", "f", "err", "size")}
                if meth.startswith("EAGLE"):
                    res["infer"][str(k)][meth]["source_in_candidates"] = float(
                        np.mean([z["in_C"] for z in lst]))
        if detail:
            res["observer_sets"] = {name: sorted(ch.nodes[i] for i in O) for name, O in deps.items()}
            res["example"] = examples
        out["by_xi"][str(xi)] = res
    return out


def main():
    t0 = time.time()
    x = willems_data.load()
    chains = x["chains"]
    res_check = willems_data.check(x)
    res = {"meta": {"seed": SEED, "t_max": T_MAX, "p_range": [P_LO, P_HI], "xis": XIS,
                    "ks": KS, "headline": HEADLINE, "realizations_featured": REALIZ_FEATURED,
                    "realizations_all": REALIZ_ALL,
                    "params": {"kappa": KAPPA, "tau": TAU, "b": [BL, BU], "upsilon": UPSILON,
                               "moths": N_MOTHS, "iterations": N_ITERS, "mutation": MUT_P,
                               "penalty_l0": PENALTY, "phi": PHI, "max_focal": MAX_FOCAL}}}
    res["data_check"] = res_check
    G, cls = load_chain(x, FEATURED)
    ch = Chain(G)
    feat = run_chain(ch, REALIZ_FEATURED, detail=True)
    feat["chainId"] = FEATURED
    feat["classes"] = pd.Series(cls).value_counts().to_dict()
    feat["sources_by_class"] = pd.Series([cls[ch.nodes[i]] for i in ch.sources]).value_counts().to_dict()
    feat["depth"] = nx.dag_longest_path_length(G)
    # layered layout for the figure
    layer = {}
    for n in nx.topological_sort(G):
        layer[n] = max([layer[p] + 1 for p in G.predecessors(n)], default=0)
    order = {}
    for L in range(max(layer.values()) + 1):
        ln = [n for n in ch.nodes if layer[n] == L]
        if L == 0:
            ln.sort(key=lambda n: -len(nx.descendants(G, n)))
        else:
            ln.sort(key=lambda n: np.mean([order[p] for p in G.predecessors(n)]))
        for k, n in enumerate(ln):
            order[n] = k / max(1, len(ln) - 1)
    feat["layout"] = {n: [layer[n], round(order[n], 4)] for n in ch.nodes}
    feat["edges"] = [[a, b] for a, b in G.edges()]
    feat["class_of"] = cls
    res["featured"] = feat
    print("featured done", round(time.time() - t0), "s")

    allc = []
    for cid in chains.chainId:
        G, _ = load_chain(x, int(cid))
        if G.number_of_nodes() < 20:
            continue
        ch = Chain(G)
        if len(ch.sources) < 5:
            continue
        r = run_chain(ch, REALIZ_ALL)
        r["chainId"] = int(cid)
        r["sic"] = str(chains[chains.chainId == cid].SIC_Description.iloc[0])
        allc.append(r)
        print(" chain", cid, ch.N, round(time.time() - t0), "s")
    res["all_chains"] = allc

    # summary across chains
    summ = {"n_chains": len(allc)}
    for xi in XIS:
        sx = {}
        deps = ["Random", "Most suppliers (degree)", "Greedy coverage", "EAGLE (moth-flame)"]
        sx["median_delay"] = {d: float(np.nanmedian([c["by_xi"][str(xi)]["deploy"][d]["mean_delay"]
                                                     for c in allc])) for d in deps}
        sx["median_never"] = {d: float(np.median([c["by_xi"][str(xi)]["deploy"][d]["never_detected_share"]
                                                  for c in allc])) for d in deps}
        sx["eagle_vs"] = {}
        for d in deps[:-1]:
            e = [c["by_xi"][str(xi)]["deploy"]["EAGLE (moth-flame)"]["mean_delay"] for c in allc]
            o = [c["by_xi"][str(xi)]["deploy"][d]["mean_delay"] for c in allc]
            ne = [c["by_xi"][str(xi)]["deploy"]["EAGLE (moth-flame)"]["never_detected_share"] for c in allc]
            no = [c["by_xi"][str(xi)]["deploy"][d]["never_detected_share"] for c in allc]
            sx["eagle_vs"][d] = {
                "fewer_undetected": int(sum(a < b - 1e-9 for a, b in zip(ne, no))),
                "more_undetected": int(sum(a > b + 1e-9 for a, b in zip(ne, no))),
                "fitness_higher": int(sum(
                    c["by_xi"][str(xi)]["deploy"]["EAGLE (moth-flame)"]["fitness"]
                    > c["by_xi"][str(xi)]["deploy"][d]["fitness"] + 1e-9 for c in allc)),
                "fitness_lower": int(sum(
                    c["by_xi"][str(xi)]["deploy"]["EAGLE (moth-flame)"]["fitness"]
                    < c["by_xi"][str(xi)]["deploy"][d]["fitness"] - 1e-9 for c in allc)),
            }
        sx["infer"] = {}
        for k in KS:
            meths = ["EAGLE", "Earliest sender", "Common upstream", "Random candidate"]
            rows = [c["by_xi"][str(xi)]["infer"][str(k)] for c in allc
                    if "EAGLE" in c["by_xi"][str(xi)]["infer"][str(k)]
                    and c["by_xi"][str(xi)]["infer"][str(k)]["runs_reaching_k"] >= 10]
            sx["infer"][str(k)] = {"chains_used": len(rows)}
            for b in ["First site's suppliers", "Common upstream (no lot records)"]:
                e = "EAGLE (no lot records)"
                sx["infer"][str(k)][f"NL_EAGLE_top1_better_than_{b}"] = int(
                    sum(r[e]["hit1"] > r[b]["hit1"] + 1e-9 for r in rows))
                sx["infer"][str(k)][f"NL_EAGLE_top1_worse_than_{b}"] = int(
                    sum(r[e]["hit1"] < r[b]["hit1"] - 1e-9 for r in rows))
                sx["infer"][str(k)][f"NL_EAGLE_err_lower_than_{b}"] = int(
                    sum(r[e]["err"] < r[b]["err"] - 1e-9 for r in rows))
                sx["infer"][str(k)][f"NL_EAGLE_f_higher_than_{b}"] = int(
                    sum(r[e]["f"] > r[b]["f"] + 1e-9 for r in rows))
            for mth in METHODS:
                sx["infer"][str(k)][mth] = {
                    key: float(np.nanmedian([r[mth][key] for r in rows]))
                    for key in ("hit1", "top3", "f", "err", "size")}
            for b in meths[1:]:
                sx["infer"][str(k)][f"EAGLE_top1_better_than_{b}"] = int(
                    sum(r["EAGLE"]["hit1"] > r[b]["hit1"] + 1e-9 for r in rows))
                sx["infer"][str(k)][f"EAGLE_top1_worse_than_{b}"] = int(
                    sum(r["EAGLE"]["hit1"] < r[b]["hit1"] - 1e-9 for r in rows))
                sx["infer"][str(k)][f"EAGLE_err_lower_than_{b}"] = int(
                    sum(r["EAGLE"]["err"] < r[b]["err"] - 1e-9 for r in rows))
        summ[str(xi)] = sx
    res["summary"] = summ
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print("done", res["meta"]["runtime_s"], "s")


if __name__ == "__main__":
    main()
