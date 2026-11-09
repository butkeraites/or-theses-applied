"""Thesis series #2 — Fabian Akkerman, "Machine Learning for Sequential Decisions in Logistics"
(University of Twente, 2025; EURO Doctoral Dissertation Award 2026), Chapter 3 (with Nils
Knofius, Matthieu van der Heijden and Martijn Mes): dual sourcing solved with Deep Controlled
Learning (DCL) and Endogenous Parameterised Learning (EPL). Applied here to an independent
pharmacy that restocks each drug category from a cheap regular order (arrives in L days) or a
same-day rush order (20% dearer plus a $15 courier fee).

Demand: real daily sales of 7 ATC drug categories from one pharmacy's point-of-sale system
(Kaggle "Pharma sales data", Milan Zdravkovic, CC BY-NC 4.0), 2014-01-02 .. 2019-10-08, rounded
to whole packs. Policies are trained/tuned on 2014-01-02 .. 2017-11-30 and tested on the real
2018-01-01 .. 2019-10-08 days (December 2017 is a warm-up). Cost parameters are illustrative.

Policies (per setting = category x pack price x regular lead time):
  BSP  single-source base stock, regular or rush, whichever is cheaper (the thesis's baseline)
  PAR  par level on the regular order + rush exactly what is owed to waiting patients
  DIP  dual index policy (Veeraraghavan & Scheller-Wolf 2008), both levels tuned by simulation.
       With equal failure rates for both sources, the thesis's IWA heuristic reduces to this.
  DCL  Deep Controlled Learning (Temizoz et al. 2025): approximate policy iteration, a neural
       classifier trained on simulation-optimal actions, one network per setting. Warm-started
       from the tuned DIP (the thesis starts from a random policy; see README)
  EPL  DCL embedded in EPL: ONE network for all settings, setting parameters in its input,
       trained on a grid that also scales demand from 0.5x to 2x

Three tests:
  stationary      fresh bootstrap paths from the training distribution (the thesis's setting)
  replay_static   the real 2018-19 days, every policy frozen as trained on 2014-17
  replay_rolling  the real 2018-19 days, re-planned every 91 days from the trailing 365 days:
                  BSP/PAR/DIP re-tuned by simulation, EPL just gets the new mean and spread as
                  input (no retraining), DCL stays frozen (it would need retraining)

Run:  uv run --with numpy --with pandas --with torch --with scipy python pharmacy_dual_sourcing.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "salesdaily.csv")
DATA_URL = "https://raw.githubusercontent.com/devgomaa/Pharma-sales-data/main/salesdaily.csv"

CATS = ["M01AB", "M01AE", "N02BA", "N05B", "N05C", "R03", "R06"]  # N02BE excluded (see README)
CAT_NAMES = {
    "M01AB": "Diclofenac-type anti-inflammatories",
    "M01AE": "Ibuprofen-type anti-inflammatories",
    "N02BA": "Aspirin-type analgesics",
    "N05B": "Anxiolytics",
    "N05C": "Sleep aids (hypnotics)",
    "R03": "Asthma and COPD drugs",
    "R06": "Antihistamines",
}
TRAIN = ("2014-01-02", "2017-11-30")
WARMUP = ("2017-12-01", "2017-12-31")
TEST = ("2018-01-01", "2019-10-08")
REPLAN_DAYS, TRAILING_DAYS = 91, 365

# ---- cost assumptions (illustrative) ------------------------------------------------------
HOLD_RATE = 0.30  # per year, of the pack price: cash, shelf space, expiry
RUSH_PREMIUM = 0.20  # a rush pack costs 20% more than the regular price ...
RUSH_FEE = 15.0  # ... plus $15 per rush order (courier), per category per day
WAIT_BASE, WAIT_FRAC = 5.0, 0.10  # $ per pack per day a patient waits: 5 + 10% of the price
QR_MAX, QE_MAX = 40, 12  # order caps per day (all policies)

TRAIN_PRICES, TRAIN_LEADS = [10.0, 60.0, 300.0], [1, 3, 5]
UNSEEN_PRICES, UNSEEN_LEADS = [25.0, 150.0], [2, 4]
EPL_SCALES = list(np.round(np.linspace(0.5, 2.0, 10), 3))  # demand-scale grid (kappa = 10)
LMAX = 5

SEED = 20261117
N_WORKERS = 8

# DCL hyper-parameters, scaled down from the thesis's Table 3.8 (see README)
DCL = dict(iters=3, n_states=1200, m_scen=128, horizon=30, warmup=10, nbr_r=3, nbr_e=4,
           hidden=[64, 64], epochs=60)
EPL = dict(iters=3, n_states=24000, m_scen=128, horizon=30, warmup=10, nbr_r=3, nbr_e=4,
           hidden=[64, 64, 64], epochs=40)  # +1 hidden layer for EPL, as in the thesis


@dataclass(frozen=True)
class Setting:
    cat: str
    price: float
    lead: int
    mu: float
    sd: float
    scale: float = 1.0  # EPL training only: demand multiplied by this factor
    dip: tuple | None = None  # (ze, zr) of the tuned dual index policy: DCL's starting policy

    @property
    def h(self):
        return HOLD_RATE * self.price / 365.0

    @property
    def b(self):
        return WAIT_BASE + WAIT_FRAC * self.price

    @property
    def prem(self):
        return RUSH_PREMIUM * self.price

    def key(self):
        return f"{self.cat}|{self.price:g}|{self.lead}"


# ---- data ----------------------------------------------------------------------------------
def load():
    if not os.path.exists(DATA):
        import urllib.request

        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    d = pd.read_csv(DATA)
    d["datum"] = pd.to_datetime(d["datum"], format="%m/%d/%Y")
    d = d.set_index("datum")
    out = {}
    for c in CATS:
        s = d[c].clip(lower=0).round().astype(int)
        out[c] = {
            "all": s.to_numpy(),
            "dates": [x.strftime("%Y-%m-%d") for x in s.index],
            "train": s.loc[TRAIN[0] : TRAIN[1]].to_numpy(),
            "warmup": s.loc[WARMUP[0] : WARMUP[1]].to_numpy(),
            "test": s.loc[TEST[0] : TEST[1]].to_numpy(),
            "i_warm": int(np.searchsorted(s.index, pd.Timestamp(WARMUP[0]))),
            "i_test": int(np.searchsorted(s.index, pd.Timestamp(TEST[0]))),
            "monthly": {k.strftime("%Y-%m"): float(v) for k, v in d[c].resample("MS").mean().items()},
        }
    return out


def boot(train, rng, B, T, scale=1.0):
    D = train[rng.integers(0, len(train), size=(B, T))].astype(float)
    if scale != 1.0:
        D = np.floor(D * scale + rng.random(D.shape))
    return D


# ---- the inventory system, vectorised over a batch -------------------------------------------
# State each morning, after the regular delivery due today: NI = packs on hand minus packs owed
# to waiting patients; pipe[:, k] = regular packs arriving on the morning of day t+1+k.
# A rush order placed this morning arrives before patients come in. Unmet demand waits.
def step(NI, pipe, qr, qe, D, s: Setting):
    NI = NI + qe
    pipe = pipe.copy()
    pipe[:, s.lead - 1] += qr
    NI = NI - D
    cost = s.h * np.maximum(NI, 0) + s.b * np.maximum(-NI, 0) + s.prem * qe + RUSH_FEE * (qe > 0)
    NI = NI + pipe[:, 0]
    pipe = np.concatenate([pipe[:, 1:], np.zeros((pipe.shape[0], 1), pipe.dtype)], axis=1)
    return NI, pipe, cost


def dip_action(NI, pipe, ze, zr):
    qe = np.clip(ze - NI, 0, QE_MAX)
    qr = np.clip(zr - (NI + qe + pipe.sum(1)), 0, QR_MAX)
    return qr, qe


class Tracker:
    """Accumulates cost components and service statistics along simulated paths."""

    def __init__(self, B):
        z = lambda: np.zeros(B)  # noqa: E731
        self.tot, self.hold, self.wait, self.rush = z(), z(), z(), z()
        self.n_rush, self.served, self.onhand, self.ru, self.rg, self.dem = z(), z(), z(), z(), z(), z()
        self.T = 0

    def add(self, NI, qr, qe, D, s):
        x = NI + qe
        end = x - D
        self.served += np.minimum(np.maximum(x, 0), D)
        self.dem += D
        self.hold += s.h * np.maximum(end, 0)
        self.wait += s.b * np.maximum(-end, 0)
        self.rush += s.prem * qe + RUSH_FEE * (qe > 0)
        self.onhand += np.maximum(end, 0)
        self.n_rush += qe > 0
        self.ru += qe
        self.rg += qr
        self.T += 1

    def result(self, paths=False):
        T = self.T
        tot = (self.hold + self.wait + self.rush) / T
        r = {
            "cost_per_day": float(tot.mean()),
            "holding_per_day": float((self.hold / T).mean()),
            "waiting_per_day": float((self.wait / T).mean()),
            "rush_per_day": float((self.rush / T).mean()),
            "fill_rate": float((self.served / np.maximum(self.dem, 1)).mean()),
            "avg_on_hand": float((self.onhand / T).mean()),
            "rush_orders_per_30d": float((self.n_rush / T * 30).mean()),
            "rush_share_units": float((self.ru / np.maximum(self.ru + self.rg, 1)).mean()),
            "days": T,
        }
        if paths:
            r["_paths"] = tot
        return r


def evaluate(policy, s: Setting, D, warm=None, paths=False):
    B = D.shape[0]
    NI = np.full(B, float(round(s.mu * (s.lead + 1))))
    pipe = np.zeros((B, LMAX))
    if warm is not None:
        for t in range(warm.shape[1]):
            qr, qe = policy(NI, pipe)
            NI, pipe, _ = step(NI, pipe, qr, qe, warm[:, t], s)
    tr = Tracker(B)
    for t in range(D.shape[1]):
        qr, qe = policy(NI, pipe)
        tr.add(NI, qr, qe, D[:, t], s)
        NI, pipe, _ = step(NI, pipe, qr, qe, D[:, t], s)
    return tr.result(paths)


# ---- baselines, tuned by simulation ----------------------------------------------------------
NEVER = -1e9


def grid_cost(s: Setting, D, warm, ze, zr):
    """Mean daily cost of dual-index rules (ze[k], zr[k]) for all k at once, common demand."""
    K, B = len(ze), D.shape[0]
    ZE, ZR = np.repeat(np.asarray(ze, float), B), np.repeat(np.asarray(zr, float), B)
    DD, WW = np.tile(D, (K, 1)), np.tile(warm, (K, 1))
    NI = np.full(K * B, float(round(s.mu * (s.lead + 1))))
    pipe = np.zeros((K * B, LMAX))
    for t in range(WW.shape[1]):
        qr, qe = dip_action(NI, pipe, ZE, ZR)
        NI, pipe, _ = step(NI, pipe, qr, qe, WW[:, t], s)
    tot = np.zeros(K * B)
    for t in range(DD.shape[1]):
        qr, qe = dip_action(NI, pipe, ZE, ZR)
        NI, pipe, c = step(NI, pipe, qr, qe, DD[:, t], s)
        tot += c
    return (tot / DD.shape[1]).reshape(K, B).mean(1)


def tune_baselines(s: Setting, train, rng):
    B, T = 24, 730
    D = boot(train, rng, B, T, s.scale)
    warm = boot(train, rng, B, 30, s.scale)
    hi = int(np.ceil(s.mu * (s.lead + 1) + 6 * s.sd * np.sqrt(s.lead + 1))) + 3
    hi_e = int(np.ceil(s.mu + 6 * s.sd)) + 3
    zr = np.arange(0, hi + 1)
    c_reg = grid_cost(s, D, warm, np.full(len(zr), NEVER), zr)
    ze = np.arange(0, hi_e + 1)
    c_rush = grid_cost(s, D, warm, ze, np.full(len(ze), NEVER))
    if c_reg.min() <= c_rush.min():
        bsp = ("regular", int(zr[c_reg.argmin()]))
    else:
        bsp = ("rush", int(ze[c_rush.argmin()]))
    c_par = grid_cost(s, D, warm, np.zeros(len(zr)), zr)
    par = int(zr[c_par.argmin()])
    best = (np.inf, None)
    for e in range(-6, int(np.ceil(s.mu + 4 * s.sd)) + 2):
        zrs = np.arange(max(e, 0), hi + 1)
        c = grid_cost(s, D, warm, np.full(len(zrs), e), zrs)
        k = int(c.argmin())
        if c[k] < best[0]:
            best = (float(c[k]), (e, int(zrs[k])))
    return {"bsp": bsp, "par": par, "dip": best[1]}


def make_bsp(spec):
    kind, z = spec
    if kind == "regular":
        return lambda NI, p: (np.clip(z - (NI + p.sum(1)), 0, QR_MAX), np.zeros_like(NI))
    return lambda NI, p: (np.zeros_like(NI), np.clip(z - NI, 0, QE_MAX))


def make_par(zr):
    return lambda NI, p: dip_action(NI, p, 0, zr)


def make_dip(spec):
    ze, zr = spec
    return lambda NI, p: dip_action(NI, p, ze, zr)


def baseline_policies(tb):
    return {"BSP": make_bsp(tb["bsp"]), "PAR": make_par(tb["par"]), "DIP": make_dip(tb["dip"])}


# ---- Deep Controlled Learning ----------------------------------------------------------------
def features(NI, pipe, s: Setting):
    """Inventory state (raw and in units of mean daily demand) + the setting's parameters."""
    B = NI.shape[0]
    mu = max(s.mu, 0.05)
    pp = pipe[:, : LMAX - 1]
    ip = NI + pipe.sum(1)
    f = np.column_stack(
        [
            NI / mu,
            pp / mu,
            ip / mu,
            ip / (mu * (s.lead + 1)),
            NI / 10.0,
            ip / 10.0,
            np.full(B, s.mu / 10.0),
            np.full(B, s.sd / mu / 3.0),
            np.full(B, s.sd / 10.0),
            np.full(B, s.lead / 5.0),
            np.full(B, np.log(s.price) / 6.0),
            np.full(B, s.b / s.price),
            np.full(B, (s.prem + RUSH_FEE) / s.b / 10.0),
        ]
    )
    return f.astype(np.float32)


class NumpyNet:
    def __init__(self, Ws, bs, mean, std):
        self.Ws, self.bs, self.mean, self.std = Ws, bs, mean, std

    def __call__(self, X):
        h = (X - self.mean) / self.std
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.maximum(h @ W + b, 0)
        out = h @ self.Ws[-1] + self.bs[-1]
        return out[:, : QR_MAX + 1].argmax(1), out[:, QR_MAX + 1 :].argmax(1)


def net_policy(net, s):
    def pol(NI, pipe):
        qr, qe = net(features(NI, pipe, s))
        return qr.astype(float), qe.astype(float)

    return pol


def textbook_policy(s: Setting):
    """pi_0: untuned single-source base stock from the newsvendor formula (no rush)."""
    cr = s.b / (s.b + s.h)
    z = int(np.ceil(s.mu * (s.lead + 1) + norm.ppf(cr) * s.sd * np.sqrt(s.lead + 1)))
    return lambda NI, p: (np.clip(z - (NI + p.sum(1)), 0, QR_MAX), np.zeros_like(NI))


def resolve_base(net, s):
    if net is not None:
        return net_policy(net, s)
    return make_dip(s.dip) if s.dip is not None else textbook_policy(s)


def simulator_best_action(NI, pipe, s, base, train, rng, cfg):
    """DCL's simulation step: try candidate actions near the base policy's action on M common
    demand scenarios over H days, following the base policy afterwards; return the cheapest."""
    qr0, _ = base(np.array([NI]), pipe[None, :])
    qr0 = int(qr0[0])
    rs = np.arange(max(0, qr0 - cfg["nbr_r"]), min(QR_MAX, qr0 + cfg["nbr_r"]) + 1)
    es = np.arange(0, min(QE_MAX, cfg["nbr_e"]) + 1)
    cand = np.array([(r, e) for r in rs for e in es], float)
    C, M, H = len(cand), cfg["m_scen"], cfg["horizon"]
    D = np.tile(boot(train, rng, M, H, s.scale), (C, 1))
    NIb = np.full(C * M, float(NI))
    pb = np.tile(pipe[None, :], (C * M, 1)).astype(float)
    NIb, pb, tot = step(NIb, pb, np.repeat(cand[:, 0], M), np.repeat(cand[:, 1], M), D[:, 0], s)
    for t in range(1, H):
        a_r, a_e = base(NIb, pb)
        NIb, pb, c = step(NIb, pb, a_r, a_e, D[:, t], s)
        tot = tot + c
    k = int(tot.reshape(C, M).mean(1).argmin())
    return int(cand[k, 0]), int(cand[k, 1])


def collect(args):
    """Collect (state features, simulation-best action) pairs along trajectories driven by the
    improved policy; the setting (EPL parameters) is resampled at every episode reset."""
    settings, trains, base_net, cfg, n, seed = args
    rng = np.random.default_rng(seed)
    X, Yr, Ye = [], [], []
    while len(X) < n:
        i = int(rng.integers(len(settings)))
        s, train = settings[i], trains[i]
        base = resolve_base(base_net, s)
        NI = np.array([float(round(s.mu * (s.lead + 1)))])
        pipe = np.zeros((1, LMAX))
        for _ in range(cfg["warmup"]):
            qr, qe = base(NI, pipe)
            NI, pipe, _ = step(NI, pipe, qr, qe, boot(train, rng, 1, 1, s.scale)[:, 0], s)
        for _ in range(50):
            ar, ae = simulator_best_action(NI[0], pipe[0], s, base, train, rng, cfg)
            X.append(features(NI, pipe, s)[0])
            Yr.append(ar)
            Ye.append(ae)
            NI, pipe, _ = step(NI, pipe, np.array([ar], float), np.array([ae], float),
                               boot(train, rng, 1, 1, s.scale)[:, 0], s)
            if len(X) >= n:
                break
    return np.array(X), np.array(Yr), np.array(Ye)


def train_classifier(X, Yr, Ye, hidden, epochs, seed):
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    yr, ye = torch.tensor(Yr, dtype=torch.long), torch.tensor(Ye, dtype=torch.long)
    layers, d = [], X.shape[1]
    for hd in hidden:
        layers += [torch.nn.Linear(d, hd), torch.nn.ReLU()]
        d = hd
    layers += [torch.nn.Linear(d, QR_MAX + 1 + QE_MAX + 1)]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    ce = torch.nn.CrossEntropyLoss()
    n, bs = len(Xt), 128
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, bs):
            idx = perm[k : k + bs]
            out = net(Xt[idx])
            loss = ce(out[:, : QR_MAX + 1], yr[idx]) + ce(out[:, QR_MAX + 1 :], ye[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    return NumpyNet([m.weight.detach().numpy().T.copy() for m in lin],
                    [m.bias.detach().numpy().copy() for m in lin],
                    mean.astype(np.float32), std.astype(np.float32))


def dcl_train(pool, settings, trains, cfg, seed, val):
    """Approximate policy iteration from pi_0 (the setting's tuned DIP); keep the iterate that is cheapest on a validation
    simulation drawn from TRAINING demand (never the test period)."""
    base, hist, best = None, [], (np.inf, None, -1)
    chunks = N_WORKERS * 2
    per = int(np.ceil(cfg["n_states"] / chunks))
    for it in range(cfg["iters"]):
        t0 = time.time()
        res = pool.map(collect, [(settings, trains, base, cfg, per, seed * 1000 + it * 100 + k)
                                 for k in range(chunks)])
        X = np.concatenate([r[0] for r in res])
        Yr = np.concatenate([r[1] for r in res])
        Ye = np.concatenate([r[2] for r in res])
        net = train_classifier(X, Yr, Ye, cfg["hidden"], cfg["epochs"], seed + it)
        vc = float(val(net))
        hist.append({"iter": it + 1, "val": vc, "n_states": int(len(X)),
                     "secs": round(time.time() - t0, 1), "rush_label_share": float((Ye > 0).mean())})
        if vc < best[0]:
            best = (vc, net, it + 1)
        base = net
    return best[1], best[2], hist


# ---- rolling re-planning on the real test days -----------------------------------------------
def replan_points(d):
    i0, n = d["i_test"], len(d["all"])
    return list(range(i0, n, REPLAN_DAYS))


def trailing_setting(s0: Setting, series, i):
    w = series[i - TRAILING_DAYS : i]
    return replace(s0, mu=float(w.mean()), sd=float(w.std())), w


def rolling_tune_job(args):
    s0, d, seed = args
    out = []
    for k, i in enumerate(replan_points(d)):
        st, w = trailing_setting(s0, d["all"], i)
        out.append(tune_baselines(st, w, np.random.default_rng(seed + k)))
    return out


def rolling_eval(s0: Setting, d, policy_at):
    """policy_at(k, setting_k) -> policy for re-plan period k. Warm-up uses period 0's policy."""
    series = d["all"].astype(float)
    pts = replan_points(d) + [len(series)]
    NI = np.array([float(round(s0.mu * (s0.lead + 1)))])
    pipe = np.zeros((1, LMAX))
    st0, _ = trailing_setting(s0, d["all"], pts[0])
    pol = policy_at(0, st0)
    for t in range(d["i_warm"], d["i_test"]):
        qr, qe = pol(NI, pipe)
        NI, pipe, _ = step(NI, pipe, qr, qe, series[t : t + 1], s0)
    tr = Tracker(1)
    for k in range(len(pts) - 1):
        st, _ = trailing_setting(s0, d["all"], pts[k])
        pol = policy_at(k, st)
        for t in range(pts[k], pts[k + 1]):
            qr, qe = pol(NI, pipe)
            tr.add(NI, qr, qe, series[t : t + 1], s0)
            NI, pipe, _ = step(NI, pipe, qr, qe, series[t : t + 1], s0)
    return tr.result()


# ---- main ----------------------------------------------------------------------------------
def main():
    import multiprocessing as mp

    t_start = time.time()
    data = load()
    stats = {}
    for c in CATS:
        tr, te = data[c]["train"], data[c]["test"]
        stats[c] = {"name": CAT_NAMES[c], "mu_train": float(tr.mean()), "sd_train": float(tr.std()),
                    "zero_days_train": float((tr == 0).mean()), "mu_test": float(te.mean()),
                    "sd_test": float(te.std()), "n_train_days": int(len(tr)), "n_test_days": int(len(te))}

    def mk(c, p, L):
        return Setting(c, float(p), int(L), stats[c]["mu_train"], stats[c]["sd_train"])

    seen = [mk(c, p, L) for c in CATS for p in TRAIN_PRICES for L in TRAIN_LEADS]
    unseen = [mk(c, p, L) for c in CATS for p in UNSEEN_PRICES for L in UNSEEN_LEADS]
    allset = seen + unseen
    print(f"{len(seen)} training settings, {len(unseen)} unseen settings", flush=True)

    def stat_set(s, k):  # stationary test paths (fresh seeds, training distribution)
        r = np.random.default_rng(SEED + 7000 + k)
        return boot(data[s.cat]["train"], r, 100, 365), boot(data[s.cat]["train"], r, 100, 30)

    def val_set(s, k):  # validation paths for picking DCL iterates (training distribution)
        r = np.random.default_rng(SEED + 3000 + k)
        return boot(data[s.cat]["train"], r, 16, 365), boot(data[s.cat]["train"], r, 16, 30)

    out = {"settings": {}}
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        tuned = pool.starmap(tune_job, [(s, data[s.cat]["train"], SEED + k) for k, s in enumerate(allset)])
        tuned = dict(zip([s.key() for s in allset], tuned))
        print(f"static baselines tuned  t={time.time()-t_start:.0f}s", flush=True)
        rolling = pool.map(rolling_tune_job, [(s, data[s.cat], SEED + 500 + k) for k, s in enumerate(allset)])
        rolling = dict(zip([s.key() for s in allset], rolling))
        print(f"rolling baselines tuned t={time.time()-t_start:.0f}s", flush=True)

        dcl = {}
        for k, s in enumerate(seen):
            vd, vw = val_set(s, k)
            s_tr = replace(s, dip=tuple(tuned[s.key()]["dip"]))
            net, it, hist = dcl_train(pool, [s_tr], [data[s.cat]["train"]], DCL, SEED + 100 * k,
                                      lambda n, s=s, vd=vd, vw=vw: evaluate(net_policy(n, s), s, vd, warm=vw)["cost_per_day"])
            dcl[s.key()] = (net, it, hist)
            print(f"DCL {s.key():16s} iter {it} val {[round(h['val'], 3) for h in hist]} t={time.time()-t_start:.0f}s", flush=True)

        # EPL: one network over every (category, price, lead, demand scale) on the grid
        epl_settings, epl_trains = [], []
        for s in seen:
            for f in EPL_SCALES:
                epl_settings.append(replace(s, mu=s.mu * f, sd=s.sd * f, scale=f))
                epl_trains.append(data[s.cat]["train"])
        epl_dip = pool.starmap(tune_job, [(e, t, SEED + 20000 + k)
                                          for k, (e, t) in enumerate(zip(epl_settings, epl_trains))])
        epl_settings = [replace(e, dip=tuple(d["dip"])) for e, d in zip(epl_settings, epl_dip)]
        print(f"EPL grid DIPs tuned t={time.time()-t_start:.0f}s", flush=True)
        vsets = [val_set(s, k) for k, s in enumerate(seen)]
        dip_val = [evaluate(make_dip(tuned[s.key()]["dip"]), s, vd, warm=vw)["cost_per_day"]
                   for s, (vd, vw) in zip(seen, vsets)]

        def epl_val(n):
            return float(np.mean([evaluate(net_policy(n, s), s, vd, warm=vw)["cost_per_day"] / dv
                                  for s, (vd, vw), dv in zip(seen, vsets, dip_val)]))

        epl_net, epl_it, epl_hist = dcl_train(pool, epl_settings, epl_trains, EPL, SEED + 99, epl_val)
        print(f"EPL iter {epl_it} val {[round(h['val'], 4) for h in epl_hist]} t={time.time()-t_start:.0f}s", flush=True)

    # ---- evaluation ----
    for k, s in enumerate(allset):
        d = data[s.cat]
        tb = tuned[s.key()]
        pols = {**baseline_policies(tb), "EPL": net_policy(epl_net, s)}
        if s.key() in dcl:
            pols["DCL"] = net_policy(dcl[s.key()][0], s)
        rec = {"setting": {**asdict(s), "h": s.h, "b": s.b, "rush_premium_per_pack": s.prem,
                           "rush_fee": RUSH_FEE, "seen": s in seen},
               "tuned_static": {"bsp": list(tb["bsp"]), "par_zr": tb["par"], "dip_ze_zr": list(tb["dip"])},
               "tuned_rolling": [{"bsp": list(r["bsp"]), "par_zr": r["par"], "dip_ze_zr": list(r["dip"])}
                                 for r in rolling[s.key()]],
               "stationary": {}, "replay_static": {}, "replay_rolling": {}}
        if s.key() in dcl:
            rec["dcl_best_iter"], rec["dcl_hist"] = dcl[s.key()][1], dcl[s.key()][2]
        sD, sW = stat_set(s, k)
        paths = {}
        test = d["test"][None, :].astype(float)
        warm = d["warmup"][None, :].astype(float)
        for name, pol in pols.items():
            r = evaluate(pol, s, sD, warm=sW, paths=True)
            paths[name] = r.pop("_paths")
            rec["stationary"][name] = r
            rec["replay_static"][name] = evaluate(pol, s, test, warm=warm)
        # paired difference vs DIP on the stationary paths (mean and 95% CI over 100 paths)
        rec["stationary_paired_vs_DIP"] = {}
        for name in pols:
            rel = (paths[name] - paths["DIP"]) / paths["DIP"]
            se = rel.std(ddof=1) / np.sqrt(len(rel))
            rec["stationary_paired_vs_DIP"][name] = [float(rel.mean()), float(rel.mean() - 1.96 * se),
                                                     float(rel.mean() + 1.96 * se)]
        R = rolling[s.key()]
        roll = {
            "BSP": lambda j, st: make_bsp(R[j]["bsp"]),
            "PAR": lambda j, st: make_par(R[j]["par"]),
            "DIP": lambda j, st: make_dip(R[j]["dip"]),
            "EPL": lambda j, st: net_policy(epl_net, st),
        }
        if s.key() in dcl:
            roll["DCL"] = lambda j, st, n=dcl[s.key()][0]: net_policy(n, s)
        for name, f in roll.items():
            rec["replay_rolling"][name] = rolling_eval(s, d, f)
        out["settings"][s.key()] = rec

    # illustration: what each policy orders for asthma drugs, $300 packs, 3-day regular lead
    s_ex = mk("R03", 300, 3)
    tb = tuned[s_ex.key()]
    pmap = {}
    for name, pol in {**baseline_policies(tb), "EPL": net_policy(epl_net, s_ex),
                      "DCL": net_policy(dcl[s_ex.key()][0], s_ex)}.items():
        NI = np.arange(-10, 41, dtype=float)
        pipe = np.zeros((len(NI), LMAX))
        pipe[:, : s_ex.lead - 1] = round(s_ex.mu)
        qr, qe = pol(NI, pipe)
        pmap[name] = {"NI": NI.tolist(), "qr": np.asarray(qr).tolist(), "qe": np.asarray(qe).tolist()}
    out["policy_map_example"] = {"setting": s_ex.key(), "pipeline_each_day": round(s_ex.mu), "map": pmap}

    out["meta"] = {
        "categories": stats, "monthly_mean_demand": {c: data[c]["monthly"] for c in CATS},
        "train": TRAIN, "warmup": WARMUP, "test": TEST,
        "replan_days": REPLAN_DAYS, "trailing_days": TRAILING_DAYS,
        "costs": {"hold_rate_per_year": HOLD_RATE, "rush_premium": RUSH_PREMIUM, "rush_fee": RUSH_FEE,
                  "wait_cost_per_pack_day": f"{WAIT_BASE} + {WAIT_FRAC} * price",
                  "qr_max": QR_MAX, "qe_max": QE_MAX},
        "grid": {"train_prices": TRAIN_PRICES, "train_leads": TRAIN_LEADS, "unseen_prices": UNSEEN_PRICES,
                 "unseen_leads": UNSEEN_LEADS, "epl_scales": EPL_SCALES},
        "dcl_cfg": DCL, "epl_cfg": EPL, "epl_best_iter": epl_it, "epl_hist": epl_hist,
        "seed": SEED, "runtime_s": round(time.time() - t_start, 1),
        "data_url": DATA_URL,
        "data_source": "Kaggle: milanzdravkovic/pharma-sales-data (CC BY-NC 4.0)",
    }
    out["summary"] = summarize(out)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"], indent=1))
    print(f"done in {time.time()-t_start:.0f}s")


def tune_job(s, train, seed):
    return tune_baselines(s, train, np.random.default_rng(seed))


def summarize(out):
    S = out["settings"]
    res = {}
    groups = {
        "seen": lambda r: r["setting"]["seen"],
        "unseen": lambda r: not r["setting"]["seen"],
        **{f"seen_lead{L}": (lambda r, L=L: r["setting"]["seen"] and r["setting"]["lead"] == L) for L in TRAIN_LEADS},
        **{f"seen_price{int(p)}": (lambda r, p=p: r["setting"]["seen"] and r["setting"]["price"] == p) for p in TRAIN_PRICES},
    }
    for gname, filt in groups.items():
        recs = [r for r in S.values() if filt(r)]
        g = {"n_settings": len(recs)}
        for view in ["stationary", "replay_static", "replay_rolling"]:
            names = [n for n in ["BSP", "PAR", "DIP", "DCL", "EPL"] if all(n in r[view] for r in recs)]
            cost = lambda r, n: r[view][n]["cost_per_day"]  # noqa: E731
            g[view] = {
                "total_cost_per_day": {n: float(sum(cost(r, n) for r in recs)) for n in names},
                "saving_of_total_vs_BSP": {n: 1 - sum(cost(r, n) for r in recs) / sum(cost(r, "BSP") for r in recs) for n in names},
                "saving_of_total_vs_DIP": {n: 1 - sum(cost(r, n) for r in recs) / sum(cost(r, "DIP") for r in recs) for n in names},
                "mean_saving_vs_DIP": {n: float(np.mean([1 - cost(r, n) / cost(r, "DIP") for r in recs])) for n in names},
                "median_saving_vs_DIP": {n: float(np.median([1 - cost(r, n) / cost(r, "DIP") for r in recs])) for n in names},
                "beats_DIP_by_1pct": {n: int(sum(cost(r, n) < 0.99 * cost(r, "DIP") for r in recs)) for n in names},
                "loses_to_DIP_by_1pct": {n: int(sum(cost(r, n) > 1.01 * cost(r, "DIP") for r in recs)) for n in names},
                "mean_fill_rate": {n: float(np.mean([r[view][n]["fill_rate"] for r in recs])) for n in names},
                "mean_rush_orders_per_30d": {n: float(np.mean([r[view][n]["rush_orders_per_30d"] for r in recs])) for n in names},
            }
        res[gname] = g
    return res


if __name__ == "__main__":
    main()
