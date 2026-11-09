"""Fairness over time for restaurant shift scheduling.

Applies the online-assignment-with-fairness-over-time framework of van Rossum, Chen & Lodi
(ATMOS 2023; Chapter 5 of B. van Rossum's PhD thesis, Erasmus 2025) to a setting the
authors did not study: a full-service restaurant near Truist Park, Atlanta, scheduling
12 servers week by week through the 2026 Braves season.

Each week t:
  1. Solve the cheapest schedule (minimum paid hours) that covers hourly demand: B_t.
  2. Pick a solution among those costing at most (1+alpha) B_t (the alpha-subproblem):
     either any cost-optimal one, or the one whose weekly "sour points" are most even
     across the 12 weekly lines (min range), as in the thesis.
  3. Assign lines to people: blindly (random), or best-to-worst (heaviest line to the
     person with the fewest sour points so far), as in the thesis.

Demand is synthetic but shaped like a casual-dining floor, with a real driver: the
2026 Braves regular-season home games (MLB Stats API) add evening or afternoon demand.
Sour points: closing shift 1 (Fri/Sat 2), closing on a game night +1, a "clopen"
(close, then open next morning) +2.

Usage: uv run --with ortools python fair_shifts.py  -> results.json
"""

from __future__ import annotations

import datetime as dt
import json
import random
import sys
import urllib.request
from pathlib import Path

from ortools.sat.python import cp_model

HERE = Path(__file__).resolve().parent
N_WORKERS = 12
OPEN, CLOSE = 10, 23  # restaurant hours, 10:00-23:00
HOURS = list(range(OPEN, CLOSE))
SHIFT_LENGTHS = (4, 6, 8)
MAX_SHIFTS, MIN_SHIFTS, MAX_HOURS = 5, 3, 40
SEASON_START = dt.date(2026, 3, 30)  # first Monday of the season
WEEKS = 26
ALPHAS = (0.0, 0.02, 0.05, 0.10)
TIME_LIMIT = 30.0


def braves_home_games():
    cache = HERE / "braves_2026_home.json"
    if not cache.exists():
        url = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=144&season=2026"
               "&gameType=R")
        data = json.load(urllib.request.urlopen(url, timeout=60))
        games = []
        for day in data["dates"]:
            for g in day["games"]:
                if g["teams"]["home"]["team"]["id"] == 144:
                    utc = dt.datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
                    local = utc.astimezone(dt.timezone(dt.timedelta(hours=-4)))  # EDT
                    games.append({"date": g["officialDate"], "local_hour": local.hour})
        cache.write_text(json.dumps(games, indent=1))
    return json.loads(cache.read_text())


def demand(date: dt.date, game_hour: int | None):
    """Servers needed per hour."""
    wd = date.weekday()  # 0 Mon .. 6 Sun
    weekend_night = wd in (4, 5)
    req = {}
    for h in HOURS:
        if h < 11:
            r = 1
        elif h < 14:
            r = 4 if wd == 6 else 3  # Sunday brunch
        elif h < 17:
            r = 2
        elif h < 21:
            r = 5 if weekend_night else 4
        else:
            r = 3 if weekend_night else 2
        req[h] = r
    if game_hour is not None:
        if game_hour >= 17:  # night game: pre-game dinner and post-game crowd
            for h in (17, 18):
                req[h] += 1
            for h in (21, 22):
                req[h] += 2
        else:  # day game: afternoon crowd
            for h in (12, 13, 14, 15, 16):
                req[h] += 1
            for h in (17, 18):
                req[h] += 1
    return req


def shifts():
    return [(s, L) for s in range(OPEN, CLOSE) for L in SHIFT_LENGTHS if s + L <= CLOSE]


def build(week_days, games, alpha=None, base_cost=None, base_sour=None):
    """CP-SAT model of one week. If alpha is None: minimise paid hours.
    Else: minimise the range of weekly sour points, s.t. hours <= (1+alpha) base_cost."""
    m = cp_model.CpModel()
    S = shifts()
    x = {(w, d, k): m.NewBoolVar(f"x{w}_{d}_{k}") for w in range(N_WORKERS)
         for d in range(7) for k in range(len(S))}
    for w in range(N_WORKERS):
        for d in range(7):
            m.Add(sum(x[w, d, k] for k in range(len(S))) <= 1)
        n_shifts = sum(x[w, d, k] for d in range(7) for k in range(len(S)))
        m.Add(n_shifts <= MAX_SHIFTS)
        m.Add(n_shifts >= MIN_SHIFTS)
        m.Add(sum(S[k][1] * x[w, d, k] for d in range(7) for k in range(len(S))) <= MAX_HOURS)
    for d, day in enumerate(week_days):
        req = demand(day, games.get(day.isoformat()))
        for h in HOURS:
            m.Add(sum(x[w, d, k] for w in range(N_WORKERS) for k, (s, L) in enumerate(S)
                      if s <= h < s + L) >= req[h])
    hours = sum(S[k][1] * x[w, d, k] for w in range(N_WORKERS) for d in range(7)
                for k in range(len(S)))
    # sour points per line
    close_k = [k for k, (s, L) in enumerate(S) if s + L == CLOSE]
    open_k = [k for k, (s, L) in enumerate(S) if s == OPEN]
    sour = []
    for w in range(N_WORKERS):
        terms = []
        for d, day in enumerate(week_days):
            weight = 2 if day.weekday() in (4, 5) else 1
            g = games.get(day.isoformat())
            if g is not None and g >= 17:
                weight += 1
            terms += [weight * x[w, d, k] for k in close_k]
            if d < 6:  # clopen within the week
                c = m.NewBoolVar(f"clopen{w}_{d}")
                closes = sum(x[w, d, k] for k in close_k)
                opens = sum(x[w, d + 1, k] for k in open_k)
                m.Add(closes + opens - 1 <= c)
                terms.append(2 * c)
        v = m.NewIntVar(0, 100, f"sour{w}")
        m.Add(v == sum(terms))
        sour.append(v)
    if alpha is None:
        # cheapest schedule first; among those, the fewest total sour points
        m.Minimize(1000 * hours + sum(sour))
    else:
        m.Add(hours <= int((1 + alpha) * base_cost + 1e-9))
        m.Add(sum(sour) <= base_sour)  # balancing must not create extra closes or clopens
        hi, lo = m.NewIntVar(0, 100, "hi"), m.NewIntVar(0, 100, "lo")
        m.AddMaxEquality(hi, sour)
        m.AddMinEquality(lo, sour)
        # range first, then fewer hours, then fewer sour points
        m.Minimize(100000 * (hi - lo) + 100 * hours + sum(sour))
    return m, hours, sour


def solve(week_days, games, alpha=None, base_cost=None, base_sour=None, seed=0):
    m, hours, sour = build(week_days, games, alpha, base_cost, base_sour)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = seed
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"week {week_days[0]}: {solver.StatusName(st)}")
    return {"hours": int(solver.Value(hours)), "sour": [int(solver.Value(v)) for v in sour],
            "status": solver.StatusName(st)}


def btw(cum, payoffs):
    """Best-to-worst: heaviest line (most sour) to the person with the fewest sour points."""
    people = sorted(range(len(cum)), key=lambda i: cum[i])
    lines = sorted(payoffs, reverse=True)
    out = cum[:]
    for p, s in zip(people, lines):
        out[p] += s
    return out


def main():
    games = {g["date"]: g["local_hour"] for g in braves_home_games()}
    weeks = []
    for t in range(WEEKS):
        days = [SEASON_START + dt.timedelta(days=7 * t + d) for d in range(7)]
        base = solve(days, games)
        row = {"week": days[0].isoformat(), "games": sum(d.isoformat() in games for d in days),
               "cost_opt": base}
        for a in ALPHAS:
            row[f"fair_{a}"] = solve(days, games, alpha=a, base_cost=base["hours"],
                                     base_sour=sum(base["sour"]))
        weeks.append(row)
        print(t, days[0], "B=", base["hours"], "sour", sorted(base["sour"]),
              {a: (row[f'fair_{a}']['hours'], sorted(row[f'fair_{a}']['sour'])) for a in ALPHAS},
              flush=True)

    rng = random.Random(2026)
    traj = {}
    # Blind rotation: cost-optimal lines, shuffled to people at random (mean of 1,000 runs)
    runs = []
    for _ in range(1000):
        cum = [0] * N_WORKERS
        path = []
        for w in weeks:
            p = w["cost_opt"]["sour"][:]
            rng.shuffle(p)
            cum = [c + s for c, s in zip(cum, p)]
            path.append(max(cum) - min(cum))
        runs.append(path)
    traj["blind"] = [sum(r[t] for r in runs) / len(runs) for t in range(WEEKS)]
    traj["blind_p90_final"] = sorted(r[-1] for r in runs)[int(0.9 * len(runs))]
    # As-solved: cost-optimal lines in solver order (worker k always gets line k)
    cum = [0] * N_WORKERS
    traj["as_solved"] = []
    for w in weeks:
        cum = [c + s for c, s in zip(cum, w["cost_opt"]["sour"])]
        traj["as_solved"].append(max(cum) - min(cum))
    final = {}
    # Best-to-worst on cost-optimal lines, and on alpha-fair lines
    for name, key in [("btw", "cost_opt")] + [(f"fair_{a}+btw", f"fair_{a}") for a in ALPHAS]:
        cum = [0] * N_WORKERS
        path = []
        for w in weeks:
            cum = btw(cum, w[key]["sour"])
            path.append(max(cum) - min(cum))
        traj[name] = path
        final[name] = {"final_cum": cum}
    total_hours = {"cost_opt": sum(w["cost_opt"]["hours"] for w in weeks)}
    for a in ALPHAS:
        total_hours[f"fair_{a}"] = sum(w[f"fair_{a}"]["hours"] for w in weeks)
    total_sour = {k: sum(sum(w[k]["sour"]) for w in weeks) for k in total_hours}
    out = {"weeks": weeks, "trajectory_range": traj, "total_hours": total_hours,
           "total_sour": total_sour, "final": final,
           "params": {"workers": N_WORKERS, "weeks": WEEKS, "alphas": ALPHAS,
                      "season_start": SEASON_START.isoformat(), "time_limit_s": TIME_LIMIT,
                      "home_games_in_window": sum(w["games"] for w in weeks)}}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v[-1] if isinstance(v, list) else v for k, v in traj.items()}, indent=1))
    print(json.dumps(total_hours, indent=1), json.dumps(total_sour, indent=1))


if __name__ == "__main__":
    sys.exit(main())
