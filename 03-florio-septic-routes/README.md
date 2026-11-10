# Florio — septic pump-out trucks with a probabilistic 10-hour day

Thesis series #2. Applies the vehicle routing problem with **stochastic demands, optimal
restocking and probabilistic duration constraints** to septic-tank pump-out trucks in Cherokee
County, Georgia. The application is mine (Renan Butkeraites, BIS), not the authors'.

## The thesis and the chapter used

- Alexandre De Macedo Florio, *Models and solution methods for stochastic vehicle routing
  problems*, PhD dissertation, Universität Wien, Fakultät für Wirtschaftswissenschaften, 2019
  (supervisor: Richard Hartl; Georg Pflug chaired the defense committee). Record: https://utheses.univie.ac.at/detail/51224/ — the full text is
  not publicly released. Winner of the EURO Doctoral Dissertation Award 2021
  (https://www.euro-online.org/web/pages/1696/edda-winner-2021).
- Part used (item 4 of the thesis abstract): A. M. Florio, R. F. Hartl, S. Minner,
  J.-J. Salazar-González, "A Branch-and-Price Algorithm for the Vehicle Routing Problem with
  Stochastic Demands and Probabilistic Duration Constraints", *Transportation Science*
  55(1):122–138, 2021, doi:10.1287/trsc.2020.1002. Read in full (copy obtained privately under
  INFORMS terms; not included in this repository). Affiliations in the paper: Florio and Hartl,
  Department of Business Decisions and Analytics, University of Vienna; Minner, TU Munich; Salazar-
  González, Universidad de La Laguna.
- Companion (thesis item 3): Florio, Hartl, Minner, "New Exact Algorithm for the Vehicle Routing
  Problem with Stochastic Demands", *Transportation Science* 54(4):1073–1090, 2020,
  doi:10.1287/trsc.2020.0976 (preprint arXiv:1806.08549, used for §3.2).

### Where this experiment departs from the paper

| Paper (section) | Here |
|---|---|
| Duration = travel time incl. restocking trips; cost = expected duration (§2.1–2.2, eq. 1) | Duration adds setup + pump time (grows with gallons), random dig time, 20 min per unload; cost = expected driving + unloading minutes |
| No vehicle cost, unlimited fleet; limited fleet also tested (§2.1, §4.1 B&P-FIX) | 600-minute planning cost per truck-day |
| Poisson / binomial / negative binomial demands (§4.1–4.4) | Custom discrete gallons from bedrooms + assumptions |
| Exact branch-and-price: backward labeling, orienteering completion bound, branching on number of routes and arcs (§3.1–3.5) | Column generation with greedy best-insertion pricing, then a MIP over all generated routes (heuristic) |
| Feasibility: Strategy I one-sided Chebyshev (confirms only), II Monte Carlo + Agresti–Coull, III exact joint PMF (§3.3, Alg. 1–2) | Cantelli bound to accept *and* reject, then the exact distribution (Strategy III analogue); Monte Carlo used only to validate final plans |
| Optimal restocking by thresholds τ_i (Yee & Golden; §3.3.1, eq. 6) | Same DP and threshold structure |

## Mapping

| Paper | Here |
|---|---|
| customer demand, revealed on arrival | gallons pumped at a house, known once the lid is open |
| vehicle capacity Q | 3,000-gallon truck tank |
| restocking trip to the depot | drive back to the yard / disposal site to unload (20 min) |
| optimal restocking policy | unload early when the next tank is likely not to fit |
| detour-to-depot policy | pump until the truck is full, then go unload |
| P(route duration ≤ L) ≥ 1 − α | P(the technician's day ≤ 600 min) ≥ 95% |

## What is implemented

- **Route cost** = expected driving + unloading minutes under the optimal unloading policy,
  computed exactly by the Yee & Golden stochastic DP (as in the paper), vectorized as
  transition matrices over remaining capacity.
- **Duration feasibility** = Cantelli (one-sided Chebyshev) bounds as a screen, then the exact
  distribution of the day's length by a forward pass over (remaining capacity, minute). The
  paper offers Chebyshev bounds, Monte Carlo with statistical inference, or analytical
  derivation of the duration distribution.
- **Routes** by column generation with a *heuristic* pricing (greedy best insertion of
  reduced cost from every seed customer), then the best set partition over all generated routes
  (HiGHS MIP via SciPy). This is price-and-branch, a heuristic, **not** the paper's exact
  branch-and-price with labeling and completion bounds.
- **Not in the paper**: (1) time on site (setup + pumping) and 20-minute unloads, plus a random
  lid-locating/digging time per job (0/15/45 min), so the day is random even without extra
  unloads; (2) a planning cost of 600 minutes per truck-day, so
  every planner first uses as few trucks as it can (the paper minimizes expected travel cost).

## Plans compared (each day, same jobs)

- `avg|dtd` — **average-day plan**: every house at its expected volume and expected service
  time, unloads placed optimally for those volumes, planned day ≤ 600 min; the driver pumps
  until full. The practitioner baseline.
- `pad|dtd` — **padded plan**: same, planned day ≤ 85% of the shift (510 min).
- `avgK|dtd` — average-day plan forced to use as many trucks as the thesis plan.
- `meanonly|opt` — optimal unloading, but only the *expected* day must fit.
- `thesis|opt` — **thesis method**: optimal unloading + P(day ≤ 600) ≥ 95%.
  `thesis90`, `thesis98`: the same at 90% and 98% (the paper's three levels).
- `*|opt` / `*|dtd`: the same routes executed with optimal unloading / pump-until-full.

Every plan is evaluated exactly under the same stochastic model and checked by Monte Carlo
(4,000 simulated days per plan per instance; `mc_*` fields).

## Data

- **Real**: housing units by number of bedrooms, Cherokee County, GA — ACS 2024 1-year, table
  B25041, via the Census Reporter API:
  https://api.censusreporter.org/1.0/data/show/latest?table_ids=B25041&geo_ids=05000US13057
  (retrieved 2026-11-06; hard-coded in the script).
- **Real rule**: Georgia rule 511-3-1-.05(3): 1,000 gallons for 1–4 bedrooms, +250 gallons per
  bedroom over four (https://www.law.cornell.edu/regulations/georgia/Ga-Comp-R-Regs-R-511-3-1-.05).
- **Synthetic** (labeled in the script): tank installed above code minimum (+0/250/500 gal with
  probability 0.55/0.30/0.15), rinse water (0/100/200 gal, 0.5/0.3/0.2), an 8% chance of a
  second tank or an oversized old system (+1,000 gal), dig time, setup time (25 min), pump rate
  (100 gal/min), 20-min unloads, 24 jobs a day at uniform random points in a box over Cherokee
  County, a hypothetical yard + disposal site in Canton.
- **Distances**: straight-line (haversine) × 1.3 detour factor at 45 km/h. No routing API used.

## Run

```
uv run --with numpy --with scipy python septic_routes.py        # 16 days -> results.json
uv run --with numpy --with scipy python figures_data.py         # adds "figures" to results.json
uv run --with numpy --with scipy python septic_routes.py 2      # quick 2-day check
```

Runtime ≈ 15 min on a laptop (905 s for the run behind `results.json`; most of it in the
90%/98% variants). Seed 2026; deterministic. `run.log` and `figs.log` are the console output.

## Headline numbers (results.json → aggregate, means over 16 days)

| plan | trucks/day | P(≥1 truck past 10 h) | E[overtime] min/day | drive+unload min/day |
|---|---|---|---|---|
| avg\|dtd (average-day) | 4.00 | 61.5% | 34.9 | 934 |
| avg\|opt | 4.00 | 50.7% | 25.3 | 878 |
| avgK\|dtd (forced to thesis truck count) | 5.00 | 42.1% | 20.1 | 917 |
| pad\|dtd (85% buffer) | 4.94 | 25.0% | 8.5 | 967 |
| meanonly\|opt | 4.00 | 48.0% | 22.7 | 857 |
| thesis90\|opt | 4.81 | 15.6% | 4.6 | 868 |
| **thesis\|opt (95%)** | **5.00** | **4.6%** | **1.1** | **878** |
| thesis\|dtd | 5.00 | 5.6% | 1.4 | 903 |
| thesis98\|opt | 5.00 | 1.3% | 0.3 | 877 |

Monte Carlo agrees with the exact numbers within 0.15 percentage points on every plan.
