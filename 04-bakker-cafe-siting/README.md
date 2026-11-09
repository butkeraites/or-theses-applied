# Bakker (KIT, 2024) applied: which café sites show up in every good plan?

Thesis: Hannah Bakker, *On the interplay between data and decisions in discrete location
problems*, Karlsruhe Institute of Technology, 2024 (referees: Prof. Stefan Nickel,
Prof. M. Grazia Speranza). Finalist, EURO Doctoral Dissertation Award 2025.
Full text (CC BY-SA 4.0): https://doi.org/10.5445/IR/1000169660

The application (a café brand choosing sites in Cobb County, GA) is mine, not the thesis's.

## What is borrowed from the thesis

| Thesis | Here |
|---|---|
| Profit-maximizing CFLP, eqs. (1.1)–(1.5) | same model; I add the standard valid inequality x_ij ≤ y_i |
| k best alternative solutions via no-good cuts, eq. (2.7) | k = 20 |
| Persistence C(i,S) (Def. 2.5), kernel persistence KER(S) (Def. 2.4, eq. 2.6) | same |
| Random counterpart E(KER) (eqs. 2.8–2.9, Appendix A) | Monte Carlo, 20,000 draws |
| Pseudo-reduced costs prc% (Def. 2.6), Dep(i)/Rep(i) (2.25–2.26), Dep_avg (Def. 2.13), weakly connected components | same, one re-solve per open site |
| ADD greedy (Algorithm 3, after Jacobsen 1983) | same idea; allocation solved exactly by LP instead of greedily |
| Ch. 7: deterministic optimum per scenario (S_det), two-stage stochastic program, expected-value solution | simplified: sample-average SP with 30 scenarios, both plans evaluated on 300 fresh scenarios; no adjustable robust counterpart |

## Data (all public)

- **Households and median household income by census tract**: U.S. Census Bureau, ACS
  2020–2024 5-year estimates, tables B11001 and B19013, from the table-based summary
  files (no API key needed):
  https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/
  Margins of error (90%) are used for the demand scenarios (SE = MOE / 1.645).
- **Tract centroids**: 2024 Census Gazetteer, Georgia tracts:
  https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_gaz_tracts_13.txt
- **County boundary**: TIGER/Line cartographic boundary, cb_2024_us_county_500k:
  https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip
- **Candidate sites**: every Publix and Kroger supermarket inside Cobb County in
  OpenStreetMap (Overpass API, `shop=supermarket`), © OpenStreetMap contributors, ODbL.
  43 sites (26 Publix, 17 Kroger), used as proxies for grocery-anchored shopping centers.
- **Driving distances**: public OSRM demo server (`router.project-osrm.org`, table service),
  tract centroid → site, fetched once and stored in `data/drive_miles.json`.

## Synthetic assumptions (mine; change them in `siting.py`)

- Demand: 0.10 addressable visits per household per week, scaled by
  (tract income / county median tract income)^0.5.
- $5 contribution margin per visit, minus $0.75 per driving mile (a stand-in for lower visit
  frequency from farther away); tracts more than 6.7 miles from a site cannot be served by it.
- Two formats: full café (4,000 visits/week, base fixed cost $8,500/week) and small counter
  (1,200 visits/week, $2,600/week). Fixed cost is scaled by 0.85–1.15 with the local
  household-weighted income rank (higher rent where incomes are higher).
- Demand scenarios: households and income drawn from their ACS sampling distributions, then
  each tract's demand multiplied by U(0.8, 1.2) (the thesis's perturbation level p = 20%).
- Practitioner rule ("trade-area count"): rank sites by households within a 3-mile drive,
  take the top ones (same number as the optimum), skipping sites within ~2 miles of a pick.

## Run

```bash
uv run --with numpy --with pyshp --with shapely python fetch_data.py   # only to refresh data/
uv run --with numpy --with scipy python siting.py                       # ~10 min, writes results.json
```

Solver: HiGHS through `scipy.optimize.milp`. All MIPs in `results.json` were solved to
optimality (relative gap 1e-6; the stochastic programs to 1e-4, both reported as optimal).

## Files

- `fetch_data.py` – downloads and filters the public data into `data/`.
- `siting.py` – the experiment; every number in the write-up comes from `results.json`.
- `data/` – tracts.csv, candidates.csv, drive_miles.json, cobb_outline.json.
