# The customer who can't leave: unboundedness in bilevel pricing

Thesis series #5. Applies the unboundedness analysis in Chapter 3 of Bárbara Rodrigues's
PhD thesis, *Linear Bilevel Optimization and Applications in the Grid Integration of Energy
Storage* (University of Edinburgh, 2025; EURO Doctoral Dissertation Award 2026 finalist;
Chapter 3 is joint work with Margarida Carvalho, Nagisa Sugishita and Miguel F. Anjos), to a
setting the authors did not study: a food distributor in metro Atlanta setting case prices
while its restaurant accounts decide where to buy. The application is mine, not the authors'.

## Run

```
uv run --with gurobipy --with numpy python captive_pricing.py   # ~1 min -> results.json
```

`gurobipy` ships with a free size-limited license, which is enough for these models (about
1,000 variables each). The first run calls the BLS public API (no key) and caches the inputs
in `inputs.json`; later runs use the cache. The mileage rate is a constant in the script
(`IRS_MILEAGE_2026`), not fetched. Seed: 20261126.

## What it does

- **Leader**: a distributor sets one weekly price per case for 7 categories.
- **Followers**: 8 restaurant accounts per "book". Each buys its weekly cases at the
  cheapest total cost: delivered from the distributor, or hauled from a cash-and-carry
  store at shelf price plus its own time and mileage, up to what its vehicle carries in the
  trips it makes. In the "fixed" model a competing distributor's delivered quote is also
  available.
- **Reformulation**: the follower LPs are replaced by their KKT conditions. Strong duality
  makes the leader's revenue linear, so the single-level model is a linear program with
  complementarity constraints (LPCC), solved with Gurobi indicator constraints (no big-M).
- **Unboundedness certificate** (adaptation of model (U), Theorem 3.9 of the thesis): find a
  point of the LPCC and a direction, components in [-1, 1], such that the whole ray keeps the
  same complementarity pattern and the leader's profit grows. Positive growth means the
  bilevel model is unbounded. In the thesis the follower's dual vertex is fixed along the ray
  (the follower's objective does not depend on the leader). Here prices are in the follower's
  objective, so the duals also get direction components. That change is mine.
- **Compared with**: the high-point relaxation (HPR), a per-item check ("does someone need
  an item only we sell?"), a per-restaurant LP ("can this restaurant meet demand without
  us?", which is exact for this particular model and is used as ground truth), and "cap and
  solve" (bound prices at 1.5x to 10x cost and solve anyway).
- **Fixed model**: bilevel prices vs cost-plus 20% and vs matching the competitor's quote,
  each evaluated by re-solving every restaurant's own LP at those prices (optimistic ties;
  also pessimistic ties and a 1% price cushion).

## Data sources

Real:
- BLS CPI Average Price Data, South urban, mean of the 12 most recent monthly values
  available on 2026-11-06 (September 2025 to September 2026; October 2025 was not
  published): APU0300FF1101 chicken breast boneless, APU0300703112 ground beef 100%,
  APU0300708111 eggs grade A large, APU0300710212 cheddar, APU0300701312 long-grain rice,
  APU0300712311 field-grown tomatoes. https://www.bls.gov/cpi/factsheets/average-prices.htm
- BLS OEWS May 2025, Atlanta-Sandy Springs-Roswell MSA, SOC 35-1012 (first-line supervisors
  of food preparation and serving workers), hourly median wage $18.87, series
  OEUM001206000000035101208. Used as the cost of the person who does the cash-and-carry run.
- IRS 2026 business standard mileage rate in effect July 1 to December 31, 2026: 76 cents
  per mile (IR-2026-29; it was 72.5 cents from January 1 to June 30, IR-2025-128). The
  experiment is set in November 2026, so it uses 76 cents.
  https://www.irs.gov/tax-professionals/standard-mileage-rates

Synthetic (my assumptions, not data):
- Cash-and-carry shelf price = 85% of the BLS retail average, per case (40 lb chicken,
  40 lb beef, 15 dozen eggs, 20 lb cheddar, 50 lb rice, 25 lb tomatoes).
- Distributor landed cost = 80% of shelf. Competitor's delivered quote = 104% of shelf.
- A "house specialty" item with cost $48/case that the cash-and-carry does not stock.
- Restaurants: size factor U(0.6, 2.0) times a base basket (4, 3, 2, 2, 1, 2 cases/week),
  each category used with probability 0.85; specialty with probability 0.025.
  Vehicle car/SUV/van (10/18/40 cases per trip; probabilities 0.3/0.4/0.3), 1 to 3 trips a
  week, 4 to 20 miles one way, 1.5 to 2.5 hours per run.
- The vehicle limit is a modelling assumption on purpose: it is the kind of constraint that
  makes a customer "captive" in a model when in reality they would make another trip or call
  another supplier.

## Results (from results.json)

- 200 books: 59 well-posed, 141 unbounded (41 because of the specialty item, 100 because a
  restaurant's weekly need exceeds its vehicle limit, with no single item to blame).
- Certificate: right status on 200/200 books; the captive restaurants it points at match
  the ground truth on 141/141; re-solving every restaurant's LP at prices along the ray
  confirms the profit keeps rising on 141/141.
- HPR: unbounded on 200/200, including the 59 well-posed books (inconclusive, as Table 2.1
  of the thesis says it can be).
- Per-item check: caught 41 of 141; missed all 100 capacity-caused ones.
- Cap and solve: on specialty books the solver priced the specialty at the cap at every cap
  (41/41). On capacity books, prices hit the cap in 0, 0, 2, 21 and 51 of 100 books at caps
  1.5x, 2x, 3x, 5x and 10x cost: at 10x, 49 broken books still looked normal.
- Speed: certificate 0.015 s per book on average (max 0.026 s); the per-restaurant LP check
  0.0004 s (timings vary run to run). For this simple model the shortcut is enough; the general certificate is
  insurance for models where no shortcut is known.
- Fixed model (competitor quote added): certificate says bounded on 200/200. Mean weekly
  gross profit per book: bilevel $2,949, cost-plus 20% $2,225, match-the-competitor $680
  (keeps 39% of volume). Restaurants pay $14,074 vs $13,349 under cost-plus. With ties
  broken against the distributor, bilevel prices earn $2,427 and keep 77% of volume; with a
  1% cushion they earn $2,808 and keep 99.9%.

## Files

- `captive_pricing.py`: the experiment.
- `inputs.json`: cached BLS/IRS inputs.
- `results.json`: every number, per book and summarized.
- `run.log`: console output of the run used for the write-up.
