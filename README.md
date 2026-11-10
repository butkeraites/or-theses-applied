# Award-winning OR theses, applied

Code and results behind the BIS series that takes an award-winning operations research
doctoral thesis and applies one of its ideas to a setting the author did not study. Each
folder is one post: a small, self-contained experiment, run locally, with the raw output
that every number in the write-up comes from. Folder numbers follow the order the drafts
were written; the # column is the order of publication.

| # | Thesis | Applied to | Write-up |
|---|---|---|---|
| 1 | Bart van Rossum, *Optimising Fair Work Allocation: Applications in Railway Crew Planning* (Erasmus University Rotterdam, 2025; EURO Doctoral Dissertation Award 2026 finalist) | Restaurant server schedules over the 2026 Braves season | [Who closes on Friday?](https://bisdecisions.com/insights/who-closes-on-friday) |
| 2 | Alexandre Florio, stochastic vehicle routing (University of Vienna, 2019; EURO Doctoral Dissertation Award 2021 winner) | Septic pump-out trucks and the ten-hour day | [Back within ten hours?](https://bisdecisions.com/insights/septic-trucks-and-the-ten-hour-day) · `03-florio-septic-routes/` |
| 3 | Fabian Akkerman, *Machine Learning for Sequential Decisions in Logistics* (University of Twente, 2025; EURO Doctoral Dissertation Award 2026 winner) | Restocking at a pharmacy: regular order or rush order | Under the author's review; see `02-akkerman-pharmacy-dual-sourcing/README.md` |
| 4 | Hannah Bakker, *Data and Decisions in Discrete Location* (KIT, 2024; EURO Doctoral Dissertation Award 2025 finalist) | Café siting in Cobb County, Georgia | Under the author's review |
| 5 | Bárbara Rodrigues, linear bilevel optimization (University of Edinburgh, 2025; EURO Doctoral Dissertation Award 2026 finalist) | Pricing for restaurants that can't switch supplier | Under the author's review |
| 6 | Tao Wen, information propagation and source localization (University of Manchester, 2025; EURO Doctoral Dissertation Award 2026 finalist) | Tracing a bad lot in a supply chain | Under the author's review |

The applications are mine, not the thesis authors'. Any error in them is mine too.

## 01 · Who closes on Friday? (`01-vanrossum-fair-shifts/`)

Applies the fairness-over-time strategy of van Rossum, Chen & Lodi
([ATMOS 2023](https://doi.org/10.4230/OASIcs.ATMOS.2023.17); Chapter 5 of the thesis) to a
12-server restaurant near Truist Park, week by week for 26 weeks of the 2026 Braves season.
Each week: solve the cheapest schedule (CP-SAT), pick the most even among the equally cheap
ones, and assign its lines best-to-worst.

```bash
cd 01-vanrossum-fair-shifts
uv run --with ortools python fair_shifts.py   # -> results.json (26 weeks x 5 solves, ~minutes)
uv run --with ortools python extras.py        # -> extras.json (perfect-information benchmark,
                                              #    one night of the even schedule)
```

- `results.json`: every weekly solve (hours, sour points per line, solver status), the gap
  trajectories for blind rotation (mean of 1,000 seasons), as-solved, best-to-worst and the
  alpha-fair strategies, and the season totals.
- `extras.json`: the offline benchmark (all 26 weeks known in advance; best possible final
  gap 1, worst in-season gap 2 with the even lines) and the shifts of the week of
  Sep 7, 2026. CP-SAT can return a different schedule with the same cost when re-solved.
- `braves_2026_home.json`: home games from the public MLB Stats API (cached).
- `run.log`: console output of the run behind `results.json`.

Demand is synthetic in shape; only the game schedule is real. Servers are interchangeable,
and the sour-point weights are my assumption.

## #3 · When is the rush order worth it? (`02-akkerman-pharmacy-dual-sourcing/`)

Applies Deep Controlled Learning and Endogenous Parameterised Learning from Chapter 3 of the
thesis (Akkerman, Knofius, van der Heijden & Mes, IJPR 2025) to an independent pharmacy
choosing between a regular order and a same-day rush order, on six years of real daily sales.
Details, data source and how to run: the folder's README.

## The others

Each folder has its own README with the data sources, the model and how to run it:
`03-florio-septic-routes/`, `04-bakker-cafe-siting/`, `05-rodrigues-captive-pricing/`,
`06-wen-supply-trace/`.

## Provenance

The code, analysis and write-ups were produced with AI assistance (Claude, Anthropic) under
the direction of Renan Butkeraites, who is responsible for the content. All experiments
ran locally.

## License

Code: MIT. Results and text: CC BY 4.0. Third-party data keeps its own terms: MLB schedule
data is included only as a cache of a public API response; the OpenStreetMap-derived files in
`04-bakker-cafe-siting/data/` are ODbL (see `LICENSE-DATA.md` there); the pharmacy sales (02)
and Willems (2008) supply-chain data (06) are downloaded on run, not redistributed.
