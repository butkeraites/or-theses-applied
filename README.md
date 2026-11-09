# Award-winning OR theses, applied

Code and results behind the BIS series that takes an award-winning operations research
doctoral thesis and applies one of its ideas to a setting the author did not study. Each
folder is one post: a small, self-contained experiment, run locally, with the raw output
that every number in the write-up comes from.

| # | Thesis | Applied to | Write-up |
|---|---|---|---|
| 1 | Bart van Rossum, *Optimising Fair Work Allocation: Applications in Railway Crew Planning* (Erasmus University Rotterdam, 2025; EURO Doctoral Dissertation Award 2026 finalist) | Restaurant server schedules over the 2026 Braves season | [Who closes on Friday?](https://bisdecisions.com/insights/who-closes-on-friday) |

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

## Provenance

The code, analysis and write-ups were produced with AI assistance (Claude, Anthropic) under
the direction of Renan Butkeraites, who is responsible for the content. All experiments
ran locally.

## License

Code: MIT. Results and text: CC BY 4.0. MLB schedule data belongs to its owner and is
included only as a cache of a public API response.
