# Thesis series #2: dual sourcing with Deep Controlled Learning, at an independent pharmacy

Code for the BIS post "When is the rush order worth it? An award-winning thesis, tested at a pharmacy" (draft), applying Chapter 3 of Fabian Akkerman's
PhD thesis *Machine Learning for Sequential Decisions in Logistics* (University of Twente,
2025; EURO Doctoral Dissertation Award 2026) to a new setting. The chapter is joint work with
Nils Knofius, Matthieu van der Heijden and Martijn Mes, published as "Solving dual sourcing
problems with supply mode dependent failure rates", *International Journal of Production
Research* (2025), doi:10.1080/00207543.2025.2489755 (preprint arXiv:2410.03887). The
authors' own code (C++, DynaPlex) is at https://github.com/frakkerman/dual_sourcing_am.
This repository is an independent Python re-implementation of the ideas, not their code.

The pharmacy application is mine (Renan Butkeraites, BIS), not the authors'.

## How to run

```
uv run --with numpy --with pandas --with torch --with scipy python pharmacy_dual_sourcing.py
python3 headline.py   # adds the derived numbers quoted in the article to results.json
```

It downloads the data on first run (to `data/`), uses 8 worker processes and wrote
`results.json` in about 23 minutes on an Apple-silicon laptop (10 cores). Seed: 20261117.

## Data

- **Demand:** "Pharma sales data" by Milan Zdravković, Kaggle,
  https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data (CC BY-NC 4.0). Built from
  600,000 point-of-sale transactions of one individual pharmacy, 2014–2019, 57 drugs grouped
  into 8 ATC categories, resampled to daily totals. The publisher notes the data was
  pre-processed (outlier treatment, missing-value imputation), which is why some daily values
  are fractional; I round them to whole packs. The script fetches the identical `salesdaily.csv`
  from a public GitHub mirror (`DATA_URL` in the script; SHA-1
  `8a23cf2a3eb364d6feb45d03237879fb00dab57a`, identical in two independent mirrors). The data
  file is not redistributed here.
- I use 7 of the 8 categories. N02BE (paracetamol-type, ~30 packs/day) is left out: its order
  quantities exceed the action space I could afford to train a classifier over.
- Train / tune: 2014-01-02 to 2017-11-30. Warm-up: December 2017. Test: the real days from
  2018-01-01 to 2019-10-08.

Nothing else is real data. All costs below are illustrative assumptions.

## Model (one drug category = one stock-keeping unit)

Each morning the pharmacist sees packs on hand minus packs owed to waiting patients, and the
regular orders already on their way. She can place

- a **regular order**: arrives L mornings later (L = 1–5), at the regular pack price;
- a **rush order**: arrives the same morning, before patients come in, at +20% per pack plus
  a $15 courier fee per order.

Unmet demand waits (backorder), at $5 + 10% of the pack price per pack per day. Holding costs
30% of the pack price per year. Cost reported = holding + waiting + the extra paid for rush
orders (the regular purchase price is the same for every policy in the long run and is left
out). Order caps: 40 regular and 12 rush packs per day, for every policy.

Grid: 7 categories × pack price {$10, $60, $300} × L {1, 3, 5} = 63 training settings, plus 28
unseen settings (price {$25, $150} × L {2, 4}) used only to test generalisation.

What is not transferred: the chapter's distinctive feature is that parts from the two sources
fail at different rates, so the sourcing decision changes future demand. A pill from the rush
order works like one from the regular order, so that part has no analogue here. With equal
failure rates, the chapter's IWA heuristic (Eqs. 3.18–3.21) reduces to the underlying dual
index policy, which is my DIP baseline.

## Policies

| Name | What | Tuning |
|---|---|---|
| BSP | single-source base stock (regular or rush, whichever is cheaper), the chapter's baseline | grid search by simulation |
| PAR | par level on the regular order + rush exactly what is owed to patients | grid search by simulation |
| DIP | dual index policy (Veeraraghavan & Scheller-Wolf 2008) | full 2-D grid search by simulation |
| DCL | Deep Controlled Learning (Temizöz et al. 2025), one network per setting | approximate policy iteration |
| EPL | DCL embedded in EPL: one network for all settings; setting parameters are inputs | approximate policy iteration on a parameter grid |

Simulation tuning uses 24 bootstrapped two-year paths of training demand per candidate.

### DCL as implemented (simplified from the thesis)

Approximate policy iteration (thesis Appendix 3.G, Algorithm 6): starting from a policy π,
visit states along a trajectory; in each state, try candidate orders on M common demand
scenarios over H days, following π afterwards, and label the state with the cheapest
candidate; train a neural classifier on the (state, best order) pairs; it becomes the next π.
Differences from the thesis, to fit a laptop:

- π₀ is the setting's tuned dual index policy (a warm start), not a random policy. See
  "First run" below for what happened with an untuned textbook start.
- M = 128 scenarios and H = 30 days (thesis: M = 2,000, H = 500–1,000); N = 1,200 states per
  iteration per setting (thesis: 5,000–20,000); 3 iterations; best iterate picked on a
  validation simulation of training demand (never the test period).
- Candidates are restricted to ±3 packs around π's regular order and 0–4 rush packs.
- The classifier has two softmax heads (regular 0–40, rush 0–12), 2×64 hidden units
  (EPL: 3×64, one extra layer as in the thesis), trained with Adam in PyTorch.
- State features follow the thesis's list (state variables, inventory position, expected
  demand and spread) plus the setting's cost parameters.

### EPL

One network trained on states drawn from all 63 training settings × 10 demand scales
(0.5× to 2× the category's demand, the κ = 10 grid of the thesis), with the setting's
parameters in its input; the setting is resampled at every episode reset (thesis §3.4.3).
24,000 states per iteration, 3 iterations. EPL's π₀ is the dual index policy tuned for each
(setting, scale) pair; at test time EPL needs no tuning, only the item's numbers.

## Tests

1. **stationary**: 100 fresh one-year bootstrap paths from the training distribution (the
   thesis's i.i.d. setting); paired comparison against DIP with 95% CIs.
2. **replay_static**: the real 2018-01-01..2019-10-08 days, every policy frozen as trained
   or tuned on 2014–2017.
3. **replay_rolling**: the same real days, re-planned every 91 days from the trailing 365
   days. BSP, PAR and DIP are re-tuned by simulation; EPL simply receives the new mean and
   spread as inputs, without retraining; DCL stays frozen (it would need retraining).

## Results

See `results.json` (`summary` for the aggregates; `settings` for every setting, policy and
test, with cost components, fill rate, stock on hand and rush orders per 30 days).

Headline numbers (`headline` block; totals over the 63 training settings):

- **Stationary** (demand like 2014–2017): the rush order is worth 0.2% (DIP vs BSP).
  DCL is 1.1% and EPL 1.2% above DIP. DCL beats DIP by >1% in
  1 of 63 settings and loses by >1% in 39.
- **Real days, frozen**: PAR 13.4% and DIP 11.0% below BSP; DCL 8.0% and EPL
  6.2% above BSP.
- **Real days, re-planned quarterly**: re-planning cuts BSP by 26.6%; PAR is a further
  9.2% lower (33.3% below frozen BSP). EPL with updated inputs, no
  retraining: 30.9% below frozen EPL, level with re-tuned BSP (0.1%).
- **By regular lead time** (PAR vs BSP, re-planned real days): 1 day 0.3%, 3 days
  4.1%, 5 days 14.8%.
- **Unseen settings** (28, never trained on): EPL 1.7% above DIP (stationary);
  4.9% below BSP on re-planned real days (PAR: 6.6%).

Not everything helped: re-planning raised costs for anxiolytics (N05B), whose sales dipped in
2017 and recovered in 2018. Learned policies never beat the tuned dual index rule in total.

### First run (kept for transparency)

`results_run1_textbook_start.json` is an earlier full run with DCL/EPL started from the untuned
textbook rule, M = 48 scenarios, ±4/0–5 candidates. Stationary totals (63 settings): BSP
144.1, DIP 143.8, DCL 150.6, EPL 156.8 (DCL 4.8% and EPL
9.0% above DIP). Policy iteration was still improving after 3 rounds, so I
switched to the warm start rather than spending hours more compute; the article reports the
second run.

