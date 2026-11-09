# Thesis series #6: Tao Wen, EAGLE (thesis Chapter 6), applied to food recalls

**Question.** A bad lot is moving through a food supply chain. Where should a few inspection
points sit so it gets caught early? And once the first few inspections come back positive,
which stage started it?

**Source.** Tao Wen, *Data-Driven Modeling and Analytics for Information Propagation and
Decision-making in Social Networks*, PhD thesis, The University of Manchester, 2025. It was
supervised by Yu-wang Chen and Tahir Abbas Syed (thesis acknowledgements; Research
Explorer) and was a finalist for the EURO Doctoral Dissertation Award 2026.

This experiment uses Chapter 6 (pp. 144–170), "Early identification of information
propagation sources through generalized Bayesian inference in large-scale social networks".
The thesis calls the method **EAGLE**. Per Sec. 1.6 it is joint work with Yu-wang Chen, Jie
Zhao, Changchun Pan, Tahir Abbas Syed, Kang Hao Cheong and Yong-Yeol Ahn, and it is under
review. This README describes it from the thesis only.

The earlier version of this draft used Chapter 4 (ERIUE, *Omega* 2024). That experiment is
kept in `ch4-eriue/` for reference. The article does not use it.

## How to run

```
uv run --with networkx --with numpy --with scipy --with pandas --with openpyxl \
    python eagle_supply.py
```

On the first run it downloads the data into `data/` (git-ignored; see "Data"), then writes
`results.json`. The seed is fixed (20261201), and the run takes about a minute (66 s) on a
laptop. To check only the data, run `python willems_data.py`.

## What EAGLE does (thesis Sec. 6.3) and how I implemented it

**1. Placing the observers (Sec. 6.3.1).** EAGLE picks |O| = ξ|N| observers. The goal is that
every candidate source has observers among its one-hop neighbours, not too few and not too
many. It maximizes F(O) = Σ_c ω_l over candidates c, where l is the number of observers next
to c (Eqs. 6.3–6.4):

| Observers next to c (l) | ω_l |
| --- | --- |
| 0 | −λ |
| 0 < l < round(k̄) | exp(−(round(k̄) − l)/2) |
| l ≥ round(k̄) | 1 |

It solves this with a discrete moth-flame optimizer (Eqs. 6.5–6.8). The thesis specifies:

- a moth made of the top-|O| nodes by degree, with the others degree-biased (ϑ × k_i);
- a search area of the top κ|O| nodes by degree, with κ = 4;
- a sigmoid crossover position with b in [−5, 5] and a random threshold γ;
- two-way exchange with the best moth for the top τ = 5 moths, one-way for the rest;
- mutation for the top τ;
- duplicates repaired.

My choices for what the thesis leaves open: 20 moths, 60 iterations, mutation probability
0.1, λ = 1. The thesis's two-way exchange could change the best moth. I keep the best moth
unless the exchange improves it, and I always keep the best solution found so far.

**2. Collecting the information (Sec. 6.3.2).**

- Each exposed observer reports (o_i, t_i, s_i): when it first received the problem, and
  from whom.
- Candidates are the non-observers that are closer on average to the exposed observers than
  to the unexposed ones (Eq. 6.9).
- Pairwise likelihood: P(v) = exp(−|ε(v)|) (Eq. 6.11). Here ε = 1 − (d(v,s_i) −
  d(v,s_j)) / (t_i − t_j), or d(v,s_i) − d(v,s_j) when the times are equal. Pairs of exposed
  observers form a ring of |O+| pairs.
- Group likelihood: the Pearson correlation between report times and distances, leaving one
  observer out each time (Eq. 6.13, used when |O+| ≥ 4).

**3. Generalized Bayesian inference (Sec. 6.3.3).**

- Each likelihood vector becomes a weighted belief distribution: candidates whose likelihoods
  are within φ of each other form one focal set. The mass is ψ × the set's mean likelihood
  (Eq. 6.18).
- Credibility ψ is (k_oi + k_oj)/(2 max k) for pairs (Eq. 6.19) and (1 − k_oi/Σk)/2 for
  groups (Eq. 6.20).
- The pieces are combined with the recursive evidential-reasoning rule (Eqs. 6.21–6.23).
  The leftover ignorance is removed (Eq. 6.24), and candidates are ranked by pignistic belief
  η_v = Σ_{θ∋v} m_θ/|θ| (Eq. 6.25).
- Focal sets holding the lowest belief are pruned (υ = 2%).

**My choices where the thesis is not specific:**

- φ = 0.05.
- I rescale each piece's cluster masses to sum to ψ, so that Σm + m_P(C) = 1 as Eq. 6.17
  requires.
- I use raw hop counts and time steps for d′ and t′. The ratio in Eq. 6.10 is
  scale-invariant.
- Negative correlations in Eq. 6.13 are set to 0, since a mass cannot be negative.
- Focal sets are capped at 400.

**What changes in a supply chain (my adaptation, not the thesis's):**

- **Direction.** The thesis's networks are undirected. Here the graph is directed (supplier →
  plant → transport → distribution center) and the problem travels only downstream.
- **Distances.** d(v, s) is the directed hop distance from v down to s, with unreachable set
  to |N|.
- **What an observer watches.** An inspection point checks what it receives, so it watches
  its upstream neighbours. "Degree" in the deployment heuristics and ψ is therefore in-degree.
- **k̄.** It is the mean out-degree. On the cereal chain round(k̄) = 1, so F reduces to
  "number of candidates watched by at least one inspection point" (minus λ times those not
  watched).
- **Propagation.** It is a heterogeneous SI model (the thesis's HSI), downstream only. Each
  link gets its own pass-on probability ~ U(0.1, 0.9). Every step (one shipping cycle), each
  contaminated stage tries each clean downstream neighbour. The model runs for at most 40
  steps. The thesis does not give HSI parameters.
- **Sources.** They are drawn uniformly from stages with at least one downstream link; a
  problem at a terminal distribution point cannot spread.

## What is compared

**Deployment** (thesis Sec. 6.4.2 compares random and degree-based):

- Random.
- Degree-based: the stages that receive from the most suppliers.
- **Greedy coverage.** This is a textbook OR baseline, not in the thesis: add the stage that
  raises F the most.
- EAGLE's moth-flame.

The metrics are:

- **Detection delay:** the first step any inspection point is contaminated, averaged (the
  thesis's Fig. 6.4 metric). An inspection point at the source counts as caught at step 0.
- **Share of runs never detected:** no inspection point is downstream of the source.

**Source inference.** All methods use the EAGLE deployment. Following the thesis, only runs
where the source is not itself an observer count. Each method sees the first k = 2, 3 or 4
positive inspections, plus any ties at the k-th report's time step.

- **EAGLE**, as above.
- **Earliest sender:** the stage(s) that shipped the first positive lot. This is the obvious
  trace-back.
- **Common upstream:** the stage(s) that can reach every flagged sender with the smallest
  mean distance (a common-supplier search).
- **Random candidate:** one random member of EAGLE's candidate set.

The metrics are:

- **Exact (top-1):** a single answer, and it is right.
- **Top-3:** the source is among the three highest-ranked. For the set baselines, the set has
  three or fewer members and contains the source.
- **Error distance:** the mean undirected hop distance from the inferred set to the source
  (Eq. 6.26).
- **F-score** (Eq. 6.27).
- **Inferred set size.**

The headline setting, ξ = 10% and k = 3, was fixed before looking at results. The featured
chain gets 400 runs; every other chain with at least 20 stages and 5 possible sources gets
100 runs, for both ξ = 10% and 20%.

## Data

**Willems (2008).** Sean P. Willems, "Data Set — Real-World Multiechelon Supply Chains Used
for Inventory Optimization", *Manufacturing & Service Operations Management* 10(1):19–23,
https://doi.org/10.1287/msom.1070.0176.

- 38 supply chains that companies had implemented in practice, with the data disguised.
- A stage is the processing of one SKU at one location. Arcs are precedence relations, and
  stages with no outgoing arcs are demand stages.
- Every stage is labelled Part_, Manuf_, Trans_, Dist_ or Retail_ (the paper's Table 2).
- Per-stage fields (Table 3): stageCost, stageTime, stDevStageTime, avgDemand,
  stdDevDemand, serviceLevel, maxServiceTime, relDepth, stageClassification, xPosition and
  yPosition. This experiment uses only the network and the classifications.

**Copy used.** The official data is the journal's online appendix (e-companion), which sits
behind a bot check. I used a **public Google Sheets copy** linked from
https://gist.github.com/sderaco/921848b968ffdd54864c80ee0ff28231. `willems_data.py`
downloads it on first run. **The data is not redistributed in this repository.** The paper's
INFORMS terms limit the article to research, teaching and private study.

**Check against the paper** (`willems_data.py`, stored in `results.json` → `data_check`):

- All 38 chains match Table 1: chain number, company identifier and SIC code. There are no
  mismatches.
- Stage counts run from 8 to 2,025 and arc counts from 10 to 16,225, exactly the ranges the
  paper states.
- All 16 stage names shown in the paper's Figure 1 (chain 12) are present.
- One quirk: the copy lists every stage of chain 36 twice (2,902 rows, 1,451 unique stages).
  The graph uses unique stages.

This is consistency with the published paper, not proof that the copy equals the appendix.

## Results

All numbers come from `results.json`; see the article and the LinkedIn kit notes for which
fields they use.

Key numbers. "Exact" means a single answer, and it is right. F is the F-score of Eq. 6.27.
Err is the error distance in hops.

**Cereal chain 12, ξ = 10% (9 inspection points), first 3 positives, 400 runs:**

- **Placement:**
  - Random placement misses the bad lot entirely in 31.5% of runs.
  - Degree-based, greedy and EAGLE's moth-flame placement all miss none.
  - Moth-flame chose exactly the nine plants that degree-based picks.
  - Mean detection delay is 1.52 steps (degree and moth-flame), 1.61 (greedy) and 2.92
    (random).
- **Inference with lot records** (142 runs reached 3 positives):

  | Method | Exact | F | Err |
  | --- | --- | --- | --- |
  | EAGLE | 38.7% | 0.48 | 1.32 |
  | Earliest sender | 82.4% | 0.82 | 0.18 |
  | Common upstream | 82.4% | 0.82 | 0.18 |
  | Random candidate | 9.9% | 0.10 | 2.37 |

- **Without lot records** (inspections know when, not from whom):

  | Method | F | Err |
  | --- | --- | --- |
  | EAGLE | 0.44 | 1.43 |
  | Common upstream | 0.39 | 1.48 |
  | First site's suppliers | 0.20 | 1.90 |

**Across 35 chains** (≥ 20 stages and ≥ 5 possible sources; ξ = 10%):

- **Placement:** median share of runs never detected is 49% for random, 19% for
  degree-based, 13% for greedy and 10% for moth-flame. Moth-flame left fewer runs undetected
  than degree-based in 19 chains and more in 5. Against greedy it left fewer in 9 and more
  in 18; greedy reached a higher fitness F in 21 chains.
- **Inference, k = 3** (28 chains with ≥ 10 runs reaching 3 positives): median F-score.

  | Method | Median F |
  | --- | --- |
  | Common upstream | 0.99 |
  | Earliest sender | 0.96 |
  | EAGLE | 0.50 |
  | Random | 0.02 |

  EAGLE was worse on exact hits than earliest sender in 27 of 28 chains.
- **Without lot records:**

  | Method | Median F |
  | --- | --- |
  | Common upstream | 0.27 |
  | EAGLE | 0.24 |
  | First site's suppliers | 0.15 |

  EAGLE beat first site's suppliers on F in 25 of 28 chains, and common upstream in 3.

**Mechanism.** Eq. 6.11 compares *differences* in distance to each observer's sender with
*differences* in report time. In these shallow chains (depth ≤ 4 on chain 12), the first
positive inspection usually received the lot straight from the source. When two reports name
the same sender, d(v,s_i) − d(v,s_j) = 0 for every candidate, and the evidence cancels. The
"who shipped it" label, read directly, already names the source. With p < 1 per link, a hop
also takes more than one step on average, so the "one hop per step" assumption behind
Eq. 6.10 is only approximate.

`ch4-eriue/` holds the earlier Chapter 4 experiment (ERIUE on the same chains). It is not
used in the article.

## License note

The MIT license of this repository covers the code only. The Willems (2008) supply-chain data is not redistributed: `willems_data.py` downloads it on first run, and its INFORMS terms (research, teaching and private study) apply.
