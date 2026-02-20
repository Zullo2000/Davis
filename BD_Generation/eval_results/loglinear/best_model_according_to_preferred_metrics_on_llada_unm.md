# Best Model Selection: LLaDA Unmasking — Preferred Metrics Analysis

**Date:** 2026-02-20
**Scope:** LLaDA unmasking only, log-linear schedule, all remasking variants
**Source data:** `eval_results/loglinear/comparison.md` (22 methods, 5 seeds each)
**Personal Best:** llada_topp0.9_remdm_confidence_tsw0.5

LLaDA unmasking with top-p 0.9 nucleus sampling, confidence-based remasking (per-position remasking probability proportional to softmax of negative confidence, no eta parameter, budget set by noise schedule), t_switch 0.5 (remasking active only in the second half of denoising), 100 sampling steps, 1000 samples per seed across 5 seeds.

---

## Priority Metrics

1. **Novelty x Mode coverage (weighted)** — high mode coverage only matters with high novelty
2. **Spatial transitivity** — physical realizability of generated graphs
3. **Cond. edge TV (weighted)** — conditional edge fidelity per room-type pair
4. **Type-cond. degree TV (weighted)** — per-type connectivity fidelity
5. **Node TV** — room type distribution fidelity

JS divergence was evaluated for metrics 3-5 and found redundant (see Section 7).

---

## 1. Novelty x Mode Coverage (weighted)

Novelty is nearly saturated for all remasking methods (>=0.9988), so the
differentiator is mode coverage (weighted). Simple product:

| Method | Novelty | Mode cov (wt.) | Product |
|--------|:-------:|:--------------:|:-------:|
| topp_no_remask | 0.975 | 69.6% | **0.678** |
| cap eta=0.2 | 1.000 | 67.2% | 0.671 |
| cap eta=0.4 | 0.999 | 71.4% | **0.713** |
| cap eta>=0.6 | 0.999 | 70.6% | 0.705 |
| conf tsw=0.3 | 0.999 | 73.2% | **0.731** |
| conf tsw=0.5 | 1.000 | 72.7% | 0.727 |
| conf tsw=0.7 | 1.000 | 72.6% | 0.725 |
| conf tsw=1.0 | 0.999 | 73.3% | **0.732** |

**Observations:**

- **Confidence remasking consistently beats cap** on this combined metric:
  0.725-0.732 vs cap's 0.671-0.713. The best cap (eta=0.4, 0.713) falls below
  the worst confidence (tsw=0.7, 0.725).
- **t_switch is a non-critical parameter for confidence.** The spread across
  tsw=0.3->1.0 is only 0.007 (0.725-0.732). The self-regulating nature of the
  confidence softmax makes this expected: at high noise levels, few positions
  have enough decoded context to generate confident predictions, so remasking
  naturally throttles itself.
- **cap eta saturates at 0.4.** eta 0.4->1.0 is essentially flat (within noise),
  meaning once you remask enough positions per step, additional remasking budget
  doesn't help -- the model simply re-predicts the same tokens from similar context.
- **cap eta=0.2 is actually worse than no remasking** on this metric (0.671 vs
  0.678). Too little remasking: not enough correction cycles to improve diversity,
  but enough to introduce edge distribution perturbation that slightly hurts coverage.

**Winner:** Confidence tsw=1.0 (0.732), closely followed by tsw=0.3 (0.731).

---

## 2. Spatial Transitivity

| Method | Transitivity | H-consistent | V-consistent |
|--------|:----------:|:----------:|:----------:|
| topp_no_remask | **99.9%** | 99.9% | 100.0% |
| cap eta=0.2 | 98.3% | 98.5% | 99.9% |
| cap eta=0.4 | 98.1% | 98.2% | 99.8% |
| cap eta>=0.6 | 98.4% | 98.5% | 99.9% |
| conf tsw=0.3 | 98.2% | 98.4% | 99.8% |
| conf tsw=0.5 | 98.5% | 98.6% | 99.8% |
| conf tsw=0.7 | 98.4% | 98.6% | 99.8% |
| conf tsw=1.0 | **98.7%** | 98.8% | 99.9% |

**Observations:**

- All remasking costs ~1.2-1.8% transitivity vs baseline. This is the price of
  stochastic re-prediction: some re-drawn edge types create horizontal ordering
  cycles that the original deterministic prediction avoided.
- **conf tsw=1.0 is the best remasking variant** at 98.7%. Always-on confidence
  remasking gives the most correction cycles, and the confidence-weighted
  allocation preferentially re-masks the uncertain (and likely contradictory)
  positions.
- The violations are almost entirely horizontal (H-consistent ~98.2-98.8% vs
  V-consistent ~99.8-99.9%). Horizontal spatial relationships (left-of, right-of)
  are harder for the model to keep cycle-free, likely because horizontal adjacency
  has more ambiguity in real floorplans.
- **Cap and confidence are indistinguishable on transitivity.** The ~0.5% spread
  across all remasking methods falls within noise (std +/-0.2-0.4%).

**Winner:** conf tsw=1.0 (98.7%), but all methods are within noise.

---

## 3. Cond. Edge TV (weighted)

| Method | Cond. edge TV (wt.) |
|--------|:-------------------:|
| topp_no_remask | **0.472** |
| conf tsw=0.3 | 0.569 |
| cap eta=0.4 | 0.571 |
| conf tsw=1.0 | 0.571 |
| cap eta>=0.6 | 0.571 |
| conf tsw=0.5 | 0.580 |
| conf tsw=0.7 | 0.580 |
| cap eta=0.2 | 0.595 |

**Observations:**

- No-remask baseline is clearly best (0.472 vs next-best 0.569). This is the
  core tradeoff: remasking improves diversity but degrades conditional edge
  fidelity by ~20%.
- **conf tsw=0.3 is the best remasking variant** (0.569), marginally ahead of
  cap eta=0.4 and conf tsw=1.0 (0.571). The conservative switch lets the model
  build an accurate draft before remasking perturbs it.
- The degradation from remasking is substantial (~0.1 TV, or ~21% relative
  increase). This reflects the mechanism: stochastic re-prediction after
  remasking draws from the learned distribution rather than taking the argmax,
  introducing variance in edge-type assignment per room-type pair.

**Winner:** conf tsw=0.3 (0.569), marginal over cap eta=0.4 and conf tsw=1.0.

---

## 4. Type-cond. Degree TV (weighted)

| Method | Type-cond. degree TV (wt.) |
|--------|:--------------------------:|
| topp_no_remask | **0.159** |
| conf tsw=0.3 | 0.165 |
| cap eta=0.2 | 0.166 |
| conf tsw=0.5 | 0.166 |
| conf tsw=1.0 | 0.169 |
| cap eta>=0.6 | 0.173 |
| conf tsw=0.7 | 0.174 |
| cap eta=0.4 | 0.174 |

**Observations:**

- This metric is remarkably stable across all methods. The worst remasking
  variant (cap eta=0.4, TV=0.174) is only 0.015 worse than the baseline
  (0.159) -- a 9% relative increase. Compare this to the 21% degradation
  on conditional edge TV.
- **Remasking barely touches per-type degree distribution.** This makes sense:
  degree (number of connections per room type) is a first-order structural
  property, while edge type assignment is second-order. Remasking re-draws
  edge types but rarely changes whether an edge exists, so degree distributions
  stay stable.

**Winner:** Tie within noise. conf tsw=0.3 marginally best (0.165).

---

## 5. Node TV

| Method | Node TV |
|--------|:-------:|
| topp_no_remask | **0.119** |
| cap eta>=0.6 | 0.197 |
| cap eta=0.4 | 0.198 |
| conf tsw=1.0 | 0.199 |
| conf tsw=0.7 | 0.200 |
| conf tsw=0.5 | 0.203 |
| conf tsw=0.3 | 0.204 |
| cap eta=0.2 | 0.210 |

**Observations:**

- Same pattern as conditional edge: baseline clearly best, remasking adds
  ~0.08 TV (~66% relative increase). Node type distribution is more sensitive
  to remasking than degree distribution but less than edge type distribution.
- All remasking methods cluster in a tight band (0.197-0.210). No meaningful
  cap vs confidence distinction.

**Winner:** Tie. All remasking methods equivalent.

---

## 6. Synthesis: Cap vs Confidence for LLaDA

| Criterion | Winner | Margin |
|-----------|--------|--------|
| Novelty x Mode cov (wt.) | **Confidence** | +0.02 (0.73 vs 0.71) |
| Spatial transitivity | **Tie** | within noise |
| Cond. edge TV (wt.) | **Confidence tsw=0.3** | marginal (0.569 vs 0.571) |
| Type-cond. degree TV (wt.) | **Tie** | within noise |
| Node TV | **Tie** | within noise |

**Confidence remasking is slightly but consistently better than cap for the
priority metrics.** The advantage concentrates on mode coverage (+2-4 pp
weighted), which is where it matters for the novelty x coverage objective.
On distributional and structural metrics, the two are equivalent.

### Recommendation

Use **confidence remasking with tsw=1.0**:

- Simplest configuration -- no tuning needed, self-regulating
- Tied-best on novelty x coverage (0.732)
- Best transitivity among remasking methods (98.7%)
- Equivalent to all other remasking variants on distributional metrics
- The main cost of any remasking on LLaDA is ~0.1 on cond. edge TV (weighted)
  -- a 21% degradation in conditional edge fidelity. This is the price for
  +4pp mode coverage and 4x archetype increase (29 -> 121).

---

## 7. JS Divergence: Redundant for All Three Metric Families

For each metric where both TV and JS were computed, we checked whether JS
changes the rank ordering or reveals information TV misses.

### Cond. edge: JS vs TV

| Method | TV (wt.) | JS (wt.) | TV rank | JS rank |
|--------|:--------:|:--------:|:-------:|:-------:|
| topp_no_remask | 0.472 | 0.175 | 1 | 1 |
| conf tsw=0.3 | 0.569 | 0.243 | 2 | 2 |
| cap eta=0.4 | 0.571 | 0.248 | 3 | 3 |
| conf tsw=1.0 | 0.571 | 0.248 | 4 | 4 |
| cap eta>=0.6 | 0.571 | 0.248 | 5 | 5 |
| conf tsw=0.5 | 0.580 | 0.254 | 6 | 6 |
| conf tsw=0.7 | 0.580 | 0.255 | 7 | 7 |
| cap eta=0.2 | 0.595 | 0.262 | 8 | 8 |

Rankings identical. **JS redundant.**

### Type-cond. degree: JS vs TV

| Method | TV (wt.) | JS (wt.) | TV rank | JS rank |
|--------|:--------:|:--------:|:-------:|:-------:|
| topp_no_remask | 0.159 | 0.033 | 1 | 2 |
| conf tsw=0.3 | 0.165 | 0.033 | 2 | 3 |
| cap eta=0.2 | 0.166 | 0.031 | 3 | 1 |
| conf tsw=0.5 | 0.166 | 0.038 | 4 | 5 |
| conf tsw=1.0 | 0.169 | 0.038 | 5 | 4 |
| cap eta>=0.6 | 0.173 | 0.038 | 6 | 6 |
| conf tsw=0.7 | 0.174 | 0.040 | 7 | 8 |
| cap eta=0.4 | 0.174 | 0.038 | 8 | 7 |

Minor rank swaps in top 3, but absolute differences are tiny (entire spread
is 0.009 on both TV and JS). **No practical conclusion changes. JS redundant.**

### Node: JS vs TV

| Method | TV | JS | TV rank | JS rank |
|--------|:--:|:--:|:-------:|:-------:|
| topp_no_remask | 0.119 | 0.023 | 1 | 1 |
| cap eta>=0.6 | 0.197 | 0.044 | 2 | 3 |
| cap eta=0.4 | 0.198 | 0.043 | 3 | 2 |
| conf tsw=1.0 | 0.199 | 0.044 | 4 | 4 |
| conf tsw=0.7 | 0.200 | 0.044 | 5 | 5 |
| conf tsw=0.5 | 0.203 | 0.046 | 6 | 7 |
| conf tsw=0.3 | 0.204 | 0.045 | 7 | 6 |
| cap eta=0.2 | 0.210 | 0.050 | 8 | 8 |

Rankings essentially identical. **JS redundant.**

### Conclusion

**Report TV only for cond. edge, type-cond. degree, and node metrics.** JS
carries the same information in all three cases and can be dropped from the
preferred metrics set without losing signal.
