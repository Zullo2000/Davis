# ReMDM Remasking Design

Reference: Schiff et al., "Remasking Discrete Diffusion Models with Inference-Time Scaling" (arXiv:2503.00307, ICLR 2025).
Code reference: https://github.com/kuleshov-group/remdm

This document covers the design choices for integrating ReMDM remasking into
our bubble diagram discrete diffusion pipeline. It explains the mechanics,
motivates each choice, and defines the experiment plan.

---

## 1. Remasking Schedules

### 1.1 Max-Capped Schedule (strategy="cap")

**Formula:** `sigma_t = min(eta, sigma_max)` where `sigma_max = clamp((1 - alpha_s) / alpha_t, 0, 1)`.

Each decoded (non-MASK, non-PAD) position is independently remasked with
probability `sigma_t`. The `eta` parameter caps the maximum remasking intensity.

**Behavior at extremes:**
- `eta = 0` → `sigma_t = 0` → no remasking → **recovers standard MDLM**.
- `eta = 1` → `sigma_t = sigma_max` → cap is ineffective, maximum possible
  remasking at every step. Very aggressive.

**File:** `bd_gen/diffusion/remasking.py:125-129`

### 1.2 Rescaled Schedule (strategy="rescale")

**Formula:** `sigma_t = eta * sigma_max`

Scales the upper bound linearly instead of capping. For small eta values
it behaves similarly to cap; for large eta it can exceed cap's remasking rate
at early timesteps.

**File:** `bd_gen/diffusion/remasking.py:131`

### 1.3 Confidence-Based Schedule (strategy="confidence")

**Paper formula:**
```
sigma_t[l] = softmax(-confidence)[l] * sigma_max
```

Per-position remasking probability is proportional to the model's *lack* of
confidence (negative of P(decoded_token)). High-confidence positions are
rarely remasked; low-confidence positions are remasked more often.

**Key difference from cap/rescale:** The confidence schedule does NOT use an
`eta` parameter. The total remasking budget is determined entirely by the noise
schedule (`sigma_max`), and the allocation across positions is determined by
model confidence.

**Implementation change needed:** Our current implementation
(`remasking.py:245-286`) uses "cap + confidence redistribution"
(`sigma_cap = min(eta, sigma_max)`, then redistribute by confidence).
We need to change this to match the paper: remove the eta dependency
and use `sigma_max` directly as the total budget.

**Why no eta for confidence?** The confidence softmax already acts as an
adaptive controller — at positions where the model is very certain, the
softmax weight is near zero (effectively no remasking). At uncertain positions,
the weight is high. Adding eta on top would double-gate the mechanism
unnecessarily.

---

## 2. ReMDM-Switch: Turning Remasking On/Off

### 2.1 The Problem

Without an on/off mechanism, remasking is active at every intermediate step
(not at the final step i=0, which our code already skips). At early timesteps
(t close to 1), very few tokens are decoded and the model has little context.
Confidence scores are unreliable. Remasking at this stage is wasteful or
harmful.

### 2.2 The Solution: ReMDM-Switch

Apply remasking **only** when `t < t_switch`:
- For `t >= t_switch`: standard MDLM decoding (no remasking)
- For `t < t_switch`: remasking enabled

This means: first build an initial draft with standard MDLM, then spend
the remaining denoising budget fixing mistakes via remasking.

### 2.3 How to Choose t_switch

At time t in MDLM, the fraction of tokens still masked is approximately
`1 - alpha(t)`. So t_switch controls how much structure must exist before
remasking activates.

| t_switch | When remasking starts | Decoded tokens at activation | Tradeoff |
|----------|-----------------------|------------------------------|----------|
| 0.7–0.9  | Very early            | ~10-30% decoded              | Max correction time, but unreliable confidence |
| 0.4–0.6  | Mid-denoising         | ~40-60% decoded              | Good balance |
| 0.1–0.3  | Near the end          | ~70-90% decoded              | Reliable confidence, but few steps left to correct |

**For our graph domain** (short sequences, ~36 positions, hard structural
constraints): the ReMDM molecule experiments used `t_on=0.25` (conservative,
structured domain like ours). We test `t_switch ∈ {0.3, 0.5, 0.7}`.

### 2.4 Implementation

Add a `t_switch` config parameter. In the sampling loop, change the
remasking call condition from:

```python
if remasking_fn is not None and i > 0:
```

to:

```python
if remasking_fn is not None and i > 0 and t_now < t_switch:
```

Where `t_now = (i + 1) / num_steps`. Approximately 3 lines of change in
`sampling.py`.

### 2.5 Switch for Cap vs Confidence

For this round of experiments:
- **Cap:** no Switch (eta already controls intensity directly)
- **Confidence:** with Switch (no eta to control intensity, so Switch is the
  mechanism to avoid aggressive early remasking)

A later round could test cap + Switch as an additional refinement.

---

## 3. Where Temperature Acts in the Sampling Loop

This section explains exactly how temperature interacts with unmasking and
remasking. Understanding this is critical for designing experiments.

### 3.1 The Sampling Loop Steps

The reverse diffusion sampling loop (`sampling.py:174-282`) performs these
steps at each denoising iteration:

```
For each step i (from num_steps-1 down to 0):
    t_now = (i+1) / num_steps    (noisier)
    t_next = i / num_steps        (cleaner)

    Step 4a: model(x_t, pad_mask, t) → node_logits, edge_logits
    Step 4d: CHOOSE TOKENS from logits          ← TEMPERATURE ACTS HERE
    Step 4f: CHOOSE WHICH POSITIONS to unmask    ← CONFIDENCE ACTS HERE (llada)
    Step 4g: Place chosen tokens at unmasked positions
    Step 4j: CHOOSE WHICH POSITIONS to remask    ← CONFIDENCE ACTS HERE (remasking)
```

### 3.2 Temperature Controls Token Selection (Step 4d)

Temperature determines **which token** goes into a position when it is
unmasked. This is the ONLY place temperature acts.

**temperature = 0 (argmax):**
```python
node_pred = node_logits.argmax(dim=-1)   # always pick highest-prob token
edge_pred = edge_logits.argmax(dim=-1)   # deterministic, same input → same output
```

**temperature > 0 (Gumbel sampling):**
```python
# Equivalent to sampling from softmax(logits / temperature)
gumbel_noise = -log(-log(uniform_random))             # in float64
perturbed = logits / temperature + gumbel_noise
pred = perturbed.argmax(dim=-1)                        # stochastic
```

**temperature = 1.0** gives the model's natural probability distribution
(`softmax(logits)`). Lower temperature sharpens it (more deterministic),
higher temperature flattens it (more random).

### 3.3 Confidence for Unmasking Uses RAW Logits (Step 4f)

In LLaDA unmasking mode (`sampling.py:226-234`), the decision "which masked
positions to unmask first" is based on:

```python
node_probs = F.softmax(node_logits, dim=-1)    # RAW logits, not temperature-scaled
confidence = probs.gather(-1, pred_token)        # P(predicted token)
# → Unmask top-k most confident positions
```

**Temperature does NOT affect the unmasking ranking.** Even with temperature=5,
the most confident positions (by raw logits) unmask first. Temperature only
changes what token gets placed there.

### 3.4 Confidence for Remasking Uses RAW Logits (Step 4j)

In confidence-based remasking (`remasking.py:255-267`), the decision "which
decoded positions to remask" is based on:

```python
node_probs = F.softmax(node_logits, dim=-1)    # RAW logits again
confidence = probs.gather(-1, current_token)     # P(current token at this position)
# → softmax(-confidence) → high remasking weight for low-confidence positions
```

**Temperature does NOT affect remasking decisions.** The same positions get
remasked regardless of temperature setting.

### 3.5 Summary Table

| Step | What it decides | Uses temperature? | Uses confidence? |
|------|----------------|-------------------|-----------------|
| 4d   | **Which token** to place at a position | **YES** | No |
| 4f   | **Which masked positions** to unmask | No | Only in llada mode (raw logits) |
| 4j   | **Which decoded positions** to remask | No | Only in confidence strategy (raw logits) |

Temperature is orthogonal to both confidence mechanisms. It controls token
identity, not position selection.

---

## 4. Why Temperature is Critical for Remasking

### 4.1 The Problem with Argmax + Remasking

With temperature=0 (argmax), the model always picks the single most probable
token given the current context. Consider what happens when remasking is
applied:

```
Step 10: position A is unmasked → argmax("left_of") — deterministic
Step 15: remasking re-masks position A
Step 16: position A is unmasked again
         → model sees similar context → argmax("left_of") again
         → remasking accomplished nothing
```

The model can only produce a different token if the surrounding context changed
significantly (e.g., multiple neighbors were also remasked simultaneously).
In practice, the top prediction is usually robust to small context changes.
**Remasking with argmax mostly just re-confirms the same mistakes.**

### 4.2 Stochastic Sampling Enables Correction

With temperature > 0 or top-p sampling, the model can sample different tokens
on re-prediction:

```
Step 10: position A is unmasked → sample("left_of")
Step 15: remasking re-masks position A
Step 16: position A is unmasked again
         → sample("above") — different draw from the distribution
         → if "above" fits the surrounding edges better, subsequent
            remasking cycles will keep it (high confidence → not remasked)
         → if "above" is worse, it gets remasked again → try again
```

This is the iterative correction loop that makes ReMDM work: explore
alternatives, keep improvements, retry failures.

### 4.3 This Applies to ALL Remasking Strategies

Temperature does not affect the remasking decision mechanics of cap, rescale,
or confidence. But it determines whether re-prediction after remasking can
actually produce different (potentially better) results. Without stochastic
sampling, any remasking strategy is severely limited.

---

## 5. Temperature vs Top-p Nucleus Sampling

### 5.1 Temperature Sampling

Scales all logits uniformly: `p(token) = softmax(logits / T)`.

- T=0: argmax (deterministic)
- T=0.5: sharpened — mostly top token, occasionally explores
- T=1.0: standard softmax — model's natural calibrated probabilities
- T>1: flattened — spreads mass to unlikely tokens

**Problem for our domain:** With small vocabularies (15 node types, 13 edge
types), even moderate temperature can push significant probability onto
structurally wrong tokens:

```
Raw logits for an edge position:
  "left_of": 5.2, "right_of": 3.1, "above": 0.3, "no_edge": -1.2, ...

temperature=0.7 → softmax is sharper, still mostly "left_of" ✓
temperature=1.0 → "above" and "no_edge" get non-trivial mass
temperature=1.5 → nearly uniform — sampling garbage
```

### 5.2 Top-p (Nucleus) Sampling

Instead of scaling all logits, top-p truncates the tail:

1. Compute softmax(logits) at temperature=1.0
2. Sort tokens by probability (descending)
3. Find smallest set whose cumulative probability >= p (e.g., 0.9)
4. Zero out everything else, renormalize, sample

```
Same edge position with top-p=0.9:
  "left_of": 0.72, "right_of": 0.19 → cumsum=0.91 >= 0.9 → KEEP
  "above": 0.05, ...               → EXCLUDED

  Sample from: {"left_of": 0.79, "right_of": 0.21}
```

**Advantages:**
- Stochasticity (remasking can explore) ✓
- No long-tail problem (structurally nonsensical tokens excluded) ✓
- Adaptive per position: where the model is very certain (one token >95%),
  top-p effectively becomes argmax. Where uncertain, it allows exploration.
  Exactly what we want.

### 5.3 Why temperature=1.0 with top-p=0.9

Top-p is designed to work with the model's natural probability distribution
(temperature=1.0 = `softmax(logits)`). If you combine lower temperature with
top-p, they fight each other:

- temperature < 1.0 sharpens the distribution first, eliminating most of the
  tail before top-p acts → top-p becomes redundant
- temperature > 1.0 flattens the distribution, pushing bad tokens into the
  "top 90%" → top-p doesn't clip enough

temperature=1.0 is the neutral point: the model's own uncertainty is preserved
and top-p does exactly its intended job.

### 5.4 Recommendation for Our Domain

**Use top-p=0.9 with temperature=1.0** for all remasking experiments.

Our domain has hard structural constraints (spatial transitivity, connectivity).
A single bad token from the tail can cascade through remasking cycles. Top-p
prevents this while still enabling the stochastic exploration that remasking
needs.

Note: **top-p is not currently implemented** — only argmax and Gumbel.
Adding it requires ~15 lines in `sampling.py`.

---

## 6. The 2×2 Interaction: Sampling × Remasking

| | argmax (T=0) | top-p=0.9 (T=1.0) |
|---|---|---|
| **No remasking (MDLM)** | Best single-shot prediction (current baseline) | Adds noise that can't be corrected → expect slightly worse |
| **Remasking (ReMDM)** | Can't explore alternatives → limited benefit | Explore + correct → the sweet spot |

Neither stochastic sampling nor remasking is sufficient alone. They are
synergistic: stochastic sampling provides the exploration that remasking
needs to correct.

**Experimental design:** We test both argmax and top-p as baselines
(Layer 1), then use top-p for all remasking experiments (Layers 2-3),
with one argmax+remasking control run to confirm the synergy.

---

## 7. fp64 Numerical Stability

All critical computations already use float64 (verified):

| Computation | fp64? | Location |
|-------------|-------|----------|
| Gumbel noise | Yes | `sampling.py:44` — `torch.rand(..., dtype=torch.float64)` |
| p_unmask | Yes | `sampling.py:194-199` — alpha in float64 |
| sigma_max (remasking) | Yes | `remasking.py:90-103` — float64, cast to float32 |
| Confidence softmax | float32 | `remasking.py:255-279` — fine, only for relative ranking |

No changes needed.

---

## 8. Why Test Higher Step Counts (Future Round)

Deferred to a later experiment round, but the motivation:

Standard MDLM has diminishing returns with more steps — finer unmasking
granularity doesn't improve prediction quality. ReMDM changes this because
more steps = more remasking/re-prediction cycles = more correction
opportunities.

For our graph domain: structural constraints (transitivity, connectivity) are
holistic — you need to see the full graph to judge local decisions. More
remasking cycles let the model iteratively refine the global structure.

The paper shows monotonic improvement up to T=4096 for text. Our sequences
are much shorter (~36 positions), so T=200 or T=500 may be sufficient.

**However:** For this round we keep T=100. At 100 steps with ~36 positions,
each step unmasks ~0.36 positions on average — already very fine-grained.
The priority is finding the right remasking strategy before scaling steps.

---

## 9. Unmasking Mode Interaction (Open Question)

We have two unmasking modes:
- **random:** each masked position independently unmasks with probability
  p_unmask (standard MDLM)
- **llada:** unmask highest-confidence positions first (LLaDA-style)

Layer 1 of the experiment plan tests both to determine which works better
as the MDLM baseline. The winner is carried forward into remasking experiments.

**Hypothesis:** llada + confidence remasking could be particularly synergistic —
llada is careful about when to unmask (high-confidence first), and confidence
remasking corrects low-confidence mistakes afterward. They're complementary.

---

## 10. Layered Experiment Design

### Fixed Parameters (All Runs)
- T = 100 (sampling steps)
- num_samples = 1000
- seeds = [42, 123, 456, 789, 1337] (multi-seed)
- All existing metrics (validity, diversity, novelty, distribution match,
  structural, conditional, denoising)

### Layer 1: Baselines (No Remasking) — 4 runs

Goal: establish reference points, determine best unmasking mode, validate
that top-p alone doesn't improve MDLM.

| Run | Unmasking | Sampling   | Remasking | Notes |
|-----|-----------|------------|-----------|-------|
| 1   | random    | argmax     | none      | Existing baseline (may already have) |
| 2   | random    | top-p=0.9  | none      | Isolate top-p effect |
| 3   | llada     | argmax     | none      | Existing baseline (may already have) |
| 4   | llada     | top-p=0.9  | none      | Isolate top-p on llada |

**Decision after L1:** Pick better unmasking mode → use for all subsequent layers.
**Expected outcome:** argmax baselines >= top-p baselines (no correction mechanism).

### Layer 2: Cap Eta Sweep — 6 runs

Goal: find best eta for cap schedule, confirm top-p is needed with remasking.

Fix: winner unmasking from L1, top-p=0.9, no Switch.

| Run | Strategy | eta  | Sampling   | Notes |
|-----|----------|------|------------|-------|
| 5   | cap      | 0.2  | top-p=0.9  | Mild remasking |
| 6   | cap      | 0.4  | top-p=0.9  | |
| 7   | cap      | 0.6  | top-p=0.9  | |
| 8   | cap      | 0.8  | top-p=0.9  | |
| 9   | cap      | 1.0  | top-p=0.9  | Maximum remasking (no cap) |
| 10  | cap      | best | argmax     | Control: confirm top-p synergy |

**Decision after L2:** Best eta for cap.
**Expected outcome:** Sweet spot around eta=0.4–0.6 for graphs. Run 10
significantly worse than same eta with top-p.

### Layer 3: Confidence + Switch Sweep — 3 runs

Goal: find best t_switch for confidence schedule.

Fix: winner unmasking from L1, top-p=0.9, confidence strategy (paper version,
no eta).

| Run | Strategy    | t_switch | Sampling   | Notes |
|-----|-------------|----------|------------|-------|
| 11  | confidence  | 0.3      | top-p=0.9  | Conservative: remask only last 30% |
| 12  | confidence  | 0.5      | top-p=0.9  | Balanced |
| 13  | confidence  | 0.7      | top-p=0.9  | Aggressive: remask from early on |
| 13b | confidence  | 1.0      | top-p=0.9  | No switch: remask at all steps (both modes) |

**Decision after L3:** Best t_switch for confidence (including no-switch baseline).

### Layer 4: Head-to-Head — 0 additional runs

Compare best cap result (from L2) vs best confidence result (from L3).
Data already collected; just compare metrics.

### Total: ~14 runs

(Minus any baselines already evaluated. Runs 1 and 3 may already exist.)

### Code Changes Required Before Running

1. **Implement top-p sampling** in `sampling.py` (~15 lines)
2. **Change confidence schedule** in `remasking.py` to paper version (remove
   eta dependency, use `sigma_max` directly)
3. **Add t_switch parameter** to config and sampling loop (~3 lines)
4. **Add top-p config option** to `configs/eval/default.yaml`

### What to Defer to Later Rounds

- Higher step counts (T=200, T=500) with winning config
- Cap + Switch combination
- Temperature grid (0.7, 0.9) with top-p
- Rescale strategy testing
- Combining cap/rescale base with confidence redistribution

---

## 11. Experiment Results & Analysis

> **Date:** 2026-02-20 (updated with tsw=1.0 runs)
> **Schedule:** Log-linear (with importance sampling)
> **Total runs:** 22 (4 baselines + 9 random remasking + 9 llada remasking)
> **Full data:** `eval_results/loglinear/comparison.md` (auto-generated, all 22 methods)

### 11.1 What we ran

The original plan (Section 10) called for running Layers 2-3 with only the
"winning" unmasking mode from Layer 1. Layer 1 did not produce a clear winner
— random and llada each dominate different metric families — so we ran Layers
2-3 with **both** unmasking modes (20 runs total instead of ~13). A 21st and 22nd run
(`random_topp0.9_remdm_confidence_tsw1.0` and `llada_topp0.9_remdm_confidence_tsw1.0`)
were added to complete the confidence sweep with no-switch baselines (t_switch=1.0,
remasking active at all steps) for both unmasking modes.

Run 10 (argmax control) was skipped: llada+argmax produces only 5 unique
archetypes (complete mode collapse from deterministic decoding), making it
uninformative as a remasking control. The top-p synergy argument is already
well-supported by comparing baselines (argmax diversity 0.005 vs top-p 0.945).

### 11.2 The fundamental tradeoff: structure vs distribution

The single most important finding is that **unmasking mode** (random vs llada)
dominates all other choices (remasking strategy, eta, t_switch). The two modes
create distinct operating regimes:

**llada unmasking** resolves high-confidence positions first. This produces
graphs with correct degree sequences (MMD-Degree 0.035-0.050) and consistent
spatial relationships (transitivity 98-100%), but with a biased edge type
distribution (Edge JS 0.106-0.214) and limited diversity (mode coverage
67-73%). The model "plays it safe" — generating structurally sound but
repetitive graphs.

**random unmasking** treats all positions equally. This better matches the
training data distribution (Edge JS 0.035-0.082, Node JS 0.004-0.006) and
produces much higher diversity (mode coverage 88-91%, 2x more archetypes),
but with worse graph structure (MMD-Degree 0.302-0.408, 10x worse) and
lower validity (97.9-99.7% vs 99.8-100%).

### 11.3 What remasking does to each mode

**For random unmasking**, remasking:
- Improves Node JS (0.006 → 0.004) and mode coverage (88.5% → 90.7%)
- Doubles unique archetypes (103 → 242)
- Costs 2x Edge JS (0.035 → 0.070) and 1.7% validity (99.7% → 98.0%)
- Worsens already-poor structure (MMD 0.302 → 0.404)
- **Net effect:** more diverse, slightly worse distribution match on edges

**For llada unmasking**, remasking:
- Dramatically improves diversity (0.945 → 0.987) and archetypes (29 → 112, 4x)
- Improves graph structure (MMD-Degree 0.050 → 0.037, best of any method)
- Costs 2x distribution match: Node JS (0.023 → 0.043), Edge JS (0.106 → 0.197)
- Minimal validity cost (100% → 99.8%)
- **Net effect:** much more diverse while maintaining structural quality

Remasking's benefit is proportional to how much room there is to improve.
llada baseline has low diversity (the bottleneck), so remasking provides a
large diversity lift. Random baseline has poor structure, but remasking
can't fix structural problems — it just re-draws from the same distribution.

### 11.4 Cap vs confidence: minimal difference

Within each unmasking mode, cap and confidence remasking produce very similar
results. The choice of remasking strategy matters far less than the choice of
unmasking mode.

**Cap eta sweep** (both modes): eta saturates at 0.4 — values 0.4 through 1.0
produce nearly identical metrics. The remasking budget hits a ceiling where
additional remasking cycles stop changing outcomes (re-masked positions get
similar predictions from similar context).

**Confidence t_switch sweep** (both modes): t_switch values 0.3, 0.5, 0.7, and 1.0
produce similar results. The tsw=1.0 case is particularly informative: it means
remasking is active at **all** timesteps (no switch-off), yet it performs
equivalently to the partial-remasking variants. This confirms that the
confidence threshold is self-regulating — at early timesteps (high noise)
few positions exceed the threshold, so remasking naturally fades without
needing an explicit t_switch cutoff. The small differences across the sweep:
- tsw=0.3 (conservative, remask only last 30%): slightly better conditional metrics
- tsw=0.5: slightly better MMD (0.035 best of all methods for llada)
- tsw=0.7 (aggressive): slightly more archetypes but worse Edge JS
- tsw=1.0 (always remask): near-identical to tsw=0.5 — llada: 99.7% validity,
  0.204 Edge JS, 0.035 MMD, 73.3% mode coverage, 121 archetypes;
  random: 98.2% validity, 0.081 Edge JS, 0.400 MMD, 90.6% coverage, 251 archetypes

### 11.5 Pareto front — candidate methods for final selection

No single method wins all metrics. The next step is to choose based on
downstream priorities. The Pareto-optimal candidates are:

| Method | Validity | Edge JS | MMD-Deg | Mode cov | Archetypes | Profile |
|---|:---:|:---:|:---:|:---:|:---:|---|
| llada_topp (no remask) | **100%** | 0.106 | 0.050 | 69.6% | 29 | Safest structure, low diversity |
| llada + cap eta=0.4 | 99.8% | 0.197 | **0.037** | 71.4% | 112 | Best structure + improved diversity |
| llada + conf tsw=0.5 | 99.8% | 0.207 | **0.035** | 72.7% | 120 | Best MMD overall |
| llada + conf tsw=1.0 | 99.7% | 0.204 | **0.035** | 73.3% | 121 | ≈ tsw=0.5, confirms always-remask viable |
| random_topp (no remask) | 99.7% | **0.035** | 0.302 | 88.5% | 103 | Best distribution match |
| random + cap eta=0.2 | 98.3% | 0.070 | 0.400 | 89.6% | 239 | Best coverage + good distribution |
| random + cap eta=0.4 | 98.0% | 0.073 | 0.404 | **90.7%** | **242** | Maximum diversity |
| random + conf tsw=1.0 | 98.2% | 0.081 | 0.400 | 90.6% | **251** | Most archetypes, higher Edge JS |

The tsw=1.0 entries are near-duplicates of existing Pareto members, confirming
that the confidence strategy's self-regulating behavior makes t_switch a
non-critical hyperparameter. In practice, one can simply set tsw=1.0 (always
remask) without tuning, and get equivalent performance.

### 11.6 Initial intuitions for method selection

These are preliminary observations to guide tomorrow's decision:

1. **If structural correctness is paramount** (e.g., downstream floorplan
   generation requires valid spatial relationships): llada + cap eta=0.4 or
   llada + conf tsw=0.5. The 4x diversity improvement over llada baseline
   with minimal validity loss and improved MMD makes this attractive.

2. **If distributional fidelity is paramount** (e.g., generated samples must
   statistically resemble the training data): random_topp without remasking.
   Remasking helps coverage but hurts Edge JS. The baseline already has
   excellent JS divergence.

3. **If maximum coverage/diversity is paramount** (e.g., exploring the full
   design space): random + cap eta=0.4. Best mode coverage and most unique
   archetypes, at the cost of ~2% validity and doubled Edge JS.

4. **The llada Edge JS problem** deserves investigation. llada's high Edge JS
   (0.106+) likely comes from its biased unmasking order: high-confidence
   positions (often common edge types) unmask first, leaving ambiguous positions
   for later. This creates a systematic edge-type distribution skew that
   remasking worsens (more re-drawing from skewed context). A hybrid approach
   (llada for initial steps, random for final steps) might combine their
   strengths.

5. **Remasking is most valuable for llada** — it addresses llada's main weakness
   (low diversity) while preserving its main strength (structural quality).
   For random, remasking provides marginal coverage gains at meaningful
   distribution cost.

6. **t_switch is a non-critical hyperparameter for confidence remasking.**
   The tsw=1.0 runs (remasking active at every timestep) perform equivalently
   to tsw=0.3-0.7, confirming the confidence threshold is self-regulating.
   This simplifies deployment: use confidence remasking with tsw=1.0 and
   avoid tuning this parameter entirely.
