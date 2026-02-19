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

**Decision after L3:** Best t_switch for confidence.

### Layer 4: Head-to-Head — 0 additional runs

Compare best cap result (from L2) vs best confidence result (from L3).
Data already collected; just compare metrics.

### Total: ~13 runs

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
