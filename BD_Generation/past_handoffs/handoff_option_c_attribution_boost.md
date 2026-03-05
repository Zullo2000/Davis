# Handoff: Implement Option C — Reward-Attributed Confidence Boosting

> **Created:** 2026-03-05
> **Session purpose:** Design and plan the implementation of Option C (reward-attributed confidence boosting) to mitigate the guidance-remasking conflict in SVDD guided sampling.

---

## What was accomplished

- **Updated `docs/guidance.md`** with a full new section "Reward-Attributed Confidence Boosting (Option C)" documenting:
  - The problem (remasking destroys guided positions)
  - The mechanism (per-position reward attribution + confidence boosting for just-unmasked positions only)
  - The self-calibrating β formula: `β(K, {r_k}) = K/(K+K₀) · 1/(σ_r + ε)` with K₀=4
  - Verification against the worked example from `00_masked_diffusion_conceptual_overview.md` §15.4.3
  - Edge cases and properties
- **Created a detailed implementation plan** at `C:\Users\Alessandro Zuliani\.claude\plans\dynamic-brewing-steele.md`
- **Thorough codebase exploration** of all relevant files with exact line numbers identified

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Where to compute attribution | In `guided_sampler.py` (not in remasking module) | Attribution needs K candidates + rewards, which only exist in the guidance loop |
| How to pass boost to remasking | Additive `confidence_boost` tensor via optional parameter | Clean separation: remasking module doesn't know about guidance; boost is just a (B, SEQ) additive tensor |
| β formula | `K/(K+K₀) / (σ_r + ε)` with K₀=4 | Self-calibrating, no sweep needed. K₀=4 is the combinatorial floor for a binary partition test. σ_r normalizes reward-scale attribution to confidence-scale. |
| Which positions get boosted | Only $\mathcal{U}_t$ (just-unmasked at current step) | Already-committed positions have identical tokens across all K candidates → attribution is trivially 0 |
| Feature flag | `attribution_boost: bool = False` parameter on `guided_sample()` + `--attribution-boost` CLI flag | Backward-compatible, opt-in |

## Current state of the codebase

- **All existing code works** — no modifications have been made yet. The plan is ready for implementation.
- **`docs/guidance.md`** has been updated with the theoretical section (status: "not yet implemented")
- **`00_masked_diffusion_conceptual_overview.md`** has the original Option C description in §15.4.3 (with the old naive `β = 1/Δr̄` suggestion — the doc was NOT updated with the new formula, only `docs/guidance.md` was)
- **Test suite is green** (51 remasking tests, 16 guided sampler tests, sampling tests all pass)

## What remains to be done — The Implementation Plan

Read the full plan at `C:\Users\Alessandro Zuliani\.claude\plans\dynamic-brewing-steele.md`. Summary of the 6 steps:

### Step 1: `bd_gen/diffusion/remasking.py` — accept confidence boost
- Add `confidence_boost: Tensor | None = None` to `__call__()` (line 159) and `_confidence_remasking()` (line 227)
- After line 290 (`confidence = torch.cat([node_conf, edge_conf], dim=1)`), add: `if confidence_boost is not None: confidence = confidence + confidence_boost`
- That's it for this file — minimal change

### Step 2: `bd_gen/diffusion/sampling.py` — pass through
- Add `confidence_boost: Tensor | None = None` to `_single_step_remask()` (line 241)
- Pass it through to `remasking_fn()` call at line 272-274
- Non-guided `sample()` (line 428) uses the default `None` — no behavior change

### Step 3: `bd_gen/guidance/guided_sampler.py` — core logic
- **3a.** Add `attribution_boost: bool = False` parameter to `guided_sample()` (line 237)
- **3b.** New helper `_compute_attribution_boost(candidates, rewards, selected_k, x_t_pre, n_max, K0=4, eps=1e-8) -> Tensor` — fully vectorized, no Python loops. The plan file has the complete pseudocode.
- **3c.** In the SVDD loop: save `x_t_pre_unmask = x_t.clone()` before step 4c (line 361). Between step 4h (line 416) and 4i (line 422), compute the boost and pass to `_single_step_remask()`.
- **3d.** Add `mean_attribution_boost` and `positions_boosted` to `GuidanceStatsStep`

### Step 4: `scripts/generate_guided.py` — CLI
- Add `--attribution-boost` flag (store_true). Pass to `guided_sample()`.

### Step 5: Tests
- **`test_remasking.py`**: 3 new tests (None is noop, boost protects low-conf positions, zero boost ignored)
- **`test_guided_sampler.py`**: 3 new tests (smoke test, reduces remasking delta, noop without remasking)

### Step 6: `docs/guidance.md`
- Change status line from "not yet implemented" to implemented

## Files to reference in next session

**Read these first (in order):**
1. `C:\Users\Alessandro Zuliani\.claude\plans\dynamic-brewing-steele.md` — the full implementation plan with pseudocode
2. `docs/guidance.md` lines 653–737 — the Option C theoretical section (already written)
3. `bd_gen/guidance/guided_sampler.py` lines 334–503 — the SVDD loop (where the core changes go)
4. `bd_gen/diffusion/remasking.py` lines 159–313 — `__call__()` and `_confidence_remasking()` (where the boost gets consumed)
5. `bd_gen/diffusion/sampling.py` lines 241–276 — `_single_step_remask()` (passthrough)

**Supporting files:**
- `scripts/generate_guided.py` lines 164–192 (argparse) and 339–354 (guided_sample call)
- `tests/test_guided_sampler.py` — existing test patterns (especially test 49 for remasking)
- `tests/test_remasking.py` — existing confidence remasking tests (lines 496–760) for test patterns

## Context for the next session

### Key architectural insight
The confidence boost is **additive** to model confidence, applied **before** the `softmax(-confidence)` in remasking. This means:
- A boosted position's effective confidence goes UP → its `exp(-conf)` goes DOWN → less likely to be remasked
- The boost is zero outside $\mathcal{U}_t$, so already-committed and still-masked positions are unaffected
- The remasking budget (σ_max) is unchanged — boosting just redistributes it away from guided positions

### The β formula derivation
`β = K/(K+4) · 1/(σ_r + ε)` where σ_r is reward std across K candidates per batch element per step.
- `1/σ_r` converts attribution (reward-scale) to z-score-like (confidence-scale)
- `K/(K+4)` is a statistical trust sigmoid — shrinks boost at low K where attribution is noisy
- K₀=4 is the minimum K for a binary partition to have 2+ samples per group
- No hyperparameter sweep needed

### Gotchas
- `candidates` tensor shape is `(K, B, SEQ)` — K is dim 0, not dim 2
- `rewards` is float64 (importance weights need precision) — make sure attribution arithmetic stays in float64
- The `x_t.clone()` for pre-unmask state must happen BEFORE the K-expansion (line 361), not after
- `_single_step_remask` is called from both `sample()` and `guided_sample()` — the new parameter must default to `None` for backward compat
- When `K < 2`, skip boosting entirely (can't form groups for attribution)
- When `σ_r < ε`, skip boosting (no reward discrimination → nothing to protect)

### Backward compatibility
All changes are additive with safe defaults. No existing tests, configs, or training code are affected. The feature is opt-in via `attribution_boost=True`.
