# Remasking Design for v2 (Learned Forward Process)

Reference: [remasking_design_with_fixed_forward_process.md](remasking_design_with_fixed_forward_process.md) (v1 design + full 22-run experiment suite)
Reference: Schiff et al., "Remasking Discrete Diffusion Models with Inference-Time Scaling" (arXiv:2503.00307, ICLR 2025)
Date: 2026-02-25

This document adapts ReMDM remasking for the v2 learned forward process
(per-position alpha from rate network). It does NOT repeat v1 analysis — it
references it, eliminates settled questions, and focuses on the technical
adaptation and the single experiment to run.

---

## 1. Motivation: Why Remasking on v2

### v2 baseline performance (llada + top-p 0.9, no remasking)

| Metric | v1 llada baseline | v1 llada + conf tsw=1.0 | v2 llada baseline |
|--------|:-----------------:|:----------------------:|:-----------------:|
| Validity | 100.0% | 99.7% | **100.0%** |
| Inside validity | 99.4% | 93.3% | **100.0%** |
| Spatial transitivity | 99.9% | 98.7% | **100.0%** |
| Edge JS | 0.106 | 0.204 | **0.035** |
| Node JS | 0.023 | 0.044 | **0.013** |
| Edge TV | 0.399 | 0.586 | **0.217** |
| Mode coverage (weighted) | 69.6% | 73.3% | **78.3%** |
| Diversity | **0.945** | **0.982** | 0.671 |
| Novelty | **0.975** | **0.999** | 0.864 |
| Unique archetypes | 28.6 | **120.8** | 26.0 |

v2 already dominates v1 (with or without remasking) on validity, distribution
fidelity, and mode coverage. But v2 generates only **26 unique archetypes** —
the learned forward process produces more structured masking trajectories that
reduce sampling stochasticity, causing highly accurate but repetitive outputs.

### The hypothesis

Remasking was the primary diversity driver in v1: archetypes jumped from 29 to
121 (4.2x) while structural quality was preserved or improved (MMD-Degree
0.050 → 0.035). Applying the same technique to v2 should yield a similar
diversity boost while preserving v2's superior quality baseline.

### Why v2 is a better remasking candidate than v1

v2 has more quality headroom to absorb remasking's costs:

| Cost metric | v2 baseline | Expected after remasking | v1 + remasking (actual) |
|-------------|:-----------:|:------------------------:|:-----------------------:|
| Edge JS | 0.035 | ~0.050–0.070 | 0.204 |
| Inside validity | 100.0% | ~93–97% | 93.3% |
| Edge TV | 0.217 | ~0.35–0.45 | 0.586 |

Even at the pessimistic end of predicted degradation, v2+remasking would still
match or exceed v1 baseline (no remasking) on distribution fidelity, while
gaining the archetype diversity boost.

### Target outcome

100+ unique archetypes (4x, matching v1 remasking gains) with inside_validity
remaining above 93% (the v1 remasking floor for LLaDA).

---

## 2. Eliminated Design Choices (v1 Lessons Applied)

The v1 experiment suite tested 22 configurations across 2 unmasking modes ×
3 remasking strategies × multiple hyperparameter values. The lessons from that
sweep eliminate most design decisions for v2, leaving a single configuration.

### 2.1 Random unmasking mode — ELIMINATED

v1 result: inside_validity collapses to **54–59%** with remasking (v1 doc
Section 11, comparison.md). Random unmasking re-masks edge positions before
node types are decided, creating architecturally forbidden containment
relationships. v2's perfect 100% inside_validity is too valuable to risk on a
mode that demonstrated catastrophic failure.

### 2.2 Cap and rescale remasking strategies — ELIMINATED

v1 result: confidence remasking is strictly better ergonomically than cap
(v1 doc Section 11.4, best_model_according_to_preferred_metrics doc Section 6):
- Confidence wins on novelty × mode coverage (+0.02 weighted composite)
- Ties on all other metrics within noise
- Cap requires eta tuning (saturates at eta >= 0.4) while confidence is
  self-regulating with zero hyperparameters
- Rescale was never fully tested in v1 — even less reason to introduce it now

### 2.3 Argmax sampling — ELIMINATED

v1 result: argmax + remasking is ineffective (v1 doc Section 4). With
deterministic token selection, remasked positions re-predict the same token
from similar context. The model cannot explore alternatives.
`llada_argmax_no_remask` produces only 5 unique archetypes (complete mode
collapse). Top-p = 0.9 is essential for remasking's explore-correct loop.

### 2.4 t_switch sweep — ELIMINATED

v1 result: t_switch is non-critical for confidence remasking (v1 doc
Section 11.4). Values 0.3, 0.5, 0.7, 1.0 all produce equivalent results
(spread < 0.007 on novelty × coverage composite). The confidence softmax is
self-regulating: at high noise (early steps), few positions have reliable
confidence, so remasking naturally throttles itself without an explicit cutoff.

**Decision: use t_switch = 1.0** (always remask — simplest, tied-best in v1).

### Summary

| Choice | Status | v1 Evidence | Reference |
|--------|:------:|-------------|-----------|
| Random unmasking | ELIMINATED | inside_validity 54–59% | comparison.md row 30 |
| Cap strategy | ELIMINATED | = confidence, but needs eta tuning | best_model doc §6 |
| Rescale strategy | ELIMINATED | Never fully tested; cap-equivalent | v1 doc §1.2 |
| Argmax sampling | ELIMINATED | 5 archetypes, mode collapse | comparison.md row 36 |
| t_switch sweep | ELIMINATED | 0.3–1.0 all equivalent | v1 doc §11.4 |
| Confidence + LLaDA + top-p | **SELECTED** | Best composite, self-regulating | best_model doc §6 |

---

## 3. The Single Configuration for v2

Every other combination was tested in v1's 22-run experiment suite and found
to be either equivalent or strictly worse. The only open question for v2 is
whether the technical adaptation of remasking to per-position alpha works
correctly and delivers the expected diversity boost.

| Parameter | Value |
|-----------|-------|
| Unmasking mode | LLaDA (confidence-based top-k) |
| Remasking strategy | confidence (per-position softmax of −confidence) |
| t_switch | 1.0 (always remask — confidence is self-regulating) |
| Sampling | top-p = 0.9, temperature = 1.0 |
| Steps | 100 |
| Seeds | [42, 123, 456, 789, 1337] |
| Samples per seed | 1000 |

**Method name:** `v2_llada_topp0.9_remdm_confidence_tsw1.0`

This is **one experiment, not a sweep**. All design choices were settled by
v1. If the metrics fall outside expected ranges (Section 5), the issue is in
the technical adaptation, not the design choice.

---

## 4. Technical Adaptation: RemaskingSchedule for Per-Position Alpha

This section specifies the exact code changes needed to make `RemaskingSchedule`
work with v2's per-position alpha from the rate network.

### 4.1 The interface problem

Currently, remasking is explicitly disabled for v2 in two places:

**`sampling.py:194–200`** — warns and sets `remasking_fn = None`:
```python
if rate_network is not None and remasking_fn is not None:
    warnings.warn(
        "rate_network and remasking_fn both provided. Remasking with "
        "learned rates is not yet supported; remasking will be skipped.",
        stacklevel=2,
    )
    remasking_fn = None
```

**`generate_samples.py:213–217`** — logs warning and sets `remasking_fn = None`:
```python
if remasking_fn is not None and is_v2:
    logger.warning(
        "Remasking not supported with v2 learned rates; disabling."
    )
    remasking_fn = None
```

The root cause: `RemaskingSchedule._compute_sigma_max()` uses
`self.noise_schedule.alpha(t)` which returns a scalar `(B,)` tensor. With v2,
alpha is per-position `(B, SEQ_LEN)` from the rate network.

### 4.2 Per-position sigma_max derivation

The remasking budget sigma_max measures "how much mask probability remains"
at the current timestep. The formula is identical for v1 and v2 — only the
shape changes:

**v1 (scalar):**
```
sigma_max = clamp((1 - alpha(t_next)) / alpha(t_now), 0, 1)    # shape: (B, 1)
```

**v2 (per-position):**
```
sigma_max_l = clamp((1 - alpha_l(t_next)) / alpha_l(t_now), 0, 1)  # shape: (B, SEQ_LEN)
```

where `alpha_l(t)` comes from `rate_network(t, pad_mask)` returning
`(B, SEQ_LEN)`.

**Physical interpretation:** positions that the rate network keeps clean longer
(higher alpha at t_next) get lower sigma_max (less remasking budget). This is
desirable — the rate network has learned which positions are "easy" and should
be kept stable.

**PAD positions:** The rate network returns `alpha = 1.0` for PAD positions
(enforced in `rate_network.py:271–272`), so `sigma_max_l = 0` for PAD. This
means PAD positions can never be remasked. This is correct and provides a
second layer of protection alongside the existing `remask_candidates` filter.

### 4.3 Interface changes to RemaskingSchedule

**Constructor** — add optional `rate_network` parameter:

```python
class RemaskingSchedule:
    def __init__(
        self,
        strategy: str,
        eta: float,
        noise_schedule: NoiseSchedule | None,  # None when using rate_network
        vocab_config: VocabConfig,
        rate_network: torch.nn.Module | None = None,  # NEW
    ) -> None:
```

When `rate_network` is provided, `noise_schedule` may be `None` — the
RemaskingSchedule will use the rate network for alpha computation instead.
Validation: exactly one of `noise_schedule` or `rate_network` must be provided.

**`_compute_sigma_max`** — add `pad_mask` parameter, branch on rate_network:

```python
def _compute_sigma_max(
    self,
    t_now: float,
    t_next: float,
    batch_size: int,
    device: torch.device,
    pad_mask: Tensor | None = None,  # NEW: required for rate_network
) -> Tensor:
    """Returns (B, 1) for v1 or (B, SEQ_LEN) for v2."""
    if self.rate_network is not None:
        # v2 path: per-position alpha from rate network
        t_now_tensor = torch.full(
            (batch_size,), t_now, dtype=torch.float32, device=device,
        )
        t_next_tensor = torch.full(
            (batch_size,), t_next, dtype=torch.float32, device=device,
        )
        alpha_t = self.rate_network(t_now_tensor, pad_mask).double()
        alpha_s = self.rate_network(t_next_tensor, pad_mask).double()
        sigma_max = (1.0 - alpha_s) / (alpha_t + 1e-8)
        sigma_max = torch.clamp(sigma_max, min=0.0, max=1.0)
        return sigma_max.float()  # (B, SEQ_LEN)
    else:
        # v1 path: scalar alpha from noise_schedule (existing code, unchanged)
        ...
        return sigma_max.float().unsqueeze(1)  # (B, 1)
```

**`__call__`** — no signature change needed. `pad_mask` is already a parameter.
It just needs to be threaded through to `_compute_sigma_max` and
`_confidence_remasking`.

### 4.4 Confidence remasking with per-position sigma_max

Walk through the modified `_confidence_remasking` method to verify correctness:

1. **sigma_max computation** (Step 1): Now `(B, SEQ_LEN)` instead of `(B, 1)`.
   No formula change — just uses rate_network instead of noise_schedule.

2. **Per-position confidence** (Step 2): Unchanged. Still
   `softmax(logits) → gather predicted token probability`. Returns
   `(B, SEQ_LEN)`.

3. **Remasking weights** (Step 3): `eta_conf = softmax(-confidence)` over
   decoded positions. Unchanged. Returns `(B, SEQ_LEN)`.

4. **Budget scaling** (Step 4): This is where per-position sigma_max matters.
   ```python
   sigma_per_pos = (sigma_max * eta_conf * n_decoded).clamp(0.0, 1.0)
   ```

   - **v1**: sigma_max is `(B, 1)` — same budget for all positions,
     redistributed by confidence.
   - **v2**: sigma_max is `(B, SEQ_LEN)` — per-position budget, FURTHER
     redistributed by confidence.

   **Broadcasting**: `sigma_max (B, SEQ_LEN) * eta_conf (B, SEQ_LEN) *
   n_decoded (B, 1)` — all shapes broadcast correctly. No code change needed.

   **Double-weighting effect**: positions with low alpha (harder, more masking
   in forward process) get higher sigma_max, meaning they get MORE remasking
   budget on top of the confidence redistribution. This is a natural double
   signal: the rate network says "this position is harder" AND the model
   confidence says "this prediction is uncertain." Both push toward more
   remasking — desirable behavior.

   **Is double-weighting too aggressive?** No. The softmax already normalizes
   across positions, and sigma_max scaling preserves relative ordering. The
   total budget is bounded by `sigma_max_l * n_decoded` per position, clamped
   to [0, 1]. If anything, per-position sigma_max makes remasking *more
   selective* (easy positions get less budget).

5. **Stochastic remasking decision** (Step 5): Unchanged.
   `rand < sigma_per_pos AND remask_candidates`.

### 4.5 Changes to sampling.py

Remove the warning + disable block at lines 194–200:
```python
# DELETE this block:
if rate_network is not None and remasking_fn is not None:
    warnings.warn(...)
    remasking_fn = None
```

The `remasking_fn` interface is unchanged — `pad_mask` is already passed in
the call at line 363. The RemaskingSchedule now has its own rate_network
reference, so it computes sigma_max internally.

### 4.6 Changes to generate_samples.py

Two changes:

1. **Pass `rate_network` to `create_remasking_schedule()`** at line 210–212:
   ```python
   remasking_fn = create_remasking_schedule(
       cfg.eval.remasking, noise_schedule, vocab_config,
       rate_network=rate_network,  # NEW
   )
   ```

2. **Remove the early-exit block** at lines 213–217:
   ```python
   # DELETE this block:
   if remasking_fn is not None and is_v2:
       logger.warning(...)
       remasking_fn = None
   ```

### 4.7 Changes to factory function

Thread `rate_network` parameter through `create_remasking_schedule()`:

```python
def create_remasking_schedule(
    config: dict,
    noise_schedule: NoiseSchedule | None,  # None allowed when rate_network provided
    vocab_config: VocabConfig,
    rate_network: torch.nn.Module | None = None,  # NEW
) -> RemaskingSchedule | None:
    if not config.get("enabled", False):
        return None
    return RemaskingSchedule(
        strategy=config["strategy"],
        eta=config["eta"],
        noise_schedule=noise_schedule,
        vocab_config=vocab_config,
        rate_network=rate_network,  # NEW
    )
```

### 4.8 Numerical precision

- Rate network returns float32. Cast to **float64** for sigma_max computation
  (same pattern as `sampling.py:256–262`). Cast back to float32 before return.
- Softmax for confidence remasking weights stays float32 (only used for
  relative ranking, not subject to cancellation).

### 4.9 Implementation summary

| File | Change | Lines |
|------|--------|:-----:|
| `bd_gen/diffusion/remasking.py` | Add `rate_network` to constructor + `_compute_sigma_max` v2 branch + thread `pad_mask` | ~30 |
| `bd_gen/diffusion/remasking.py` (factory) | Add `rate_network` param to `create_remasking_schedule` | ~5 |
| `bd_gen/diffusion/sampling.py` | Remove disable block (lines 194–200) | −7 |
| `scripts/generate_samples.py` | Pass `rate_network` to factory, remove early-exit (lines 213–217) | ~5 |
| `tests/test_remasking.py` | Test per-position sigma_max, PAD sigma=0, shape validation, backward compat | ~50 |
| `tests/test_sampling_v2.py` | Test remasking NOT disabled with rate_network | ~15 |
| **Total** | | ~110 |

### 4.10 v1 Backward Compatibility & Reversibility

All changes follow the v2 additive convention from
`planning_T1_with_learned_forward_process.md` Section 1.5: every new parameter
defaults to `None`, and v1 code paths are never entered when `rate_network` is
absent.

**v1 callers are unaffected:**
- `RemaskingSchedule("cap", 0.1, noise_schedule=sched, vocab_config=vc)` —
  works exactly as before (`rate_network=None` by default).
- `create_remasking_schedule(cfg, noise_schedule, vc)` — same, `rate_network`
  defaults to `None`.
- `_compute_sigma_max(t_now, t_next, B, device)` — `pad_mask` defaults to
  `None`, v1 path doesn't use it.
- Removed guard blocks in `sampling.py` and `generate_samples.py` only
  triggered when `rate_network is not None` — never executed for v1.

**v2 removal checklist (if v2 is abandoned):**

| File | Revert action |
|------|---------------|
| `remasking.py` | Remove `rate_network` param from constructor + factory, remove v2 branch in `_compute_sigma_max`, remove `pad_mask` param |
| `sampling.py` | Re-add guard block (optional — harmless when rate_network is never passed) |
| `generate_samples.py` | Remove `rate_network=rate_network` kwarg, re-add early-exit block (optional) |
| `tests/test_remasking.py` | Delete `TestV2SigmaMax`, `TestV2ConfidenceRemasking`, `TestV2Factory` classes + v2 fixtures |
| `tests/test_sampling_v2.py` | Revert `TestV2RemaskingIntegration` to `TestV2RemaskingWarning` |

All v1 tests continue to pass unchanged. The v2 code additions are isolated
and can be deleted without affecting v1 behavior.

---

## 5. Expected Outcomes

Predictions based on v1 remasking patterns, adjusted for v2's stronger
baseline.

### 5.1 Diversity predictions

| Metric | v2 baseline | Predicted v2+remasking | Basis |
|--------|:-----------:|:---------------------:|-------|
| Unique archetypes | 26 | 100–120 | v1: 29 → 121 (4.2x) |
| Diversity | 0.671 | ~0.95–0.98 | v1: 0.945 → 0.982 |
| Novelty | 0.864 | ~0.99 | v1: 0.975 → 0.999 |
| Mode coverage (wt.) | 78.3% | ~80–85% | v1: 69.6% → 73.3% (+3.7pp) |

Mode coverage gain may be smaller than v1's +3.7pp because v2 already starts
higher (78.3% vs 69.6%) — less room for improvement.

### 5.2 Validity predictions

| Metric | v2 baseline | Predicted v2+remasking | Basis |
|--------|:-----------:|:---------------------:|-------|
| Validity | 100.0% | ~99.7% | v1: 100% → 99.7% |
| Inside validity | 100.0% | ~93–97% | v1: 99.4% → 93.3% (−6.1pp) |
| Spatial transitivity | 100.0% | ~98.5–99% | v1: 99.9% → 98.7% (−1.2pp) |

v2's stronger denoiser (higher accuracy at all timesteps: +11–18% vs v1) may
produce a smaller inside_validity drop than v1's 6.1pp.

### 5.3 Distribution predictions

| Metric | v2 baseline | Predicted v2+remasking | v1+remasking (actual) |
|--------|:-----------:|:---------------------:|:---------------------:|
| Edge JS | 0.035 | ~0.050–0.070 | 0.204 |
| Edge TV | 0.217 | ~0.35–0.45 | 0.586 |
| Node JS | 0.013 | ~0.030–0.045 | 0.044 |

Even at the pessimistic end (Edge JS 0.070), v2+remasking would still be
**3x better** than v1+remasking (0.204) and better than v1 baseline (0.106).
This is the key advantage: v2 has enough distribution headroom to absorb
remasking's cost.

### 5.4 Summary comparison

| Metric | v1 baseline | v1+remask | v2 baseline | v2+remask (predicted) |
|--------|:-----------:|:---------:|:-----------:|:---------------------:|
| Inside validity | 99.4% | 93.3% | 100.0% | **93–97%** |
| Edge JS | 0.106 | 0.204 | 0.035 | **0.050–0.070** |
| Mode cov. (wt.) | 69.6% | 73.3% | 78.3% | **80–85%** |
| Archetypes | 29 | 121 | 26 | **100–120** |
| Transitivity | 99.9% | 98.7% | 100.0% | **98.5–99%** |

If the predictions hold, v2+remasking would be the **clear Pareto winner**:
best on every metric except raw Edge JS (where it still dominates v1).

---

## 6. Evaluation Plan

Uses the split pipeline: GPU generation → saved tokens → CPU metrics.

### 6.1 Generation (GPU, jabiru)

```bash
# On jabiru:
cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation

python scripts/generate_samples.py \
    eval.checkpoint_path=outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt \
    noise=learned \
    eval.unmasking_mode=llada \
    eval.top_p=0.9 \
    eval.remasking.enabled=true \
    eval.remasking.strategy=confidence \
    eval.remasking.eta=0.0 \
    eval.remasking.t_switch=1.0
```

Output: `eval_results/learned/v2_llada_topp0.9_remdm_confidence_tsw1.0_samples.pt`

**Directory note:** `noise=learned` causes `schedule_tag = "learned"`, so
samples land in `eval_results/learned/`. For comparison table compatibility
(all 23 existing methods are in `loglinear/`), manually move or symlink the
`_samples.pt` file to `eval_results/loglinear/` before running `evaluate.py`.
Same procedure used for the v2 baseline (`implementation_state_T1.md`, v2
Phase 8 issues).

### 6.2 Evaluation (CPU, jabiru or local)

```bash
python scripts/evaluate.py \
    --schedule loglinear \
    --model v2_llada_topp0.9_remdm_confidence_tsw1.0 \
    --update-comparison
```

Output:
- `eval_results/loglinear/v2_llada_topp0.9_remdm_confidence_tsw1.0.json`
- Updated `eval_results/loglinear/comparison.md` (now 24 methods)

### 6.3 Copy results locally

```bash
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear/v2_*.json BD_Generation/eval_results/loglinear/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear/comparison.md BD_Generation/eval_results/loglinear/
```

### 6.4 Comparison targets

| Label | Method name |
|-------|-------------|
| v2 baseline | `v2_llada_topp0.9_no_remask` |
| v1 + best remasking | `llada_topp0.9_remdm_confidence_tsw1.0` |
| v1 baseline | `llada_topp0.9_no_remask` |

### 6.5 Success and failure thresholds

| Metric | v2 baseline | Success (>=) | Failure (<) |
|--------|:-----------:|:------------:|:-----------:|
| Unique archetypes | 26 | 80 (3x) | 50 |
| Diversity | 0.671 | 0.95 | 0.80 |
| Inside validity | 100.0% | 93% | 90% (abort) |
| Edge JS | 0.035 | < 0.080 | > 0.120 |
| Mode coverage (wt.) | 78.3% | > 80% | < 75% |

### 6.6 One run, not a sweep

Total runs: **1** (not 22 like v1). All design choices were settled by v1's
22-run experiment. The only question is whether the per-position alpha
adaptation works. If it does, metrics will fall in the expected ranges from
Section 5. If it fails (below failure thresholds), the issue is in the
technical adaptation, not the design choice.

---

## 7. Risk Analysis

### Risk 7.1: Inside validity degradation (MEDIUM)

v2 starts at 100% — any drop is visible. v1 confidence remasking dropped
inside_validity by 6.1pp (99.4% → 93.3%).

**Mitigating factors:**
- v2's stronger denoiser (higher accuracy at all timesteps) means re-predictions
  after remasking are more likely correct
- Per-position sigma_max could help: rate network positions that are "easy"
  (high alpha) get less remasking budget, potentially preserving
  inside/surrounding edge integrity
- Confidence strategy's self-regulation naturally protects high-confidence
  correct predictions from remasking

**Mitigation if below 90%:** try reduced top-p (0.8) to limit exploration
range, or add inside_validity post-hoc filtering.

### Risk 7.2: Diversity gain smaller than expected (MEDIUM)

v2's lower baseline diversity (0.671 vs v1's 0.945) comes from more
structured masking trajectories from the rate network.

**Concern:** if the rate network produces very deterministic schedules
(alpha_l very close to 0 or 1 at most timesteps), sigma_max_l will be small
for most positions, limiting the total remasking budget. This would constrain
the explore-correct loop.

**Diagnostic:** after generating samples, compare mean sigma_max across
positions between v1 and v2 at representative timesteps (t = 0.3, 0.5, 0.7).
If v2's mean sigma_max is significantly smaller, the remasking budget is
constrained by the rate network.

**Mitigation if archetypes < 50:** consider a hybrid approach — use v1's global
sigma_max for remasking budget but keep v2's per-position alpha for unmasking.
This decouples the remasking budget from the rate network.

### Risk 7.3: Double-weighting too aggressive (LOW)

sigma_max_l (from rate network) × eta_conf (from confidence softmax) could
over-concentrate remasking on a few positions while leaving most untouched.

This is actually the desired behavior for confidence remasking (target
uncertain positions). If pathological, would show up as very few positions
being remasked per step with no diversity improvement — diagnosable from
the archetype count falling below 50.

### Risk 7.4: Rate network forward pass cost (LOW)

Remasking calls `_compute_sigma_max` once per step, which now requires two
rate_network forward passes (for t_now and t_next). The sampling loop already
calls `rate_network(t_tensor, pad_mask)` and `rate_network(t_next_tensor,
pad_mask)` for p_unmask computation (sampling.py lines 256–260).

**Option A:** pass alpha_now and alpha_next from sampling loop to remasking_fn
(avoid redundant computation). Requires expanding the remasking_fn interface.

**Option B:** let RemaskingSchedule recompute — rate network is tiny (~5K
params, polynomial evaluation). Cost is negligible vs the denoiser forward pass
(~1.28M params).

**Recommendation:** Option B for simplicity. The rate network forward pass is a
single matrix multiply + polynomial evaluation — ~0.1ms per call. At 100 steps,
that's ~20ms total overhead (200 extra rate_network calls), negligible compared
to ~100 denoiser passes at ~5ms each = ~500ms.

---

## 8. Post-Experiment Decision Tree

```
archetypes > 80 AND inside_validity > 93%
  → SUCCESS. Use v2_llada_topp0.9_remdm_confidence_tsw1.0 as the recommended
    model. Update best_model_according_to_preferred_metrics_on_llada_unm.md.

archetypes > 80 BUT inside_validity < 93%
  → PARTIAL SUCCESS. Try reduced top-p (0.8) to limit exploration range.
    If still below 93%, accept the tradeoff or add post-hoc filtering.

archetypes < 50
  → REMASKING NOT EFFECTIVE. Investigate:
    (a) Is mean sigma_max_l too small? → Rate network over-constrains budget.
    (b) Is the double-weighting suppressing remasking? → Try v1 global
        sigma_max for budget, keep v2 per-position alpha for unmasking.
    (c) Is the low baseline diversity (0.671) a denoiser problem, not a
        remasking problem? → Try increased top-p (0.95) without remasking first.

metrics worse than v2 baseline across the board
  → BUG IN IMPLEMENTATION. Check sigma_max computation, broadcasting,
    float64 precision. Verify RemaskingSchedule receives rate_network correctly.
```
