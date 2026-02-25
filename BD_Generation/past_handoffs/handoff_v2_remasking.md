# Handoff: v2 Remasking (Learned Forward Process)

> **Created:** 2026-02-25
> **Session purpose:** Enable ReMDM confidence remasking for v2 learned forward process (per-position sigma_max from rate network)

---

## What was accomplished

- **Adapted `RemaskingSchedule` for v2**: Added `rate_network=None` kwarg to constructor and factory. New v2 branch in `_compute_sigma_max()` returns per-position `(B, SEQ_LEN)` sigma instead of scalar `(B, 1)`. PAD positions get sigma=0 automatically (rate_network returns alpha=1 for PAD).
- **Removed remasking blockers**: Deleted the guard block in `sampling.py` (lines 194-200) that disabled remasking when `rate_network` was provided. Deleted the early-exit block in `generate_samples.py` (lines 213-217).
- **14 new tests**: 11 in `test_remasking.py` (TestV2SigmaMax, TestV2ConfidenceRemasking, TestV2Factory) + 3 in `test_sampling_v2.py` (TestV2RemaskingIntegration replacing TestV2RemaskingWarning).
- **Design doc updated**: Added Section 4.10 (v1 backward-compat & reversibility) to `remasking_design_with_learned_forward_process.md`.
- **Implementation state updated**: New section "v2 — Remasking with Per-Position Sigma_max" in `implementation_state_T1.md`.

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Single configuration | llada + confidence + t_switch=1.0 + top-p=0.9 | v1's 22-run experiment eliminated all other options (Section 2 of design doc) |
| Option B for rate_network cost | Let RemaskingSchedule recompute alpha (don't pass from sampling loop) | Rate network is tiny (~5K params, ~0.1ms/call). 20ms total overhead vs ~500ms for denoiser |
| All changes additive | `rate_network=None` defaults everywhere | v1 backward-compat per planning doc Section 1.5; v2 removable |

## Current state of the codebase

- **Code complete and tested**: 54/54 pass (test_remasking + test_sampling_v2), 598/601 full suite (3 pre-existing unrelated failures in test_metrics.py), ruff clean
- **NOT committed**: All changes are unstaged. The user may want to review before committing.
- **NOT evaluated**: No samples generated yet — needs GPU run on jabiru

## What remains to be done

1. **Generate samples on jabiru (GPU)**:
   ```bash
   cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation
   # First: git pull or scp the changed files to jabiru
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

2. **Move samples to loglinear dir** (for comparison table compatibility):
   ```bash
   cp eval_results/learned/v2_llada_topp0.9_remdm_confidence_tsw1.0_samples.pt \
      eval_results/loglinear/
   ```

3. **Run evaluation (CPU)**:
   ```bash
   python scripts/evaluate.py \
       --schedule loglinear \
       --model v2_llada_topp0.9_remdm_confidence_tsw1.0 \
       --update-comparison
   ```

4. **Copy results locally**:
   ```bash
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear/v2_*.json BD_Generation/eval_results/loglinear/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear/comparison.md BD_Generation/eval_results/loglinear/
   ```

5. **Interpret results** against expected ranges (design doc Section 5):
   - Archetypes: expect 100-120 (success >= 80, failure < 50)
   - Inside validity: expect 93-97% (abort if < 90%)
   - Edge JS: expect 0.050-0.070 (failure > 0.120)
   - See decision tree in design doc Section 8

6. **If successful**: Update `best_model_according_to_preferred_metrics_on_llada_unm.md`

## Files to reference in next session

1. `BD_Generation/remasking_design_with_learned_forward_process.md` — Full design doc with evaluation commands (Section 6), expected outcomes (Section 5), decision tree (Section 8)
2. `BD_Generation/implementation_state_T1.md` — "v2 — Remasking with Per-Position Sigma_max" section has the configuration and test summary
3. `BD_Generation/eval_results/loglinear/comparison.md` — Current 23-method comparison table (will become 24 after eval)
4. `BD_Generation/eval_results/loglinear/best_model_according_to_preferred_metrics_on_llada_unm.md` — Method selection analysis to update if results are good

## Context for the next session

- **noise=learned causes schedule_tag="learned"**: Samples land in `eval_results/learned/`, not `eval_results/loglinear/`. Must manually copy/symlink to `loglinear/` before running `evaluate.py`. Same issue encountered during v2 Phase 8 (documented in implementation_state_T1.md).
- **t_switch=1.0 skips the first step**: `t_now < t_switch` at t=1.0 is False, so remasking doesn't run at the very first step. This is correct — everything is masked at t=1.0, nothing to remask.
- **The 3 failing tests in test_metrics.py** are pre-existing: `_aggregate_multi_seed` was renamed to `aggregate_multi_seed` in save_utils.py but the test still imports the old name. Not related to our changes.
- **v1 is fully preserved**: If v2+remasking results are bad, v1+remasking (`llada_topp0.9_remdm_confidence_tsw1.0`) remains the best diversity method with 121 archetypes and 93.3% inside validity.
