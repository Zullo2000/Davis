# Handoff: v2 Learned Forward Process Implementation

> **Created:** 2026-02-20
> **Session purpose:** Implement MELD learned forward process (v2) per spec `planning_T1_with_learned_forward_process.md`

---

## What was accomplished

All core v2 modules are implemented and tested (588 tests pass, 0 failures):

- **Phase 1 DONE:** `bd_gen/diffusion/rate_network.py` — RateNetwork class with polynomial parameterization, per-position alpha, analytical derivative, PAD handling. Tests: `tests/test_rate_network.py` (12 tests)
- **Phase 2 DONE:** Added to `bd_gen/diffusion/forward_process.py` — `stgs_sample()`, `forward_mask_learned()` (training), `forward_mask_eval_learned()` (eval), `STGSOutput` TypedDict. Tests: `tests/test_forward_process_v2.py` (13 tests)
- **Phase 3 DONE:** Added `ELBOLossV2` class to `bd_gen/diffusion/loss.py` — separate node/edge normalization, per-position weights, lambda_edge. Tests: `tests/test_loss_v2.py` (9 tests)
- **Phase 4 DONE:** Added `pre_embedded` optional parameter to `BDDenoiser.forward()` in `bd_gen/model/denoiser.py` — v1 callers unaffected (default None)
- **Phase 5 DONE:** Modified `bd_gen/diffusion/sampling.py` — added `rate_network` parameter to `sample()`, per-position p_unmask branch, llada budget adaptation, remasking incompatibility warning. Tests: `tests/test_sampling_v2.py` (10 tests)
- **Phase 6 DONE:** `scripts/train_v2.py` (full training loop), `configs/noise/learned.yaml`, `configs/training/v2.yaml`
- **`bd_gen/diffusion/__init__.py`** updated with all new exports (STGSOutput, forward_mask_learned, forward_mask_eval_learned, stgs_sample, ELBOLossV2)

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Denoiser change | Single `pre_embedded` param | Minimal v1 impact, default None preserves all v1 behavior |
| Training script | Separate `train_v2.py` | Fundamentally different loop (STGS, dual model, gumbel temp) |
| Checkpoint format | `rate_network_state_dict` key | Auto-detect v2 vs v1 checkpoint in evaluate.py |
| Gumbel temp schedule | Linear decay by default | Simpler, configurable via `gumbel_temperature_decay` |
| Dummy noise schedule | LogLinearSchedule passed to sample() even for v2 | sample() signature requires it; v2 path ignores it when rate_network provided |

## Current state of the codebase

- **All 588 tests pass** (0 failures, 2 skips)
- **ruff** should be clean (agents ran ruff during implementation)
- All v1 code paths are 100% backward compatible
- The `__init__.py` was updated by the linter to include the new v2 exports

## What remains to be done

### Phase 7: Evaluation Integration (evaluate.py) — ~5 targeted edits

1. Add `RateNetwork` import (DONE — line 57)
2. Add `_load_v2_checkpoint()` helper function (was about to write — detects v2 by `rate_network_state_dict` key)
3. Replace model loading block (lines 352-354) with v2-aware loading that calls `_load_v2_checkpoint()`
4. Add `rate_network=None` parameter to `_generate_and_evaluate_single_seed()` signature, pass to `sample()` call
5. Pass `rate_network` in multi-seed loop (line 435) and sample-saving section (line 580)
6. Add `v2_` prefix to method_name when rate_network is not None (line 529)

See spec Section 12.4 and the `_load_v2_checkpoint` function I started writing (it was rejected mid-edit).

### Documentation updates

7. Update `implementation_state_T1.md` — add v2 Phase summaries (1-7) following existing format
8. Update `docs/diffusion.md` — add v2 section at end covering rate_network, STGS, forward_mask_learned, ELBOLossV2
9. Update `docs/model.md` — add v2 section noting pre_embedded parameter
10. Update `docs/training.md` — add v2 section covering train_v2.py, configs, gumbel temp
11. Update `docs/evaluation.md` — add v2 section covering v2 checkpoint loading, rate_network in sampling

**Important:** User asked that all doc additions explicitly state they belong to v2 and go at the end.

## Files to reference in next session

1. `BD_Generation/planning_T1_with_learned_forward_process.md` — THE SPEC (Sections 9, 10, 12, 13, 14)
2. `BD_Generation/implementation_state_T1.md` — current state (needs v2 phases added)
3. `BD_Generation/scripts/evaluate.py` — needs Phase 7 edits
4. `BD_Generation/scripts/train_v2.py` — just created, reference for Phase 7
5. `BD_Generation/bd_gen/diffusion/rate_network.py` — for _load_v2_checkpoint pattern
6. `BD_Generation/bd_gen/diffusion/sampling.py` — to see rate_network parameter placement

## Context for the next session

- The `sample()` function's `rate_network` parameter was added AFTER `remasking_fn` and BEFORE `num_rooms_distribution` — match this order in all callers
- When `rate_network is not None`, `sample()` warns and disables remasking_fn (not yet supported)
- v2 checkpoint detection: presence of `rate_network_state_dict` key in the checkpoint dict
- The `noise_schedule` parameter to `sample()` is still required even for v2 — pass a dummy `LogLinearSchedule()` when using rate_network
- All agents that created code ran tests successfully. The full suite was verified: 588 passed
- Config override for v2 training: `python scripts/train_v2.py noise=learned training=v2`
