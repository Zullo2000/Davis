# Handoff: G3 Guided Sampler Implementation

> **Created:** 2026-02-27
> **Session purpose:** Implement Phase G3 (Guided Sampler) of the SVDD inference-time guidance system

---

## What was accomplished

- **Refactored `bd_gen/diffusion/sampling.py`**: Extracted loop body into `_single_step_unmask()` (steps 4c–4h) and `_single_step_remask()` (step 4i). `sample()` calls both sequentially — zero behavior change. All 38 existing sampling tests pass.
- **Implemented `bd_gen/guidance/guided_sampler.py`**: Full SVDD K-candidate reweighting loop with `guided_sample()` function, `GuidanceStats` dataclass, and scoring helpers for both soft and hard reward modes. Includes per-step diagnostics (ESS, weight entropy, reward trajectories, remasking delta, per-constraint violations).
- **Updated `bd_gen/guidance/__init__.py`**: Added exports for `guided_sample` and `GuidanceStats`.
- **Created `tests/test_guided_sampler.py`**: 16 tests covering spec tests 40–53 (K=1 unguided match, PAD/MASK invariants, output shapes, constraint effectiveness, ESS sanity, stats completeness, v1/v2 compatibility, remasking, hard mode, top-p). All pass.
- **Created `scripts/generate_guided.py`**: CLI script following `generate_samples.py` pattern. Loads model + guidance config + calibration, calls `guided_sample()` in batches, saves results with `guidance_stats` key.

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| v2 detection in `_single_step_unmask` | `p_unmask.shape[-1] > 1` instead of `rate_network is not None` | Helper shouldn't need to know about rate networks; shape check is equivalent (v2 gives (B,SEQ), v1 gives (B,1)) |
| Scoring loop | Python loop over K×B with per-sample calls | Spec says vectorized in practice but per-sample is correct first; optimization deferred |
| `final_mean_violation_when_failed` | Computed from total_violation / num_failed | Avoids double-counting satisfied samples |

## Current state of the codebase

- **Full test suite**: 711 passed, 3 pre-existing failures (`_aggregate_multi_seed` import — unrelated), 2 skipped. **No regressions from G3.**
- **G1 COMPLETE**, **G2 COMPLETE**, **G3 code COMPLETE** — pending user approval to mark G3 as COMPLETE.
- Config YAMLs (`configs/guidance/default.yaml`, `example_basic.yaml`) already existed from G1 with correct fields.

## What remains to be done

1. **Update `docs/guidance.md`** with G3 section (guided sampler docs, diagnostics table, generation script usage, compute cost table). Edit was drafted but user interrupted before applying — the content is described in the session.
2. **Update `implementation_state_T1_guidance.md`** — mark G3 status, record files created/modified, test results, deviations from spec.
3. **User approval** to mark G3 as COMPLETE.
4. **Phase G4** (Calibration + Evaluation): `calibration.py`, `calibrate_constraints.py`, `evaluate.py` extension. Can run in parallel with nothing — only depends on G1.
5. **Phase G5** (End-to-End experiments): No new code. Sweep α, K, model variants.

## Files to reference in next session

1. `BD_Generation/CLAUDE.md` — agent rules (read first)
2. `BD_Generation/implementation_state_T1_guidance.md` — dynamic state (needs G3 update)
3. `BD_Generation/planning_T1_guidance.md` — static spec (sections 8, 10–14 for G3/G4)
4. `BD_Generation/bd_gen/guidance/guided_sampler.py` — core G3 implementation
5. `BD_Generation/tests/test_guided_sampler.py` — 16 tests, all pass
6. `BD_Generation/scripts/generate_guided.py` — generation script
7. `BD_Generation/bd_gen/diffusion/sampling.py` — refactored with `_single_step_unmask` + `_single_step_remask`
8. `BD_Generation/docs/guidance.md` — needs G3 section added

## Context for the next session

- The `docs/guidance.md` update was drafted but the edit was rejected by the user (they wanted to create a handoff instead). The G3 docs section should cover: sampling refactoring, SVDD loop pseudocode, diagnostics table, generation script usage, compute cost table.
- The `implementation_state_T1_guidance.md` G3 section template already exists (lines 96–121) with placeholders — just needs filling in with actual results.
- `generate_guided.py` uses `argparse` for guidance-specific args (`--guidance-config`, `--alpha`, `--K`, `--guidance-tag`) and passes remaining args to Hydra. This is different from `generate_samples.py` which uses pure Hydra.
- The scoring loop in `guided_sampler.py` is a Python loop over K×B — works correctly but could be optimized later with batched `build_effective_probs_batch`. The spec acknowledges this is acceptable for K=8, B=64.
