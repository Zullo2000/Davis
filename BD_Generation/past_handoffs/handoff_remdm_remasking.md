# Handoff: ReMDM Remasking Implementation

> **Created:** 2026-02-18
> **Session purpose:** Implement ReMDM-style remasking (inference-only) on top of MDLM bubble diagram generator, evaluate and compare against baseline.

---

## What was accomplished

- **Baseline evaluation captured**: Ran MDLM v1 with seed=42, 1000 samples, 100 steps. Results saved to `eval_results/mdlm_baseline.json`.
- **ReMDM remasking implemented**: New `bd_gen/diffusion/remasking.py` with `RemaskingSchedule` class supporting "cap" and "rescale" strategies. Factory function `create_remasking_schedule()` for Hydra config wiring.
- **Sampling hook expanded**: `bd_gen/diffusion/sampling.py` remasking hook signature changed from `(x_t, t)` to `(x_t, t_now, t_next, pad_mask)` with `i > 0` guard (no remasking at final step).
- **Evaluation pipeline updated**: `scripts/evaluate.py` wires remasking config from Hydra, saves structured JSON to `eval_results/`.
- **Config extended**: `configs/eval/default.yaml` now has `unmasking_mode` and nested `remasking` section (enabled, strategy, eta).
- **19 remasking tests**: `tests/test_remasking.py` — PAD protection stress test, sigma formulas, mask token correctness, integration with `sample()`, float64 precision.
- **ReMDM evaluation run**: Same seed=42 config + cap strategy eta=0.1. Results in `eval_results/remdm_cap_eta0.1.json`.
- **Comparison table**: `eval_results/comparison.md` with full side-by-side metrics and analysis.
- **Documentation updated**: Both `planning_T1_with_fixed_forward_process.md` (2 surgical changes) and `implementation_state_T1.md` (new section appended).
- **Eval results infrastructure**: `eval_results/save_utils.py` with `save_eval_result()`, `load_eval_result()`, `build_comparison_table()`.

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Remasking approach | Post-hoc (not full ReMDM two-distribution posterior) | Simpler, captures key error-correction benefit, extensible later |
| Hook signature | `(x_t, t_now, t_next, pad_mask)` | sigma_t needs both alpha_t/alpha_s; PAD protection is critical invariant |
| Last-step guard | `i > 0` check in sampling loop | Without it, remasked positions remain as MASK in output |
| sigma_t precision | float64 | Matches existing p_unmask numerical stability pattern |
| Default eta | 0.1 for cap strategy | Recommended in ReMDM paper, good diversity/validity trade-off |
| Results format | JSON per run + markdown comparison | JSON is machine-readable; markdown for documentation/papers |

## Current state of the codebase

### What works
- **All 453 tests pass** (including 19 new remasking tests)
- Full v1 MDLM pipeline: data → training → evaluation → visualization
- ReMDM remasking: configurable via Hydra CLI overrides
- Evaluation comparison infrastructure: save/load/compare across methods

### Uncommitted changes
All remasking work is **uncommitted** (staged + untracked). Key changes:
- **Modified**: `bd_gen/diffusion/__init__.py`, `bd_gen/diffusion/sampling.py`, `configs/eval/default.yaml`, `implementation_state_T1.md`, `planning_T1_with_fixed_forward_process.md`, `scripts/evaluate.py`, `tests/test_sampling.py`
- **New files**: `bd_gen/diffusion/remasking.py`, `eval_results/` (directory with 5 files), `tests/test_remasking.py`
- **Also uncommitted (pre-existing)**: `handoff_float64_stability.md`, `handoff_zero_masking.md`, `changes_for_sound_comparisons.md`, `remasking_doubts.md`

### Evaluation results summary
| Metric | MDLM Baseline | ReMDM-cap (eta=0.1) |
|---|---|---|
| Validity | 99.5% | 98.5% (-1.0%) |
| Diversity | 0.977 | 0.993 (+0.016) |
| Novelty | 0.943 | 0.976 (+0.033) |
| Unique archetypes | 62 | 126 (2x) |
| Mode coverage (unweighted) | 7.5% | 11.5% (+3.9%) |
| Cond. edge KL (weighted) | 0.4253 | 0.3975 (-0.028, better) |

## What remains to be done

Per the "Next Steps" in `eval_results/comparison.md`:
1. **Sweep lower eta values** (0.02–0.05) to find sweet spot between diversity gain and validity preservation
2. **Try "rescale" strategy** (already implemented, just needs evaluation run)
3. **Combine remasking with confidence-based unmasking** (`unmasking_mode: "confidence"` — already implemented in sampling.py)
4. **Full ReMDM two-distribution posterior** (separate distributions for masked vs unmasked positions — more complex, deferred)
5. **Commit the remasking work** — all changes are uncommitted

## Files to reference in next session

Priority reading order:
1. `BD_Generation/implementation_state_T1.md` — dynamic state, last section covers ReMDM
2. `BD_Generation/eval_results/comparison.md` — results analysis and next steps
3. `BD_Generation/bd_gen/diffusion/remasking.py` — core remasking implementation
4. `BD_Generation/bd_gen/diffusion/sampling.py` — the sampling loop with remasking hook
5. `BD_Generation/configs/eval/default.yaml` — remasking config structure
6. `BD_Generation/planning_T1_with_fixed_forward_process.md` — static spec (only if architectural questions arise)

## Context for the next session

- **Remasking is inference-only** — no model retraining needed. Just flip `eval.remasking.enabled=true` in the evaluate.py CLI.
- **The "rescale" strategy is already coded** in `RemaskingSchedule._compute_sigma_t()` — just pass `eval.remasking.strategy=rescale`.
- **`notebooks/04_sample_analysis.ipynb`** has modified cells but those are from the previous session's analysis work, not from this remasking session.
- **PAD protection is the #1 invariant** — any new sampling modifications must respect `pad_mask`. The stress test in `test_remasking.py::TestPADProtection` runs 1000 iterations.
- **Hydra config overrides** for evaluation: `python scripts/evaluate.py eval.checkpoint_path=checkpoint_final.pt seed=42 eval.num_samples=1000 eval.sampling_steps=100 eval.remasking.enabled=true eval.remasking.strategy=cap eval.remasking.eta=0.1 wandb.mode=disabled`
- **There are scratch files** in the repo (`changes_for_sound_comparisons.md`, `remasking_doubts.md`, `handoff_float64_stability.md`, `handoff_zero_masking.md`) — these are working notes, not part of the codebase.
