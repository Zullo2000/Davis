# Handoff: Evaluation Pipeline Split (Generate vs Evaluate)

> **Created:** 2026-02-24
> **Session purpose:** Split monolithic evaluate.py into GPU-only generation + CPU-only metric computation, so new metrics don't require re-generating 23 models × 5 seeds × 1000 samples.

---

## What was accomplished

- **Split the evaluation pipeline into two scripts:**
  - `scripts/generate_samples.py` (NEW) — GPU-only: loads model, generates tokens per seed, saves `{method}_samples.pt`
  - `scripts/evaluate.py` (NEW) — CPU-only: loads saved tokens, computes ALL metrics, saves/updates `{method}.json`
- **Moved shared utilities to `eval_results/save_utils.py`:**
  - `_make_json_serializable` → `make_json_serializable` (public)
  - Added `aggregate_multi_seed()` (extracted from old evaluate.py)
- **Renamed old evaluate.py → `scripts/generate_and_evaluate.py`** (transition backup)
- **Updated `implementation_state_T1.md`** with new "Evaluation Pipeline Split" section
- All files pass syntax checks and import verification

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Save format | tokens + pad_masks as `.pt` per method | Compact (~1.6 MB per method), canonical representation, can recompute everything from tokens |
| File location | `eval_results/{schedule}/{method}_samples.pt` alongside `{method}.json` | Co-location, same naming logic, `.pt` files don't interfere with `*.json` glob |
| New evaluate.py CLI | argparse (not Hydra) | No model/config needed for CPU-only metrics; simpler interface |
| Metric filtering | Always compute ALL metrics | All are CPU-cheap; no reason to skip any |
| Old evaluate.py | Renamed to `generate_and_evaluate.py` | Transition backup while backfilling; delete once all 23 models have `_samples.pt` |

## Current state of the codebase

- **Code complete, NOT yet committed.** All changes are unstaged.
- **No `_samples.pt` files exist yet** — the 23 existing models need a one-time GPU backfill run
- Existing eval results: 23 JSON files in `eval_results/loglinear/`, 12 in `eval_results/linear/`
- `generate_and_evaluate.py` is unchanged from the original evaluate.py (just renamed)
- Imports verified: all function signatures match, data paths match config defaults

## What remains to be done

1. **Commit the current changes** (pipeline split code)
2. **Backfill all models with `_samples.pt` files** — run `generate_samples.py` for each of the 23+12 model configs on GPU (each config uses different checkpoint + sampling params)
   - This requires access to the GPU server (jabiru) and all checkpoints
   - The 23 loglinear configs and 12 linear configs can be found by examining the existing JSON files' `config` sections
3. **Verify consistency** — run `evaluate.py` on a backfilled model and compare metrics to the existing JSON (should be identical since same tokens + same metric code)
4. **Delete `generate_and_evaluate.py`** once all models are backfilled
5. **Add any new metrics** to `evaluate.py`'s `compute_all_metrics()` function, then re-run on all models (CPU-only, instant)

## Files to reference in next session

1. `BD_Generation/implementation_state_T1.md` — last section has the pipeline split status
2. `BD_Generation/scripts/generate_samples.py` — GPU generation script (NEW)
3. `BD_Generation/scripts/evaluate.py` — CPU metric script (NEW)
4. `BD_Generation/eval_results/save_utils.py` — shared utilities (MODIFIED)
5. `BD_Generation/scripts/generate_and_evaluate.py` — old monolithic script (transition backup)
6. `BD_Generation/configs/eval/default.yaml` — eval config with seeds, sampling params

## Context for the next session

- **Backfill strategy:** Each existing JSON file's `config` section contains the exact Hydra overrides needed to reproduce that run (checkpoint, unmasking_mode, top_p, remasking settings, noise type). You can parse these to generate the `generate_samples.py` commands.
- **v2 models:** v2 checkpoints are auto-detected by `generate_samples.py` (presence of `rate_network_state_dict` key). The method name gets `v2_` prefix automatically.
- **v2 noise type quirk:** v2 uses `noise.type=learned` but results are stored in `eval_results/loglinear/` (manually placed). `generate_samples.py` derives the output dir from `cfg.noise.type`, so v2 runs will save to `eval_results/learned/` unless you override `noise.type=loglinear`.
- **The 23 loglinear models** include: 4 baselines (random/llada × argmax/topp), 10 cap-eta runs, 8 confidence-tsw runs, 1 v2 model. See `eval_results/loglinear/comparison.md` for the full list.
- **No tests were written** for the new scripts (they're CLI tools, not library code). Verification is via metric comparison against existing JSONs.
