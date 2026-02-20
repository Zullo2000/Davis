# Handoff: Evaluation Infrastructure Upgrade — COMPLETE

> **Created:** 2026-02-18
> **Session purpose:** Implement all 5 batches of the evaluation infrastructure upgrade (JS/TV/W1 metrics, denoising eval, multi-seed, stratified, scoreboard prefixes)

---

## What was accomplished

All 5 batches from the evaluation upgrade plan are **complete** (494/494 tests pass):

### Batch 1 — Distance primitive tests
- Added 16 unit tests for `_total_variation`, `_js_divergence`, `_wasserstein1_1d_discrete` (functions existed from prior session)
- Tests cover: identical distributions, disjoint, symmetry, bounds, different lengths, zero entries

### Batch 2 — JS/TV/W1 metric extensions
- Extended `distribution_match()` → now returns `node_js`, `edge_js`, `node_tv`, `edge_tv`, `rooms_w1` alongside existing KL keys
- Extended `conditional_edge_kl()` → now returns JS/TV mean+weighted variants
- Added `conditional_edge_distances_topN()` — KL/JS/TV for top-N most frequent room-type pairs
- Extended `type_conditioned_degree_kl()` → now returns JS/TV mean+weighted variants
- All existing KL keys preserved (backward compatible, additive only)

### Batch 3 — Denoising evaluation module
- Created `bd_gen/eval/denoising_eval.py` with `denoising_eval()` and `denoising_val_elbo()`
- Sampler-independent model quality: accuracy + cross-entropy at configurable noise levels
- 4 tests in `tests/test_denoising_eval.py`; docs in `docs/denoising_eval.md`

### Batch 4 — Multi-seed + stratified + evaluate.py refactor
- Rewrote `scripts/evaluate.py`: extracted `_generate_and_evaluate_single_seed()`, added `_aggregate_multi_seed()` (mean/std over 5 seeds), integrated denoising eval + stratified metrics
- Added `validity_by_num_rooms()`, `spatial_transitivity_by_num_rooms()`, `edge_present_rate_by_num_rooms()` to metrics.py
- New JSON output structure: `{meta, per_seed, summary, denoising}`

### Batch 5 — Scoreboard prefixes + documentation
- Added `_prefix_metrics()` for wandb grouping (`denoise/*`, `sampler/validity/*`, etc.)
- Updated `docs/evaluation.md` with all new sections
- Updated `implementation_state_T1.md` with "Post-v1 — Evaluation Infrastructure Upgrade" section

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| JS convention | 0·log(0/x)=0, no epsilon | Mathematically correct, avoids smoothing artifacts |
| KL retention | Keep alongside JS/TV | Backward compatibility, diagnostic value |
| W1 scope | num_rooms only | W1 requires ordinal variable; JS/TV for categorical |
| Multi-seed | mean±std over 5 seeds, no bootstrap | Simple, sufficient for comparing methods |
| Denoising eval | Runs once, seed-independent | Only sampler quality varies per seed |
| Stratified scope | validity, transitivity, edge_present_rate | Most actionable drill-down metrics |
| Key-set assertions | Changed from `==` to `issubset` | Forward-compatible as new keys are added |

## Current state of the codebase

- **All 494 tests pass** (0 failures, 1 unrelated warning about lr_scheduler)
- `ruff check` should be clean (no new lint issues introduced)
- All new code has corresponding tests and documentation
- `implementation_state_T1.md` is up to date
- Config `configs/eval/default.yaml` has all new fields with sensible defaults
- **No uncommitted breaking changes** — all extensions are additive

### Uncommitted files (git status)
**Modified:** `bd_gen/diffusion/__init__.py`, `sampling.py`, `bd_gen/eval/__init__.py`, `metrics.py`, `configs/eval/default.yaml`, `docs/evaluation.md`, `implementation_state_T1.md`, `scripts/evaluate.py`, `tests/test_metrics.py`, `tests/test_sampling.py`, `planning_T1_with_fixed_forward_process.md`, `notebooks/04_sample_analysis.ipynb`

**New:** `bd_gen/diffusion/remasking.py`, `bd_gen/eval/denoising_eval.py`, `docs/denoising_eval.md`, `tests/test_denoising_eval.py`, `tests/test_remasking.py`, `eval_results/` (comparison data)

## What remains to be done

1. **Commit all changes** — the eval upgrade + ReMDM remasking work is complete but uncommitted
2. **Run full evaluation** with the upgraded pipeline (`python scripts/evaluate.py`) on the trained checkpoint to generate multi-seed results with JS/TV/W1 metrics
3. **Compare MDLM vs ReMDM** using the new statistically sound metrics (mean±std across 5 seeds)
4. **Optional:** Tune ReMDM eta parameter using the new denoising eval as a model-quality baseline
5. **Optional:** Analyze stratified metrics to identify if certain num_rooms values have systematically worse performance

## Files to reference in next session

1. `BD_Generation/implementation_state_T1.md` — full project state (always read first per CLAUDE.md)
2. `BD_Generation/scripts/evaluate.py` — the refactored evaluation pipeline
3. `BD_Generation/bd_gen/eval/metrics.py` — all metric functions
4. `BD_Generation/bd_gen/eval/denoising_eval.py` — denoising evaluation module
5. `BD_Generation/configs/eval/default.yaml` — evaluation configuration
6. `BD_Generation/docs/evaluation.md` — evaluation documentation
7. `BD_Generation/changes_for_sound_comparisons.md` — the original change spec that drove this upgrade

## Context for the next session

- **OmegaConf requirement**: `denoising_eval.py` uses `OmegaConf.create()` to wrap noise config dicts — `get_noise()` expects attribute access (`.type`), not dict access (`["type"]`)
- **Noise schedule**: denoising eval uses `"cosine"` schedule (not `"loglinear"` which doesn't exist)
- **Test key assertions**: use `expected_keys.issubset(result.keys())` not `==` for forward compatibility
- **Scoreboard prefixes**: `_prefix_metrics()` maps flat eval keys to grouped wandb metric names — see the function in `evaluate.py` for the full mapping
- **Multi-seed flow**: denoising eval runs once outside the seed loop; sampler metrics run per-seed then aggregate
