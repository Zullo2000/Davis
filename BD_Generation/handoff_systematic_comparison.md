# Handoff: Systematic Evaluation Comparison Infrastructure

> **Created:** 2026-02-18
> **Session purpose:** Fix the disconnect between the evaluation pipeline (which correctly computes JS/TV/W1 + multi-seed) and the results/comparison layer (which was still single-seed KL-only)

---

## What was accomplished

### Code changes
- **Rewrote `eval_results/save_utils.py`** — V2 JSON format with `{format_version, per_seed, summary, denoising}`, backward-compatible V1 auto-upgrade, structured `build_comparison_table()` with metric family grouping (Validity, Coverage, Distribution, Structure, Conditional, Denoising)
- **Updated `scripts/evaluate.py`** (lines 508-528) — `save_eval_result()` now receives structured multi-seed data (`per_seed_metrics`, `summary_metrics`, `denoising_metrics`) instead of `flat_metrics`
- **Created `scripts/compare.py`** — CLI utility that auto-discovers `eval_results/*.json` and generates `comparison.md`
- **Created `tests/test_save_utils.py`** — 20 tests covering V2 roundtrip, V1 upgrade, formatting, table generation

### Evaluations run
- **MDLM baseline**: 5 seeds x 1000 samples x 100 steps → `eval_results/mdlm_baseline.json` (V2 format)
- **ReMDM-cap eta=0.1**: 5 seeds x 1000 samples x 100 steps → `eval_results/remdm_cap_eta0.1.json` (V2 format)
- **Generated `eval_results/comparison.md`** — auto-generated with all metric families, mean +/- std, JS/TV/W1 as primary

### Test results
- **514 tests pass** (0 failures), including 20 new `test_save_utils.py` tests

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| JSON format version | V2 with `format_version: 2` field | Enables backward-compatible V1 auto-upgrade |
| V1 upgrade behavior | `std: 0.0` for all metrics, `_upgraded_from_v1` flag | No misleading uncertainty, clear footnote in comparison |
| Unicode avoidance | `+/-` instead of `±`, `--` instead of `—` | Windows console encoding issues |
| Metric family registry | Hardcoded list of `(key, display_name, is_pct, is_diagnostic)` tuples | Ensures consistent table structure across runs |
| Integer formatting | `mean == int(mean)` check → display as int | "Unique archetypes: 120" not "120.0000" |
| KL as diagnostic | Marked `(diag.)` in table, hideable with `--primary-only` | JS/TV/W1 are the headline metrics per spec |

## Current state of the codebase

- All evaluation infrastructure works end-to-end: `evaluate.py` → V2 JSON → `compare.py` → `comparison.md`
- `eval_results/` contains V2 JSONs for both methods with full per-seed data + summary + denoising
- `comparison.md` is auto-generated with 7 sections, mean +/- std, delta column
- **All changes are uncommitted** — see git status above
- Previous handoff files (`handoff_eval_upgrade.md`, `handoff_eval_upgrade_complete.md`, `handoff_remdm_remasking.md`) document earlier sessions

## What remains to be done

1. **Commit all changes** — the eval upgrade + remasking + systematic comparison work is uncommitted
2. **Optional: Lower eta values** — try ReMDM with eta=0.02-0.05 for better validity/diversity tradeoff
3. **Optional: Rescale strategy** — try `remasking.strategy=rescale` instead of `cap`
4. **Optional: Confidence-based unmasking** — combine remasking with `unmasking_mode=confidence`
5. **Update `implementation_state_T1.md`** — add section on systematic comparison infrastructure

## Files to reference in next session

1. `BD_Generation/eval_results/comparison.md` — the auto-generated comparison (final output)
2. `BD_Generation/eval_results/save_utils.py` — V2 format, metric families, `build_comparison_table()`
3. `BD_Generation/scripts/compare.py` — CLI for regenerating comparison
4. `BD_Generation/scripts/evaluate.py` — full eval pipeline (multi-seed, denoising, structured save)
5. `BD_Generation/changes_for_sound_comparisons.md` — the original spec that drove both upgrades
6. `BD_Generation/eval_results/mdlm_baseline.json` — V2 result with per-seed data
7. `BD_Generation/eval_results/remdm_cap_eta0.1.json` — V2 result with per-seed data

## Context for the next session

- **No GPU available** — evaluations ran on CPU (~8 min per method with 5 seeds x 1000 samples)
- **Denoising metrics are nearly identical** between MDLM and ReMDM — confirms differences are sampler-only, not model quality
- **ReMDM's conditional edge JS/TV improved** (lower = better match to training dist.) while **type-conditioned degree JS/TV degraded** — remasking helps spatial relationships but hurts per-type connectivity patterns
- **Rooms W1 is high-variance** (std ~0.012-0.023) — ordinal metric on a 5-value support is inherently noisy with 1000 samples
- **`_format_mean_std` integer detection**: `mean == int(mean)` can be fragile for float imprecision — works now but watch for edge cases
- **`conditional_topN_pairs: 20`** in config means conditional metrics use top-20 most frequent room-type pairs (not all eligible pairs)
