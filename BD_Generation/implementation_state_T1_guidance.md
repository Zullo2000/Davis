
# Implementation State — Guidance (SVDD)

> Updated after each phase. Coordinator reads this + the spec before starting work.
> Rule: keep each phase summary under 60 lines. Capture decisions and deviations, not raw logs.
> Spec: `planning_T1_guidance.md` v1.0 (2026-02-27)


## Overall Status
- Current phase: G5 IN PROGRESS (Rounds 1–6; adaptive EMA lock implemented; Round 6 awaiting GPU run)
- Dependencies: v1 pipeline complete, v2 (MELD) trained and evaluated

### Dependency Graph

```
G1 ──→ G2 ──→ G3 ──→ G5
 │                  ↗
 └──→ G4 ──────────┘
```

G4 (calibration) can run in parallel with G2/G3 — it only needs G1 (hard violations on decoded graphs).

---

## Phase G1 — Constraint Primitives + Hard Violations
Status: COMPLETE

### What was built
1. **Constraint ABC + ConstraintResult** (`constraints.py`): Base class with `hard_violation()` and `soft_violation()` stub (→ `NotImplementedError`). Fields: `name`, `weight`, `p90_normalizer`.
2. **4 constraint primitives** (`constraints.py`): `hard_violation()` implemented, `soft_violation()` stubs for G2.
   - ExactCount, CountRange, RequireAdj, ForbidAdj — all per spec Sections 5.2–5.5.
3. **RewardComposer** (`reward.py`): `compute_energy_hard()`, `compute_reward_hard()`, `load_calibration()`. Soft methods exist but delegate to G2 stubs. Three phi functions: linear, quadratic, log1p.
4. **Pydantic schema** (`constraint_schema.py`): `ExactCountSpec`, `CountRangeSpec`, `RequireAdjSpec`, `ForbidAdjSpec` with field validators. Discriminated union via `Field(discriminator="type")`. `GuidanceConfig`, `compile_constraints()`, `load_guidance_config()`.
5. **`__init__.py`**: Re-exports all public API.
6. **Config YAMLs**: `configs/guidance/default.yaml`, `configs/guidance/example_basic.yaml`.
7. **Docs**: `docs/guidance.md`.

### Files created
- `bd_gen/guidance/__init__.py`, `constraints.py`, `reward.py`, `constraint_schema.py`
- `tests/test_constraints.py` (32 tests), `tests/test_constraint_schema.py` (30 tests)
- `configs/guidance/default.yaml`, `configs/guidance/example_basic.yaml`
- `docs/guidance.md`

### Test results
- 62 new tests: **all pass**
- Full suite: 659 passed, 3 pre-existing failures (unrelated `_aggregate_multi_seed` import), 2 skipped
- **No regressions**

### Deviations from spec
None. Implementation follows spec exactly.

---

## Phase G2 — Soft Violations from Logits
Status: COMPLETE

### What was built
1. **`build_effective_probs()`** (`soft_violations.py`): Per-position probability distributions — PAD → zeros, committed → one-hot, MASK → softmax(logits). Fully vectorized via `torch.where`, no Python loops. All float64.
2. **`build_effective_probs_batch()`** (`soft_violations.py`): Batched version for (K*B, SEQ_LEN) tensors. Same logic with leading batch dimension.
3. **`hard_decode_x0()`** (`soft_violations.py`): Argmax-decode masked positions, keep committed. Supports single and batched inputs.
4. **`_compute_adj_terms()`** (`soft_violations.py`): Shared helper for RequireAdj/ForbidAdj — computes `p_ij = p_types_ij * P_adj_ij` for all edge positions. Uses `torch.triu_indices` for vectorized edge-pair indexing (verified to match VocabConfig ordering).
5. **Soft violation implementations** in `constraints.py` (replaced `NotImplementedError` stubs):
   - ExactCount: `v = |Σ q_i(type) - target|`
   - CountRange: `v = max(0, lo - n̂) + max(0, n̂ - hi)`
   - RequireAdj: `v = 1 - P(exists)` via `log1p(-p_ij)` accumulation, clamped for numerical stability
   - ForbidAdj: `v = Σ p_ij` (expected count of forbidden adjacencies)
6. **RewardComposer soft mode** enabled — `compute_energy_soft()` / `compute_reward_soft()` now work end-to-end. Fixed float64 for no-constraints edge case.

### Files created
- `bd_gen/guidance/soft_violations.py`
- `tests/test_soft_violations.py` (36 tests)

### Files modified
- `bd_gen/guidance/constraints.py` (soft_violation implementations + import `_compute_adj_terms`)
- `bd_gen/guidance/reward.py` (float64 fix, docstring cleanup)
- `bd_gen/guidance/__init__.py` (added exports: `build_effective_probs`, `build_effective_probs_batch`, `hard_decode_x0`)
- `tests/test_constraints.py` (replaced `test_soft_violation_raises_not_implemented` with `test_soft_violation_returns_tensor`)
- `docs/guidance.md` (added G2 soft violations section)

### Test results
- 36 new tests: **all pass**
- Full suite: 695 passed, 3 pre-existing failures (unrelated `_aggregate_multi_seed` import), 2 skipped
- **No regressions**

### Deviations from spec
None. Implementation follows spec exactly.

### Key design decisions
- `torch.triu_indices(n_max, n_max, offset=1)` for vectorized edge-pair indexing — avoids Python loop, matches VocabConfig's row-major upper-triangle ordering (verified by test)
- `_compute_adj_terms` placed in `soft_violations.py` (not `constraints.py`) to keep adjacency probability logic reusable and avoid coupling
- RequireAdj log-space accumulation uses `torch.log1p(-p_ij_clamped)` with `eps=1e-15` clamp for numerical stability
- PAD safety: `build_effective_probs` zeros PAD rows; soft violations additionally mask via `pad_mask[:n_max]` as safety net

---

## Phase G3 — Guided Sampler
Status: COMPLETE

### What was built
1. **Refactored `sampling.py`**: Extracted `_single_step_unmask()` (steps 4c–4h: token selection + unmasking + PAD clamp + inpainting) and `_single_step_remask()` (step 4i: remasking) and `_clamp_pad()` helper. `sample()` calls both sequentially — zero behavior change. All 38 existing sampling tests pass.
2. **`guided_sample()`** (`guided_sampler.py`): SVDD K-candidate reweighting loop — single shared model call → expand to K*B → `_single_step_unmask` → score via RewardComposer (soft or hard mode) → importance weights `softmax(r/α)` → multinomial resample → `_single_step_remask` on winner → per-step diagnostics. Returns `(final_tokens, GuidanceStats)`.
3. **`GuidanceStats` / `GuidanceStatsStep` / `GuidanceStatsStepPerSample`**: Diagnostics dataclasses with ESS, weight entropy, reward trajectories, remasking delta, per-constraint violations, final satisfaction rates, violation histograms.
4. **Scoring helpers**: `_score_candidates_soft`, `_score_candidates_hard`, `_score_single_soft`, `_score_single_hard` — Python loop over K×B (correct first, optimization deferred).
5. **`generate_guided.py`** script: argparse for guidance args (`--guidance-config`, `--alpha`, `--K`, `--guidance-tag`) + Hydra passthrough. Loads model + constraints + calibration → `guided_sample()` in batches → saves with `guidance_stats` key.

### Files modified
- `bd_gen/diffusion/sampling.py` (extracted `_single_step_unmask`, `_single_step_remask`, `_clamp_pad`)
- `bd_gen/guidance/__init__.py` (added `guided_sample`, `GuidanceStats` exports)

### Files created
- `bd_gen/guidance/guided_sampler.py` (575 lines)
- `scripts/generate_guided.py` (411 lines)
- `tests/test_guided_sampler.py` (16 tests)

### Test results
- 16 new tests: **all pass** (spec tests 40–53)
- Full suite: 711 passed, 3 pre-existing failures (`_aggregate_multi_seed` import — unrelated), 2 skipped
- **No regressions**

### Deviations from spec
- Scoring loop is Python loop over K×B (not vectorized) — spec acknowledges this is acceptable for K≤16, B≤64. Optimization deferred.
- v2 detection in `_single_step_unmask` uses `p_unmask.shape[-1] > 1` (shape-based) instead of `rate_network is not None` — helper doesn't need to know about rate networks.
- `final_mean_violation_when_failed` computed as `total_violation / num_failed` (avoids double-counting satisfied samples).

### Key design decisions
- Shared model call (1x cost) with K-fold expansion only for stochastic transition + scoring
- `_single_step_unmask`/`_single_step_remask` extracted as module-level functions (not methods) for reuse by both `sample()` and `guided_sample()`
- Config YAMLs (`default.yaml`, `example_basic.yaml`) already existed from G1 — no changes needed

**Depends on:** G1, G2

---

## Phase G4 — Calibration + Evaluation
Status: COMPLETE

### What was built
1. **`calibrate_from_samples()`** (`calibration.py`): Takes decoded graph_dicts + compiled constraints → computes `hard_violation()` on all samples → P90 = 90th percentile of non-zero violations (via `numpy.percentile`). If all violations 0, P90 = 1.0. Returns `{name: p90}` dict.
2. **`save_calibration()` / `load_calibration()`** (`calibration.py`): JSON save/load with `mkdir(parents=True)` for nested output paths.
3. **`calibrate_constraints.py`** script: CLI loads `{model}_samples.pt` from `eval_results/{schedule}/`, detokenizes all seeds, compiles constraints from YAML, runs calibration, saves JSON.
4. **`compute_constraint_metrics()`** (`evaluate.py`): New function computing per-constraint satisfaction metrics on decoded graphs. Called from `compute_all_metrics()` when `constraints` parameter is provided.
5. **`--guidance-config` CLI flag** in `evaluate.py`: Loads + compiles constraints, threads through `evaluate_method()` → `compute_all_metrics()`.

### Metrics added to evaluate.py
- `constraint/satisfaction_{name}`: fraction with `hard_violation == 0`
- `constraint/satisfaction_overall`: fraction where ALL constraints simultaneously satisfied
- `constraint/mean_violation_{name}`: mean hard violation (all samples)
- `constraint/mean_violation_when_failed_{name}`: mean violation conditioned on failure
- `constraint/violation_histogram_{name}`: distribution as `{"0": count, "1": count, "2": count, "3+": count}`

### Files created
- `bd_gen/guidance/calibration.py`
- `scripts/calibrate_constraints.py`
- `tests/test_calibration.py` (10 tests)

### Files modified
- `bd_gen/guidance/__init__.py` (added `calibrate_from_samples`, `save_calibration`, `load_calibration` exports)
- `scripts/evaluate.py` (added `compute_constraint_metrics()`, `--guidance-config` flag, threaded `constraints` through `compute_all_metrics` and `evaluate_method`)

### Test results
- 10 new tests: **all pass** (spec tests 54–56 + edge cases)
- Full suite: 721 passed, 3 pre-existing failures (`_aggregate_multi_seed` import — unrelated), 2 skipped
- **No regressions**

### Deviations from spec
None. Implementation follows spec exactly.

**Depends on:** G1 (can run in parallel with G2/G3)

---

## Phase G5 — End-to-End Integration + Tuning
Status: IN PROGRESS (K* sweep complete, next: α fine-tuning at chosen K*)

### What was built (prep)
1. **CLI overrides** in `generate_guided.py`: Added `--reward-mode` (soft/hard) and `--calibration` (JSON path) flags. Consistent with existing `--alpha`/`--K` pattern. Enables soft vs hard comparison without separate YAML files.
2. **`run_g5_experiments.sh`**: Comprehensive experiment automation for jabiru (full 60-config grid). 5 steps dispatched via `bash run_g5_experiments.sh step1|step2|step3|step4|step5`.
3. **`run_g5_pilot.sh`**: Pilot experiment script. Updated for Round 2: α∈{0.01,0.05,0.15,0.3} × K∈{16,24} = 8 configs. Includes 5 steps: calibrate, generate, evaluate, compare, analyze.
4. **`analyze_guidance_stats.py`**: Reads `_samples.pt` guidance diagnostics. Single-model analysis (ESS, max_weight, reward_gap, remasking_delta, per-constraint trajectories). `--compare-modes` for soft vs hard decision table. `--export-tsv` for plotting. **New: `--plot-analysis` outlier-aware pipeline** (see below).
5. **Constraint metrics in comparison tables**: Added `_build_constraint_table()` to `save_utils.py` — dynamically detects `constraint/*` keys and renders satisfaction rates, mean violations in comparison.md. Also added guidance config params (K, alpha, reward_mode) to config table.

### Pilot v1 results (6 configs, completed 2026-02-28)
Ran α∈{0.1,1.0,5.0} × K∈{4,16}, v1 + llada + top-p=0.9 + no remasking, soft reward. Old constraint set (one_kitchen, one_living, kitchen_near_living, no_bath_kitchen).
- **α=0.1 is the sweet spot**: K=16 α=0.1 → 77% satisfaction (2× baseline 43.3%). α≥1.0 barely moves (~48%).
- **Quality tradeoff mild**: diversity -4%, cond. edge TV +0.045 at α=0.1. 100% validity everywhere.
- **one_living always 100%** → uninformative constraint, replaced (see below).
- Dynamic analysis written up in `docs/guidance.md` §Pilot Results (ESS, reward trajectory, reward gap, violations, α→0 discussion).

### Constraint set revision (2026-03-02)
Replaced `one_living` (trivially satisfied) with `between_2_and_3_bathrooms` (CountRange, Bathroom ∈ [2,3]) in `configs/guidance/example_basic.yaml`. New set:
1. `one_kitchen` — ExactCount(Kitchen, 1)
2. `kitchen_near_living` — RequireAdj(Kitchen, LivingRoom)
3. `no_bath_kitchen` — ForbidAdj(Bathroom, Kitchen)
4. `between_2_and_3_bathrooms` — CountRange(Bathroom, 2–3)

**All existing pilot results and calibration are stale** — must re-calibrate and re-run.

### Analysis pipeline upgrade (2026-03-02)
Distribution plots (histograms at final step) were uninformative — values concentrated around discrete points. Replaced with outlier-aware pipeline (`--plot-analysis`):
1. **Outlier detection**: P1 of final-step reward (bottom 1% = outliers). Configurable via `--outlier-percentile`.
2. **Trimmed scalar means**: final-step ESS, reward, reward gap, per-constraint violations — computed on clean samples (P1+ only).
3. **Two trajectory plots per config**: `*_trajectories_outliers.png` (2 random P1 samples) and `*_trajectories_clean.png` (2 random non-outlier samples). Reproducible via `--analysis-seed`.
4. Legacy `--plot-distributions` and `--plot-trajectories` preserved for backward compat.

### Round 2 setup (2026-03-02)
Fine α sweep with revised constraints. Combines constraint revision + finer grid into a single round.
- **Grid**: α ∈ {0.01, 0.05, 0.15, 0.3} × K ∈ {16, 24} = 8 configs
- **Variant**: v1 + llada + top-p=0.9 + no remasking, soft reward
- **Constraints**: one_kitchen, kitchen_near_living, no_bath_kitchen, between_2_and_3_bathrooms
- **Script**: `run_g5_pilot.sh` updated with 5 steps: calibrate → generate → evaluate → compare → analyze
- **Output**: `comparison_guided_round2.md` + per-config `*_trajectories_outliers.png` / `*_trajectories_clean.png`
- **Rationale**: Round 1 showed α=0.1 is sweet spot but only tested {0.1, 1.0, 5.0}. Need finer resolution around 0.1 to find optimal. K=24 tests whether more candidates further improves satisfaction beyond K=16.

### Experiment plan (revised 2026-03-02)
1. ~~**Round 1 — Coarse sweep** (6 configs)~~: DONE. α=0.1 sweet spot, K=16 > K=4. Old constraint set (included trivially-satisfied `one_living`).
2. ~~**Constraint set revision**~~: DONE. Replaced `one_living` → `between_2_and_3_bathrooms`.
3. ~~**Analysis pipeline upgrade**~~: DONE. Outlier-aware `--plot-analysis`.
4. ~~**Round 2 — Fine α sweep**~~: SUPERSEDED by K* sweep (Round 2 was too slow at 5000 samples).
5. ~~**K* sweep — find minimal K for good constraint satisfaction**~~ (16 configs): DONE (2026-03-02).
   - **Result (no-remasking)**: K*≈12. Plateau at ~56% satisfaction for K=12–24.
   - **Result (confidence)**: No plateau — curve still climbing at K=24 (61%). K*>24.
   - **Key finding**: remasking shifts K* UP (fights guidance, needs more candidates).
   - **Runtime**: 53 min (no-remask) + 48 min (confidence) generation under GPU contention.
   - **Details**: see `docs/guidance.md` §Round 2 — K* sweep.
6. ~~**Round 3 — α fine-tuning at K=16**~~: DONE (script: `run_g5_alpha.sh`).
   - Swept α ∈ {0.01, 0.05, 0.1, 0.15, 0.3, 0.5} at K=16 for both no-remask and confidence variants.
   - 3 seeds × 200 samples = 600 per config.
7. ~~**Round 4 — Remasking × Reward-mode at K=16, α=0.01**~~: DONE (2026-03-05).
   - **Result**: Option C (RACB) failed to rescue confidence remasking (56% vs 69% no-remask).
   - Hard reward mode conclusively bad (~35%). Soft reward only going forward.
   - Script: `run_g5_round4.sh`. Output: `comparison_guided_round4.md`.
8. ~~**Round 5 — Option A vs Option B at K=16, α=0.01**~~: DONE (2026-03-06).
   - **Result**: Both A and B match Option C at ~55–56% — none closes the gap to no-remask (69%).
   - Option A (fresh logits, 2× cost): 55.0%. Fresh model call produces same confidence rankings.
   - Option B (protect just-unmasked, 0 cost): 56.0%. Protection too narrow (1 step).
   - Trajectories confirm persistent oscillation in all remasking variants.
   - **Conclusion**: Structural conflict — no single-step mitigation works. No-remask is best for SVDD.
   - Script: `run_g5_round5.sh`. Output: `comparison_guided_round5.md`.
9. **Round 6 — Option A + EMA Lock vs Option B + EMA Lock** (2026-03-06): SCRIPT READY, awaiting GPU.
   - Adaptive EMA lock: remasking ON early → lock when reward plateaus → no-remask late.
   - Grid: 2 configs (Option A + lock, Option B + lock) at K=16, α=0.01.
   - Results dir: `eval_results/loglinear_noise_sc/round6_guid/`
10. Expand to v2 variants if warranted.

### Option C — Reward-Attributed Confidence Boosting (2026-03-05)
Implemented the mitigation for the guidance-remasking conflict: confidence remasking destroys guided positions (low model confidence = high attribution). Option C boosts confidence of reward-aligned just-unmasked positions before remasking.

**Mechanism:**
- Per-position reward attribution: for each just-unmasked position, compare mean reward of candidates matching vs not matching the winner's token
- Self-calibrating β = K/(K+K₀) / (σ_r + ε) with K₀=4 — no hyperparameter sweep needed
- Additive boost to model confidence before `softmax(-confidence)` in remasking
- Only positions in U_t (just-unmasked) get non-zero boosts; positive attribution only (clamp min=0)
- Guards: K<2 → skip; σ_r<ε → skip (no reward discrimination)

**Files modified:**
- `bd_gen/diffusion/remasking.py` — `confidence_boost: Tensor | None = None` param on `__call__()` and `_confidence_remasking()`
- `bd_gen/diffusion/sampling.py` — `confidence_boost` passthrough in `_single_step_remask()`
- `bd_gen/guidance/guided_sampler.py` — `_compute_attribution_boost()` helper (vectorized, float64), `attribution_boost: bool = False` param on `guided_sample()`, SVDD loop integration, new diagnostics (`mean_attribution_boost`, `positions_boosted`)
- `scripts/generate_guided.py` — `--attribution-boost` CLI flag, stored in payload config
- `docs/guidance.md` — status updated from "not yet implemented" to "implemented"

**Files modified (tests):**
- `tests/test_remasking.py` — 3 new tests: `TestConfidenceBoost` (None noop, boost protects low-conf, zero noop)
- `tests/test_guided_sampler.py` — 3 new tests: `TestAttributionBoost` (smoke, reduces remasking delta, noop without remasking)

**Test results:**
- 6 new tests: **all pass**
- Full suite: 727 passed, 3 pre-existing failures (`_aggregate_multi_seed` import), 2 skipped
- **No regressions**

**Backward compatibility:** All changes additive with safe defaults. Feature is opt-in via `attribution_boost=True` / `--attribution-boost`.

### Round 5 — Option A vs Option B (2026-03-06)

**Result**: Both Option A (55.0%) and Option B (56.0%) match Option C (56.0%) — no mitigation closes the gap to no-remask (69%).

**Root cause confirmed**: Confidence remasking is structurally anti-correlated with guidance. The model assigns low confidence to reward-steered tokens (they diverge from model likelihood), so remasking preferentially destroys guided positions. This is not a same-step feedback issue (Option B fails) nor a stale-logits issue (Option A fails) — it is intrinsic to the confidence criterion.

**Trajectories**: ESS erratic (2–16), remasking delta oscillates (-2 to +1), per-constraint violations show saw-tooth pattern, reward non-monotonic. Identical qualitative behavior across all three mitigations.

**Quality**: All remasking variants comparable. Option B best inside validity (97.8% vs 95.2–95.3%). No-remask best constraint satisfaction but lowest diversity (0.905 vs 0.990–0.993).

**Files created:**
- `scripts/run_g5_round5.sh` — 2-config experiment script

**Files modified:**
- `docs/guidance.md` — Round 5 results, verdict, open question on selective early-stage remasking
- `implementation_state_T1_guidance.md` — Round 5 marked DONE, open question added

**Conclusion**: No-remask + soft reward at K=16, α=0.01 is the current best (69%, 5.2× baseline). Remasking mitigation investigation concluded. Next direction: adaptive EMA lock for early-stage remasking.

### Adaptive EMA Remasking Lock (2026-03-06)

Per-sample adaptive lock: start with remasking ON, track EMA(reward), lock (permanently disable remasking) when d(EMA) ≤ 0 for 3 consecutive steps or at hard deadline t=0.5. β=0.85, per-sample granularity.

**Mechanism:** Save winner tokens before remasking; restore locked samples after. Skip Option A's model call when all locked.

**Files modified:**
- `bd_gen/guidance/guided_sampler.py` — 4 new params (`ema_lock`, `ema_beta`, `ema_lock_consecutive`, `ema_lock_deadline`), EMA state init, per-step lock logic, save/restore for locked samples, diagnostics (`samples_locked`, `ema_reward`, `locked`, `lock_steps`)
- `scripts/generate_guided.py` — 4 new CLI flags (`--ema-lock`, `--ema-beta`, `--ema-lock-consecutive`, `--ema-lock-deadline`), stored in payload config
- `scripts/analyze_guidance_stats.py` — `_extract_sample_trajectory` extracts `ema_reward` and `_lock_step`; `_plot_trajectory_figure` overlays EMA on reward subplot, draws t_lock dashed vertical line on all subplots
- `docs/guidance.md` — Adaptive EMA Lock section + Round 6 setup

**Files created:**
- `scripts/run_g5_round6.sh` — Round 6 experiment (Option A + EMA lock, Option B + EMA lock)

**Tests:** 4 new tests in `tests/test_guided_sampler.py` (smoke, deadline, diagnostics, noop). Full suite: 731 passed.

**Backward compatibility:** All changes additive with safe defaults. Feature is opt-in via `ema_lock=True` / `--ema-lock`.

### Round 6 — Option A + EMA Lock vs Option B + EMA Lock (2026-03-06)

**Status:** Script ready, awaiting GPU run on jabiru.
**Grid:** K=16, α=0.01, soft reward, confidence remasking + EMA lock. 2 configs: Option A + lock, Option B + lock.
**Results dir:** `eval_results/loglinear_noise_sc/round6_guid/`

### Files modified
- `scripts/generate_guided.py` (added `--reward-mode`, `--calibration` CLI overrides)
- `eval_results/save_utils.py` (added `_build_constraint_table()` for dynamic constraint metrics in comparison, added guidance config params to config table)
- `scripts/analyze_guidance_stats.py` (added `--plot-analysis` outlier-aware pipeline: `_classify_outliers`, `_collect_trimmed_final_means`, `plot_analysis`, `_plot_trajectory_figure` helper; refactored `plot_trajectories` to use shared helper)
- `configs/guidance/example_basic.yaml` (replaced `one_living` with `between_2_and_3_bathrooms`)
- `docs/guidance.md` (added: Soft vs Hard reward mode section, Pilot Results dynamic analysis §1–5, aggregate results table with reward mode column)
- `scripts/run_g5_pilot.sh` (Round 2: updated grid to α∈{0.01,0.05,0.15,0.3}×K∈{16,24}, added `analyze` step, output → `comparison_guided_round2.md`)

### Files created
- `scripts/run_g5_experiments.sh` (experiment automation, 5 steps — full grid)
- `scripts/run_g5_pilot.sh` (6-config pilot experiment)
- `scripts/run_g5_kstar.sh` (K* sweep: K ∈ {4,8,10,12,14,16,20,24} × 2 variants, 200 samples/config)
- `scripts/analyze_guidance_stats.py` (trajectory diagnostics + soft/hard comparison)

### Test results
- 114 guidance tests: **all pass** (no regressions from analysis pipeline or constraint config changes)

### Parallelization
Round 2: 4 α × 2 K = 8 configs (single variant, soft reward). Future full grid: 4 variants × N α × M K, independent GPU runs.

### What to monitor
- ESS(t) curves across α (active but not degenerate?)
- Per-constraint violation(t) (when does each resolve?)
- reward_gap(t) (is guidance steering?)
- reward_remasking_delta (remasking cooperating or fighting?)
- Inside validity, edge TV, diversity vs unguided baseline

### CPU vs GPU split

| Step | Script | Compute | Why |
|------|--------|---------|-----|
| 1. Calibrate | `calibrate_constraints.py` | **CPU** | Python loop over graph dicts, integer counting / set lookups. No tensors. |
| 2. Soft vs Hard | `generate_guided.py` | **GPU** | Denoiser forward pass at each of 100 steps. |
| 3. Full grid | `generate_guided.py` | **GPU** | Same — only GPU-bound operation in G5. |
| 4. Evaluate | `evaluate.py --guidance-config` | **CPU** | Detokenization + metrics are dict/list ops, small graph kernels (n≤8). |
| 5. Comparison | `compare_selected.py` | **CPU** | String formatting / I/O. |

All steps run on jabiru because `_samples.pt` files and checkpoints live there. Only `generate_guided.py` actually uses the GPU.

### Reusing existing unguided samples

35 unguided models already have `_samples.pt` on jabiru (1000 samples × 5 seeds each):
```
eval_results/loglinear_noise_sc/*_samples.pt  (21 files)
eval_results/learned_noise_sc/*_samples.pt    (2 files)
eval_results/linear_noise_sc/*_samples.pt     (12 files)
```

These are used for:
1. **Calibration input** — `calibrate_constraints.py` reads existing `_samples.pt` to compute P90 normalizers. No new generation needed.
2. **Evaluation baselines** — existing JSON results in `eval_results/` are the comparison targets for guided models (quality/diversity degradation check).

Existing samples **cannot** substitute for guided generation — SVDD reweighting steers at every denoising step, producing fundamentally different outputs.

### Jabiru directory migration (2026-02-27)

Eval result directories were renamed locally (`loglinear/` → `loglinear_noise_sc/`, `linear/` → `linear_noise_sc/`). On jabiru, `_samples.pt` files are gitignored and were not tracked by git, so they stayed in the old `loglinear/` directory. During sync we ran:
```bash
mv eval_results/loglinear/*_samples.pt eval_results/loglinear_noise_sc/
```
The `linear/` samples need the same treatment if not already moved:
```bash
mv eval_results/linear/*_samples.pt eval_results/linear_noise_sc/
```
After confirming all `_samples.pt` files are in the `*_noise_sc/` directories, the old empty directories can be deleted.

### Jabiru reference

SSH: `ssh amine.chraibi@jabiru.polytechnique.fr`
Working dir: `cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation`

Checkpoints:
```
loglinear (v1): outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt
v2 (MELD):      outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt
```

### 4 model variants — Hydra overrides

| Variant | Schedule | Checkpoint | Key overrides |
|---------|----------|------------|---------------|
| v1 no-remask | loglinear | `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt` | `noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=false` |
| v1 confidence | loglinear | same | `noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.t_switch=1.0` |
| v2 no-remask | learned | `outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt` | `noise=learned eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=false` |
| v2 confidence | learned | same | `noise=learned eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.t_switch=1.0` |

### Step-by-step commands (on jabiru)

**Step 1 — Calibrate (CPU, run once)**
```bash
# v1 baselines (from existing unguided loglinear samples)
python scripts/calibrate_constraints.py \
    --schedule loglinear_noise_sc \
    --model llada_topp0.9_no_remask \
    --constraints configs/guidance/example_basic.yaml \
    --output configs/guidance/calibration_v1_no_remask.json

python scripts/calibrate_constraints.py \
    --schedule loglinear_noise_sc \
    --model llada_topp0.9_remdm_confidence_tsw1.0 \
    --constraints configs/guidance/example_basic.yaml \
    --output configs/guidance/calibration_v1_confidence.json

# v2 baselines (from existing unguided learned samples)
python scripts/calibrate_constraints.py \
    --schedule learned_noise_sc \
    --model v2_llada_topp0.9_no_remask \
    --constraints configs/guidance/example_basic.yaml \
    --output configs/guidance/calibration_v2_no_remask.json

python scripts/calibrate_constraints.py \
    --schedule learned_noise_sc \
    --model v2_llada_topp0.9_remdm_confidence_tsw1.0 \
    --constraints configs/guidance/example_basic.yaml \
    --output configs/guidance/calibration_v2_confidence.json
```

**Step 2 — Soft vs Hard comparison (GPU, 8 runs at α=1.0, K=8)**
```bash
# Example: v1 no-remask, soft mode
python scripts/generate_guided.py \
    eval.checkpoint_path=outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt \
    noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 \
    eval.remasking.enabled=false \
    --guidance-config configs/guidance/example_basic.yaml \
    --alpha 1.0 --K 8 --guidance-tag soft_test

# Repeat for all 4 variants × 2 modes (soft/hard via reward_mode in YAML).
# Analyze GuidanceStats from all 8 runs → pick reward mode → proceed.
```

**Step 3 — Full grid (GPU, 60 runs)**
```bash
# Template (substitute variant overrides + alpha + K):
python scripts/generate_guided.py \
    eval.checkpoint_path=<ckpt> <variant_overrides> \
    --guidance-config configs/guidance/example_basic.yaml \
    --alpha <ALPHA> --K <K> --guidance-tag basic
```
4 variants × 5 α × 3 K = 60 independent runs. All parallelizable across GPU slots.

**Step 4 — Evaluate all guided models (CPU)**
```bash
python scripts/evaluate.py --schedule loglinear_noise_sc \
    --guidance-config configs/guidance/example_basic.yaml \
    --update-comparison

python scripts/evaluate.py --schedule learned_noise_sc \
    --guidance-config configs/guidance/example_basic.yaml \
    --update-comparison
```

**Step 5 — Copy results back locally**
```bash
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*.json BD_Generation/eval_results/loglinear_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*.md BD_Generation/eval_results/loglinear_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/learned_noise_sc/*.json BD_Generation/eval_results/learned_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/learned_noise_sc/*.md BD_Generation/eval_results/learned_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/configs/guidance/calibration_*.json BD_Generation/configs/guidance/
```

**Depends on:** G1, G2, G3, G4
