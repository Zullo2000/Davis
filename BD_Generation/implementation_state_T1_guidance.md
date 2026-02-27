
# Implementation State — Guidance (SVDD)

> Updated after each phase. Coordinator reads this + the spec before starting work.
> Rule: keep each phase summary under 60 lines. Capture decisions and deviations, not raw logs.
> Spec: `planning_T1_guidance.md` v1.0 (2026-02-27)


## Overall Status
- Current phase: G5 IN PROGRESS (prep complete, ready for jabiru runs)
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
Status: IN PROGRESS (pilot run: 6 configs ready for jabiru)

### What was built (prep)
1. **CLI overrides** in `generate_guided.py`: Added `--reward-mode` (soft/hard) and `--calibration` (JSON path) flags. Consistent with existing `--alpha`/`--K` pattern. Enables soft vs hard comparison without separate YAML files.
2. **`run_g5_experiments.sh`**: Comprehensive experiment automation for jabiru (full 60-config grid). 5 steps dispatched via `bash run_g5_experiments.sh step1|step2|step3|step4|step5`.
3. **`run_g5_pilot.sh`**: Focused 6-config pilot experiment (v1 + llada + top-p=0.9 + no remasking, α∈{0.1,1.0,5.0} × K∈{4,16}). Validates metrics before committing to full grid.
4. **`analyze_guidance_stats.py`**: Reads `_samples.pt` guidance diagnostics. Single-model analysis (ESS, max_weight, reward_gap, remasking_delta, per-constraint trajectories). `--compare-modes` for soft vs hard decision table. `--export-tsv` for plotting.
5. **Constraint metrics in comparison tables**: Added `_build_constraint_table()` to `save_utils.py` — dynamically detects `constraint/*` keys and renders satisfaction rates, mean violations in comparison.md. Also added guidance config params (K, alpha, reward_mode) to config table.

### Experiment plan (revised)
1. **Pilot run** (6 configs): v1 + llada + top-p=0.9 + no remasking, α∈{0.1,1.0,5.0} × K∈{4,16}. Validate that constraint satisfaction metrics, comparison tables, and GuidanceStats diagnostics work correctly before scaling up.
2. **Full grid** (after pilot validation): Extend to 4 variants × 5 α × 3 K = 60 configs.
3. **Generate guided samples** across 4 model variants:
   - v1 + llada + top-p=0.9 + no remasking
   - v1 + llada + top-p=0.9 + confidence remasking tsw=1.0
   - v2 + llada + top-p=0.9 + no remasking
   - v2 + llada + top-p=0.9 + confidence remasking tsw=1.0
4. **α grid**: [0.1, 0.5, 1.0, 2.0, 5.0]
5. **K grid**: [4, 8, 16]
6. **Full evaluation** with `--guidance-config` → comparison tables.

### Files modified
- `scripts/generate_guided.py` (added `--reward-mode`, `--calibration` CLI overrides)
- `eval_results/save_utils.py` (added `_build_constraint_table()` for dynamic constraint metrics in comparison, added guidance config params to config table)

### Files created
- `scripts/run_g5_experiments.sh` (experiment automation, 5 steps — full grid)
- `scripts/run_g5_pilot.sh` (6-config pilot experiment)
- `scripts/analyze_guidance_stats.py` (trajectory diagnostics + soft/hard comparison)

### Test results
- 124 guidance tests: **all pass** (no regressions from CLI additions — generate_guided.py is a script, not unit-tested directly)

### Parallelization
4 variants × 5 α × 3 K = 60 configs, independent GPU runs. Soft-vs-hard comparison (step 2) completes first to select reward mode for remaining grid.

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
