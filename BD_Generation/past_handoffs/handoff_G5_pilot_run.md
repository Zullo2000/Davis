# Handoff: G5 Pilot Run — Diagnostics Upgrade & Ready to Execute

> **Created:** 2026-02-27
> **Session purpose:** Address user feedback on fake examples doc, upgrade diagnostics with per-sample distribution plots and time-evolution trajectory plots, prepare for 6-config pilot run on jabiru.

---

## What was accomplished

- **Answered user questions about the fake examples doc:**
  - Constraints come from user-authored YAML (`configs/guidance/example_basic.yaml`) — the 4 in the example are a starting set, fully user-configurable
  - Baseline = same model (v1 + llada + top-p=0.9 + no remasking) WITHOUT guidance, re-evaluated with `--guidance-config` so constraint metrics appear
  - Satisfaction rate = (# samples where constraint satisfied) / (1000 per seed), mean ± std across 5 seeds

- **Fixed seeds** in fake examples: `(42, 43, 44, 45, 46)` → `(42, 123, 456, 789, 1337)` to match `configs/eval/default.yaml`

- **Replaced Distribution metrics with Priority Metrics** in comparison table:
  - Removed: Distribution (JS/TV/W1) section from guided comparisons
  - Added `_PRIORITY_METRICS` in `save_utils.py`: mode coverage (weighted), spatial transitivity, cond. edge TV (weighted), type-cond. degree TV (weighted), node TV
  - Added `GUIDED_METRIC_FAMILIES` constant and `guided=` parameter to `build_comparison_table()`
  - `compare_selected.py` now has `--guided` CLI flag
  - `run_g5_pilot.sh` compare step passes `--guided`

- **Added per-sample GuidanceStats saving** in `generate_guided.py`:
  - Now saves `batch_stats_per_sample` alongside batch-averaged stats
  - Per-sample tensors of shape `(B,)` for ESS, rewards, violations at every step
  - Enables distribution plots and per-sample trajectory analysis

- **Added plotting to `analyze_guidance_stats.py`:**
  - `--plot-distributions`: Histograms over all 5000 samples at final denoising step (ESS, reward_gap, remasking_delta, per-constraint violations)
  - `--plot-trajectories --traj-seed 42`: Time evolution for 2 individual samples from one seed, overlaid in same figure per metric
  - Both save PNG files to `eval_results/<schedule>/`

- **Rewrote `fake_examples_guidance_stats.md`** with all changes: corrected seeds, explained constraints/baseline, Priority Metrics table, distribution plot layout (Part B), trajectory plot layout (Part C) with detailed interpretation guide

- **Tests:** 721 passed, 3 pre-existing failures (unchanged `_aggregate_multi_seed` import), no regressions

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Replace Distribution section for guided | `GUIDED_METRIC_FAMILIES` with Priority Metrics | User wants focused metrics: mode coverage weighted, spatial transitivity, cond. edge TV, type-cond. degree TV, node TV |
| Per-sample stats storage | Save full `steps_per_sample` tensors | ~18 MB extra per run; needed for distribution plots and trajectory analysis |
| Trajectory sample selection | Sample 0 and sample `batch_size` (from different batches) | Gets diversity in sample characteristics; could also be 0 and 1 |
| Distribution plot: final step only | Histogram at last denoising step | Most informative for constraint satisfaction; per-step distributions would be too many plots |

## Current state of the codebase

### What exists and works
- Full guidance module (G1-G4): `bd_gen/guidance/` — constraints, soft violations, reward composer, guided sampler, calibration
- All scripts ready: `generate_guided.py` (now with per-sample stats), `calibrate_constraints.py`, `evaluate.py`, `analyze_guidance_stats.py` (now with plotting), `run_g5_pilot.sh` (now with `--guided`)
- Comparison tables: `--guided` flag switches to focused metric families
- 721 tests pass

### What is partially done
- G5 experiments: all prep complete, **not yet run on jabiru**
- No calibration JSONs yet (step 1 on jabiru)
- `fake_examples_guidance_stats.md` reviewed and updated — user approved the format

### Known issues
- 3 pre-existing test failures in `test_metrics.py` (`_aggregate_multi_seed` import renamed). Unrelated to guidance.
- `--alpha 0` gotcha in `generate_guided.py` line 219: `0.0` is falsy, falls through to YAML default. Not relevant for pilot (α=0.1 minimum).

## What remains to be done

1. **Sync code to jabiru** — `git push` + `ssh jabiru` + `git pull`
2. **Run pilot on jabiru:**
   ```bash
   cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation
   bash scripts/run_g5_pilot.sh all    # or step-by-step: calibrate → generate → evaluate → compare
   ```
3. **Run distribution + trajectory plots on jabiru:**
   ```bash
   for model in llada_topp0.9_no_remask_guided_basic_K{4,16}_a{0.1,1.0,5.0}; do
     python scripts/analyze_guidance_stats.py \
       --schedule loglinear_noise_sc --model "$model" \
       --plot-distributions --plot-trajectories --traj-seed 42
   done
   ```
4. **Copy results back locally:** scp the JSONs, comparison_guided_pilot.md, and PNG plots
5. **Review pilot results** — check satisfaction vs quality tradeoff using Priority Metrics, inspect distribution histograms and trajectory plots
6. **Decide next steps:** expand to full grid (60 configs) if pilot looks good, or adjust α/K range

## Files to reference in next session

1. `BD_Generation/scripts/run_g5_pilot.sh` — the 6-config pilot script (ready to run)
2. `BD_Generation/fake_examples_guidance_stats.md` — approved output format reference
3. `BD_Generation/scripts/analyze_guidance_stats.py` — new `--plot-distributions` and `--plot-trajectories` flags
4. `BD_Generation/scripts/generate_guided.py` — saves per-sample stats now
5. `BD_Generation/eval_results/save_utils.py` — `GUIDED_METRIC_FAMILIES`, `_PRIORITY_METRICS`, `build_comparison_table(guided=True)`
6. `BD_Generation/implementation_state_T1_guidance.md` — G5 status

## Context for the next session

- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`
- **v1 checkpoint**: `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt`
- **Seeds**: `[42, 123, 456, 789, 1337]` from `configs/eval/default.yaml`
- **Pilot grid**: α ∈ {0.1, 1.0, 5.0} × K ∈ {4, 16} = 6 configs, all v1 + llada + top-p=0.9 + no remasking
- **7 models in comparison**: 1 unguided baseline + 6 guided
- **Remasking delta will always be 0** for this pilot (no-remasking variant)
- **φ = linear** (default). If one constraint dominates energy in results, consider log1p later
- **matplotlib required** on jabiru for plotting — should already be installed in `.venv`
- **The user is most interested in the trajectory plots** (Part C) — they want to see if violations bounce back after being resolved, and whether guidance fights the sampling process
