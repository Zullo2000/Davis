# Handoff: G5 Guided Generation Experiments

> **Created:** 2026-02-27
> **Session purpose:** Prepare G5 experiment infrastructure and run SVDD guided generation experiments on jabiru

---

## What was accomplished

- Added `--reward-mode` (soft/hard) and `--calibration` (JSON path) CLI overrides to `scripts/generate_guided.py`, consistent with existing `--alpha`/`--K` pattern
- Created `scripts/run_g5_experiments.sh` — full experiment automation for jabiru with 5 independent steps (calibrate, soft-vs-hard, full grid, evaluate, compare)
- Created `scripts/analyze_guidance_stats.py` — reads `_samples.pt` diagnostics, produces ESS/reward/violation trajectory analysis and soft-vs-hard comparison tables
- Verified all 124 guidance tests pass, full suite 721 passed (3 pre-existing failures unrelated to guidance, 2 skipped)
- Updated `implementation_state_T1_guidance.md` to reflect G5 IN PROGRESS status

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Reward mode toggle | CLI `--reward-mode` flag (not separate YAMLs) | Avoids config duplication, matches `--alpha`/`--K` pattern |
| Calibration override | CLI `--calibration` flag | Each of 4 variants needs its own P90 normalizers |
| Experiment structure | 5 sequential steps with explicit pause after step 2 | Soft vs hard decision must be made before committing to 60-run grid |
| No code changes to guidance core | Only CLI and scripts | G5 is experiments-only per spec |

## Current state of the codebase

### What exists and works
- **Full guidance module** (G1-G4 complete): `bd_gen/guidance/` — constraints, soft violations, reward composer, guided sampler, calibration
- **Scripts ready**: `generate_guided.py`, `calibrate_constraints.py`, `evaluate.py --guidance-config`, `analyze_guidance_stats.py`, `run_g5_experiments.sh`
- **Configs**: `configs/guidance/example_basic.yaml` (4 constraints: ExactCount Kitchen=1, ExactCount LivingRoom=1, RequireAdj Kitchen-LivingRoom, ForbidAdj Bathroom-Kitchen)
- **No calibration JSONs yet** — these are produced by step 1 on jabiru

### Known issues
- 3 pre-existing test failures in `test_metrics.py` (`_aggregate_multi_seed` import — renamed to `aggregate_multi_seed` but tests not updated). Unrelated to guidance.

## What remains to be done

### On jabiru (SSH, GPU server)

1. **Sync code** from local to jabiru (git push + pull, or rsync)
2. **Step 1 — Calibrate** (CPU, ~2 min):
   ```bash
   cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation
   bash scripts/run_g5_experiments.sh step1
   ```
3. **Step 2 — Soft vs Hard comparison** (GPU, ~40 min, 8 runs):
   ```bash
   bash scripts/run_g5_experiments.sh step2
   ```
4. **Analyze soft vs hard** (CPU):
   ```bash
   python scripts/analyze_guidance_stats.py --compare-modes
   ```
   Decision: pick `soft` or `hard` based on ESS stability, reward gap, satisfaction rates.

5. **Step 3 — Full grid** (GPU, ~5 hours, 60 runs):
   ```bash
   bash scripts/run_g5_experiments.sh step3 <soft|hard>
   ```
6. **Step 4 — Evaluate** (CPU, ~10 min):
   ```bash
   bash scripts/run_g5_experiments.sh step4
   ```
7. **Step 5 — Comparison tables** (CPU):
   ```bash
   bash scripts/run_g5_experiments.sh step5
   ```

### Back on local machine

8. **Copy results back**:
   ```bash
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*.json BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/learned_noise_sc/*.json BD_Generation/eval_results/learned_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/configs/guidance/calibration_*.json BD_Generation/configs/guidance/
   ```
9. **Analyze and document findings** — update `implementation_state_T1_guidance.md` with experiment results, best hyperparameters, and conclusions

## Files to reference in next session

1. `BD_Generation/implementation_state_T1_guidance.md` — current state (read first)
2. `BD_Generation/planning_T1_guidance.md` — spec (Section 13 = G5 details, Section 11 = diagnostics)
3. `BD_Generation/scripts/run_g5_experiments.sh` — experiment automation
4. `BD_Generation/scripts/analyze_guidance_stats.py` — results analysis
5. `BD_Generation/scripts/generate_guided.py` — guided generation script (CLI flags)
6. `BD_Generation/configs/guidance/example_basic.yaml` — constraint config

## Context for the next session

- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`
- **Checkpoints**: v1 at `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt`, v2 at `outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt`
- **4 variants**: v1 no-remask, v1 confidence (tsw=1.0), v2 no-remask, v2 confidence (tsw=1.0). All use llada + top-p=0.9.
- **Grid**: 5 alphas [0.1, 0.5, 1.0, 2.0, 5.0] × 3 Ks [4, 8, 16] = 15 configs per variant, 60 total
- **Pathology detection**: ESS < 2 = α too aggressive; ESS = K = guidance ineffective; max_weight ~ 1 = diversity destroyed; negative remasking_delta = remasking fights guidance
- **The `--reward-mode` flag** overrides the YAML setting, so soft vs hard can be toggled without config duplication
- **Existing unguided samples** (35 models) are used for calibration input and evaluation baselines only — they cannot substitute for guided generation
