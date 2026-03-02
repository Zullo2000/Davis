# Handoff: G5 Round 2 — Fine α Sweep Running on Jabiru

> **Created:** 2026-03-02
> **Session purpose:** Launch Round 2 guidance experiments with revised constraints and finer α grid on jabiru.

---

## What was accomplished

- **Pilot script updated** for Round 2: `scripts/run_g5_pilot.sh` now runs α ∈ {0.01, 0.05, 0.15, 0.3} × K ∈ {16, 24} = 8 configs, with a new `analyze` step (step 5) for `--plot-analysis`.
- **Implementation state updated** (`implementation_state_T1_guidance.md`) with Round 2 setup, revised experiment plan, updated parallelization note.
- **Committed and pushed** two commits to main:
  - `9e17974` — Round 1 results + Round 2 prep (constraint revision, analysis pipeline, all Round 1 eval JSONs/PNGs/handoffs)
  - `6b4830a` — Reduced grid from 12 to 8 configs (dropped α=0.1 and α=0.2)
- **Launched Round 2 on jabiru** via `nohup bash scripts/run_g5_pilot.sh all > run_g5_round2.log 2>&1 &`
- **Calibration completed successfully** — new P90 values: `one_kitchen=1.0, kitchen_near_living=1.0, no_bath_kitchen=2.0, between_2_and_3_bathrooms=1.0`

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| α grid | {0.01, 0.05, 0.15, 0.3} | α=0.1 already in Round 1 data, α=0.2 redundant near 0.15/0.3 |
| K values | {16, 24} | K=16 was best in Round 1, K=24 tests if more candidates helps further |
| Drop K=4 | Yes | Round 1 showed K=4 clearly worse than K=16 |
| Output file | `comparison_guided_round2.md` | Separate from Round 1's `comparison_guided_pilot.md` |

## Current state of the codebase

- **All G1–G4 complete and tested** (114 guidance tests pass)
- **Round 2 generation is running on jabiru** but very slow (~306s/batch) due to GPU contention with another user's SFT training job (PID 1246636, `louis-a+`, using 16.9GB of 24.5GB A5000)
- **Calibration JSON is fresh** (`configs/guidance/calibration_v1_no_remask.json`) with new 4-constraint set
- **Round 1 results are stale** — computed with old constraint set (included trivially-satisfied `one_living`). Files preserved in `eval_results/loglinear_noise_sc/` for reference but not comparable to Round 2.
- **1 pre-existing test failure**: `test_metrics.py::TestAggregateMultiSeed::test_mean_std_basic` — unrelated `_aggregate_multi_seed` import rename

## What remains to be done

1. **Wait for GPU to be free** — check if Louis's SFT training (PID 1246636) has finished:
   ```bash
   nvidia-smi
   ```
   If still running, kill our job and restart when GPU is free:
   ```bash
   kill $(pgrep -f run_g5_pilot)
   # After GPU is free:
   nohup bash scripts/run_g5_pilot.sh generate > run_g5_round2.log 2>&1 &
   ```
   Calibration already done — can skip straight to `generate`.

2. **Check generation progress**:
   ```bash
   grep "^--- Run\|Seed\|Generation complete" run_g5_round2.log
   ```

3. **Run remaining steps** (if generation completes but evaluate/compare/analyze didn't run):
   ```bash
   bash scripts/run_g5_pilot.sh evaluate
   bash scripts/run_g5_pilot.sh compare
   bash scripts/run_g5_pilot.sh analyze
   ```

4. **Copy results back locally**:
   ```bash
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*round2* BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*guided_basic_K16_a0.01* BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*guided_basic_K16_a0.05* BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*guided_basic_K16_a0.15* BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*guided_basic_K16_a0.3* BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*guided_basic_K24* BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*trajectories*.png BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/configs/guidance/calibration_v1_no_remask.json BD_Generation/configs/guidance/
   ```

5. **Analyze Round 2 results** — document findings in `docs/guidance.md` under "### Round 2 — Fine α sweep with revised constraints"

6. **Next experiment decisions** based on Round 2:
   - If satisfaction plateaus at low α → optimal α found, proceed to other model variants
   - If K=24 >> K=16 → consider even larger K
   - Consider hard reward mode comparison at best α

## Files to reference in next session

1. **This handoff** — `BD_Generation/past_handoffs/handoff_G5_round2_running.md`
2. **Implementation state** — `BD_Generation/implementation_state_T1_guidance.md` (G5 section)
3. **Pilot script** — `BD_Generation/scripts/run_g5_pilot.sh` (Round 2 grid, 5 steps)
4. **Guidance docs** — `BD_Generation/docs/guidance.md` (Round 1 findings, design choices)
5. **Constraint config** — `BD_Generation/configs/guidance/example_basic.yaml`
6. **Round 1 comparison** — `BD_Generation/eval_results/loglinear_noise_sc/comparison_guided_pilot.md`

## Context for the next session

- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`, activate: `source .venv/bin/activate && cd BD_Generation`
- **Jabiru Python**: 3.9 — avoid backslashes in f-string expressions
- **GPU contention**: Another user (louis-a+) was running SFT training (PID 1246636) using 16.9GB on the A5000. Check `nvidia-smi` before starting GPU jobs. Our generation was ~60x slower than expected due to this.
- **nohup log**: `BD_Generation/run_g5_round2.log` on jabiru — check with `grep "^--- Run" run_g5_round2.log` for progress
- **Calibration is done** — `calibration_v1_no_remask.json` already regenerated with new constraints. No need to re-run calibrate step.
- **`--alpha 0` gotcha**: `0.0` is falsy in `generate_guided.py` line 219, falls through to YAML default. Not relevant for current α range.
- **`repeat_interleave` bug was fixed** (commit `693a07a`) — already in main.
- **Round 1 result files** in `eval_results/loglinear_noise_sc/` are from old constraint set (one_living). Keep for reference but not comparable to Round 2.
- **Expected Round 2 runtime**: ~40 min for 8 configs with full GPU, ~52h if sharing with another job.
