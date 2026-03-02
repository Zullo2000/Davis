# Handoff: G5 Round 2 — Re-calibrate & Re-run with Revised Constraints + New Analysis Pipeline

> **Created:** 2026-03-02
> **Session purpose:** Prepare and run the next round of SVDD guidance experiments with revised constraint set and outlier-aware analysis pipeline.

---

## What was accomplished

- **Pilot v1 results analyzed** (Round 1, 6 configs): α=0.1 confirmed as only effective temperature, K=16 > K=4. Dynamic analysis (ESS, reward trajectory, reward gap, violation trajectories) fully written up.
- **Constraint set revised**: replaced trivially-satisfied `one_living` (ExactCount LivingRoom=1, always 100%) with `between_2_and_3_bathrooms` (CountRange Bathroom ∈ [2,3]) in `configs/guidance/example_basic.yaml`.
- **Analysis pipeline upgraded**: replaced uninformative distribution histograms with outlier-aware pipeline (`--plot-analysis`) in `scripts/analyze_guidance_stats.py`:
  - P1 outlier detection on final-step reward
  - Trimmed scalar means (excluding outliers)
  - Two trajectory plots per config: `*_trajectories_outliers.png` + `*_trajectories_clean.png`
- **Documentation restructured**: `docs/guidance.md` now has "Gradual Discoveries" section with Round 1 findings, design choices table, comparison metrics, and permanent dynamics explanations. Ready for Round 2 entries.
- **Implementation state updated**: `implementation_state_T1_guidance.md` reflects all changes.

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Outlier criterion | P1 of final-step reward (bottom 1%) | Asymmetric cut — only flags bad tail. ~50 outliers out of 5000, enough for trajectory sampling |
| Outlier metric | Final reward_selected | Directly tied to what guidance optimizes; monotone with total violation |
| Replace one_living | between_2_and_3_bathrooms (CountRange) | one_living was 100% in all configs including baseline — zero signal |
| Soft vs Hard reward mode | Soft for all pilot runs | Hard mode comparison deferred to Round 2 or 3 |
| Analysis output | Trajectory plots only (no histograms) | Histograms concentrated at discrete values, uninformative |

## Current state of the codebase

- **All G1–G4 complete and tested** (114 guidance tests pass)
- **G5 Round 1 results are stale** — old constraint set (one_living). Existing `_samples.pt`, calibration JSON, and eval result JSONs/PNGs in `eval_results/loglinear_noise_sc/` are from the old constraints.
- **`calibration_v1_no_remask.json`** uses old constraints — must be regenerated
- **Code is ready** for Round 2: new constraint YAML committed, analysis pipeline upgraded, generate/evaluate/calibrate scripts all work
- **1 pre-existing test failure**: `test_metrics.py::TestAggregateMultiSeed::test_mean_std_basic` — unrelated `_aggregate_multi_seed` import rename, not caused by guidance changes

## What remains to be done

1. **Re-calibrate** with new constraint set on jabiru:
   ```bash
   python scripts/calibrate_constraints.py \
       --schedule loglinear_noise_sc \
       --model llada_topp0.9_no_remask \
       --constraints configs/guidance/example_basic.yaml \
       --output configs/guidance/calibration_v1_no_remask.json
   ```

2. **Re-run pilot** (6 configs, same α×K grid as Round 1) with new constraints:
   ```bash
   # Use run_g5_pilot.sh or manual commands — same as Round 1 but with updated YAML
   # α ∈ {0.1, 1.0, 5.0} × K ∈ {4, 16}, v1 + llada + top-p=0.9 + no remasking, soft
   ```

3. **Evaluate** all 6 guided + 1 baseline:
   ```bash
   python scripts/evaluate.py --schedule loglinear_noise_sc \
       --guidance-config configs/guidance/example_basic.yaml \
       --update-comparison
   ```

4. **Run new analysis pipeline**:
   ```bash
   python scripts/analyze_guidance_stats.py \
       --schedule loglinear_noise_sc \
       --model llada_topp0.9_no_remask_guided_basic_K16_a0.1 \
       --plot-analysis
   ```

5. **Document Round 2 findings** in `docs/guidance.md` under "### Round 2 — Revised constraints"

6. **Finer α sweep** at K=16: α ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.5}

7. **Consider**: hard reward mode comparison at α=0.1, expansion to other model variants (v1+remasking, v2)

## Files to reference in next session

1. **This handoff** — `BD_Generation/past_handoffs/handoff_G5_round2_prep.md`
2. **Implementation state** — `BD_Generation/implementation_state_T1_guidance.md` (G5 section, full jabiru commands)
3. **Guidance docs** — `BD_Generation/docs/guidance.md` (Gradual Discoveries section for Round 2 entry)
4. **Constraint config** — `BD_Generation/configs/guidance/example_basic.yaml` (new 4-constraint set)
5. **Analysis script** — `BD_Generation/scripts/analyze_guidance_stats.py` (`plot_analysis`, `_classify_outliers`)
6. **Pilot shell script** — `BD_Generation/scripts/run_g5_pilot.sh` (may need updating for new constraints)
7. **Calibration script** — `BD_Generation/scripts/calibrate_constraints.py`

## Context for the next session

- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`, activate: `source .venv/bin/activate && cd BD_Generation`
- **Jabiru Python**: 3.9 — avoid backslashes in f-string expressions (commit `a8b59e9` fixed this)
- **v1 checkpoint**: `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt`
- **`--alpha 0` gotcha**: `0.0` is falsy in `generate_guided.py` line 219, falls through to YAML default. Not relevant for current α range but beware if testing α=0.
- **`repeat_interleave` bug was fixed** (commit `693a07a`) — K-candidate expansion was mixing batch elements. Already in main.
- **Old Round 1 result files** in `eval_results/loglinear_noise_sc/` are from the old constraint set (one_living). They can be kept for reference but are not comparable to Round 2 results. The guided `_samples.pt` files on jabiru need to be regenerated.
- **Calibration P90 values will change** because the new constraint `between_2_and_3_bathrooms` was not in the old set. Must re-calibrate before generating.
- **721 tests pass** (3 pre-existing failures unrelated to guidance, 1 from `_aggregate_multi_seed` rename)
