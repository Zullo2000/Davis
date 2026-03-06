# Handoff: Vectorized K-Candidate Scoring + Round 7 Experiment

> **Created:** 2026-03-06
> **Session purpose:** Vectorize the SVDD scoring loop to enable K=50 experiments, then run Round 7 with the new parallelized method.

---

## What was accomplished

- **Vectorized soft scoring pipeline** — replaced the Python double loop `for k in K: for b in B:` with fully batched tensor operations. Scoring time is now nearly independent of K.
  - `bd_gen/guidance/soft_violations.py` — added `_compute_adj_terms_batch()` for `(KB, n_max, V)` → `(KB, n_edges)`
  - `bd_gen/guidance/constraints.py` — added `soft_violation_batch()` abstract method + implementations on all 4 constraint classes (ExactCount, CountRange, RequireAdj, ForbidAdj)
  - `bd_gen/guidance/reward.py` — added `compute_energy_soft_batch()` and `compute_reward_soft_batch()` to RewardComposer
  - `bd_gen/guidance/guided_sampler.py` — replaced `_score_candidates_soft` and `_score_single_soft` with batched versions using `build_effective_probs_batch` + `compute_reward_soft_batch`
- **8 new tests** in `tests/test_soft_violations.py` — all verify batched results match per-sample to `atol=1e-12`
- **Updated docs** — `docs/guidance.md` compute cost table, `implementation_state_T1_guidance.md` with vectorization section
- **Full test suite passes**: 747 passed, 3 pre-existing failures (unrelated `_aggregate_multi_seed` import), 2 skipped

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Batch method placement | New `soft_violation_batch()` alongside existing `soft_violation()` | Backward compat — single-sample methods preserved for tests and hard mode |
| Hard mode scoring | Left as Python loop (not vectorized) | Round 4 concluded hard mode is bad (~35% vs 69%). Only soft mode used. |
| Memory for K=50 | No concern | RPLAN K=50 B=64: ~12 MB float64 for probs — negligible vs GPU memory |

## Current state of the codebase

- **Working**: All guidance phases G1–G4 complete. G5 in progress (Rounds 1–6b done).
- **Ready to run**: Round 7 script `scripts/run_g5_round7.sh` exists but tests decay schedule at K=16 only.
- **Next experiment needed**: Round 7 should be updated/extended to also test K=50 with the vectorized scoring. The script currently has 2 configs (Option A + decay p=3, Option B + decay p=3) both at K=16.
- **No failing tests from our changes**.

## What remains to be done

1. **Update `scripts/run_g5_round7.sh`** (or create a new round script) to include K=50 configs:
   - Option B + decay p=3, K=50, α=0.01 (the main experiment)
   - Optionally also K=50 no-remasking as a comparison
2. **Sync code to jabiru**: `git push` or `scp` the vectorized scoring changes
3. **Run the experiment on jabiru** (GPU required):
   - Calibration already exists from previous rounds
   - Generate: `python scripts/generate_guided.py ... --K 50 ...`
   - Evaluate + compare + analyze
4. **Analyze results**: Does K=50 improve satisfaction beyond the K=16 plateau? Especially for `no_bath_kitchen` (the bottleneck constraint).
5. **Expected timing**: K=50 Option B + decay ~10-13 min for 600 samples (faster than old K=16 confidence at ~21 min, thanks to vectorized scoring).

## Files to reference in next session

1. `implementation_state_T1_guidance.md` — current state (read at session start per CLAUDE.md rules)
2. `planning_T1_guidance.md` — static spec
3. `bd_gen/guidance/guided_sampler.py` — the main guided sampling loop (vectorized scoring at lines 103-154)
4. `scripts/run_g5_round7.sh` — existing Round 7 script to extend with K=50
5. `scripts/generate_guided.py` — CLI for guided generation (--K flag)
6. `docs/guidance.md` — Round 2 K* sweep results (K=24 still climbing for confidence remasking)

## Context for the next session

- The vectorization changes are **on the local machine only** — need to be synced to jabiru before running experiments.
- Round 7 was already scripted for K=16 with decay p=3. The next session should either modify this script or create a new `run_g5_round8.sh` for the K=50 experiment.
- Key hypothesis being tested: K=16 may have too high variance for SVDD reweighting. K=50 gives more candidates per step, reducing variance. With vectorized scoring, the compute cost is nearly the same.
- The `no_bath_kitchen` (ForbidAdj) constraint is the bottleneck — it kept improving with K up to K=24 in Round 2. K=50 should test whether it continues improving.
- Jabiru SSH: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis/BD_Generation`
- Checkpoint: `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt` (v1 loglinear)
