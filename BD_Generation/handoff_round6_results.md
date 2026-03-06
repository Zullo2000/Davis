# Handoff: Round 6 — EMA Lock Results Analysis

> **Created:** 2026-03-06
> **Session purpose:** Implement adaptive EMA remasking lock, run Round 6 experiment, analyze results.

---

## What was accomplished
- **Adaptive EMA remasking lock implemented** in `bd_gen/guidance/guided_sampler.py`: per-sample lock that tracks EMA(reward) and permanently disables remasking when d(EMA) <= 0 for 3 consecutive steps or at hard deadline t=0.5. Beta=0.85.
- **4 CLI flags** added to `scripts/generate_guided.py`: `--ema-lock`, `--ema-beta`, `--ema-lock-consecutive`, `--ema-lock-deadline`
- **EMA + lock visualization** in `scripts/analyze_guidance_stats.py`: EMA(reward) overlaid as dashed line on reward subplot, t_lock vertical dashed line on all subplots
- **4 new tests** in `tests/test_guided_sampler.py` (smoke, deadline, diagnostics, noop) — all 611 pass (excluding 3 pre-existing `_aggregate_multi_seed` failures)
- **Round 6 experiment script** created: `scripts/run_g5_round6.sh`
- **Docs updated**: `docs/guidance.md` (EMA lock section + Round 6 setup), `implementation_state_T1_guidance.md`
- **Code committed and pushed** to `origin/main` (commit `8f17824`)
- **Round 6 launched on jabiru** via `tmux new -s round6` running `bash scripts/run_g5_round6.sh all`

## Key decisions made
| Decision | Choice | Rationale |
|---|---|---|
| EMA beta | 0.85 (fixed, no sweep) | User decision; middle value from prior analysis |
| Lock granularity | Per-sample | Each sample may stabilize at different steps; infrastructure already supports per-sample tracking |
| Lock mechanism | Save/restore (clone before remask, restore locked samples after) | Minimal code change; no need to modify `_single_step_remask` or remasking functions |
| Option A optimization | Skip fresh-logits model call when all samples locked | Avoids wasteful GPU computation in late steps |
| Round 6 grid | 2 configs: Option A + lock, Option B + lock | Directly comparable to Round 5 (same configs without lock) |

## Current state of the codebase
- **G1-G4 COMPLETE**, G5 IN PROGRESS (Rounds 1-6)
- **Round 6 running on jabiru** in tmux session `round6` — `bash scripts/run_g5_round6.sh all`
  - Calibrate (reuses existing) -> Generate (GPU, 2 configs) -> Evaluate -> Compare -> Analyze -> Organize
  - Results will be in `eval_results/loglinear_noise_sc/round6_guid/`
- All 611 tests pass (excluding 3 pre-existing `_aggregate_multi_seed` import failures)
- **No regressions** — EMA lock is fully opt-in with safe defaults

## What remains to be done
1. **Check Round 6 completion on jabiru**: `tmux attach -t round6`
2. **Pull results locally** (after organize step moves files to `round6_guid/`):
   ```bash
   scp -r amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/round6_guid/ BD_Generation/eval_results/loglinear_noise_sc/round6_guid/
   ```
3. **Analyze Round 6 results**: Read `round6_guid/comparison_guided_round6.md` and inspect trajectory PNGs (should show EMA overlay + lock step dashed lines)
4. **Compare to prior rounds**:
   - No-remask (R4): 69% — upper bound
   - Option A no lock (R5): 55%, Option B no lock (R5): 56%
   - Option A + lock (R6): ?, Option B + lock (R6): ?
5. **Write up results** in `docs/guidance.md` and `implementation_state_T1_guidance.md`
6. **Decide next direction**: If lock helps → tune deadline/consecutive. If not → close remasking investigation, proceed to v2 or constraint scaling.

## Files to reference in next session
- `BD_Generation/implementation_state_T1_guidance.md` — dynamic state (always read first per CLAUDE.md)
- `BD_Generation/planning_T1_guidance.md` — static spec
- `BD_Generation/docs/guidance.md` — full experiment history including EMA lock design
- `BD_Generation/eval_results/loglinear_noise_sc/round6_guid/` — Round 6 results (comparison.md + PNGs)
- `BD_Generation/bd_gen/guidance/guided_sampler.py` — EMA lock implementation (lines 447-568 for state init + lock logic)
- `BD_Generation/scripts/run_g5_round6.sh` — experiment script (for reference on configs/tags)

## Context for the next session
- **Round 6 model tags**: `r6lockA` (Option A + EMA lock), `r6lockB` (Option B + EMA lock)
- **Full model names**: `llada_topp0.9_remdm_confidence_tsw1.0_guided_r6lockA_K16_a0.01` and `...r6lockB...`
- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, workdir: `/Data/amine.chraibi/Davis/BD_Generation`
- **tmux session**: `round6` — check with `tmux attach -t round6`
- **The EMA lock tracks `reward_pre` (pre-remask reward of the selected winner)** — this is the right signal since it captures guidance quality before remasking corrupts it
- **The `_lock_step` in trajectory extraction** is stored as an int (not a list) — special key convention in `_extract_sample_trajectory`
- **Analysis PNGs will show**: EMA(reward) as dashed line on "Reward (selected)" subplot, vertical dashed line at t_lock on all subplots
- **If the script fails at the `compare` step**: Round 5 `_samples.pt` files were accidentally deleted on jabiru in a previous session. The compare step references R5 models — if their JSON metrics exist the comparison will work, but if not, you may need to remove R5 models from the comparison list in the script
