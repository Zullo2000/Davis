# Handoff: Round 6 — Adaptive Remasking Lock Implementation

> **Created:** 2026-03-06
> **Session purpose:** Design adaptive "remask-then-lock" strategy, organize eval results, prepare for Round 6 experiment.

---

## What was accomplished
- **Round 5 results committed**: Options A/B/C all fail equally (~55-56%) vs no-remask (69%). Remasking mitigation investigation concluded.
- **Eval results reorganized**: Guided experiment files moved into `eval_results/loglinear_noise_sc/round{1..5}_guid/` subfolders for easier navigation. Non-guided baselines remain in the parent directory.
- **EMA trajectory analysis script** (`scripts/plot_ema_trajectories.py`): 3 plots — EMA comparison, ESS/K thresholds, reward-derivative lock criterion.
- **Adaptive lock criterion designed** through discussion: `t_lock = min(t=0.5, first step where d(EMA_reward) ≤ 0 for 3 consecutive steps)`
- **Round 5 `_samples.pt` files were accidentally deleted on jabiru** (by an overly broad `rm *r5opt*` glob during git pull conflict resolution). Round 4 `_samples.pt` files are intact and were used for the EMA analysis instead.

## Key decisions made
| Decision | Choice | Rationale |
|---|---|---|
| Remasking strategy for Round 6 | Option B (protect just-unmasked) | Free (0 cost), provides buffer during early remasking phase |
| Lock criterion | d(EMA_reward) ≤ 0 for 3 consecutive steps | ESS-based thresholds failed (ESS/K starts high, no clean crossing). Reward derivative directly measures whether remasking is still helping. |
| Hard deadline | t = 0.5 (step 50/100) | Safety net — lock by midpoint regardless of reward dynamics |
| EMA β | TBD — script shows 3 values (0.7, 0.85, 0.95) | User needs to inspect `ema_reward_lock.png` on jabiru to pick β |
| Per-sample vs batch lock | Not yet decided | Per-sample is more precise; batch-level is simpler. Discuss in next session. |

## Current state of the codebase
- **G1-G4 COMPLETE**, G5 IN PROGRESS (Rounds 1-5 done)
- All 727 tests pass (3 pre-existing failures unrelated to guidance)
- `scripts/plot_ema_trajectories.py` committed and pushed — **not yet run with the latest version on jabiru** (the reward-derivative lock plot was just pushed)
- The `_samples.pt` files on jabiru for Round 4 configs are intact:
  - `llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01_samples.pt`
  - `llada_topp0.9_no_remask_guided_r4soft_K16_a0.01_samples.pt`
  - (plus r4hard variants and all kstar/alpha/basic samples)

## What remains to be done
1. **Run `plot_ema_trajectories.py` on jabiru** to generate `ema_reward_lock.png` — inspect to pick β
2. **Decide per-sample vs batch-level lock**
3. **Implement adaptive lock in `guided_sampler.py`**:
   - Add `ema_lock: bool` parameter (+ `ema_beta`, `ema_lock_consecutive`, `ema_lock_deadline` params)
   - In SVDD loop: compute EMA(reward), track derivative streak, flip per-sample `remasking_active` flag
   - When locked: pass `remasking_fn=None` to `_single_step_remask`
   - Log lock step in GuidanceStats
4. **Add `--ema-lock` CLI flag** to `generate_guided.py`
5. **Implement the `t_switch` inversion** — currently `t_now < t_switch` activates remasking late; for "early remasking" we need `t_now >= t_switch`. See `bd_gen/diffusion/sampling.py:277`. BUT: with EMA-based adaptive lock, we may not need `t_switch` at all — the lock replaces it.
6. **Round 6 experiment on jabiru**: Run Option B + adaptive lock at K=16, α=0.01, compare to no-remask (69%) and plain confidence (56%)
7. **Tests**: Add tests for adaptive lock logic (EMA computation, streak detection, lock trigger)

## Files to reference in next session
- `BD_Generation/implementation_state_T1_guidance.md` — dynamic state (always read first per CLAUDE.md)
- `BD_Generation/planning_T1_guidance.md` — static spec
- `BD_Generation/docs/guidance.md` — full experiment history, §Round 5 verdict, §Open Question
- `BD_Generation/scripts/plot_ema_trajectories.py` — EMA + lock criterion visualization (has `find_lock_step()` reference implementation)
- `BD_Generation/bd_gen/guidance/guided_sampler.py:331` — `guided_sample()` SVDD loop where lock must be integrated
- `BD_Generation/bd_gen/diffusion/sampling.py:241-284` — `_single_step_remask()` with `t_switch` logic
- `BD_Generation/bd_gen/diffusion/remasking.py` — `confidence_boost` and `protect_mask` params (Option B/C)

## Context for the next session
- **Adaptive lock design**: Remasking ON at start (Option B: confidence + protect just-unmasked). At each step, compute `ema_reward[t] = β * ema_reward[t-1] + (1-β) * reward_selected[t]`. Track `d_ema = ema[t] - ema[t-1]`. When `d_ema ≤ 0` for 3 consecutive steps → lock (disable remasking for this sample permanently). Hard deadline at t=0.5 (step 50).
- **t_switch semantics**: t goes 1.0→0.0. `t_now < t_switch` activates remasking at LOW t (late). We want remasking at HIGH t (early). The adaptive lock may replace t_switch entirely — just start with `t_switch=1.0` (always remask) and let the EMA lock handle the off-switch.
- **Option B implementation**: Already exists — `protect_mask` param in `_single_step_remask()` and `remasking.py`. Just need to enable it in the SVDD loop.
- **Round 5 `_samples.pt` deleted on jabiru**: If needed, re-run via `bash scripts/run_g5_round5.sh` (~5-10 min GPU). But Round 4 samples suffice for analysis.
- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, workdir: `/Data/amine.chraibi/Davis/BD_Generation`
- **The user will define what "Round 6" consists of** — don't assume the experiment grid. Wait for instructions.
