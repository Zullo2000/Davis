# Handoff: G5 Round 5 Results + Remasking Strategy Discussion

> **Created:** 2026-03-06
> **Session purpose:** Run Round 5 experiments (Option A vs B), analyze results, update docs, begin discussion on selective remasking for constraint scaling.

---

## What was accomplished
- **Round 5 executed on jabiru**: 2 configs (confidence + Option A, confidence + Option B), K=16, α=0.01, soft reward, 3 seeds × 200 samples = 600 per config
- **Results copied locally** and moved to `eval_results/loglinear_noise_sc/` (6 files: 2 JSONs, 4 trajectory PNGs, 1 comparison MD)
- **docs/guidance.md updated**: Round 5 results table, findings, verdict ("remasking mitigation investigation concluded"), and new "Open Question — Selective remasking for constraint scaling" section
- **implementation_state_T1_guidance.md updated**: Round 5 marked DONE, experiment plan items 8-10 updated, new "Round 5 — Option A vs Option B" summary section added

## Key decisions made
| Decision | Choice | Rationale |
|---|---|---|
| Remasking mitigations A/B/C | All fail equally (~55-56%) | Structural conflict: model assigns low confidence to reward-steered tokens → remasking preferentially destroys guided positions |
| Best config | No-remask + soft, K=16, α=0.01 (69%) | 5.2× unguided baseline (13.3%); all remasking variants plateau at ~56% |
| Next direction | Selective early-stage remasking | No-remask is irreversible — problematic as constraint count grows |

## Current state of the codebase
- **G1-G4 COMPLETE**, G5 IN PROGRESS (Rounds 1-5 done, more experiments possible)
- All 727 tests pass (3 pre-existing failures unrelated to guidance)
- Round 5 results committed locally but **not yet pushed to remote** — needs `git add` + `git commit`

## What remains to be done
1. **Discuss and design "remask-then-lock" strategy** (early remasking + late no-remasking)
   - Key issue: current `t_switch` semantics are **inverted** for our use case
   - `t_now < t_switch` activates remasking → `t_switch=0.3` means remasking in the **last 30 steps** (low t), not the first 70
   - Need to either: (a) add `remasking_early_only` flag that flips to `t_now >= t_switch`, or (b) add two thresholds `t_start`/`t_stop`
   - See `bd_gen/diffusion/sampling.py:277` for the condition
2. **Implement the inverted t_switch** (or new parameter)
3. **Round 6 experiment**: sweep `t_switch ∈ {0.2, 0.3, 0.4, 0.5}` at K=16, α=0.01 with early-only remasking
4. **Commit Round 5 results + docs updates** (user hasn't requested yet)
5. Eventually: expand guidance to v2 variants

## Files to reference in next session
- `BD_Generation/implementation_state_T1_guidance.md` — dynamic state (always read first per CLAUDE.md)
- `BD_Generation/planning_T1_guidance.md` — static spec
- `BD_Generation/docs/guidance.md` — full experiment history, especially §Round 5 verdict and §Open Question
- `BD_Generation/bd_gen/diffusion/sampling.py:241-282` — `_single_step_remask()` with `t_switch` logic
- `BD_Generation/bd_gen/guidance/guided_sampler.py:331` — `t_switch` parameter in `guided_sample()`
- `BD_Generation/eval_results/loglinear_noise_sc/comparison_guided_round5.md` — Round 5 comparison table

## Context for the next session
- **t_switch semantics gotcha**: `t` goes from 1.0→0.0 during denoising. Current condition `t_now < t_switch` means remasking activates when t is LOW (late in denoising). We want the opposite for "early remasking": remasking when t is HIGH (early), then lock when t drops below threshold.
- The user wants to **think together** about the remasking strategy before implementing — don't jump straight to coding
- The user's intuition: without remasking, as constraints grow, early-committed violating tokens can never be corrected. Early-stage remasking provides error correction when most tokens are still MASK and violations are fixable.
- All Round 4 reference results are already on jabiru and locally — no need to re-run baselines for Round 6
- Jabiru SSH: `ssh amine.chraibi@jabiru.polytechnique.fr`, workdir: `/Data/amine.chraibi/Davis/BD_Generation`
