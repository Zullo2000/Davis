# Handoff: Round 4 — Remasking × Reward-mode (continued)

> **Created:** 2026-03-05
> **Session purpose:** Fix device bugs blocking confidence+RACB configs, analyze soft vs hard reward mode results, continue Round 4 experiments

---

## What was accomplished

- **Fixed 2 device mismatch bugs** in `bd_gen/guidance/guided_sampler.py` that prevented confidence+RACB configs from running on GPU:
  - Commit `941a6fc`: `_score_candidates_soft` (line ~125) and `_score_candidates_hard` (line ~166) — `torch.zeros(K, B)` missing `device=candidates.device`
  - Commit `33ee5f7`: `_score_single_soft` (line ~202) and `_score_single_hard` (line ~224) — `torch.zeros(B)` missing `device=x_t.device`
  - Both commits pushed to remote (jabiru can `git pull`)
- **Analyzed configs 1 & 2** (no-remasking × {soft, hard} reward):
  - Soft reward: 69% overall satisfaction (kitchen 95%, adj 93%, forbid 93%, bathroom 81%)
  - Hard reward: 35% overall satisfaction (kitchen 96%, adj 56%, forbid 80%, bathroom 72%)
  - Soft wins decisively — granular probability signal steers better than binary argmax
- **Documented soft reward mode decision** in `docs/guidance.md` (lines ~92–111): "Why we use soft reward mode (Round 4 finding)" section with comparison table and rationale
- **Copied results locally** for configs 1 & 2 (JSON, _samples.pt, trajectory PNGs — 8 files total)

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Soft vs hard reward mode | Always use soft | 69% vs 35% satisfaction — granular probability signal outperforms binary argmax |
| Device placement pattern | Use `device=candidates.device` / `device=x_t.device` | Follows existing patterns in codebase, ensures scoring works on any device |

## Current state of the codebase

### Working
- All 4 device bugs fixed, 19 guided sampler tests pass
- Full test suite: 727+ passed, 3 pre-existing failures (`_aggregate_multi_seed` import — unrelated), 2 skipped
- Configs 1 & 2 results available locally in `eval_results/loglinear_noise_sc/`
- Git: main at `33ee5f7`, pushed to remote

### Partially done — Round 4 experiment grid
| Config | Variant | Reward | Status |
|--------|---------|--------|--------|
| 1 | no-remask | soft | DONE — results copied locally, evaluated, analyzed |
| 2 | no-remask | hard | DONE — results copied locally, evaluated, analyzed |
| 3 | confidence + RACB | soft | NEEDS RE-RUN — was failing due to device bugs (now fixed) |
| 4 | confidence + RACB | hard | NOT YET STARTED — blocked by config 3 (sequential generation) |

### Local result files (configs 1 & 2)
```
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01.json
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01_samples.pt
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01_trajectories_clean.png
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01_trajectories_outliers.png
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01.json
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01_samples.pt
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01_trajectories_clean.png
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01_trajectories_outliers.png
```

## What remains to be done

1. **On jabiru**: `git pull` to get both device fixes (`941a6fc` + `33ee5f7`)
2. **Re-run configs 3 & 4** on jabiru:
   ```bash
   cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation
   tmux new -s round4
   # Config 3: confidence + soft + RACB
   python scripts/generate_guided.py \
       eval.checkpoint_path=outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt \
       noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 \
       eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.t_switch=1.0 \
       eval.seeds=[42,123,456] eval.num_samples=200 \
       wandb.mode=disabled \
       --guidance-config configs/guidance/example_basic.yaml \
       --calibration configs/guidance/calibration_v1_confidence.json \
       --alpha 0.01 --K 16 --reward-mode soft --attribution-boost --guidance-tag r4soft

   # Config 4: confidence + hard + RACB
   python scripts/generate_guided.py \
       eval.checkpoint_path=outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt \
       noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 \
       eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.t_switch=1.0 \
       eval.seeds=[42,123,456] eval.num_samples=200 \
       wandb.mode=disabled \
       --guidance-config configs/guidance/example_basic.yaml \
       --calibration configs/guidance/calibration_v1_confidence.json \
       --alpha 0.01 --K 16 --reward-mode hard --attribution-boost --guidance-tag r4hard
   ```
3. **Evaluate each config** as it finishes (from second terminal):
   ```bash
   # Config 3
   python scripts/evaluate.py --schedule loglinear_noise_sc \
       --model llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 \
       --guidance-config configs/guidance/example_basic.yaml
   python scripts/analyze_guidance_stats.py --schedule loglinear_noise_sc \
       --model llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01 --plot-analysis

   # Config 4
   python scripts/evaluate.py --schedule loglinear_noise_sc \
       --model llada_topp0.9_remdm_confidence_tsw1.0_guided_r4hard_K16_a0.01 \
       --guidance-config configs/guidance/example_basic.yaml
   python scripts/analyze_guidance_stats.py --schedule loglinear_noise_sc \
       --model llada_topp0.9_remdm_confidence_tsw1.0_guided_r4hard_K16_a0.01 --plot-analysis
   ```
4. **Copy results locally** (scp JSONs, PNGs, _samples.pt for configs 3 & 4)
5. **Run full 4-config comparison** (step 4 of `run_g5_round4.sh compare`)
6. **Analyze all 4 configs**: Does RACB mitigate the guidance-remasking conflict? Compare no-remask vs confidence+RACB satisfaction rates.
7. **Update `implementation_state_T1_guidance.md`** with Round 4 findings
8. **Decide next steps**: Expand to v2 variants if warranted (item 8 in experiment plan)

## Files to reference in next session

1. `past_handoffs/handoff_round4_remasking_reward_continued.md` — this file
2. `implementation_state_T1_guidance.md` — full experiment state (read first per CLAUDE.md)
3. `scripts/run_g5_round4.sh` — Round 4 experiment script (5-step pipeline)
4. `docs/guidance.md` — guidance module docs (includes soft vs hard section)
5. `bd_gen/guidance/guided_sampler.py` — core guided sampling (device fixes applied)
6. `eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01.json` — config 1 results
7. `eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01.json` — config 2 results

## Context for the next session

- **Device bugs only manifest with confidence+RACB** (configs 3 & 4) because that's the only code path calling `_compute_attribution_boost()` and computing `reward_remasking_delta`. No-remasking configs never hit those scoring functions with GPU tensors.
- **Configs run sequentially** in the `run_g5_round4.sh generate` step (4 configs × 3 seeds × ~6 min/seed ≈ ~72 min total). You can evaluate each config from a second SSH terminal as it finishes.
- **Calibration is already done** — `calibration_v1_no_remask.json` and `calibration_v1_confidence.json` exist on jabiru. No need to re-run step 1.
- **The soft reward mode decision is already documented** in `docs/guidance.md`. Future experiments should always use soft unless there's a specific reason to revisit.
- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`, activate `.venv`, then `cd BD_Generation`.
- **Modified file in git status** (`00_masked_diffusion_conceptual_overview.md`): Pre-existing change, not related to Round 4 work.
