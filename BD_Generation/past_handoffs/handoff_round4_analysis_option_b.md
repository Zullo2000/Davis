# Handoff: Round 4 Complete — Option B Next

> **Created:** 2026-03-05
> **Session purpose:** Complete Round 4 experiment grid (4 configs), analyze results, discuss next mitigation options for guidance-remasking conflict

---

## What was accomplished

- **Completed all 4 Round 4 configs** on jabiru (K=16, α=0.01):
  - Config 1: no-remask + soft → **69% overall satisfaction** (best)
  - Config 2: no-remask + hard → 35%
  - Config 3: confidence+RACB + soft → 56%
  - Config 4: confidence+RACB + hard → 36%
- **Copied all results locally** (JSON, _samples.pt, trajectory PNGs for all 4 configs)
- **Analyzed static + dynamic metrics** for configs 3 & 4:
  - Static: RACB helped kitchen counting (+2.3) and adjacency (+5.7) but ForbidAdj collapsed (93%→79%), dragging overall from 69%→56%
  - Dynamic: Remasking delta wildly oscillating (-2 to +1 for soft, -3 to +2 for hard), reward trajectories chaotic/non-converging, ESS spiking to 1 (degenerate), violations never settling
- **Diagnosed why Option C (RACB) failed**: Additive confidence boost is too weak — softmax normalization washes it out. It reduces remasking probability but doesn't prevent it, so the tug-of-war persists
- **Analyzed Options A/B/C** from `00_masked_diffusion_conceptual_overview.md` §15.4:
  - Option B (protect just-unmasked for 1 step) is the most promising — directly breaks the same-step feedback loop
  - Option B+C together could be optimal
  - Option A (fresh logits, 2× cost) is gold standard upper bound

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Soft vs hard reward | Always soft | 69% vs 35% (no-remask), 56% vs 36% (confidence) — consistent ~20pp gap |
| Option C (RACB) verdict | Insufficient alone | Oscillation undamped, ForbidAdj collapses, overall satisfaction drops 13pp vs no-remask |
| Next mitigation to try | Option B (protect just-unmasked 1 step) | Zero cost, no hyperparameters, directly prevents same-step remask of guided positions |

## Current state of the codebase

### Working
- All code on main at `33ee5f7`, pushed to remote
- Full test suite: 727+ passed, 3 pre-existing failures (`_aggregate_multi_seed` import), 2 skipped
- Option C (RACB) fully implemented and functional (just not effective enough)
- All 4 Round 4 result files locally in `eval_results/loglinear_noise_sc/`

### Round 4 result files (all local)
```
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01.json
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01_samples.pt
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4soft_K16_a0.01_trajectories_*.png
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01.json
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01_samples.pt
eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_r4hard_K16_a0.01_trajectories_*.png
eval_results/loglinear_noise_sc/llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01.json
eval_results/loglinear_noise_sc/llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01_samples.pt
eval_results/loglinear_noise_sc/llada_topp0.9_remdm_confidence_tsw1.0_guided_r4soft_K16_a0.01_trajectories_*.png
eval_results/loglinear_noise_sc/llada_topp0.9_remdm_confidence_tsw1.0_guided_r4hard_K16_a0.01.json
eval_results/loglinear_noise_sc/llada_topp0.9_remdm_confidence_tsw1.0_guided_r4hard_K16_a0.01_samples.pt
eval_results/loglinear_noise_sc/llada_topp0.9_remdm_confidence_tsw1.0_guided_r4hard_K16_a0.01_trajectories_*.png
```

### Not yet done
- `implementation_state_T1_guidance.md` not yet updated with Round 4 final conclusions
- `docs/guidance.md` not yet updated with Round 4 remasking analysis
- Option B not implemented
- No comparison.md generated for the 4-config grid (step 4 of `run_g5_round4.sh`)

## What remains to be done

1. **Decide direction**: Either:
   - **(a) Implement Option B** (protect just-unmasked positions 1 step) → run Round 5 experiment to test it
   - **(b) Accept no-remasking as the winner** and move forward to v2 variants or declare G5 complete
   - **(c) Option B+C combo** — keep RACB, add Option B as additional protection

2. **If implementing Option B**:
   - Modify `bd_gen/diffusion/sampling.py` `_single_step_remask()` to accept a `protect_mask` parameter (positions to exclude from remasking candidates)
   - Modify `bd_gen/guidance/guided_sampler.py` to pass `was_mask_before` as `protect_mask`
   - Add tests, add `--protect-just-unmasked` CLI flag
   - Run Round 5: confidence+B vs confidence+C vs confidence+B+C vs no-remask baseline
   - Key question: does Option B eliminate the oscillation AND recover the 69% no-remask satisfaction?

3. **Update docs**: Round 4 conclusions in `implementation_state_T1_guidance.md` and `docs/guidance.md`

4. **Generate comparison table**: `run_g5_round4.sh compare` step (or manual `compare_selected.py`)

## Files to reference in next session

1. `implementation_state_T1_guidance.md` — full experiment state (read per CLAUDE.md)
2. `past_handoffs/handoff_round4_analysis_option_b.md` — this file
3. `00_masked_diffusion_conceptual_overview.md` §15 (lines 853–1096) — Options A/B/C analysis
4. `bd_gen/guidance/guided_sampler.py` — core guided sampling (Option C/RACB already here)
5. `bd_gen/diffusion/sampling.py` — `_single_step_remask()` (where Option B would be implemented)
6. `bd_gen/diffusion/remasking.py` — remasking logic (`confidence_boost` param already exists)
7. `docs/guidance.md` — guidance module docs
8. `scripts/run_g5_round4.sh` — Round 4 experiment script

## Context for the next session

- **Why Option C failed**: The additive confidence boost before `softmax(-confidence)` gets washed out by normalization. It reduces remasking probability but doesn't prevent it. The oscillation in trajectories is undamped — remasking delta swings wildly throughout all 100 steps, never converging.
- **ForbidAdj is the worst case**: ForbidAdj constraints require token combinations the model doesn't naturally produce (low base confidence) — exactly where guidance adds value but remasking strikes hardest. ExactCount constraints are less affected.
- **How to evaluate remasking trajectories**: Look for (1) narrowing oscillation envelope, (2) >50% positive remasking delta steps, (3) late-step convergence (t<30), (4) ESS stability. Config 3 fails all four criteria.
- **Soft reward mode is settled**: Always use soft. Documented in `docs/guidance.md` lines ~92-111.
- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, dir: `/Data/amine.chraibi/Davis`, activate `.venv`, `cd BD_Generation`. Git is at `33ee5f7` on both local and remote.
- **eta=0.1 in remasking config**: Harmless default from `configs/eval/default.yaml:15`, ignored by confidence strategy. Prints in logs but has no effect.
