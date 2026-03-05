# Handoff: Round 4 — Remasking × Reward-mode Experiment

> **Created:** 2026-03-05
> **Session purpose:** Set up and launch Round 4 experiment comparing no-remasking vs confidence remasking (with Reward-Attributed Confidence Boosting) × soft vs hard reward mode.

---

## What was accomplished

- **Created `scripts/run_g5_round4.sh`** — Round 4 experiment script (5-step pipeline: calibrate → generate → evaluate → compare → analyze)
- **Committed and pushed** all pending work: Option C attribution boost code, Round 3 results, Round 4 script (commits `4f14264`, `505bc56`, `44be13e`)
- **Launched Round 4 on jabiru** via tmux session `round4` — currently running `bash scripts/run_g5_round4.sh all`

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Confidence remasking always uses Reward-Attributed Confidence Boosting | `--attribution-boost` on all confidence configs | Vanilla confidence remasking uses stale model logits that don't account for guidance reweighting — fights the guidance signal |
| Naming: dropped "Option C" | Now called "Reward-Attributed Confidence Boosting" (RACB) | It's the only remasking mode we use with guidance, not an "option" anymore |
| Tags: `r4soft` / `r4hard` | Encode reward mode in guidance tag | Method names differentiate: `..._guided_r4soft_K16_a0.01` vs `..._guided_r4hard_K16_a0.01` |

## Round 4 experiment design

**Fixed:** K=16, α=0.01, v1 loglinear checkpoint
**Grid:** 2 variants × 2 reward modes = 4 configs

| # | Remasking | Reward mode | RACB |
|---|-----------|-------------|------|
| 1 | none | soft (softmax) | n/a |
| 2 | none | hard (argmax) | n/a |
| 3 | confidence (tsw=1.0) | soft (softmax) | yes |
| 4 | confidence (tsw=1.0) | hard (argmax) | yes |

**Samples:** 3 seeds (42, 123, 456) × 200 = 600 per config
**Output:** `eval_results/loglinear_noise_sc/comparison_guided_round4.md` + per-config trajectory plots

## Current state

- **Running on jabiru** in tmux session `round4` (`tmux attach -t round4` to check)
- Estimated runtime: ~2-3h total
- All code is committed and pushed to `main` at `44be13e`
- No local uncommitted changes

## What remains to be done

1. **Wait for Round 4 to complete on jabiru** — check with `tmux attach -t round4`
2. **Copy results back locally:**
   ```bash
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*r4*  BD_Generation/eval_results/loglinear_noise_sc/
   scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/comparison_guided_round4.md  BD_Generation/eval_results/loglinear_noise_sc/
   ```
3. **Analyze Round 4 results** — compare the 4 configs:
   - Does RACB-based remasking improve over no-remasking? (previous rounds showed remasking fights guidance)
   - Does hard reward mode outperform soft? (hard uses argmax decode, more expensive but possibly more accurate)
   - Check constraint satisfaction, quality metrics, trajectory plots
4. **Update `implementation_state_T1_guidance.md`** with Round 4 findings
5. **Decide next steps:** expand to v2 variants if warranted (item 8 in experiment plan)

## Files to reference in next session

1. `implementation_state_T1_guidance.md` — dynamic state (read per CLAUDE.md rules)
2. `scripts/run_g5_round4.sh` — the experiment script (understand what was run)
3. `eval_results/loglinear_noise_sc/comparison_guided_round4.md` — results (once available)
4. `bd_gen/guidance/guided_sampler.py` — SVDD loop + attribution boost logic
5. `bd_gen/diffusion/remasking.py` — confidence remasking + boost integration

## Context for the next session

- **Jabiru SSH:** `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis/BD_Generation`
- **venv:** `source .venv/bin/activate` (must activate before running scripts)
- **tmux:** session `round4` — `tmux attach -t round4` to check status
- **Git sync gotcha:** jabiru has `_samples.pt` files (gitignored) that generate → evaluate depends on. Don't delete those accidentally.
- **Previous findings:** Round 3 showed α=0.01 best for satisfaction at K=16; K* sweep showed remasking shifts K* up (fights guidance). Round 4 tests whether RACB mitigates this.
