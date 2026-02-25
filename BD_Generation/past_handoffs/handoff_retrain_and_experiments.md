# Handoff: Retrain Model & Run 13-Run Experiment Suite

> **Created:** 2026-02-19
> **Session purpose:** Fix confidence remasking (ReMDM paper), add top-p sampling + t_switch, prepare batch script for retraining + 13 layered experiments on university server GPU.

---

## What was accomplished

- **Fixed confidence remasking** in `bd_gen/diffusion/remasking.py`: budget now uses `sigma_max` directly (paper formula) instead of `min(eta, sigma_max)`. Confidence strategy ignores `eta` parameter entirely.
- **Added top-p nucleus sampling** in `bd_gen/diffusion/sampling.py`: `_top_p_sample()` helper, wired via `top_p` parameter to `sample()`. Top-p takes priority over temperature when set.
- **Added t_switch (ReMDM-Switch)** in `bd_gen/diffusion/sampling.py`: remasking only activates when `t_now < t_switch`. Default `t_switch=1.0` is backward compatible.
- **Updated evaluate.py**: wires `top_p` and `t_switch`, new method naming scheme `{unmasking}_{sampling}_{remasking}` (e.g., `random_topp0.9_remdm_cap_eta0.4`).
- **Updated configs**: `eval/default.yaml` has `top_p: null` and `t_switch: 1.0`.
- **Updated save_utils.py**: config table includes `top_p` and `remasking_t_switch`.
- **Updated implementation_state_T1.md**: 3 post-v1 sections marked as MISTAKE (code retained, results discarded), new "Retraining After Previously Committed Changes" section added (Status: PENDING).
- **Created `scripts/run_experiments.sh`**: batch script for retrain + 12 eval runs + comparison table.
- **Created `remasking_design_with_fixed_forward_process.md`**: design document covering remasking strategies, experiment layers, top-p rationale.
- **Tests**: 533 pass, 2 skipped (data-dependent, will unskip after experiments), 1 pre-existing warning (lr_scheduler ordering in test_integration.py — cosmetic, not blocking).
- **Commit**: `840dfb6` — all changes in one clean commit.

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Confidence budget formula | `sigma_max * softmax(-conf) * n_decoded` | Matches ReMDM paper (arXiv:2503.00307), no eta capping |
| Top-p priority | Top-p overrides temperature when set | Cleaner than combining both; temperature=0.0 still works as argmax fallback |
| t_switch default | 1.0 (= remasking at all steps except first/last) | Backward compatible with existing cap behavior |
| t_switch boundary | `t_now < t_switch` (strict less-than) | At t=1.0 everything is masked anyway, no point remasking |
| Method naming | `{unmasking}_{sampling}_{remasking}` | Encodes full config in filename for easy comparison |
| Run 10 deferred | Manual after Layer 2 analysis | Need to pick best eta first |
| Eval results fresh start | Deleted old JSONs | Old results used wrong confidence formula + argmax-only |

## Current state of the codebase

- **All code changes committed** (`840dfb6`), ready to push/transfer to server
- **No eval results yet** — `eval_results/` is empty (old results deleted)
- **No v2 checkpoint yet** — existing `checkpoint_final.pt` was trained before SUBS zero masking, float64 ELBO, and importance sampling
- **Unstaged deletions**: old `eval_results/*.json`, `comparison.md`, `docs/mdlm_comparison.md`, `remasking_doubts.md`, `update_planning.md` — these are intentional cleanup from fresh start
- **533 tests pass**, 2 skip (need eval result JSONs), 1 warning (pre-existing, non-blocking)

## What remains to be done

### On University Server (GPU required)

1. **Transfer code** to server (git pull or rsync)
2. **Run the batch script**:
   ```bash
   cd BD_Generation
   bash scripts/run_experiments.sh
   ```
   This does everything in sequence (~70-90 min on RTX A5000):
   - Phase 0: Retrain (`python scripts/train.py wandb.mode=disabled`) → new checkpoint
   - Phase 1 (Layer 1): 4 baselines — random/llada × argmax/top-p=0.9, no remasking
   - Phase 2 (Layer 2): 5 cap eta sweep — eta ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, random+top-p=0.9
   - Phase 3 (Layer 3): 3 confidence+switch — t_switch ∈ {0.3, 0.5, 0.7}, random+top-p=0.9
   - Phase 4: `python scripts/compare.py` → `eval_results/comparison.md`

3. **After batch completes — analyze results**:
   - Layer 1: Does llada unmasking beat random? If yes, re-run L2/L3 with `eval.unmasking_mode=llada`
   - Layer 2: Pick best eta from cap sweep
   - Run 10 (manual): `$EVAL_CMD $COMMON $CKPT eval.unmasking_mode=random eval.temperature=0.0 eval.top_p=null eval.remasking.enabled=true eval.remasking.strategy=cap eval.remasking.eta=<BEST> eval.remasking.t_switch=1.0`
   - Layer 4: Head-to-head comparison of best cap vs best confidence

4. **Update implementation_state_T1.md** with:
   - Retraining section → Status: COMPLETE (with training loss, time)
   - New section for experiment results and analysis

### Script details to know
- The script uses `set -euo pipefail` — stops on first error
- It auto-detects the latest checkpoint via `ls -td outputs/*/checkpoints/checkpoint_final.pt`
- Each eval run saves to `eval_results/` with method-encoded filename
- 5 seeds per eval run: [42, 123, 456, 789, 1337]
- 1000 samples per seed, 100 sampling steps

## Files to reference in next session

**Read these first (in order):**
1. `BD_Generation/implementation_state_T1.md` — lines 507-531 for retraining section, lines 333-406 for remasking status
2. `BD_Generation/scripts/run_experiments.sh` — the full batch script
3. `BD_Generation/remasking_design_with_fixed_forward_process.md` — Section 10 for experiment design rationale
4. `BD_Generation/configs/eval/default.yaml` — eval config with all parameters

**Key implementation files (if debugging needed):**
5. `BD_Generation/bd_gen/diffusion/remasking.py` — confidence fix (lines 245-286)
6. `BD_Generation/bd_gen/diffusion/sampling.py` — top-p + t_switch (lines 170-280)
7. `BD_Generation/scripts/evaluate.py` — method naming + wiring
8. `BD_Generation/eval_results/save_utils.py` — JSON format + comparison table builder

## Context for the next session

- **The 3 retraining changes are already in the code** — just need to run `python scripts/train.py`. No code changes needed for retraining.
- **Importance sampling was enabled in config** (commit `0760aca`) but never actually used for training. The v1 checkpoint was trained without it.
- **The batch script handles everything end-to-end** — no manual intervention needed until Run 10 (after Layer 2 analysis).
- **Top-p and temperature interaction**: when `top_p` is set (not null), it takes priority. `temperature=0.0` alone still means argmax. The experiment script uses Hydra overrides like `eval.top_p=0.9` and `eval.temperature=0.0`.
- **t_switch boundary gotcha**: at `t_now = 1.0` (first denoising step), remasking is skipped because `1.0 < 1.0` is false. This is correct — at t=1.0 everything is masked, remasking is meaningless.
- **2 skipped tests** (`test_loads_existing_v1_file`, `test_with_existing_files`) will auto-unskip once eval result JSONs exist after experiments.
- **Pre-existing warning** in `test_integration.py:248` about lr_scheduler ordering — cosmetic, doesn't affect training correctness.
