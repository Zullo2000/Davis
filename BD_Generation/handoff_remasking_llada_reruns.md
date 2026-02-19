# Handoff: Remasking Experiments — LLaDA Re-runs

> **Created:** 2026-02-19
> **Session purpose:** Analyzed linear vs log-linear schedule results, diagnosed IS+linear failure, ran log-linear 12-experiment suite. Next: re-run Layers 2-3 with llada unmasking on jabiru.

---

## What was accomplished

### Analysis & diagnosis
- **Copied linear eval results** from jabiru to `eval_results/linear/` (12 JSON + comparison.md)
- **Diagnosed IS + linear schedule failure**: importance sampling concentrates training near t->0 where linear schedule has alpha~1 (nothing masked), leaving model untrained at high masking rates used during generation. Caused validity to drop from 99.3% (v1) to 94.6% (v2), and llada to catastrophically fail (30% validity)
- **Confirmed log-linear schedule fixes everything**: validity restored to 99.9%, llada restored to 100%, model accuracy at t=0.5 improved from 3% to 54% for edges

### Log-linear training & eval (on jabiru)
- **Trained** log-linear model: 500 epochs, checkpoint at `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt`
- **Ran all 12 eval experiments** (Layers 1-3 with random unmasking)
- **Copied results** to local `eval_results/loglinear/`
- **Committed and pushed** log-linear code + linear results (commit `7723df3`)

### Code changes committed
- `LogLinearSchedule` in `noise_schedule.py`, config `configs/noise/loglinear.yaml`
- `evaluate.py` saves to `eval_results/<schedule>/` based on noise type
- `compare.py` accepts `--schedule` arg
- `run_experiments.sh` accepts schedule as first argument
- 11 new tests for LogLinearSchedule (544 total pass)

### Documentation
- Updated `implementation_state_T1.md` with:
  - "Retraining & Noise Schedule Comparison" section marked COMPLETE
  - New "Remasking Design Comparison (Log-Linear Schedule)" section with Layer 1-3 results

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Log-linear over linear | Log-linear is the correct schedule with IS | alpha(0.5)=0.5 vs 0.007; IS+linear leaves model untrained at generation-time masking rates |
| Skip Run 10 for linear | Not informative | Remasking doesn't help with linear schedule at all |
| Re-run L2/L3 with llada | Pending | llada_topp has best structural quality (100% validity, MMD-Degree 0.050, type-cond deg JS 0.033) — worth testing with remasking |
| Best eta for cap | 0.2 or 0.4 | Eta 0.4-1.0 all plateau; 0.2 least Edge JS damage, 0.4 best Node JS + coverage |

## Current state of the codebase

- **Local**: `implementation_state_T1.md` has uncommitted updates (the two new sections). Log-linear eval results in `eval_results/loglinear/` are uncommitted. Several handoff/analysis docs uncommitted.
- **Server (jabiru)**: Has all code (git pulled). Log-linear checkpoint + 12 eval results in place. tmux session may still be active.
- **Tests**: 544 pass, 2 skip (auto-unskip once v1 eval JSONs exist), 1 pre-existing warning

### Key log-linear Layer 1 results

| Metric | llada_topp | random_topp | Winner |
|---|:---:|:---:|:---:|
| Validity | **100%** | 99.7% | llada |
| Node JS | 0.023 | **0.006** | random |
| Edge JS | 0.106 | **0.035** | random |
| Spatial transit. | **99.9%** | 97.8% | llada |
| MMD-Degree | **0.050** | 0.302 | llada |
| Diversity | 0.945 | **1.0** | random |
| Mode cov (wt) | 69.6% | **88.5%** | random |

### Layer 2 cap eta (random+topp): eta saturates at 0.4
### Layer 3 confidence: t_switch has minimal effect, slightly worse than cap

## What remains to be done

### Immediate: re-run Layers 2-3 with llada unmasking on jabiru
1. SSH to jabiru, start tmux, activate venv
2. Run 8 eval commands (5 cap eta + 3 confidence t_switch) with `eval.unmasking_mode=llada`
3. Use checkpoint: `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt`
4. Results will save to `eval_results/loglinear/` with `llada_` prefix in filename
5. Copy results locally, re-run `compare.py --schedule loglinear`

**Commands for jabiru:**
```bash
cd /Data/amine.chraibi/Davis/BD_Generation
tmux new -s experiments
source /Data/amine.chraibi/Davis/.venv/bin/activate

CKPT="eval.checkpoint_path=outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt"
COMMON="wandb.mode=disabled"
NOISE="noise=loglinear"
EVAL="python scripts/evaluate.py"

# Layer 2 with llada: cap eta sweep
for ETA in 0.2 0.4 0.6 0.8 1.0; do
  $EVAL $COMMON $NOISE $CKPT eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=true eval.remasking.strategy=cap eval.remasking.eta=$ETA eval.remasking.t_switch=1.0
done

# Layer 3 with llada: confidence + t_switch sweep
for TSW in 0.3 0.5 0.7; do
  $EVAL $COMMON $NOISE $CKPT eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.eta=0.0 eval.remasking.t_switch=$TSW
done
```

### After llada re-runs
6. Run 10: cap best_eta + argmax control (confirm top-p synergy)
7. Final comparison: best cap vs best confidence, across both unmasking modes
8. Update `implementation_state_T1.md` with llada remasking results
9. Commit all results and updated docs

## Files to reference in next session

**Read first (in order):**
1. `BD_Generation/implementation_state_T1.md` — lines 572-638 for remasking comparison status
2. `BD_Generation/eval_results/loglinear/comparison.md` — current 12-run results table
3. `BD_Generation/remasking_design.md` — Section 10 for experiment design rationale

**Key implementation files:**
4. `BD_Generation/bd_gen/diffusion/noise_schedule.py` — LogLinearSchedule
5. `BD_Generation/scripts/evaluate.py` — schedule-aware result saving
6. `BD_Generation/scripts/compare.py` — per-schedule comparison
7. `BD_Generation/scripts/run_experiments.sh` — parameterized by schedule

## Context for the next session

- **Server access**: SSH to `jabiru.polytechnique.fr` as `amine.chraibi`, code at `/Data/amine.chraibi/Davis`, venv at `/Data/amine.chraibi/Davis/.venv`
- **tmux**: Use `tmux new -s experiments` for long-running tasks. The previous session may still exist (`tmux ls` to check)
- **wandb broken on jabiru**: Use `wandb.mode=disabled` always
- **scp from local PowerShell** uses full hostname: `amine.chraibi@jabiru.polytechnique.fr:` (not the `jabiru` SSH alias)
- **The `env` error** on jabiru (`/users/eleves-a/2022/amine.chraibi/.local/bin/env: No such file or directory`) is cosmetic — does not block anything
- **Eval filenames** include unmasking mode: llada runs will be `llada_topp0.9_remdm_cap_eta0.2.json` etc., distinct from existing `random_topp0.9_remdm_cap_eta0.2.json`
- **compare.py** auto-discovers all JSONs in the schedule directory, so after llada runs it will generate an expanded comparison table with all 20 methods
