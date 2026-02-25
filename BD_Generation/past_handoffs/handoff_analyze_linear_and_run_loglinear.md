# Handoff: Analyze Linear Results & Run Log-Linear Experiments

> **Created:** 2026-02-19
> **Session purpose:** Linear schedule training + 12 eval runs completed on jabiru server. Log-linear schedule implemented locally but not yet pushed/trained. Next: analyze linear results, commit+push log-linear code, retrain with log-linear, run same 12 evals.

---

## What was accomplished

### Server work (jabiru)
- **Retrained model** with 3 post-v1 changes (SUBS zero masking, float64 ELBO, importance sampling) using **linear noise schedule**, 500 epochs on Polytechnique `jabiru` server
- **Ran full 12-experiment suite** (`bash scripts/run_experiments.sh`):
  - Layer 1: 4 baselines (random/llada x argmax/top-p, no remasking)
  - Layer 2: 5 cap eta sweep (0.2, 0.4, 0.6, 0.8, 1.0)
  - Layer 3: 3 confidence + t_switch sweep (0.3, 0.5, 0.7)
- **Results are on the server** at `/Data/amine.chraibi/Davis/BD_Generation/eval_results/*.json` — need to be copied locally

### Local code changes (NOT yet committed/pushed)
- **Added `LogLinearSchedule`** to `noise_schedule.py` — matches MDLM/ReMDM default schedule where `alpha(t) = 1 - (1-eps)*t` (masking probability increases linearly)
- **Added `configs/noise/loglinear.yaml`** — `type: loglinear, eps: 1e-3`
- **Bifurcated eval_results** into per-schedule subdirectories: `eval_results/linear/` and `eval_results/loglinear/`
- **Updated `evaluate.py`** — saves to `eval_results/<noise_schedule_type>/` based on `cfg.noise.type`
- **Updated `compare.py`** — accepts `--schedule` arg for per-schedule comparison
- **Updated `run_experiments.sh`** — accepts schedule as first argument: `bash scripts/run_experiments.sh loglinear`
- **Added 11 tests** for LogLinearSchedule (544 total pass, 2 skip, 1 pre-existing warning)
- **Updated `implementation_state_T1.md`** — retraining section now covers both schedules
- **Updated `docs/training.md`** — Section 14 with v2 training observations (high loss with IS, validation as reliable metric, importance sampling rationale, jabiru server)

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Linear schedule first | Complete training + eval before switching | Already running; provides baseline for comparison |
| Log-linear schedule | `alpha(t) = 1 - (1-eps)*t`, eps=1e-3 | MDLM/ReMDM default; our linear schedule has 99.3% masking at t=0.5 vs 50% for log-linear |
| Eval results bifurcation | `eval_results/<schedule>/` subdirs | Clean separation; each gets own comparison.md |
| Server switch | albatros → jabiru | `/Data` partition on albatros was 100% full (1TB) |
| wandb | Disabled on jabiru | wandb package corruption from rsync; not blocking |

## Current state of the codebase

- **Local**: 9 modified files + 1 new config, all uncommitted. 544 tests pass.
- **Server (jabiru)**: Has OLD code (pre-loglinear). Linear results in `eval_results/` (flat, not in `linear/` subdir). New code needs to be pushed+pulled before log-linear run.
- **Linear eval results**: On server only, need to be copied to local `eval_results/linear/`
- **IMPORTANT**: The linear run on jabiru used the OLD evaluate.py (saves to `eval_results/` flat). After pulling new code, results need to be moved to `eval_results/linear/` before running log-linear.

### Training observations (linear schedule, epoch 369-499)
- Training loss ~41-55 (expected with importance sampling — 15x higher due to w(t) inflation)
- Validation loss ~2.5-2.8 (comparable to v1's 2.4)
- Node accuracy ~20-22% (v1 final: 28.9%)
- Edge accuracy ~6-9% (v1 final: 27.5%) — **concerning but may be v1 was measured at different t**
- Oscillation in both train and val loss is normal (random t sampling)

## What remains to be done

### Immediate (next session)
1. **Copy linear results from jabiru** to local:
   ```bash
   mkdir -p BD_Generation/eval_results/linear
   scp jabiru:/Data/amine.chraibi/Davis/BD_Generation/eval_results/*.json BD_Generation/eval_results/linear/
   ```
2. **Analyze linear results**: read `comparison.md` or generate it locally with `python scripts/compare.py --schedule linear`
3. **Commit and push** the log-linear schedule + all local changes
4. **On jabiru**: `git pull`, move old flat results to `eval_results/linear/`, then run:
   ```bash
   bash scripts/run_experiments.sh loglinear
   ```
5. **Copy log-linear results** back to local
6. **Cross-schedule comparison**: compare best methods from linear vs log-linear

### Analysis questions to answer
- Layer 1: Does llada unmasking beat random? If yes, re-run L2/L3 with llada
- Layer 2: Which cap eta value is best?
- Run 10 (manual): best eta + argmax control to isolate top-p contribution
- Layer 3: Which t_switch is best for confidence remasking?
- Layer 4: Head-to-head best cap vs best confidence
- **Critical**: Does log-linear schedule significantly improve edge accuracy?

## Files to reference in next session

**Read first (in order):**
1. `BD_Generation/eval_results/linear/comparison.md` — the main results table (after copying from server)
2. `BD_Generation/implementation_state_T1.md` — lines 507-550 for retraining status
3. `BD_Generation/docs/training.md` — Section 14 for v2 training observations
4. `BD_Generation/scripts/run_experiments.sh` — now parameterized by schedule

**Key implementation files:**
5. `BD_Generation/bd_gen/diffusion/noise_schedule.py` — LogLinearSchedule (lines 103-161)
6. `BD_Generation/scripts/evaluate.py` — schedule-aware result saving (lines 531-535)
7. `BD_Generation/scripts/compare.py` — per-schedule comparison
8. `BD_Generation/remasking_design_with_fixed_forward_process.md` — Section 10 for experiment design rationale

## Context for the next session

- **Server access**: SSH to `jabiru.polytechnique.fr` as `amine.chraibi`, env at `/Data/amine.chraibi/Davis`, activate with `source .venv/bin/activate`
- **tmux**: Use `tmux new -s experiments` for long-running tasks. Detach: `Ctrl+B, D`. Reattach: `tmux attach -t experiments`
- **wandb broken on jabiru**: Package corruption from rsync. Use `wandb.mode=disabled`. Not worth fixing now.
- **The ELBO is theoretically schedule-invariant** (MDLM Section 3.4) in continuous time, but with 100 discrete sampling steps the schedule determines how work is distributed — log-linear is expected to perform better.
- **The 2 skipped tests** (`test_loads_existing_v1_file`, `test_with_existing_files`) auto-unskip once eval result JSONs exist.
- **Pre-existing warning** in `test_integration.py:248` about lr_scheduler ordering — cosmetic.
