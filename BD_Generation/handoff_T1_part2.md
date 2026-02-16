# Handoff: GPU Training Launched → Awaiting Checkpoint → Phase 5

> **Created:** 2026-02-16
> **Session purpose:** Set up GPU training on university SSH server, fix Python 3.9 compatibility, launch training

---

## What was accomplished

- **GitHub repo created**: `https://github.com/Zullo2000/Davis.git` (private), pushed all code
- **SSH connection to Polytechnique server**: `albatros.polytechnique.fr` via VSCode Remote-SSH
- **Python 3.9 compatibility fix**: lowered `requires-python` to `>=3.9`, added `from __future__ import annotations` to `test_tokenizer.py`, changed ruff target to `py39`. All 309 tests still pass locally.
- **Server environment set up**: venv at `/Data/amine.chraibi/Davis/.venv` with torch (cu121), all deps installed
- **Dataset cached on server**: `prepare_data.py` ran successfully
- **Training launched** (or about to launch) inside tmux: `python scripts/train.py wandb.mode=disabled`
- **Google Cloud VM attempted but abandoned**: free tier blocks GPUs, billing upgrade not completed
- **`docs/training.md` updated**: Option A (university SSH) and Option B (Google Cloud) with all corrections
- **numpy pinned to 1.26.4** on server (wandb 0.16.6 incompatible with numpy 2.0)

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| GPU server | Polytechnique `albatros` (RTX A5000, 24GB) | Google Cloud free tier blocks GPUs |
| Python compat | Lower to >=3.9, use `__future__` annotations | Server has Python 3.9.25 |
| wandb | Disabled (`wandb.mode=disabled`) | Old wandb versions reject new 86-char API keys; server pip too old to install latest wandb |
| numpy | Pinned to 1.26.4 on server | wandb 0.16.6 uses removed `np.float_` from numpy 2.0 |
| tmux | Used for training persistence | SSH disconnect would kill training otherwise |

## Current state of the codebase

### Git state
- Branch: `main` at `5ddd77f`
- **Pushed to GitHub**: `https://github.com/Zullo2000/Davis.git` (private)
- Tags: `v0.1.0`–`v0.5.0` (Phases 0–4)
- Uncommitted: `handoff_T1.md` update, `training.md` wandb/tmux updates

### Server state
- **Path**: `/Data/amine.chraibi/Davis`
- **Venv**: `/Data/amine.chraibi/Davis/.venv` (Python 3.9, torch 2.5.1+cu121)
- **GPU**: NVIDIA RTX A5000, 24GB VRAM
- **Training**: launched in tmux session `train` with `wandb.mode=disabled`
- **SSH config** (local `~/.ssh/config`):
  ```
  Host albatros
    HostName albatros.polytechnique.fr
    User amine.chraibi
    PreferredAuthentications password
    PubkeyAuthentication no
  ```

### Training output location
```
/Data/amine.chraibi/Davis/BD_Generation/outputs/<timestamp>/
├── config.yaml
└── checkpoints/
    ├── checkpoint_epoch_0049.pt
    ├── ...
    └── checkpoint_final.pt
```

### Known issues
- **wandb broken on server**: old version + new API key format incompatibility. Disabled for now.
- **Server Python is 3.9**: code works but some newer libraries may have compat issues
- `implementation_state_T1.md` still needs Phase 4 marked as COMPLETE

## What remains to be done

### Immediate: Retrieve Training Results
1. SSH into server: `Ctrl+Shift+P` → Remote-SSH → albatros
2. Check tmux: `tmux attach -t train`
3. If training finished, copy checkpoint locally:
   ```bash
   scp albatros:/Data/amine.chraibi/Davis/BD_Generation/outputs/*/checkpoints/checkpoint_final.pt .
   ```
4. Verify checkpoint loads on local machine

### Next: Phase 5 — Evaluation and Metrics
Per `planning_T1.md` (lines 913–937), branch `eval/metrics-and-validity`:
1. `bd_gen/eval/validity.py` — validity checker for generated graphs
2. `bd_gen/eval/metrics.py` — validity, novelty, diversity, distribution match
3. `scripts/evaluate.py` — generate N samples, compute all metrics
4. `scripts/sample.py` — generate and save/visualise samples
5. `bd_gen/viz/graph_viz.py` — bubble diagram visualisation
6. Tests: `test_validity.py`, `test_metrics.py`
7. `notebooks/04_sample_analysis.ipynb`

Phase 5 code can be developed on CPU with a debug checkpoint. Real evaluation needs the GPU-trained checkpoint.

## Files to reference in next session

1. `BD_Generation/handoff_T1_part2.md` — this file
2. `BD_Generation/handoff_T1.md` — previous handoff (Phase 4 context)
3. `BD_Generation/implementation_state_T1.md` — dynamic state
4. `BD_Generation/planning_T1.md` — static spec (Phase 5 at lines 913–937)
5. `BD_Generation/CLAUDE.md` — agent rules
6. `BD_Generation/docs/training.md` — training docs (Section 12: GPU setup)
7. `BD_Generation/scripts/train.py` — training script
8. `BD_Generation/bd_gen/diffusion/sampling.py` — `sample()` function (Phase 5 entry point)

## Context for the next session

### Environment facts
- **Local**: Python 3.14.2 on Windows 11, PyTorch 2.10.0
- **Server**: Python 3.9.25, PyTorch 2.5.1+cu121, RTX A5000 24GB
- **Dataset**: 80,788 graphs, cached on both local and server
- **Model**: `model=small` ~1.28M params (default), `model=base` ~5M params
- **Training estimate**: 1.5–3 hours on RTX A5000

### Gotchas
- **GitHub repo is private**: must temporarily make public to pull on server, or use token
- **`outputs/` is gitignored**: checkpoint must be transferred via `scp`, not git
- **numpy 1.26.4 on server**: pinned for wandb compat; don't upgrade
- **wandb**: if needed later, must install wandb >=0.19+ that supports new API key format, but that may need newer pip/Python
- **tmux session**: `tmux attach -t train` to reconnect; `Ctrl+B, D` to detach
