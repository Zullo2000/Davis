# Handoff: Phase 4 Complete → GPU Training → Phase 5

> **Created:** 2026-02-13
> **Session purpose:** Implement Phase 4 (Training Loop), prepare for GPU training and Phase 5 (Evaluation)

---

## What was accomplished

- **Phase 4 fully implemented, tested, merged to `main`, tagged `v0.5.0`**
- Created `bd_gen/utils/seed.py` — deterministic seeding (torch/numpy/random/CUDA)
- Created `bd_gen/utils/checkpoint.py` — save/load training checkpoints with OmegaConf
- Created `bd_gen/utils/logging_utils.py` — wandb init, metric logging, git hash capture
- Created `scripts/train.py` — full training loop (Hydra Compose API, validation, sampling, checkpointing, wandb)
- Created `tests/test_integration.py` — 5 integration tests (train step, loss decrease, checkpoint roundtrip, seed, LR warmup)
- Created `docs/training.md` — extensive documentation (~390 lines: architecture, config ref, usage, troubleshooting, Google Cloud quickstart)
- Created `notebooks/03_training_monitoring.ipynb` — wandb dashboard plotting
- **309/309 tests pass**, ruff clean
- **CPU debug training verified**: 2 epochs, loss decreased 7.41 → 6.37, checkpoints created

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Hydra CLI | Compose API instead of `@hydra.main` | Python 3.14 argparse incompatibility with Hydra 1.3.2 |
| Output dir | `BD_Generation/outputs/<timestamp>/` | Manual management since Compose API doesn't set up Hydra cwd |
| Windows workers | Auto-detect and force `num_workers=0` | PyTorch DataLoader multiprocessing unreliable on Windows |
| Loss decrease test | Fixed `t=0.5` for 50 steps | Random t causes high ELBO weight variance with 4-sample batch |
| Sample logging | Counts only, no images | `bd_gen/viz/` doesn't exist until Phase 5 |
| Node class weights | `None` (unweighted) in v1 | Spec says optional; edge weights are the critical ones |

## Current state of the codebase

### Package structure (Phases 0–4 complete)
```
bd_gen/
  data/       → dataset, tokenizer, vocab, graph2plan_loader (Phase 1)
  model/      → BDDenoiser, embeddings, transformer (Phase 2)
  diffusion/  → forward_mask, ELBOLoss, sample, noise schedules (Phase 3)
  utils/      → seed, checkpoint, logging_utils (Phase 4)
  eval/       → empty __init__.py (Phase 5 placeholder)
  viz/        → empty __init__.py (Phase 5 placeholder)
scripts/
  prepare_data.py  → download + cache dataset
  train.py         → full training loop
tests/
  309 tests total, all passing
```

### Git state
- Branch: `main` at `ffb683b`
- Tags: `v0.1.0` (Phase 0), `v0.3.0` (Phase 2), `v0.4.0` (Phase 3), `v0.5.0` (Phase 4)
- **Local only — nothing pushed to remote yet**

### Known issues
- Hydra 1.3.2 is the latest available; `@hydra.main` broken on Python 3.14 (workaround in place via Compose API)
- `implementation_state_T1.md` needs updating to mark Phase 4 as COMPLETE (user hasn't given explicit approval yet per CLAUDE.md rules)

## What remains to be done

### Immediate: GPU Training (user action required)
1. **User sets up Google Cloud VM** (T4 or V100) — see `docs/training.md` Section 12
2. **User had questions about Cloud VM setup** — resume this conversation thread
3. User runs full 500-epoch training: `python scripts/train.py`
4. User monitors via wandb, downloads `checkpoint_final.pt`

### Next: Phase 5 — Evaluation and Metrics
Per `planning_T1.md` (lines 913–937), branch `eval/metrics-and-validity`:
1. `bd_gen/eval/validity.py` — validity checker for generated graphs
2. `bd_gen/eval/metrics.py` — validity, novelty, diversity, distribution match
3. `scripts/evaluate.py` — generate N samples, compute all metrics, log to wandb
4. `scripts/sample.py` — generate and save/visualise samples
5. `bd_gen/viz/graph_viz.py` — bubble diagram visualisation
6. Tests: `test_validity.py`, `test_metrics.py`
7. `notebooks/04_sample_analysis.ipynb`

Phase 5 code can be developed on CPU with a debug checkpoint. Real evaluation needs the GPU-trained checkpoint.

## Files to reference in next session

1. `BD_Generation/implementation_state_T1.md` — dynamic state (read first per CLAUDE.md)
2. `BD_Generation/planning_T1.md` — static spec (Phase 5 at lines 913–937)
3. `BD_Generation/CLAUDE.md` — agent rules
4. `BD_Generation/docs/training.md` — Phase 4 docs (Google Cloud quickstart at Section 12)
5. `BD_Generation/scripts/train.py` — training script (Phase 5 uses same model/dataset interfaces)
6. `BD_Generation/bd_gen/diffusion/sampling.py` — `sample()` function (Phase 5 calls this)
7. `BD_Generation/bd_gen/data/dataset.py` — `num_rooms_distribution` attribute (needed for sampling)

## Context for the next session

### User's pending questions
- User wanted to ask about **Cloud VM setup** before proceeding with GPU training. The session ended before those questions were discussed. Resume with those questions first.

### Environment facts
- **Python 3.14.2** on Windows 11, PyTorch 2.10.0
- **Dataset**: 80,788 graphs cached at `BD_Generation/data_cache/graph2plan_nmax8.pt` (13.5MB). All graphs have 4–8 rooms.
- **Model sizes**: `model=small` ~1.28M params, `model=base` ~5M params. Both fit on a T4.
- **Training estimate**: batch_size=256 on GPU → ~252 batches/epoch × 500 epochs = ~126K steps. Should take 2–4 hours on T4.
- **wandb project**: `bd-generation`. User needs `wandb login` on the VM.

### Gotchas
- **Python 3.14 + Hydra**: If cloud VM has Python 3.14, the Compose API workaround in `train.py` handles it. If Python 3.12/3.13, both approaches work.
- **No remote push yet**: All commits are local. User needs to set up a remote and `git push` before cloning on the VM.
- **`outputs/` is gitignored**: Training outputs won't be committed.
- **Windows `num_workers=0`**: Auto-detected in `train.py`. On Linux VMs, the default `num_workers=4` from config will be used.
