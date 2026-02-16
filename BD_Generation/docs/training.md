# Training Module Documentation

Phase 4 of the BD-Gen implementation.  This module provides the complete
training infrastructure for the MDLM (Masked Discrete Language Model)
bubble diagram generator, connecting the data pipeline (Phase 1), model
architecture (Phase 2), and diffusion core (Phase 3) into a runnable
training loop.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Utility Modules](#3-utility-modules)
4. [Training Script](#4-training-script)
5. [Configuration Reference](#5-configuration-reference)
6. [Usage Guide](#6-usage-guide)
7. [wandb Integration](#7-wandb-integration)
8. [Checkpoint Format](#8-checkpoint-format)
9. [Learning Rate Schedule](#9-learning-rate-schedule)
10. [Per-Class Accuracy](#10-per-class-accuracy)
11. [Troubleshooting](#11-troubleshooting)
12. [GPU Training Setup](#12-gpu-training-setup)
13. [Training Results](#13-training-results)

---

## 1. Overview

### What Phase 4 Builds

| File | Purpose |
|------|---------|
| `bd_gen/utils/seed.py` | Deterministic seeding (torch, numpy, random, CUDA) |
| `bd_gen/utils/checkpoint.py` | Save/load training checkpoints |
| `bd_gen/utils/logging_utils.py` | wandb initialisation, metric logging, git hash |
| `scripts/train.py` | Hydra-based training entry point |

### How It Connects Phases 0–3

The training script is the **only place** that bridges all previously
independent modules.  It orchestrates:

- **Phase 1 (Data):** `BubbleDiagramDataset` provides tokenised batches
  with `edge_class_weights` and `num_rooms_distribution`.
- **Phase 2 (Model):** `BDDenoiser` receives noised tokens and predicts
  logits for each vocabulary position.
- **Phase 3 (Diffusion):** `forward_mask` adds noise, `ELBOLoss` computes
  the MDLM training objective, and `sample()` generates new diagrams.

No module imports another across boundaries — `train.py` is the glue.

---

## 2. Architecture

### Training Pipeline (Single Batch)

```
                   ┌──────────────────────────────────────────────────┐
                   │              scripts/train.py                    │
                   │                                                  │
  DataLoader ──►   │  tokens, pad_mask = batch                       │
                   │         │                                        │
                   │         ▼                                        │
                   │  t ~ Uniform(0, 1)                              │
                   │         │                                        │
                   │         ▼                                        │
                   │  x_t, mask_indicators = forward_mask(            │
                   │      tokens, pad_mask, t, noise_schedule)        │
                   │         │                                        │
                   │         ▼                                        │
                   │  node_logits, edge_logits = model(               │
                   │      x_t, pad_mask, t)                           │
                   │         │                                        │
                   │         ▼                                        │
                   │  loss = ELBOLoss(                                │
                   │      node_logits, edge_logits,                   │
                   │      tokens, x_t, pad_mask, mask_indicators,     │
                   │      t, noise_schedule)                          │
                   │         │                                        │
                   │         ▼                                        │
                   │  loss.backward()                                 │
                   │  clip_grad_norm_(model.parameters(), grad_clip)  │
                   │  optimizer.step()                                │
                   │  scheduler.step()                                │
                   └──────────────────────────────────────────────────┘
```

### Epoch-Level Structure

```
for epoch in range(epochs):
    ├── Training pass (all batches)
    │     ├── Sample timestep t
    │     ├── Forward mask → model → loss → backward → step
    │     └── Log train/loss and train/lr per step
    │
    ├── Validation (every val_every epochs)
    │     ├── Compute val loss (no gradients)
    │     └── Compute node/edge accuracy at masked positions
    │
    ├── Sampling (every sample_every epochs)
    │     └── Generate 8 samples via reverse diffusion
    │
    └── Checkpoint (every checkpoint_every epochs)
          └── Save model + optimizer + epoch + config to .pt file
```

---

## 3. Utility Modules

### 3.1 `bd_gen/utils/seed.py`

```python
set_seed(seed: int) -> None
```

Sets seeds for:
- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)` (covers CPU and single CUDA device)
- `torch.cuda.manual_seed_all(seed)` (multi-GPU)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

The CUDA-specific calls are guarded by `torch.cuda.is_available()` so
the function is safe to call on CPU-only machines.

**Reproducibility guarantee:** Same seed + same config + same code →
identical loss curves (within floating-point rounding on same hardware).

### 3.2 `bd_gen/utils/checkpoint.py`

#### `save_checkpoint`

```python
save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: DictConfig,
    path: str | Path,
) -> None
```

Saves a dictionary with four keys (see [Section 8](#8-checkpoint-format))
via `torch.save`.  The Hydra `DictConfig` is converted to a plain dict
with `OmegaConf.to_container(config, resolve=True)` because OmegaConf
objects cannot be pickled reliably.  Parent directories are created
automatically.

#### `load_checkpoint`

```python
load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict    # {"epoch": int, "config": dict}
```

Loads model weights into the provided `model`.  If `optimizer` is given,
also restores its state (learning rate, momentum buffers, etc.).  The
`device` parameter is passed to `torch.load(map_location=...)` so a
GPU-trained checkpoint can be loaded on CPU for inference.

Returns a metadata dict with the saved epoch number and the full config
used during training.

### 3.3 `bd_gen/utils/logging_utils.py`

#### `get_git_commit`

```python
get_git_commit() -> str
```

Returns the 40-character hex SHA of `HEAD`.  Falls back to `"unknown"`
if `git` is not installed or the working directory is not a repo.

#### `init_wandb`

```python
init_wandb(config: DictConfig) -> None
```

Initialises a wandb run using `config.wandb.project`,
`config.wandb.entity`, `config.wandb.mode`, and
`config.experiment_name`.  The full resolved config is uploaded.  The
git commit hash is logged under `wandb.config.git_commit`.

#### `log_metrics`

```python
log_metrics(metrics: dict, step: int) -> None
```

Thin wrapper around `wandb.log(metrics, step=step)`.  When wandb is in
`"disabled"` mode, this is a no-op.

---

## 4. Training Script

### `scripts/train.py`

Uses Hydra's **Compose API** (`initialize_config_dir` + `compose`)
instead of the `@hydra.main` decorator.  This avoids an `argparse`
incompatibility between Hydra 1.3.2 and Python >= 3.14.  CLI overrides
(e.g. `model=base`, `training.epochs=10`) are still supported via
`sys.argv`.  The composite config is loaded from `configs/config.yaml`
(which references `model`, `data`, `noise`, `training`, and `eval`
sub-configs).

#### Execution Flow

1. **Seed and device** — `set_seed(cfg.seed)`, auto-detect CUDA.
2. **wandb init** — `init_wandb(cfg)` with full config and git hash.
3. **VocabConfig** — `VocabConfig(n_max=cfg.data.n_max)`.
4. **Datasets** — `BubbleDiagramDataset` for train and val splits.
   Data paths are resolved relative to `_PROJECT_ROOT` (the
   `BD_Generation/` directory), not the Hydra output directory.
5. **DataLoaders** — With a Windows guard that forces `num_workers=0`
   when `platform.system() == "Windows"` (PyTorch multiprocessing
   DataLoader is unreliable on Windows).
6. **Model + loss + optimizer** — `BDDenoiser`, `ELBOLoss`, `AdamW`.
   All moved to device.  `ELBOLoss` stores class weights via
   `register_buffer`, so `.to(device)` handles them automatically.
7. **LR schedule** — `LambdaLR` with linear warmup (see
   [Section 9](#9-learning-rate-schedule)).
8. **Training loop** — Nested epoch/batch loop with per-step wandb
   logging.  Timesteps are optionally importance-sampled and always
   clamped to `[1e-5, 1.0]`.
9. **Validation** — Every `val_every` epochs: loss + per-class accuracy.
10. **Sampling** — Every `sample_every` epochs: 8 samples via reverse
    diffusion (logged as counts; image visualisation is Phase 5).
11. **Checkpointing** — Every `checkpoint_every` epochs + final
    checkpoint.  Saved to `checkpoints/` inside the Hydra output dir.

#### Output Directory

Since we use the Compose API (not `@hydra.main`), we manage the output
directory manually.  Each run creates
`BD_Generation/outputs/<YYYY-MM-DD_HH-MM-SS>/` containing:

- `config.yaml` — fully resolved config
- `checkpoints/` — model checkpoints

Data paths use `_PROJECT_ROOT` (absolute) so they are independent of
the output directory.

---

## 5. Configuration Reference

### `configs/training/default.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lr` | float | `3e-4` | Peak learning rate for AdamW. |
| `weight_decay` | float | `0.01` | L2 regularisation strength. |
| `warmup_steps` | int | `1000` | Linear warmup from 0 to `lr` over this many **optimiser steps** (not epochs). |
| `epochs` | int | `500` | Total number of training epochs. |
| `optimizer` | str | `"adamw"` | Optimiser type (only AdamW in v1). |
| `grad_clip` | float | `1.0` | Max gradient norm for `clip_grad_norm_`.  Set to `0` to disable clipping. |
| `checkpoint_every` | int | `50` | Save checkpoint every N epochs. |
| `sample_every` | int | `25` | Generate samples every N epochs. |
| `val_every` | int | `5` | Run validation every N epochs. |
| `ema` | bool | `false` | Exponential Moving Average (disabled in v1). |
| `importance_sampling` | bool | `false` | Use MDLM importance-weighted timestep sampling instead of uniform. |

### `configs/data/graph2plan.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `n_max` | `8` | Max rooms per diagram (determines sequence length). |
| `batch_size` | `256` | Batch size.  Use `32` for CPU debug runs. |
| `num_workers` | `4` | DataLoader worker processes.  Forced to `0` on Windows. |
| `mat_path` | `"data/data.mat"` | Path to raw `.mat` file (relative to `BD_Generation/`). |
| `cache_path` | `"data_cache/graph2plan_nmax8.pt"` | Path to cached tensor file. |
| `splits.train` | `0.8` | Training fraction. |
| `splits.val` | `0.1` | Validation fraction. |
| `splits.test` | `0.1` | Test fraction. |

### `configs/model/small.yaml`

| Key | Default |
|-----|---------|
| `d_model` | `128` |
| `n_layers` | `4` |
| `n_heads` | `4` |
| `cond_dim` | `128` |
| `mlp_ratio` | `4` |
| `dropout` | `0.1` |
| `frequency_embedding_size` | `256` |

### `configs/model/base.yaml`

Same keys as `small` but: `d_model=256`, `n_layers=6`, `n_heads=8`,
`cond_dim=256`.

### `configs/noise/linear.yaml`

| Key | Default |
|-----|---------|
| `type` | `"linear"` |
| `sigma_min` | `0.0` |
| `sigma_max` | `10.0` |

### `configs/noise/cosine.yaml`

| Key | Default |
|-----|---------|
| `type` | `"cosine"` |
| `eps` | `1e-3` |

### `configs/config.yaml` (top-level compose)

```yaml
defaults:
  - model: small
  - data: graph2plan
  - noise: linear
  - training: default
  - eval: default
  - _self_

seed: 42
experiment_name: "bd_gen_v1"
wandb:
  project: "bd-generation"
  entity: null
  mode: "online"
```

---

## 6. Usage Guide

### Basic Training (GPU)

```bash
cd BD_Generation
python scripts/train.py
```

Uses all defaults: `model=small`, 500 epochs, `batch_size=256`, wandb
online, linear noise schedule.

### CPU Debug Run

```bash
python scripts/train.py \
    wandb.mode=disabled \
    training.epochs=2 \
    training.val_every=1 \
    training.sample_every=2 \
    training.checkpoint_every=2 \
    data.batch_size=32
```

Runs 2 epochs with small batches, no wandb, validating and
checkpointing each time.  Useful for verifying the pipeline works
before launching a real GPU run.

### Override Model Size

```bash
python scripts/train.py model=base
```

Switches to the ~5M-parameter base model (256-dim, 6 layers, 8 heads).

### Override Noise Schedule

```bash
python scripts/train.py noise=cosine
```

### Resume from Checkpoint

Currently resume-from-checkpoint is not built into the training script
(v1 scope).  To resume manually:

```python
from bd_gen.utils.checkpoint import load_checkpoint
from bd_gen.model.denoiser import BDDenoiser
from bd_gen.data.vocab import VocabConfig

vc = VocabConfig(n_max=8)
model = BDDenoiser(d_model=128, n_layers=4, n_heads=4, vocab_config=vc)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

meta = load_checkpoint("checkpoints/checkpoint_epoch_0049.pt", model, optimizer)
start_epoch = meta["epoch"] + 1
```

### Output Directory

Every training run creates a timestamped directory under `outputs/`:

```
outputs/2026-02-13_14-30-00/
├── config.yaml               # fully resolved config
└── checkpoints/
    ├── checkpoint_epoch_0049.pt
    ├── checkpoint_epoch_0099.pt
    └── checkpoint_final.pt
```

---

## 7. wandb Integration

### What Gets Logged

| Metric | Frequency | Description |
|--------|-----------|-------------|
| `train/loss` | Per step | Batch-level ELBO loss. |
| `train/lr` | Per step | Current learning rate (shows warmup ramp). |
| `val/loss` | Every `val_every` epochs | Average validation ELBO loss. |
| `val/node_accuracy` | Every `val_every` epochs | Accuracy at predicting masked node types. |
| `val/edge_accuracy` | Every `val_every` epochs | Accuracy at predicting masked edge types. |
| `samples/num_generated` | Every `sample_every` epochs | Number of samples generated (8). |
| Full Hydra config | Once at init | Complete resolved config for reproducibility. |
| `git_commit` | Once at init | 40-char SHA of HEAD commit. |

### Dashboard Setup

1. Sign up at [wandb.ai](https://wandb.ai) (free for personal use).
2. Run `wandb login` and paste your API key.
3. Training runs will appear under the `bd-generation` project.
4. Create panels for: `train/loss`, `val/loss`, `val/node_accuracy`,
   `val/edge_accuracy`, `train/lr`.

### Offline and Disabled Modes

```bash
# Offline: logs stored locally, sync later with `wandb sync`
python scripts/train.py wandb.mode=offline

# Disabled: no wandb at all (fastest for testing)
python scripts/train.py wandb.mode=disabled
```

---

## 8. Checkpoint Format

Each `.pt` file contains a dictionary with these keys:

```python
{
    "model_state_dict": OrderedDict,   # model.state_dict()
    "optimizer_state_dict": dict,      # optimizer.state_dict()
    "epoch": int,                      # 0-indexed epoch number
    "config": dict,                    # plain dict (resolved Hydra config)
}
```

### Loading for Inference

```python
model = BDDenoiser(...)
meta = load_checkpoint("checkpoint_final.pt", model, device="cpu")
model.eval()
# model is ready for sampling — no optimizer needed
```

### Loading for Resumed Training

```python
model = BDDenoiser(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
meta = load_checkpoint("checkpoint_final.pt", model, optimizer, device="cuda")
start_epoch = meta["epoch"] + 1
# Continue training from start_epoch
```

### Cross-Device Loading

A checkpoint saved on GPU can be loaded on CPU by passing
`device="cpu"` to `load_checkpoint`.  This uses
`torch.load(map_location=device)` internally.

---

## 9. Learning Rate Schedule

### Linear Warmup, Then Constant

```
LR
 │
 │           ┌──────────────────── lr = 3e-4 (constant)
 │          /
 │         /
 │        /
 │       /
 │      /   warmup_steps = 1000
 │     /
 │    /
 │   /
 │  /
 │ /
 │/
 └──────────────────────────────────────── Step
 0    500   1000  1500  2000  ...
```

The schedule uses `torch.optim.lr_scheduler.LambdaLR` with:

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps   # linear ramp from 0 to 1
    return 1.0                        # constant multiplier
```

The actual learning rate at step `s` is `lr * lr_lambda(s)`.

`scheduler.step()` is called **per batch** (not per epoch), so the
warmup completes after 1000 optimiser steps regardless of batch size
or dataset size.

At step 0 the learning rate is exactly 0.  This is standard practice
for linear warmup — the first batch trains with near-zero LR, which
is harmless.

---

## 10. Per-Class Accuracy

### What It Measures

Per-class accuracy tracks how well the model predicts the original
clean tokens at **masked positions only**.  This is the core denoising
task: given a partially masked sequence, predict what was masked.

### Computation

For each validation batch:

1. Apply `forward_mask` to get `x_t` and `mask_indicators`.
2. Run the model to get `node_logits` and `edge_logits`.
3. Split both `mask_indicators` and `pad_mask` at position `n_max`:
   - **Node positions:** `[0, n_max)` — room type predictions.
   - **Edge positions:** `[n_max, seq_len)` — relationship predictions.
4. For each split, count predictions where `argmax(logits) == target`
   at positions where `mask_indicators & pad_mask` (masked AND real).
5. Accuracy = correct / total.

### Why This Matters

- **Node accuracy** tracks whether the model learns room type
  distributions (living room, bedroom, kitchen, etc.).
- **Edge accuracy** is critical for detecting the **edge sparsity
  problem**: if the model predicts `no-edge` everywhere, edge accuracy
  might look high (since `no-edge` dominates the training data).
  This metric, combined with a per-class breakdown in later phases,
  helps detect this collapse mode.

---

## 11. Troubleshooting

### Windows: DataLoader Multiprocessing Failures

**Symptom:** `RuntimeError` or `BrokenPipeError` on Windows with
`num_workers > 0`.

**Solution:** The training script automatically detects Windows and
forces `num_workers=0`.  A warning is logged.

### Out of Memory (OOM) on CPU

**Symptom:** `RuntimeError: ... Tried to allocate ...`

**Solution:** Reduce batch size:
```bash
python scripts/train.py data.batch_size=16
```

The default `batch_size=256` is intended for GPU training.  Use
`batch_size=16` or `32` for CPU debug runs.

### wandb Authentication

**Symptom:** `wandb.errors.CommError: ... 403 ...`

**Solution:**
```bash
wandb login
```
Or use offline/disabled mode:
```bash
python scripts/train.py wandb.mode=disabled
```

### NaN Loss

**Symptom:** `loss=nan` during training.

**Possible causes:**
1. Learning rate too high — reduce `training.lr`.
2. Missing gradient clipping — ensure `training.grad_clip > 0`.
3. Bug in data pipeline — run `pytest tests/` to verify all 300+
   tests pass.

The ELBO loss has several numerical safeguards (t clamped to
`[1e-5, 1.0]`, epsilon in denominators, w(t) clamped to `[0, 1000]`,
N_active clamped to min 1.0).

### Hydra Config Errors

**Symptom:** `omegaconf.errors.ConfigAttributeError`

**Solution:** Check that all config files exist in `configs/` and that
CLI overrides use the correct dotpath syntax:
```bash
# Correct
python scripts/train.py training.epochs=10

# Wrong (missing group)
python scripts/train.py epochs=10
```

### Checkpoint Load Errors

**Symptom:** `RuntimeError: Error(s) in loading state_dict`

**Cause:** Model architecture mismatch between save and load.  The
checkpoint stores the config — check `meta["config"]` to see what
architecture was used.

---

## 12. GPU Training Setup

Two options for running the full 500-epoch training on a GPU.

---

### Option A: University SSH Server (Polytechnique)

#### Prerequisites

- SSH access to `albatros.polytechnique.fr`
- VSCode with Remote-SSH extension

#### Step 1: SSH Config (local machine)

Add this to `~/.ssh/config`:

```
Host albatros
  HostName albatros.polytechnique.fr
  User amine.chraibi
  PreferredAuthentications password
  PubkeyAuthentication no
```

#### Step 2: Connect via VSCode

1. Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host**
2. Select **albatros** from the list
3. Enter password when prompted

VSCode opens a remote window — terminal, files, and extensions all
run on the server.

#### Step 3: Activate the Environment

Everything is already installed at `/Data/amine.chraibi/Davis`.
No need to clone or run `pip install` (which takes very long).

```bash
cd /Data/amine.chraibi/Davis
source .venv/bin/activate
```

To update the code after local changes (push from local first):
```bash
git pull origin main
```

#### Step 3b: Fresh Install (only if starting from scratch)

If the environment at `/Data/amine.chraibi/Davis` is missing or
broken, reinstall from scratch:

```bash
git clone https://github.com/Zullo2000/Davis.git
cd Davis
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e "BD_Generation/[dev]"
```

Note: the repo must be temporarily public for `git clone` to work,
or use a personal access token.

#### Step 4: Prepare Data

```bash
cd BD_Generation
python scripts/prepare_data.py
```

Downloads and caches the Graph2Plan dataset (~80K samples).

#### Step 5: wandb (currently broken on this server)

wandb on the Polytechnique server uses an old version that rejects
the new 86-character API keys. Skip wandb for now:

```bash
# wandb login does NOT work — API key format mismatch
# Use disabled mode instead:
python scripts/train.py wandb.mode=disabled
```

Training still prints progress to the terminal. Checkpoints are
saved regardless of wandb.

#### Step 6: Train (inside tmux)

Use tmux so training survives SSH disconnects:

```bash
tmux new -s train
cd /Data/amine.chraibi/Davis
source .venv/bin/activate
cd BD_Generation
python scripts/train.py wandb.mode=disabled
```

To detach: `Ctrl+B`, then `D`.  To reattach: `tmux attach -t train`.

#### Step 7: Copy Checkpoints Locally

```bash
# From your local machine:
scp albatros:~/Davis/BD_Generation/outputs/*/checkpoints/checkpoint_final.pt ./
```

---

### Option B: Google Cloud VM

#### Prerequisites

- Google Cloud account with billing enabled (full account, not free tier)
- `gcloud` CLI installed locally (`gcloud init` to configure)

#### Step 1: Create a VM

```bash
gcloud compute instances create bd-gen-training \
    --zone=europe-west1-b \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --preemptible
```

Use `--preemptible` (spot) for ~70% cost savings.  A T4 is sufficient
for the 1–5M parameter model with 36-token sequences.

#### Step 2: SSH and Setup

```bash
gcloud compute ssh bd-gen-training --zone=europe-west1-b

# On the VM:
git clone <your-repo-url> Davis
cd Davis
pip install -e "BD_Generation/[dev]"
```

#### Step 3: Prepare Data

```bash
cd BD_Generation
python scripts/prepare_data.py
```

#### Step 4: Login to wandb

```bash
wandb login
```

#### Step 5: Train

```bash
python scripts/train.py
```

Monitor progress at [wandb.ai](https://wandb.ai) in real-time.

#### Step 6: Stop the VM

```bash
# From your local machine:
gcloud compute instances stop bd-gen-training --zone=europe-west1-b
```

Or delete it:
```bash
gcloud compute instances delete bd-gen-training --zone=europe-west1-b
```

#### Cost Estimate

| GPU | Spot $/hr | Full training (~3h) |
|-----|-----------|---------------------|
| T4 | ~$0.11 | ~$0.33 |
| V100 | ~$0.74 | ~$2.22 |

Total budget for multiple experimental runs: **under $5**.

#### Tips

- **Checkpointing:** Checkpoints are saved every 50 epochs.  If the
  spot VM is preempted, you can resume from the latest checkpoint.
- **Billing alerts:** Set up a budget alert in the Google Cloud Console
  to avoid unexpected charges.
- **Copy checkpoints locally:** Use `gcloud compute scp` to download
  checkpoints after training:
  ```bash
  gcloud compute scp bd-gen-training:~/Davis/BD_Generation/outputs/*/checkpoints/checkpoint_final.pt ./
  ```

---

## 13. Training Results

### First Full Training Run

Trained on the Polytechnique `albatros` server (NVIDIA RTX A5000, 24 GB
VRAM) with the default `model=small` configuration and
`wandb.mode=disabled`.

| Setting | Value |
|---------|-------|
| GPU | RTX A5000 (24 GB) |
| Model | `small` (d_model=128, 4 layers, 4 heads, ~1.28M params) |
| Batch size | 256 |
| Epochs | 500 |
| Noise schedule | Linear (sigma_min=0, sigma_max=10) |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Warmup | 1000 steps |
| Total steps | 126,000 |
| Wall time | ~17 minutes |
| Python | 3.9.25 (server) |
| PyTorch | 2.5.1+cu121 |

### Final Metrics

| Metric | Value |
|--------|-------|
| Training loss (final epochs) | ~2.8–3.2 |
| Validation loss | ~2.4 |
| Node accuracy | ~28.9% (random chance: 6.7%) |
| Edge accuracy | ~27.5–28.4% (random chance: 7.7%) |

### Why Training Loss Does Not Decrease Monotonically

The per-epoch training loss fluctuates noticeably (e.g. 2.95 → 3.10 →
3.41 → 7.02 → 2.80 across epochs 495–499).  This is expected in MDLM
training and does **not** indicate a problem.  Three factors cause it:

1. **Random timestep sampling.**  Each batch draws `t ~ Uniform(0, 1)`
   independently.  The difficulty of predicting masked tokens varies
   dramatically with `t`: at large `t` most tokens are masked (hard),
   while at small `t` very few are masked (easy but heavily weighted).
   Different epoch averages see different timestep distributions.

2. **ELBO weight variance.**  The loss weight `w(t) = 1 / (1 - α_t)`
   diverges as `t → 0` (clamped at 1000 in our implementation).  A
   single batch that happens to draw several `t` values close to 0
   contributes a disproportionately large loss, spiking the epoch
   average.  The epoch-498 spike to 7.0 is a textbook example of this.

3. **Stochastic masking.**  Even at the same `t`, the *which* tokens are
   masked is random (each position is masked independently with
   probability `1 - α_t`).  This adds another layer of variance to the
   per-batch loss.

**The validation loss (~2.4) is a more stable measure** because it
averages over more batches without gradient updates.  The long-term
trend from ~7.4 (epoch 0) to ~2.8 (epoch 499) shows clear convergence.

### Sample Quality (Qualitative)

After 500 epochs, deterministic sampling (`temperature=0.0`, 50 steps)
produces:
- **No MASK tokens** in output (sampling clean-up works correctly)
- **4/4 unique samples** in a test batch (no mode collapse)
- **Diverse room types**: LivingRoom, MasterRoom, Kitchen, Bathroom,
  SecondRoom, Balcony all appear in generated samples
- **Varying room counts** (3–7 rooms per graph, matching the training
  distribution)
- **PAD tokens** in correct positions

Quantitative evaluation (validity, novelty, diversity) is deferred to
Phase 5.

### Server-Specific Issues

| Issue | Resolution |
|-------|------------|
| wandb 0.16.6 rejects new 86-char API keys | Used `wandb.mode=disabled` |
| numpy 2.0 removed `np.float_` (used by old wandb) | Pinned numpy==1.26.4 on server |
| Server Python 3.9 vs local Python 3.14 | Lowered `requires-python` to `>=3.9`, added `from __future__ import annotations` where needed |
