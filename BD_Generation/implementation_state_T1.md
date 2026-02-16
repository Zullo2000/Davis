
# Implementation State

> Updated after each phase. Coordinator reads this + the spec before starting work.
> Rule: keep each phase summary under 60 lines. Capture decisions and deviations, not raw logs.


## Overall Status
- Current phase: Phase 5 (NOT STARTED)
- Last completed: Phase 4
- Spec corrections: vocab.py NODE_TYPES/EDGE_TYPES name-to-index mappings corrected (Phase 1 Step 0); loss_mask formula corrected (Phase 3)

## Phase 0 — Scaffold
Status: COMPLETE
Branch: `setup/repo-scaffold` → merged to `main`, tagged `v0.1.0`

### Deliverables (all verified)
- `pip install -e "BD_Generation/[dev]"` succeeds (Python 3.14.2, PyTorch 2.10.0)
- `from bd_gen.data.vocab import NODE_VOCAB_SIZE, RPLAN_VOCAB_CONFIG` works
- 54/54 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean
- All 8 Hydra YAML configs parse with OmegaConf

### Files created (25 new, 1 modified)
- `.gitignore` — rewritten with whitelist pattern (`/*` then `!/BD_Generation/`)
- `pyproject.toml`, `Makefile`, `README.md` — build tooling
- `bd_gen/` — 7 subpackages with `__init__.py` files
- `bd_gen/data/vocab.py` — all vocab constants, VocabConfig, pad mask, edge mapping
- `tests/test_vocab.py`, `tests/conftest.py` — 54 tests + shared fixtures
- `configs/` — 8 YAML files (model, data, noise, training, eval)
- `docs/vocab.md` — module documentation
- `scripts/.gitkeep`, `notebooks/.gitkeep` — placeholder dirs

### Deviations from spec
- None. All implementations match planning_T1.md exactly.

### Issues resolved
- hydra-core 1.3.2 installs fine on Python 3.14.2 (no compatibility issue)
- All dependencies resolved without conflicts

### Key decisions
- `.gitignore` uses `/*` exclusion pattern per spec Section 11.1
- Existing BD_Generation/ markdown files committed alongside scaffold
- `dummy_model()` fixture uses `pytest.skip()` until Phase 2

## Phase 1 — Data Pipeline
Status: COMPLETE
Branch: `data/graph2plan-loader` → merged to `main`

### Deliverables (all verified)
- 167/167 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean (all source, tests, scripts)
- `prepare_data.py` runs end-to-end: 80,788 graphs parsed, cached as `.pt`
- DataLoader smoke test passes (batch iteration works)

### Files created/modified (8 new, 1 modified)
- `bd_gen/data/vocab.py` — MODIFIED: corrected NODE_TYPES/EDGE_TYPES names (verified against Graph2Plan source)
- `bd_gen/data/graph2plan_loader.py` — NEW: .mat parser with caching, self-loop filtering, validation
- `bd_gen/data/tokenizer.py` — NEW: tokenize/detokenize with PAD vs no-edge invariant
- `bd_gen/data/dataset.py` — NEW: BubbleDiagramDataset with splits, class weights, num_rooms_distribution
- `tests/test_loader.py` — NEW: 23 tests (synthetic + real data integration)
- `tests/test_tokenizer.py` — NEW: 56 tests (roundtrip, PAD correctness, edge cases)
- `tests/test_dataset.py` — NEW: 34 tests (splits, weights, PAD invariant, DataLoader)
- `scripts/prepare_data.py` — NEW: auto-download + parse + cache + stats

### Key decisions
- Data is 0-based; no index subtraction needed (verified: min(rType)=0, max(rType)=12)
- Edge inverse: `9 - r` (verified against symmetric pairing in Graph2Plan vocab)
- Graph2Plan room types 13/14 repurposed as MASK/PAD (never appear in bubble data)
- n_max filtering not needed for RPLAN (all 80,788 graphs have 4-8 rooms)
- Class weights: inverse-frequency, PAD excluded; MASK class gets weight 0
- num_rooms_distribution: normalised histogram from training split, index k = P(k+1 rooms)

### Deviations from spec
- None. All implementations match planning_T1.md specs.

## Phase 2 — Model Architecture
Status: COMPLETE
Branch: `model/transformer-denoiser` → merged to `main`, tagged `v0.3.0`

### Deliverables (all verified)
- 222/222 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean
- `from bd_gen.model import BDDenoiser` works
- Small config: ~1.28M params, Base config: ~5.0M params (within 1-5M target)
- Forward shapes verified: `(4, 8, 15)` node logits + `(4, 28, 13)` edge logits

### Files created/modified (7 new, 2 modified)
- `bd_gen/model/embeddings.py` — NEW: NodeEmbedding, EdgeEmbedding, CompositePositionalEncoding, TimestepEmbedding
- `bd_gen/model/transformer.py` — NEW: MultiHeadSelfAttention, AdaLNBlock (adaLN-Zero)
- `bd_gen/model/denoiser.py` — NEW: BDDenoiser top-level model (11-step forward pass)
- `bd_gen/model/__init__.py` — MODIFIED: exports all 7 public classes
- `tests/test_embeddings.py` — NEW: 27 tests across all 4 embedding classes
- `tests/test_denoiser.py` — NEW: 28 tests (shapes, gradients, zero-init, PAD mask, timestep)
- `tests/conftest.py` — MODIFIED: `dummy_model()` returns real BDDenoiser(d_model=32, n_layers=1, n_heads=2)
- `docs/model.md` — NEW: detailed architecture documentation (516 lines)

### Key decisions
- Attention: `F.scaled_dot_product_attention` (only 36 tokens; flash unnecessary)
- adaLN-Zero: zero-init weights AND bias → identity modulation + zero gate at init
- Final layer: adaLN with 2 params (shift+scale), no gate (no residual at final layer)
- Classification heads: zero-init weights AND bias → uniform initial logits
- PAD mask: `(B,1,1,S)` float additive mask with -inf, broadcasts across heads and queries
- QKV: single `Linear(d_model, 3*d_model)` combined projection
- Positional encoding: composite learned (entity_type + node_index + pair_index), not sinusoidal
- Outer SiLU applied in BDDenoiser.forward() before adaLN, following DiT/DiDAPS convention

### Deviations from spec
- None. All implementations match planning_T1.md specs.

### Issues resolved
- `test_different_pad_masks_produce_different_outputs` failed initially because zero-init gates cause blocks to contribute nothing → fixed by randomizing all weights to simulate trained model

## Phase 3 — Diffusion Core
Status: COMPLETE
Branch: `diffusion/noise-and-loss` → merged to `main`, tagged `v0.4.0`

### Deliverables (all verified)
- 304/304 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean (all source and tests)
- `from bd_gen.diffusion import forward_mask, ELBOLoss, sample, get_noise` works
- Full pipeline: forward_mask → BDDenoiser → ELBOLoss → backward() flows gradients
- Sampling: `sample(model, schedule, vc, B=4, num_steps=10)` produces (4, 36) with no MASK tokens

### Files created/modified (8 new, 1 modified)
- `bd_gen/diffusion/noise_schedule.py` — NEW: NoiseSchedule ABC, LinearSchedule, CosineSchedule, get_noise factory
- `bd_gen/diffusion/forward_process.py` — NEW: forward_mask with PAD protection
- `bd_gen/diffusion/loss.py` — NEW: ELBOLoss with dual-vocabulary CE, ELBO weighting, per-sample normalization
- `bd_gen/diffusion/sampling.py` — NEW: reverse sampling loop with guidance/inpainting/remasking hooks
- `bd_gen/diffusion/__init__.py` — MODIFIED: exports all 7 public symbols
- `tests/test_forward_process.py` — NEW: 34 tests (schedules, forward mask, PAD stress, masking rate)
- `tests/test_loss.py` — NEW: 27 tests (PAD exclusion, class weighting, adversarial cases, gradient balance)
- `tests/test_sampling.py` — NEW: 21 tests (shapes, temperature, num_rooms, PAD, guidance, adversarial)
- `tests/conftest.py` — MODIFIED: added 4 diffusion fixtures (linear_schedule, cosine_schedule, edge_class_weights, elbo_loss)
- `docs/diffusion.md` — NEW: module documentation

### Key decisions
- NoiseSchedule inherits abc.ABC + nn.Module (follows DiDAPS pattern, enables register_buffer for device handling)
- CosineSchedule overrides alpha(t) with direct formula (avoids exp(-(-log(x))) numerical roundtrip)
- Importance sampling: clamp sigma_min ≥ 1e-4 to avoid log(0) when sigma_min=0
- loss_mask = mask_indicators AND pad_mask (CORRECTED spec typo: spec said AND NOT pad_mask)
- forward_mask and ELBOLoss accept vocab_config parameter (spec omitted it but n_max is needed for node/edge split)
- guidance_fn receives/returns (node_logits, edge_logits) tuple (spec said single tensor but model returns tuple)
- Gumbel noise in float64 for numerical stability
- w(t) double-clamped: t ≥ 1e-5 AND w ≤ 1000
- N_active clamped to min 1.0 for safe division

### Deviations from spec
- Added `vocab_config` parameter to `forward_mask` and `ELBOLoss` (spec signatures omitted it)
- Corrected loss_mask formula: `mask_indicators & pad_mask` (not `mask_indicators & ~pad_mask`)
- guidance_fn receives tuple of logits, not single tensor (matches model output structure)

### Issues resolved
- importance_sampling_transformation NaN at t=1 when sigma_min=0 (0 × -inf = NaN) → clamped sigma_min ≥ 1e-4
- Degenerate Gumbel test: zero-init model has near-uniform logits → tested with synthetic peaked logits instead

## Phase 4 — Training Loop
Status: COMPLETE
Branch: `training/basic-loop` → merged to `main`, tagged `v0.5.0`

### Deliverables (all verified)
- 309/309 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean
- CPU debug training: 2 epochs, loss decreased 7.41 → 6.37, checkpoints created
- Full GPU training: 500 epochs on RTX A5000, loss ~7.4 → ~2.8, checkpoint_final.pt verified
- Checkpoint loads on CPU, generates 4/4 unique samples with no MASK tokens

### Files created/modified (7 new)
- `bd_gen/utils/seed.py` — NEW: deterministic seeding (torch, numpy, random, CUDA)
- `bd_gen/utils/checkpoint.py` — NEW: save/load training checkpoints with OmegaConf serialization
- `bd_gen/utils/logging_utils.py` — NEW: wandb init, metric logging, git hash capture
- `scripts/train.py` — NEW: full training loop (Hydra Compose API, validation, sampling, checkpointing)
- `tests/test_integration.py` — NEW: 5 integration tests (train step, loss decrease, checkpoint roundtrip, seed, LR warmup)
- `docs/training.md` — NEW: training documentation (~900 lines: architecture, config ref, usage, troubleshooting, GPU setup, results)
- `notebooks/03_training_monitoring.ipynb` — NEW: wandb dashboard plotting

### Key decisions
- Hydra Compose API instead of `@hydra.main` (Python 3.14 argparse incompatibility with Hydra 1.3.2)
- Manual output directory (`BD_Generation/outputs/<timestamp>/`) since Compose API doesn't set up Hydra cwd
- Windows auto-detect: force `num_workers=0` on Windows (PyTorch DataLoader multiprocessing unreliable)
- Node class weights: `None` (unweighted) in v1 — spec says optional; edge weights are the critical ones
- Sample logging: counts only, no images — `bd_gen/viz/` doesn't exist until Phase 5
- Loss decrease test: fixed `t=0.5` for 50 steps — random `t` causes high ELBO weight variance with 4-sample batch
- `requires-python` lowered to `>=3.9` for university server compatibility (Python 3.9.25)
- numpy pinned to 1.26.4 on server (wandb 0.16.6 uses removed `np.float_` from numpy 2.0)
- wandb disabled on university server (old version rejects new 86-char API keys)

### Deviations from spec
- **Hydra Compose API**: spec assumed `@hydra.main` decorator, replaced with Compose API (`initialize_config_dir` + `compose`) due to Python 3.14 argparse incompatibility. CLI overrides still work via `sys.argv`.
- **No resume-from-checkpoint in training script**: spec mentions checkpoint save/load but the training script does not implement automatic resume from a checkpoint. Manual resume is documented in `docs/training.md`.
- **`noise` config key**: spec uses `schedule` as the config name (e.g. `noise=linear`), but `get_noise()` expects `config.type` attribute — the config YAML files use `type: "linear"` internally.
- **GPU training on university server instead of Google Cloud**: Google Cloud free tier blocks GPU allocation. Training ran on Polytechnique `albatros` (RTX A5000, 24GB) with wandb disabled.

### Training results (500 epochs, RTX A5000)
- Training loss: ~7.4 (epoch 0) → ~2.8–3.2 (epoch 499), with expected ELBO weight variance
- Validation loss: ~2.4
- Node accuracy: ~28.9% (vs 6.7% random), Edge accuracy: ~27.5–28.4% (vs 7.7% random)
- Wall time: ~17 minutes
- checkpoint_final.pt: 77 parameter tensors, loads and samples correctly on CPU

### Issues resolved
- Python 3.14 + Hydra 1.3.2 argparse incompatibility → Compose API workaround
- Server Python 3.9 → `from __future__ import annotations` in `test_tokenizer.py`, ruff target `py39`
- numpy 2.0 breaks wandb 0.16.6 → pinned numpy==1.26.4 on server
- wandb API key format mismatch → `wandb.mode=disabled`

## Phase 5 — Evaluation
Status: NOT STARTED
