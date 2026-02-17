
# Implementation State

> Updated after each phase. Coordinator reads this + the spec before starting work.
> Rule: keep each phase summary under 60 lines. Capture decisions and deviations, not raw logs.


## Overall Status
- Current phase: All phases complete (v1 pipeline) + post-v1 enhancements
- Last completed: Phase 5
- Post-v1: Confidence-based unmasking mode added to sampling (BichraiX); SUBS zero masking probabilities added to denoiser; float64 numerical stability fix (arXiv:2409.02908)
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
Status: COMPLETE
Branch: `eval/metrics-and-validity` → merged to `main`

### Deliverables (all verified)
- 348/348 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean
- `from bd_gen.eval import check_validity, validity_rate, novelty, diversity, distribution_match` works
- `from bd_gen.viz import draw_bubble_diagram, draw_bubble_diagram_grid` works

### Files created/modified (11 new, 3 modified)
- `bd_gen/eval/validity.py` — NEW: graph validity checker (connectivity BFS, room-type constraints, MASK/range checks)
- `bd_gen/eval/metrics.py` — NEW: evaluation metrics (validity_rate, diversity, novelty, distribution_match, per_class_accuracy)
- `bd_gen/viz/graph_viz.py` — NEW: bubble diagram visualization (networkx + matplotlib, 13-color room-type palette)
- `tests/test_validity.py` — NEW: 16 validity tests
- `tests/test_metrics.py` — NEW: 23 metrics tests
- `scripts/sample.py` — NEW: generate + visualize samples from checkpoint (Hydra Compose API)
- `scripts/evaluate.py` — NEW: full evaluation pipeline (generate → validate → metrics → wandb → JSON)
- `docs/evaluation.md` — NEW: module documentation
- `notebooks/04_sample_analysis.ipynb` — NEW: sample analysis notebook
- `bd_gen/eval/__init__.py` — MODIFIED: added exports for all eval functions
- `bd_gen/viz/__init__.py` — MODIFIED: added exports for draw_bubble_diagram, draw_bubble_diagram_grid
- `configs/eval/default.yaml` — MODIFIED: expanded with checkpoint_path, batch_size, viz options

### Key decisions
- `vocab_config` param added to `check_validity` (same precedent as Phase 3: `detokenize` requires it)
- `novelty` metric uses exact-match hash, not GED (GED is NP-hard; impractical at scale)
- `consistent` check simplified — upper-triangle format inherently prevents contradictions
- Visualization backend: `matplotlib.use("Agg")` for headless server compatibility
- Room-type constraints: at most 1 LivingRoom (idx 0), at most 1 Entrance (idx 10)

### Deviations from spec
- Added `vocab_config` parameter to `check_validity` (spec signature omitted it)
- `novelty` uses hash-based exact match instead of GED (practical constraint)

## Post-v1 — Confidence-Based Unmasking
Status: COMPLETE
Commit: `499a588` (BichraiX, 2026-02-17)

### Summary
Added an `unmasking_mode` parameter to `sample()` in `bd_gen/diffusion/sampling.py`. The original MDLM sampling uses random coin-flips to decide which MASK positions to unmask at each step. The new `"confidence"` mode (inspired by LLaDA — Nie et al.) instead unmasks positions where the model is most confident first, deferring ambiguous positions to later steps when more context is available.

### Motivation
Random unmasking treats all positions equally regardless of model certainty. This can cause the model to commit early to low-confidence predictions (e.g., ambiguous edge types between distant rooms) that propagate errors to subsequent steps. Confidence-based unmasking lets the model resolve easy, structural decisions first — obvious room types, dominant spatial relationships — and tackle harder predictions later with richer context. This ordering has been shown to improve sample quality in masked diffusion models (LLaDA, MDLM++ variants).

### Files modified (2)
- `bd_gen/diffusion/sampling.py` — added `unmasking_mode` param (`"random"` | `"confidence"`), restructured step 4d–4f
- `tests/test_sampling.py` — added tests for confidence mode

### Key design decisions
- Default `"random"` preserves full backward compatibility
- Token prediction moved before unmasking decision (confidence mode needs predicted probabilities)
- Budget per step: `p_unmask × num_remaining_masked` positions, selected by top-k confidence
- Final step (`i == 0`): unmasks all remaining positions regardless of confidence
- Per-sample top-k loop (has a `TODO` for vectorization at scale)

## Post-v1 — SUBS Zero Masking Probabilities
Status: COMPLETE

### Summary
Added the MDLM SUBS "zero masking probabilities" hard constraint to `BDDenoiser.forward()`. MASK and PAD logits are now clamped to `−∞` before the model returns, so `softmax` assigns them exactly zero probability. This makes an implicit guarantee (no training signal to predict MASK) into an explicit architectural constraint that applies to both training and inference.

### Files modified (4)
- `bd_gen/model/denoiser.py` — imported `NODE_MASK_IDX`, `NODE_PAD_IDX`, `EDGE_MASK_IDX`, `EDGE_PAD_IDX` from vocab; added step 11 clamping MASK+PAD logits to `−∞` at end of `forward()`
- `tests/test_denoiser.py` — added `TestZeroMaskingProbabilities` class with 5 tests (MASK/PAD logits are `−inf` for both node and edge; real-token logits remain finite)
- `bd_gen/diffusion/sampling.py` — updated Step 5 comment to note this cleanup is now a safety net (expected no-op given the architectural constraint)
- `docs/mdlm_comparison.md` — updated section 2.2 ("both implemented") and summary table row for SUBS zero masking probs

### Key design decisions
- Clamp both MASK and PAD logits (PAD should also never be predicted for real positions)
- Clamp in model `forward()`, not in sampling (architectural constraint > post-hoc fix)
- Keep sampling final cleanup as defense-in-depth safety net (costs nothing)
- Existing checkpoint (`checkpoint_final.pt`) loads unchanged — architecture is the same, only forward-pass behavior changes

## Post-v1 — Float64 Numerical Stability (arXiv:2409.02908)
Status: COMPLETE

### Summary
Fixed catastrophic cancellation in MDLM transition probability computation. The sampling formula `p_unmask = (α(t_next) − α(t_now)) / (1 − α(t_now))` loses precision in float32 when `α(t_next) ≈ α(t_now)` (high num_steps). Fix: pass `t.double()` to existing `alpha()` — PyTorch auto-promotes float32 buffers + float64 input to float64 output.

Reference: Zheng et al., "Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling" (arXiv:2409.02908).

### Inference vs training
- **`sampling.py` (inference)**: p_unmask computed in float64. Testable immediately with any existing checkpoint.
- **`loss.py` (training)**: w(t) ELBO weight computed in float64. Only benefits future training runs.

### Files modified (4)
- `bd_gen/diffusion/sampling.py` — `p_unmask` computed via `alpha(t.double())` instead of `alpha(t)` (float32)
- `bd_gen/diffusion/loss.py` — `_compute_w` uses `alpha(t.double())` and `alpha_prime().double()` for denominator precision
- `tests/test_sampling.py` — added `TestFloat64Precision` class (3 tests: accuracy at N=10000, nonzero at N=50000, entropy preservation)
- `tests/test_loss.py` — added `test_w_t_float64_precision_near_zero` to `TestELBOWeight`
- `docs/diffusion.md` — updated Numerical Stability table, added "Float64 Precision for Transition Probabilities" subsection

### Key design decisions
- No new methods on NoiseSchedule — `alpha(t.double())` leverages PyTorch auto-promotion
- `.float()` cast after float64 arithmetic — downstream comparisons (rand, clamp) expect float32
- Model forward pass untouched — stays entirely float32
- Gumbel sampling already used float64 (unchanged)

## Pre-Retrain Adjustments
Status: COMPLETE

### Summary
Two changes before retraining with the post-v1 fixes (SUBS zero masking, float64, safe CE targets).

### 1. Importance sampling enabled
Changed `training.importance_sampling` from `false` to `true` in `configs/training/default.yaml`.

**What it does:** Instead of sampling timesteps `t ~ Uniform(0,1)` during training, maps uniform samples through a CDF derived from the noise schedule's sigma function. This concentrates more timesteps where the ELBO weight `w(t) = -alpha'(t)/(1-alpha(t))` has high variance (near `t → 0`), reducing gradient noise without bias.

**Reference:** Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (MDLM), Section 3.2. Implementation: `NoiseSchedule.importance_sampling_transformation()` in `noise_schedule.py` (lines 85-100), wired in `train.py` (lines 314-315). The sigma_min=0 singularity is already handled (clamped to 1e-4).

**Why now:** The v1 training used uniform sampling to establish a baseline. With float64 `w(t)` precision now in place, importance sampling can safely operate in the regime near `t → 0` without numerical issues.

### 2. Gradient finiteness assertion added
Added `torch.isfinite(param.grad).all()` assertion to `test_full_train_step` in `tests/test_integration.py`. Documents the invariant that the -inf logit safety chain (SUBS zero masking + safe CE targets + loss_mask) produces finite gradients through the full training pipeline.

### No other training changes
- LR schedule: unchanged (linear warmup 1000 steps → constant 3e-4)
- EMA: unchanged (disabled, deferred per spec)
- Model architecture: unchanged
- Optimizer: unchanged (AdamW, weight_decay=0.01, grad_clip=1.0)
