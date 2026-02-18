
# Implementation State

> Updated after each phase. Coordinator reads this + the spec before starting work.
> Rule: keep each phase summary under 60 lines. Capture decisions and deviations, not raw logs.


## Overall Status
- Current phase: All phases complete (v1 pipeline) + post-v1 enhancements
- Last completed: Phase 5 + Eval Upgrade + Systematic Comparison
- Post-v1: Confidence-based unmasking mode added to sampling (BichraiX); SUBS zero masking probabilities added to denoiser; float64 numerical stability fix (arXiv:2409.02908); ReMDM remasking (cap strategy); Evaluation upgrade (JS/TV/W1, multi-seed, denoising eval, stratified drill-down); Systematic comparison infrastructure (V2 JSON, compare.py)
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

---

## Post-v1 — ReMDM Remasking (Inference-Time Scaling)
Status: COMPLETE

### Summary
Implemented ReMDM-cap remasking (arXiv:2503.00307) as an inference-only
enhancement. Previously decoded tokens can now be stochastically re-masked
during sampling, enabling error correction without retraining. At each
denoising step, already-decoded positions are re-masked with probability
sigma_t = min(eta, (1 - alpha_s) / alpha_t), then re-predicted in subsequent
steps with richer context.

### Files created/modified (6 new, 4 modified)
- `bd_gen/diffusion/remasking.py` — NEW: RemaskingSchedule callable class
  (cap + rescale strategies), create_remasking_schedule() factory function
- `eval_results/save_utils.py` — NEW: structured JSON save/load helpers
  for evaluation comparison across methods
- `tests/test_remasking.py` — NEW: 19 tests (PAD protection stress test,
  correct MASK token per position type, sigma formulas for both strategies,
  last-step guard, integration with sample())
- `eval_results/comparison.md` — NEW: side-by-side metrics table
  (MDLM baseline vs ReMDM-cap)
- `eval_results/mdlm_baseline.json` — NEW: baseline evaluation results
  (seed=42, 1000 samples, 100 steps)
- `eval_results/remdm_cap_eta0.1.json` — NEW: remasking evaluation results
  (same config + eta=0.1)
- `bd_gen/diffusion/sampling.py` — MODIFIED: remasking hook signature
  expanded from (x_t, t) to (x_t, t_now, t_next, pad_mask); added i>0
  guard to skip remasking at the final denoising step
- `bd_gen/diffusion/__init__.py` — MODIFIED: exports RemaskingSchedule,
  create_remasking_schedule
- `scripts/evaluate.py` — MODIFIED: wires remasking config from Hydra,
  constructs RemaskingSchedule, saves structured JSON with method metadata
- `configs/eval/default.yaml` — MODIFIED: added unmasking_mode field and
  nested remasking section (enabled, strategy, eta)

### Key decisions
- **Post-hoc remasking** (not full ReMDM two-distribution posterior):
  simpler, captures the key error-correction benefit, extensible to full
  formulation later. The full ReMDM uses separate distributions for masked
  vs unmasked positions; our approach applies standard MDLM unmasking then
  stochastically re-masks some decoded positions.
- **Hook signature change** from (x_t, t) to (x_t, t_now, t_next, pad_mask):
  required for sigma_t computation (needs both alpha_t and alpha_s) and
  PAD protection (the most critical invariant in the codebase).
- **i > 0 guard**: remasking is NOT applied at the final step (i=0), ensuring
  all tokens are finalized. Without this, remasked positions would remain
  as MASK in the output.
- **sigma_t in float64**: follows the same numerical stability pattern as
  p_unmask computation (arXiv:2409.02908 fix).
- **Default eta=0.1 for cap strategy**: recommended value from ReMDM paper,
  balances error correction vs sampling efficiency.
- **eval_results/ directory**: JSON per run for machine-readable comparison,
  markdown table for human-readable documentation.

### Deviations from spec
- Changed remasking_fn hook signature in sampling.py from 2-arg to 4-arg.
  The original spec (planning_T1.md Section 5.3 and Appendix A) defined
  `remasking_fn(x_t, t)` but this is insufficient — sigma_t computation
  requires both t_now and t_next, and PAD protection requires pad_mask.
  Spec updated accordingly.

### Evaluation results
See eval_results/comparison.md for full side-by-side metrics table.
Key findings (seed=42, 1000 samples, 100 steps, cap strategy, eta=0.1):
- Validity: 99.5% → 98.5% (-1.0%)
- Diversity: 0.977 → 0.993 (+0.016)
- Novelty: 0.943 → 0.976 (+0.033)
- Mode coverage (unweighted): 7.5% → 11.5% (+3.9%)
- Unique archetypes: 62 → 126 (2x improvement)
- Conditional edge KL (weighted): 0.4253 → 0.3975 (-0.028, better)
- Rooms KL: 0.0065 → 0.0011 (-0.005, better)
ReMDM trades ~1% validity for substantially improved diversity and mode
coverage. The 2x archetype count suggests the baseline suffers from some
mode collapse that remasking alleviates.

## Post-v1 — Evaluation Infrastructure Upgrade
Status: COMPLETE

### Summary
Comprehensive evaluation upgrade to make MDLM vs ReMDM comparisons
statistically sound. Adds JS/TV/W1 distance metrics, multi-seed evaluation
with mean/std aggregation, sampler-independent denoising evaluation, and
stratified drill-down metrics by num_rooms.

### Files created (3 new)
- `bd_gen/eval/denoising_eval.py` — denoising_eval() + denoising_val_elbo()
- `tests/test_denoising_eval.py` — 4 tests (keys, accuracy, PAD exclusion, max_batches)
- `docs/denoising_eval.md` — module documentation

### Files modified (5)
- `bd_gen/eval/metrics.py` — Added: _total_variation(), _js_divergence(),
  _wasserstein1_1d_discrete(), conditional_edge_distances_topN(),
  validity_by_num_rooms(), spatial_transitivity_by_num_rooms(),
  edge_present_rate_by_num_rooms(). Extended: distribution_match (JS/TV/W1
  keys), conditional_edge_kl (JS/TV keys), type_conditioned_degree_kl
  (JS/TV keys).
- `bd_gen/eval/__init__.py` — Added exports for all new functions
- `scripts/evaluate.py` — Refactored: _generate_and_evaluate_single_seed(),
  _aggregate_multi_seed(), _prefix_metrics(), multi-seed loop, denoising
  eval integration, stratified metrics, scoreboard prefixes
- `configs/eval/default.yaml` — Added: seeds, conditional_topN_pairs,
  stratified, run_denoising_eval, denoising_t_grid, denoising_max_batches
- `tests/test_metrics.py` — Added: TestTotalVariation (5), TestJSDivergence
  (6), TestWasserstein1 (5), TestConditionalEdgeDistancesTopN (4),
  TestValidityByNumRooms (2), TestSpatialTransitivityByNumRooms (2),
  TestEdgePresentRateByNumRooms (3), TestAggregateMultiSeed (3), plus
  additional JS/TV tests for distribution_match, conditional_edge_kl,
  type_conditioned_degree_kl
- `docs/evaluation.md` — Updated: architecture diagram, metrics table,
  new sections for JS/TV/W1, denoising eval, multi-seed, stratified,
  scoreboard prefixes, config reference

### Key decisions
- JS divergence uses 0*log(0/x)=0 convention (no epsilon smoothing needed)
- KL retained as diagnostic alongside JS/TV (backward compatible, additive only)
- W1 used only for num_rooms (ordinal; JS/TV for categorical distributions)
- Multi-seed: mean +/- std over 5 seeds [42, 123, 456, 789, 1337]; no bootstrap
- Denoising eval runs once (seed-independent); only sampler quality varies per seed
- Stratified metrics: validity, transitivity, edge-present rate by num_rooms only
- Exact key-set test assertions changed to issubset (forward-compatible)
- Scoreboard prefix mapping: denoise/*, sampler/validity/*, sampler/coverage/*,
  sampler/distribution/*, sampler/structure/*, sampler/conditional/*

## Post-v1 — Systematic Comparison Infrastructure
Status: COMPLETE

### Summary
Built a structured results pipeline for reproducible, multi-method comparison.
V2 JSON format stores per-seed data + summary statistics + denoising metrics.
Auto-generated comparison tables with metric family grouping.

### Files created (3 new)
- `eval_results/save_utils.py` — V2 JSON format with `{format_version,
  per_seed, summary, denoising}`, backward-compatible V1 auto-upgrade,
  `build_comparison_table()` with metric family grouping (Validity, Coverage,
  Distribution, Structure, Conditional, Denoising)
- `scripts/compare.py` — CLI utility that auto-discovers `eval_results/*.json`
  and generates `eval_results/comparison.md`
- `tests/test_save_utils.py` — 20 tests (V2 roundtrip, V1 upgrade,
  formatting, table generation)

### Files modified (1)
- `scripts/evaluate.py` (lines 508-528) — `save_eval_result()` now receives
  structured multi-seed data (`per_seed_metrics`, `summary_metrics`,
  `denoising_metrics`) instead of `flat_metrics`

### Evaluations run
- MDLM baseline: 5 seeds x 1000 samples x 100 steps (V2 JSON)
- ReMDM-cap eta=0.1: 5 seeds x 1000 samples x 100 steps (V2 JSON)
- Auto-generated `eval_results/comparison.md` with 7 sections, mean +/- std

### Key decisions
- V2 format with `format_version: 2` field enables backward-compatible V1
  auto-upgrade (V1 files get `std: 0.0` + `_upgraded_from_v1` flag)
- Unicode avoidance: `+/-` instead of special characters (Windows console)
- Metric family registry: hardcoded list of `(key, display_name, is_pct,
  is_diagnostic)` tuples for consistent table structure
- Integer formatting: `mean == int(mean)` check for clean display
- KL marked as `(diag.)` in table, hideable with `--primary-only`

### Key findings (5-seed mean +/- std)
- ReMDM-cap eta=0.1 vs MDLM baseline:
  - Validity: 99.3% → 98.8% (-0.5%, within noise)
  - Diversity: 0.976 → 0.991 (+0.015)
  - Unique archetypes: 77 → 120 (+55%)
  - Cond. edge JS (weighted): 0.083 → 0.075 (-10%, better)
  - Type-cond. degree JS (weighted): 0.048 → 0.069 (+44%, worse)
  - Denoising metrics nearly identical (confirms sampler-only difference)
