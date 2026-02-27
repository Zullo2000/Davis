
# Implementation State

> Updated after each phase. Coordinator reads this + the spec before starting work.
> Rule: keep each phase summary under 60 lines. Capture decisions and deviations, not raw logs.


## Overall Status
- Current phase: All phases complete (v1 pipeline) + post-v1 enhancements + v2 learned forward process (MELD) trained and evaluated + v2 remasking enabled
- Last completed: v2 Remasking (per-position sigma_max from rate_network)
- Post-v1: Confidence-based unmasking mode added to sampling (BichraiX); SUBS zero masking probabilities added to denoiser; float64 numerical stability fix (arXiv:2409.02908); ReMDM remasking (cap strategy); Evaluation upgrade (JS/TV/W1, multi-seed, denoising eval, stratified drill-down); Systematic comparison infrastructure (V2 JSON, compare.py)
- v2 (MELD): Learned per-position forward process — rate network, STGS, per-position ELBO loss, sampling v2, train_v2.py, evaluate.py integration. Trained 500 epochs on jabiru. Evaluated with llada+top-p 0.9, 5 seeds. Results: dramatically better distribution fidelity (Edge JS 3x better), better denoising accuracy (+11-18%), but lower diversity (0.67 vs 0.95).
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
- None. All implementations match planning_T1_with_fixed_forward_process.md exactly.

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
- None. All implementations match planning_T1_with_fixed_forward_process.md specs.

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
- None. All implementations match planning_T1_with_fixed_forward_process.md specs.

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
Status: MADE A MISTAKE - CONFUSED CONFIDENCE UNMASKING (LLADA) WITH CONFIDENCE REMASKING SCHEDULE
Note: The remasking infrastructure code (remasking.py, sampling hook, tests) is structurally sound. The confidence strategy's budget formula was wrong (used eta-capped budget instead of sigma_max). Previous eval results produced with argmax-only sampling and the wrong confidence formulation are discarded. Being redone with corrected confidence strategy, top-p sampling, and t_switch.

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
  The original spec (planning_T1_with_fixed_forward_process.md Section 5.3 and Appendix A) defined
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
Status: MADE A MISTAKE - CONFUSED CONFIDENCE UNMASKING (LLADA) WITH CONFIDENCE REMASKING SCHEDULE
Note: The evaluation infrastructure code (JS/TV/W1 metrics, multi-seed, denoising eval, stratified drill-down, save_utils V2 JSON) is correct and retained. Only the evaluation results produced with the wrong confidence remasking and argmax-only sampling are discarded.

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
Status: MADE A MISTAKE - CONFUSED CONFIDENCE UNMASKING (LLADA) WITH CONFIDENCE REMASKING SCHEDULE
Note: The comparison infrastructure code (save_utils.py V2 JSON, compare.py, metric family grouping) is correct and retained. Only the comparison results produced with the wrong confidence remasking and argmax-only sampling are discarded.

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

---

## Post-v1 — Retraining & Noise Schedule Comparison
Status: COMPLETE

### Summary
Retrained the model with three post-v1 code changes (SUBS zero masking, float64
ELBO, importance sampling), then ran the full 12-run experiment suite with both
linear and log-linear noise schedules. 500 epochs each on jabiru.

### Changes taking effect (both schedules)
1. **SUBS Zero Masking Probabilities**: MASK/PAD logits clamped to `-inf` in
   `BDDenoiser.forward()`. (Commit: `47e17cb`)
2. **Float64 ELBO Weights**: precision fix near `t → 0`. (Commit: `47e17cb`)
3. **Importance Sampling Enabled**: `training.importance_sampling=true`.
   (Commit: `0760aca`)

### Finding: importance sampling + linear schedule = broken

IS concentrates training timesteps where w(t) is high — near t→0. With the
linear schedule, alpha(t→0) ≈ 1 (almost nothing masked), so the model trains
mostly on trivially easy inputs and barely sees heavily-masked data. During
generation (starting from t=1, fully masked), the model operates in an
untrained regime.

**v1 vs v2 comparison (random+argmax):**

| Metric | v1 (no IS) | v2 linear+IS | v2 loglinear+IS |
|---|:---:|:---:|:---:|
| Validity | 99.3% | 94.6% (-4.7%) | **99.9%** (+0.6%) |
| Edge JS | 0.042 | 0.107 (2.5x worse) | **0.035** (17% better) |
| Node JS | 0.016 | 0.020 (25% worse) | 0.017 (~same) |

**Model quality (denoising accuracy):**

| Timestep | Linear+IS | Log-linear+IS |
|---|:---:|:---:|
| acc_edge@t=0.1 | 0.47 | **0.79** |
| acc_edge@t=0.5 | 0.03 | **0.54** (+1700%) |
| acc_node@t=0.5 | 0.16 | **0.56** (+250%) |

The linear model was essentially untrained at medium-high masking rates. Log-
linear distributes masking uniformly (alpha(0.5)=0.5 vs 0.007), making IS
concentrate training where predictions are meaningfully difficult.

**Impact on llada unmasking:**
- Linear+IS: 30% validity (cascading confidence errors at high masking rates)
- Log-linear+IS: **100% validity** (confidence scores are reliable at all t)

### Schedule conclusion
**Log-linear is the correct schedule** for our pipeline with IS enabled.
Linear schedule is only viable without IS (v1 configuration). All subsequent
work uses log-linear.

### Log-linear schedule implementation
- `LogLinearSchedule` class in `noise_schedule.py` (matching MDLM codebase)
- Config: `configs/noise/loglinear.yaml` (type: loglinear, eps: 1e-3)
- 11 new tests (544 total pass)
- Eval/compare scripts parameterized by `--schedule` argument
- Results: `eval_results/linear_noise_sc/`, `eval_results/loglinear_noise_sc/`, `eval_results/learned_noise_sc/`

### Training config (both runs)
- 500 epochs, lr=3e-4, AdamW, grad_clip=1.0, importance_sampling=true
- Server: Polytechnique `jabiru` (moved from `albatros` — disk full)
- Checkpoints: `outputs/2026-02-19_15-24-27/` (linear), `outputs/2026-02-19_16-58-23/` (loglinear)

---

## Post-v1 — Remasking Design Comparison (Log-Linear Schedule)
Status: COMPLETE (22 runs; final method selection pending)

### Experiment design
Layered experiment plan from `remasking_design_with_fixed_forward_process.md` Section 10. All runs use
log-linear schedule, 100 sampling steps, 1000 samples, 5 seeds.

- **Layer 1**: 4 baselines (random/llada x argmax/top-p, no remasking)
- **Layer 2**: 5 cap eta sweep (0.2, 0.4, 0.6, 0.8, 1.0) — run with both random+top-p and llada+top-p
- **Layer 3**: 3 confidence + t_switch sweep (0.3, 0.5, 0.7) — run with both random+top-p and llada+top-p
- **Layer 3b**: 2 confidence without t_switch (t_switch=1.0) — random+top-p and llada+top-p
- **Run 10**: skipped (argmax mode-collapses with llada; top-p synergy already demonstrated)

Total: 22 runs (4 baselines + 9 random remasking + 9 llada remasking).
Completed: 22/22. Full results: `eval_results/loglinear_noise_sc/comparison.md` (auto-generated, 22 methods).

### Layer 1 results — llada vs random baselines (log-linear)

| Metric | llada_argmax | llada_topp | random_argmax | random_topp |
|---|:---:|:---:|:---:|:---:|
| Validity | **100%** | **100%** | 99.9% | 99.7% |
| Node JS | 0.036 | 0.023 | 0.017 | **0.006** |
| Edge JS | 0.196 | 0.106 | 0.053 | **0.035** |
| Spatial transit. | **100%** | 99.9% | 99.7% | 97.8% |
| MMD-Degree | 0.121 | **0.050** | 0.104 | 0.302 |
| Type-cond deg JS | 0.128 | **0.033** | 0.069 | 0.161 |
| Diversity | 0.005 | 0.945 | 0.997 | **1.0** |
| Mode cov (wt) | 14.5% | 69.6% | 75.0% | **88.5%** |

**Two strong candidates, different strengths:**
- **random+topp**: best distribution match (JS), best diversity/coverage
- **llada_topp**: best validity (100%), best structural quality (MMD, transitivity,
  type-conditional degree)

llada_argmax mode-collapses (5 archetypes — fully deterministic process produces
one output per num_rooms). Not a viable standalone method.

### Layer 2 results — cap eta sweep (random+top-p)

| Metric | no_remask | eta=0.2 | eta=0.4 | eta=0.6-1.0 |
|---|:---:|:---:|:---:|:---:|
| Validity | **99.7%** | 98.3% | 98.0% | ~97.9% |
| Node JS | 0.0063 | 0.0045 | **0.0042** | ~0.0044 |
| Edge JS | **0.035** | 0.070 | 0.073 | ~0.073 |
| Spatial transit. | 97.8% | 96.9% | **97.3%** | ~97.2% |
| Mode cov (wt) | 88.5% | 89.6% | **90.7%** | ~90.2% |
| Unique archetypes | 103 | **239** | 242 | ~241 |

Key findings (random):
- Eta values 0.4-1.0 all plateau — remasking budget saturates early
- Node JS improves with remasking (0.006 → 0.004)
- Edge JS doubles (0.035 → 0.070) — main cost
- Structural damage minimal (transitivity -0.5-1%)
- Best eta: 0.2 (least Edge JS damage) or 0.4 (best Node JS + coverage)

### Layer 2 results — cap eta sweep (llada+top-p)

| Metric | no_remask | eta=0.2 | eta=0.4 | eta=0.6-1.0 |
|---|:---:|:---:|:---:|:---:|
| Validity | **100%** | 99.9% | 99.8% | ~99.8% |
| Node JS | **0.023** | 0.050 | 0.043 | ~0.044 |
| Edge JS | **0.106** | 0.214 | 0.197 | ~0.199 |
| MMD-Degree | 0.050 | 0.041 | 0.037 | **~0.036** |
| Type-cond deg JS | **0.033** | 0.031 | 0.038 | ~0.038 |
| Diversity | 0.945 | 0.971 | **0.987** | ~0.987 |
| Mode cov (wt) | 69.6% | 67.2% | **71.4%** | ~70.6% |
| Unique archetypes | 29 | 99 | **112** | ~112 |

Key findings (llada):
- Remasking improves diversity (0.945 → 0.987) and archetypes (29 → 112, 4x)
- Remasking hurts distribution match: Node JS doubles (0.023 → 0.043), Edge JS
  doubles (0.106 → 0.197) — remasking disrupts llada's careful ordering
- Remasking improves graph structure: MMD-Degree 0.050 → 0.037 (best of any method)
- Eta saturates at 0.4 (same pattern as random)
- Validity stays near-perfect (99.8%), much higher than random+remasking (98%)

### Layer 3 results — confidence + t_switch (random+top-p)

All three t_switch values (0.3, 0.5, 0.7) produce nearly identical results,
comparable to cap eta=0.4-1.0. t_switch has minimal effect. Confidence
remasking is slightly worse than cap on Edge JS (0.078-0.082 vs 0.070-0.073).

### Layer 3 results — confidence + t_switch (llada+top-p)

| Metric | tsw=0.3 | tsw=0.5 | tsw=0.7 |
|---|:---:|:---:|:---:|
| Validity | **99.9%** | 99.8% | 99.8% |
| MMD-Degree | 0.039 | **0.035** | 0.037 |
| MMD-Clustering | 0.030 | **0.027** | 0.028 |
| Type-cond deg JS | **0.033** | 0.038 | 0.040 |
| Diversity | 0.977 | 0.983 | **0.985** |
| Unique archetypes | 113 | 120 | **125** |

Key findings (llada confidence):
- Similar performance to cap, with slightly better MMD at tsw=0.5
- tsw=0.3 has best type-conditioned degree JS (0.033, matching baseline)
- tsw=0.7 has most archetypes (125)
- Edge JS range 0.201-0.207 (comparable to cap)

### Cross-cutting findings: llada vs random with remasking

Remasking **amplifies** the existing tradeoff between unmasking modes:

| Category | llada + remasking | random + remasking |
|---|---|---|
| Validity | **99.8-99.9%** | 97.9-98.3% |
| Graph structure (MMD) | **0.035-0.041** | 0.400-0.408 (10x worse) |
| Type-cond degree JS | **0.031-0.040** | 0.236-0.246 (6x worse) |
| Spatial transitivity | **98.1-98.5%** | 96.9-97.7% |
| Node JS | 0.043-0.050 | **0.004-0.005** (10x better) |
| Edge JS | 0.197-0.214 | **0.070-0.082** (3x better) |
| Mode coverage (wt) | 67-73% | **89-91%** |
| Unique archetypes | 99-125 | **239-257** (2x more) |

**No single best method.** There is a Pareto front: llada dominates on structural
quality, random dominates on distribution match and coverage. Cap and confidence
remasking perform similarly within each unmasking mode.

### What remains
1. **Final method selection** — choose from Pareto front based on downstream
   task priorities (structural correctness vs distributional fidelity)
2. Commit all results and updated docs

---

## v2 — Learned Forward Process (MELD)

> Spec: `planning_T1_with_learned_forward_process.md`
> All v1 code paths are 100% backward compatible. v2 is purely additive.

### v2 Phase 1 — Rate Network Module
Status: COMPLETE

**File:** `bd_gen/diffusion/rate_network.py` (NEW)
**Tests:** `tests/test_rate_network.py` (12 tests)

Implemented `RateNetwork` class with polynomial parameterization of per-position keeping probabilities `α_l(t)`. Each of the 36 sequence positions (8 nodes + 28 edges) gets its own learned monotonic schedule via softplus-positive polynomial coefficients predicted from structural embeddings. Includes `forward()`, `alpha_prime()`, and `forward_with_derivative()` (efficient single-pass). PAD positions forced to α=1.0 / α'=0.0. ~5K params (0.4% of denoiser).

### v2 Phase 2 — STGS and Forward Process v2
Status: COMPLETE

**File:** `bd_gen/diffusion/forward_process.py` (MODIFIED — new functions added)
**Tests:** `tests/test_forward_process_v2.py` (13 tests)

Added `stgs_sample()` (Straight-Through Gumbel-Softmax for discrete masking with gradient flow), `forward_mask_learned()` (training path: STGS + soft embeddings), and `forward_mask_eval_learned()` (eval path: discrete masking with per-position α). PAD invariant enforced in both paths. `STGSOutput` TypedDict exported.

### v2 Phase 3 — ELBO Loss v2
Status: COMPLETE

**File:** `bd_gen/diffusion/loss.py` (MODIFIED — new class added)
**Tests:** `tests/test_loss_v2.py` (9 tests)

Added `ELBOLossV2` class with per-position ELBO weights `w_l(t) = -α̇_l/(1-α_l)`, separate node/edge normalization (prevents 28 edge positions from drowning out 8 node positions), `lambda_edge` weighting, float64 weight computation, safe CE targets, and edge class weights.

### v2 Phase 4 — Denoiser Change
Status: COMPLETE

**File:** `bd_gen/model/denoiser.py` (MODIFIED — one optional param)

Added `pre_embedded: Tensor | None = None` parameter to `BDDenoiser.forward()`. When provided, skips token embedding and uses the pre-embedded tensor directly (used by v2 STGS training). Default `None` preserves 100% v1 behavior. No new parameters in the model — v1 checkpoints load unchanged.

### v2 Phase 5 — Sampling v2
Status: COMPLETE

**File:** `bd_gen/diffusion/sampling.py` (MODIFIED)
**Tests:** `tests/test_sampling_v2.py` (10 tests)

Added `rate_network: nn.Module | None = None` parameter to `sample()`. When provided, computes per-position `p_unmask_l = (α_l(t_next) - α_l(t_now)) / (1 - α_l(t_now))` in float64. Both "random" and "llada" unmasking modes adapted for per-position alpha. LLaDA budget computed as sum of per-position probabilities over masked positions. Remasking incompatibility warning when both `rate_network` and `remasking_fn` provided.

### v2 Phase 6 — Training Script v2
Status: COMPLETE

**Files:** `scripts/train_v2.py` (NEW), `configs/noise/learned.yaml` (NEW), `configs/training/v2.yaml` (NEW)

Full training loop with joint optimization of denoiser + rate network. Single AdamW optimizer over all parameters. STGS forward masking with Gumbel temperature annealing (linear decay from 1.0 to 0.1). Validation uses discrete masking (no STGS). Checkpoints save both `model_state_dict` and `rate_network_state_dict`. Usage: `python scripts/train_v2.py noise=learned training=v2`.

### v2 Phase 7 — Evaluation Integration
Status: COMPLETE

**File:** `scripts/evaluate.py` (MODIFIED)

Added `_load_v2_checkpoint()` helper that auto-detects v2 checkpoints (presence of `rate_network_state_dict` key), instantiates RateNetwork with hyperparams from checkpoint config, and loads weights. Model loading block replaced with v2-aware loading. `rate_network` parameter threaded through `_generate_and_evaluate_single_seed()` and both `sample()` call sites (metrics loop + sample saving). Method name prefixed with `v2_` for v2 checkpoints. Remasking disabled with warning for v2 mode. Dummy `LogLinearSchedule` passed to `sample()` (required by signature; v2 path ignores it).

### v2 Key Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Denoiser change | Single `pre_embedded` param | Minimal v1 impact, default None preserves all v1 behavior |
| Training script | Separate `train_v2.py` | Fundamentally different loop (STGS, dual model, gumbel temp) |
| Checkpoint format | `rate_network_state_dict` key | Auto-detect v2 vs v1 checkpoint in evaluate.py |
| Gumbel temp schedule | Linear decay 1.0 → 0.1 | Simpler, configurable via config |
| v2 Removal | Delete 4 new files, optionally remove 4 optional params | All changes are additive; v1 unaffected |

### v2 Phase 8 — Training & Evaluation
Status: COMPLETE

**Server:** Polytechnique `jabiru` (GPU)
**Checkpoint:** `outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt`
**Config:** 500 epochs, lr=3e-4, AdamW, grad_clip=1.0, uniform t, Gumbel temp 1.0→0.1 linear decay, lambda_edge=1.0

**Evaluation:** llada + top-p 0.9, 100 steps, 1000 samples, 5 seeds [42, 123, 456, 789, 1337], no remasking.
**Results file:** `eval_results/learned_noise_sc/v2_llada_topp0.9_no_remask.json`

### v2 Results vs v1 (primary comparison: llada_topp0.9_no_remask)

| Metric | v1 | v2 | Change |
|---|:---:|:---:|---|
| Validity | 100.0% | 100.0% | Same |
| Spatial transitivity | 99.9% | 100.0% | Same |
| Edge JS | 0.106 | **0.035** | 3x better |
| Node JS | 0.023 | **0.013** | 44% better |
| Edge TV | 0.399 | **0.217** | 46% better |
| Cond. edge JS (wt) | 0.175 | **0.155** | 11% better |
| Type-cond degree JS | 0.033 | 0.036 | ~Same |
| Mode coverage (wt) | 69.6% | **78.3%** | +8.7% |
| Denoising acc_edge@0.5 | 0.54 | **0.60** | +11% |
| Denoising acc_node@0.5 | 0.57 | **0.67** | +18% |
| Diversity | **0.945** | 0.671 | Much worse |
| Novelty | **0.975** | 0.864 | Worse |
| Unique archetypes | 28.6 | 26.0 | Slightly worse |
| MMD-Degree | **0.050** | 0.104 | 2x worse |
| MMD-Clustering | **0.032** | 0.090 | 3x worse |

### Issues resolved
- Boolean indexing bug in `forward_mask_learned()`: `gumbel_weights[~pad_mask, 0]` interpreted as two separate index args on 3D tensor. Fixed by boolean-masking to (N_pad, 2) slice, then assigning `[1.0, 0.0]`. (Commit `729ea6c`)
- Eval results now in `eval_results/learned_noise_sc/` (previously placed in `loglinear/` for comparison).

### Interpretation
v2 learned rates deliver exactly what MELD promises: reduced state clashing produces a better denoiser (higher accuracy at all timesteps, better distribution match). Edge JS improved 3x to match `random_topp` levels while retaining llada's perfect validity and transitivity. However, more structured masking trajectories reduce sampling stochasticity, causing diversity/novelty regression. The model generates more accurate but less varied outputs.

### Next steps
1. ~~Add remasking on top of v2 (main diversity driver in v1)~~ — DONE (see v2 Remasking below)
2. Try `random` unmasking mode with v2
3. Increase top-p (e.g., 0.95) to inject more sampling stochasticity

---

## v2 — Remasking with Per-Position Sigma_max
Status: COMPLETE (code + tests; evaluation pending)

> Design doc: `remasking_design_with_learned_forward_process.md`
> All changes are additive (v1 backward-compatible, v2-removable per Section 1.5 of planning doc)

### Summary
Enabled ReMDM confidence remasking for v2 learned forward process. The key
adaptation: `sigma_max` is now per-position `(B, SEQ_LEN)` from the rate
network instead of scalar `(B, 1)` from the noise schedule. Positions the
rate network keeps clean longer (high alpha) get less remasking budget — a
natural double signal alongside confidence-based redistribution.

### Files modified (4 production, 2 test, 1 doc)
- `bd_gen/diffusion/remasking.py` — `rate_network=None` kwarg in constructor + factory; v2 branch in `_compute_sigma_max` returning `(B, SEQ_LEN)`; `pad_mask` threaded to sigma computation
- `bd_gen/diffusion/sampling.py` — Removed guard block that disabled remasking when `rate_network` provided (−7 lines); removed unused `warnings` import
- `scripts/generate_samples.py` — Pass `rate_network` to `create_remasking_schedule()`; removed early-exit block
- `tests/test_remasking.py` — 11 new tests: `TestV2SigmaMax` (7), `TestV2ConfidenceRemasking` (2), `TestV2Factory` (2)
- `tests/test_sampling_v2.py` — Replaced `TestV2RemaskingWarning` with `TestV2RemaskingIntegration` (3 tests: call count, no-MASK output, PAD protection)
- `remasking_design_with_learned_forward_process.md` — Added Section 4.10 (v1 backward-compat & reversibility)

### v1 backward compatibility
All new parameters default to `None`. Existing v1 callers are unaffected:
- `RemaskingSchedule("cap", 0.1, noise_schedule=sched, vc)` — unchanged
- `create_remasking_schedule(cfg, noise_schedule, vc)` — unchanged
- `sample(..., rate_network=None)` — unchanged (guard block was never reached anyway)
- All 42 existing v1 remasking tests pass unchanged

### v2 removal checklist (if v2 abandoned)
Delete `rate_network` kwarg from constructor + factory, remove v2 branch in
`_compute_sigma_max`, remove `pad_mask` param. Optionally re-add guard blocks.

### Test results
- 54/54 pass (test_remasking.py + test_sampling_v2.py)
- 598/601 full suite pass (3 pre-existing failures in test_metrics.py unrelated)
- `ruff check` clean on all modified files

### Configuration for evaluation run
```
method: v2_llada_topp0.9_remdm_confidence_tsw1.0
unmasking: llada, top-p=0.9, remasking: confidence, t_switch=1.0, eta=0.0
checkpoint: outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt
noise=learned, 100 steps, 1000 samples, 5 seeds
```

See `remasking_design_with_learned_forward_process.md` Section 6 for full
evaluation commands (generation on jabiru GPU, evaluation CPU).

---

## Post-v1 — Evaluation Pipeline Split (Generate vs Evaluate)
Status: COMPLETE

### Problem
The monolithic `evaluate.py` coupled GPU generation and CPU metric computation
in one script. Adding a new metric required re-running generation for all 23
models × 5 seeds × 1000 samples, even though generation is the only expensive
step and all metrics are pure CPU.

### Solution: two scripts with clear responsibilities
1. **`scripts/generate_samples.py`** (GPU): loads model, generates tokens per
   seed, saves `{method}_samples.pt` to `eval_results/{schedule}/`. No metrics.
2. **`scripts/evaluate.py`** (CPU only): loads saved tokens, detokenizes,
   computes ALL metrics unconditionally, saves/updates `{method}.json`.

### Files created (2 new)
- `scripts/generate_samples.py` — GPU generation, Hydra CLI, saves `.pt`
- `scripts/evaluate.py` — CPU metrics, argparse CLI (`--schedule`, `--model`,
  `--list`, `--update-comparison`)

### Files modified (1)
- `eval_results/save_utils.py` — `_make_json_serializable` → `make_json_serializable`
  (public), added `aggregate_multi_seed()` (moved from old evaluate.py)

### Files renamed (1)
- `scripts/evaluate.py` → `scripts/generate_and_evaluate.py`
  **Transition backup only.** Kept so existing workflows don't break while the 23
  models are backfilled with `_samples.pt` files. Once all models have saved
  samples, this file is dead code and should be deleted.

### `.pt` format
```python
{format_version: 1, method, n_max, seeds, num_samples, config,
 per_seed: {str(seed): {tokens: Tensor(N, S), pad_masks: Tensor(N, S)}}}
```
Storage: ~1.6 MB per method (5 seeds × 1000 × 36 tokens). 23 methods ≈ 37 MB.

### Backfill completed (2026-02-24)
All 35 models (23 loglinear + 12 linear) backfilled with `_samples.pt` on jabiru.
Old monolithic `generate_and_evaluate.py` deleted. `.gitignore` updated to exclude
`_samples.pt` files. All JSONs re-evaluated with new `inside_validity` metric.

### Saved sample files on jabiru
All `_samples.pt` files live on the jabiru GPU server (NOT in git — too large):
```
jabiru:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*_samples.pt  (21 files)
jabiru:/Data/amine.chraibi/Davis/BD_Generation/eval_results/linear_noise_sc/*_samples.pt   (12 files)
jabiru:/Data/amine.chraibi/Davis/BD_Generation/eval_results/learned_noise_sc/*_samples.pt  (2 files)
```
SSH: `ssh amine.chraibi@jabiru.polytechnique.fr`
Path: `cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation`

### Checkpoints on jabiru
```
loglinear (v1): /Data/amine.chraibi/Davis/BD_Generation/outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt
linear (v1):    /Data/amine.chraibi/Davis/BD_Generation/outputs/2026-02-19_15-24-27/checkpoints/checkpoint_final.pt
v2 (MELD):      /Data/amine.chraibi/Davis/BD_Generation/outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt
```

### Re-evaluating after adding a new metric (CPU only, no GPU needed)
```bash
# On jabiru (has _samples.pt files + venv):
python scripts/evaluate.py --schedule loglinear_noise_sc --update-comparison
python scripts/evaluate.py --schedule linear_noise_sc --update-comparison
python scripts/evaluate.py --schedule learned_noise_sc --update-comparison
# Then copy updated JSONs + comparison.md back locally:
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*.json BD_Generation/eval_results/loglinear_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*.md BD_Generation/eval_results/loglinear_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/linear_noise_sc/*.json BD_Generation/eval_results/linear_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/linear_noise_sc/*.md BD_Generation/eval_results/linear_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/learned_noise_sc/*.json BD_Generation/eval_results/learned_noise_sc/
scp amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/learned_noise_sc/*.md BD_Generation/eval_results/learned_noise_sc/
```
