# BD Generation — Implementation Phase

> **Purpose of this document:** Serve as the complete implementation blueprint for the MDLM-based bubble diagram generator. Every decision about repository structure, version control, module design, testing, and deployment is specified here so that the coding phase can proceed step-by-step without ambiguity. This document is also the primary onboarding reference for team members joining after v1.0.
>
> **Prerequisite:** `research_T1.md` contains all architectural decisions (MDLM framework, SEQ_LEN-token representation, transformer denoiser, PAD handling, extensibility requirements). This document does not repeat those decisions — it specifies *how* to implement them.
>
> **Scope:** v1 implements **unconditional bubble diagram generation** only. The `condition=None` argument in `BDDenoiser.forward` is the only future-proofing needed now. See Appendix A for the full extensibility map.

---

## 1. Git Branching Strategy

Trunk-based development with short-lived feature branches. All work on feature branches; `main` is always passing tests. Merge back with `git merge --no-ff`.

### 1.1 Branch Naming Convention

```
<category>/<short-description>

Categories:
  setup/       — repo scaffolding, CI, dependency management
  data/        — data pipeline, dataset classes, preprocessing
  model/       — transformer architecture, embeddings, heads
  diffusion/   — noise schedule, forward process, loss, sampling
  training/    — training loop, optimizer, logging
  eval/        — evaluation metrics, validity checker
  viz/         — visualization utilities
  experiment/  — throwaway explorations (may never merge)
  fix/         — bug fixes
  docs/        — documentation updates
```

Examples:
- `setup/repo-scaffold` — initial package structure, pyproject.toml, configs
- `data/graph2plan-loader` — .mat parsing and tokenization
- `model/transformer-denoiser` — composite positional encoding + dual embeddings + encoder
- `diffusion/noise-and-loss` — noise schedules, forward masking, ELBO loss
- `experiment/focal-loss-edges` — testing focal loss for edge sparsity (may not merge)

### 1.2 Workflow Rules

1. Never commit directly to `main`. Branch from `main`, work, merge back with `--no-ff`.
2. Keep branches short-lived (1–3 days, one phase milestone). All tests must pass before merging.
3. `experiment/*` branches are exempt from merge-back — can be long-lived, abandoned, or cherry-picked.

### 1.3 Tagging Milestones

Annotated tags at each phase completion:

```
v0.1.0  — repo scaffold complete, package installable
v0.2.0  — data pipeline works, BubbleDiagramDataset returns correct shapes
v0.3.0  — model forward pass validated
v0.4.0  — diffusion core works (masking, loss, sampling)
v0.5.0  — first training run completes, loss decreases
v0.6.0  — evaluation metrics implemented, first quality results on RPLAN
v1.0.0  — unconditional BD generation, paper-ready results on RPLAN
```

Tag command: `git tag -a v0.2.0 -m "Phase 1 complete: data pipeline works"`

### 1.4 Commit Message Convention

[Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

Types:
  feat     — new feature or module
  fix      — bug fix
  refactor — code restructuring without behavior change
  test     — adding or modifying tests
  docs     — documentation changes
  config   — configuration changes
  data     — data pipeline changes

Scope: the module affected (data, model, diffusion, eval, viz, training)

Examples:
  feat(data): add Graph2Plan .mat loader with caching
  fix(diffusion): ensure PAD positions are never masked in forward process
  test(loss): add PAD exclusion stress test with 10K samples
  config: add cosine noise schedule config
```

### 1.5 ML Experimentation

Code changes (architecture variants, new losses) → git branches. Hyperparameter sweeps → same code, different Hydra config overrides, tracked in wandb. Git tracks code; wandb tracks runs.

---

## 2. Repository Structure

> **Version control scope:** Only the `BD_Generation/` directory is tracked in git. All files outside this directory (including `Others/`, `DiDAPS_COPY/`, `Papers/`, etc.) are personal workflow artifacts and should be excluded via `.gitignore`.

```
Davis/                                              # Git root (existing)
├── .gitignore                                      # Excludes everything except BD_Generation/ (see Section 11)
├── README.md                                       # Existing
│
└── BD_Generation/                                  # Project root (ONLY directory tracked in git)
    ├── README.md                                   # Project README: setup, architecture, quick-start
    ├── pyproject.toml                              # Package metadata + all dependencies (install via pip install -e .)
    ├── Makefile                                    # Common commands (see targets below)
    │
    ├── # Project documentation
    ├── research_T1.md                              # Architectural decisions
    ├── planning_T1.md                              # THIS DOCUMENT (implementation blueprint)
    ├── discussion_research_T1.md                   # Research findings
    ├── discussion_research_T1_old_version.md       # Initial research output
    ├── google_cloud_usage_efficiency.md            # GPU cost strategy
    │
    ├── bd_gen/                                     # Main Python package (importable library code ONLY)
    │   ├── __init__.py                             # Package init, version string
    │   │
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── vocab.py                            # Node/edge vocabulary, PAD/MASK indices, helpers, compute_pad_mask
    │   │   ├── graph2plan_loader.py                # Load .mat → intermediate format dicts
    │   │   ├── tokenizer.py                        # Intermediate format → SEQ_LEN-token flat sequence
    │   │   └── dataset.py                          # BubbleDiagramDataset (PyTorch Dataset), splits, class freq stats
    │   │
    │   ├── model/
    │   │   ├── __init__.py
    │   │   ├── embeddings.py                       # NodeEmbedding, EdgeEmbedding, CompositePositionalEncoding
    │   │   ├── transformer.py                      # Transformer encoder blocks with adaLN
    │   │   └── denoiser.py                         # BDDenoiser: wraps embeddings + transformer + heads
    │   │
    │   ├── diffusion/
    │   │   ├── __init__.py
    │   │   ├── noise_schedule.py                   # Direct implementation of schedule classes (linear, cosine)
    │   │   ├── forward_process.py                  # Masking logic with PAD protection
    │   │   ├── loss.py                             # MDLM ELBO loss with PAD exclusion + class weighting
    │   │   └── sampling.py                         # Reverse process, guidance hooks, inpainting
    │   │
    │   ├── eval/
    │   │   ├── __init__.py
    │   │   ├── validity.py                         # Graph validity checker
    │   │   └── metrics.py                          # Novelty, diversity, distribution match
    │   │
    │   ├── viz/
    │   │   ├── __init__.py
    │   │   └── graph_viz.py                        # Draw BDs as labeled graphs
    │   │
    │   └── utils/
    │       ├── __init__.py
    │       ├── logging_utils.py                    # wandb setup, metric logging helpers
    │       ├── checkpoint.py                       # Save/load model checkpoints
    │       └── seed.py                             # Reproducibility utilities
    │
    ├── configs/                                    # Hydra configuration hierarchy
    │   ├── config.yaml                             # Main config (composes defaults)
    │   ├── model/
    │   │   ├── small.yaml                          # d_model=128, L=4, heads=4
    │   │   └── base.yaml                           # d_model=256, L=6, heads=8
    │   ├── data/
    │   │   └── graph2plan.yaml                     # Paths, splits, batch size
    │   ├── noise/
    │   │   ├── linear.yaml                         # Linear schedule params
    │   │   └── cosine.yaml                         # Cosine schedule params
    │   ├── training/
    │   │   └── default.yaml                        # LR, epochs, optimizer, checkpointing
    │   └── eval/
    │       └── default.yaml                        # Num samples, which metrics to compute
    │
    ├── scripts/                                    # Entry points (Hydra-decorated)
    │   ├── train.py                                # Training entry point
    │   ├── evaluate.py                             # Evaluation entry point
    │   ├── sample.py                               # Generate and visualize samples
    │   ├── prepare_data.py                         # Download + preprocess → .pt cache
    │   └── inspect_data.py                         # Print dataset statistics
    │
    ├── tests/                                      # pytest test suite
    │   ├── __init__.py
    │   ├── conftest.py                             # Shared fixtures (sample_batch, etc.)
    │   ├── test_vocab.py
    │   ├── test_tokenizer.py
    │   ├── test_dataset.py
    │   ├── test_embeddings.py
    │   ├── test_denoiser.py
    │   ├── test_forward_process.py
    │   ├── test_loss.py
    │   ├── test_sampling.py
    │   ├── test_validity.py
    │   └── test_integration.py
    │
    └── notebooks/                                  # Jupyter notebooks (clear outputs before committing)
        ├── 01_data_exploration.ipynb
        ├── 02_model_sanity.ipynb
        ├── 03_training_monitoring.ipynb
        └── 04_sample_analysis.ipynb
```

### 2.1 Makefile Targets

```makefile
install:    pip install -e ".[dev]"
test:       pytest tests/ -v
lint:       ruff check bd_gen/ tests/
format:     ruff format bd_gen/ tests/
data:       python scripts/prepare_data.py
train:      python scripts/train.py
evaluate:   python scripts/evaluate.py
sample:     python scripts/sample.py
```

---

## 3. Module Breakdown

Each module is described with its files, responsibilities, and key implementation details. **For complete typed signatures, see Section 5 (Interface Contracts).** For architectural decisions (vocabulary sizes, PAD handling rules, denoiser architecture), refer to `research_T1.md`.

### 3.1 `bd_gen/data/` — Data Pipeline

| File | Responsibility |
|------|---------------|
| `vocab.py` | `VocabConfig(n_max)` frozen dataclass: single source of truth for dataset-dependent sizing. Properties: `n_edges`, `seq_len`. Methods: `compute_pad_mask()`, `edge_position_to_pair()`, `pair_to_edge_position()`. Preset: `RPLAN_VOCAB_CONFIG = VocabConfig(n_max=8)`. Module-level constants remain for N_MAX-independent values: NODE_TYPES, EDGE_TYPES, MASK/PAD indices, VOCAB_SIZES. |
| `graph2plan_loader.py` | Loads Graph2Plan's `data.mat` using `scipy.io.loadmat`. Iterates over floorplan records, extracts `rType` (room types) and `rEdge` (edge triples), maps from 1-based to 0-based indexing. Returns a list of intermediate-format dicts: `{"node_types": List[int], "edge_triples": List[Tuple[int,int,int]], "num_rooms": int}`. Caches the parsed result as a `.pt` file to avoid re-parsing. |
| `tokenizer.py` | Converts intermediate-format dicts → SEQ_LEN-token flat sequences with correct PAD placement. `tokenize(graph_dict, vocab_config)` returns `(tokens, pad_mask)` tensors. `detokenize(tokens, pad_mask, vocab_config)` converts back for evaluation/visualization. Enforces the critical distinction: `[PAD]` for positions that don't exist vs `no-edge` for real rooms that aren't adjacent. |
| `dataset.py` | `BubbleDiagramDataset(torch.utils.data.Dataset)` — loads preprocessed `.pt` cache (or triggers loading/tokenization if missing). `__getitem__` returns a `BatchDict` (see Section 4). Handles train/val/test split (80/10/10, configurable). Computes and exposes: (1) `edge_class_weights` attribute (`Tensor(13,)`, dtype=float32, inverse-frequency weights over edge vocab), (2) `node_class_weights` attribute (`Tensor(15,)`, dtype=float32, inverse-frequency weights over node vocab — optional for v1 but infrastructure is ready), (3) `num_rooms_distribution` attribute (`Tensor(vocab_config.n_max,)`, dtype=float32, histogram of room counts in training split — used by sampling to match the training distribution). These attributes are read by the training script and passed to loss/sampling constructors (see dataflow note below). |

**Vocabulary index mappings (exact values for `vocab.py`):**

> **PROVISIONAL WARNING:** The room type and edge type lists below are based on the Graph2Plan paper. They **must be verified against the actual `data.mat` file** before implementation. The Graph2Plan paper lists room types that may differ from what's stored in the data (e.g., the paper mentions "GuestRoom" and "ChildRoom" which are not in this list, while this list includes "ExternalArea" and "ExternalWall"). **Phase 1, Task 1 (data verification)** must run `load_graph2plan` on the real data, print all unique `rType` and `rEdge` values, and update this vocabulary to match. The loader must include an assertion that every loaded value maps to a known vocabulary entry.

```python
# Node vocabulary — index to type mapping
# Indices 0–12 match Graph2Plan's rType values after converting from 1-based to 0-based
# ⚠ VERIFY AGAINST data.mat — see PROVISIONAL WARNING above
NODE_TYPES = [
    "MasterRoom",      # 0
    "SecondRoom",       # 1
    "LivingRoom",       # 2
    "Kitchen",          # 3
    "Bathroom",         # 4
    "Balcony",          # 5
    "Entrance",         # 6
    "DiningRoom",       # 7
    "StudyRoom",        # 8
    "StorageRoom",      # 9
    "WallIn",           # 10
    "ExternalArea",     # 11
    "ExternalWall",     # 12
]
NODE_MASK_IDX = 13      # [MASK] — used during diffusion forward/reverse process
NODE_PAD_IDX = 14       # [PAD] — position doesn't exist in this graph
NODE_VOCAB_SIZE = 15    # 13 room types + MASK + PAD

# Edge vocabulary — index to type mapping
# Indices 0–9 match Graph2Plan's rEdge relationship values (after 1-based → 0-based)
# ⚠ VERIFY AGAINST data.mat — Graph2Plan paper uses "outside" where we write "surrounding"
EDGE_TYPES = [
    "left-of",          # 0
    "right-of",         # 1
    "above",            # 2
    "below",            # 3
    "left-above",       # 4
    "left-below",       # 5
    "right-above",      # 6
    "right-below",      # 7
    "inside",           # 8
    "surrounding",      # 9  — may be called "outside" in Graph2Plan; verify
]
EDGE_NO_EDGE_IDX = 10   # real rooms that are not adjacent (meaningful signal)
EDGE_MASK_IDX = 11       # [MASK] — used during diffusion forward/reverse process
EDGE_PAD_IDX = 12        # [PAD] — position doesn't exist (involves a PAD node)
EDGE_VOCAB_SIZE = 13     # 10 relationships + no-edge + MASK + PAD
```

```python
# Dataset-dependent sizing — configurable per dataset via VocabConfig
@dataclass(frozen=True)
class VocabConfig:
    """Single source of truth for N_MAX-derived constants. Immutable after creation."""
    n_max: int   # max rooms per graph (8 for RPLAN, 14 for ResPlan)

    @property
    def n_edges(self) -> int:
        return self.n_max * (self.n_max - 1) // 2

    @property
    def seq_len(self) -> int:
        return self.n_max + self.n_edges

    def compute_pad_mask(self, num_rooms: int) -> Tensor: ...
    def edge_position_to_pair(self, pos: int) -> tuple[int, int]: ...
    def pair_to_edge_position(self, i: int, j: int) -> int: ...

# Convenience presets
RPLAN_VOCAB_CONFIG = VocabConfig(n_max=8)    # SEQ_LEN=36
RESPLAN_VOCAB_CONFIG = VocabConfig(n_max=14)  # SEQ_LEN=105
```

**Note:** Graph2Plan's `.mat` file uses 1-based indexing for both `rType` and `rEdge`. The loader (`graph2plan_loader.py`) converts to 0-based during parsing: `rType_value - 1` for room types, `rEdge_value - 1` for edge types. Verify this mapping against the Graph2Plan source code when implementing the loader.

**Edge class weights computation formula:**

```python
# Computed over training split only, excluding PAD positions.
# For each edge class c in [0, EDGE_VOCAB_SIZE):
#   count[c] = number of non-PAD edge positions with value c across the training set
#   weight[c] = total_non_pad_edges / (EDGE_VOCAB_SIZE * count[c])
# This produces weights inversely proportional to frequency, normalized so mean = 1.0.
# PAD positions are excluded from both counting and the denominator.
```

**Class weights and statistics dataflow:** The training script (`scripts/train.py`) is responsible for bridging the data and diffusion modules. The flow is:
1. `BubbleDiagramDataset` computes `edge_class_weights`, `node_class_weights`, and `num_rooms_distribution` from training split statistics during initialization (Phase 1).
2. `scripts/train.py` reads `dataset.edge_class_weights` (and optionally `dataset.node_class_weights`) and passes them to the `ELBOLoss` constructor (Phase 4).
3. `scripts/train.py` reads `dataset.num_rooms_distribution` and passes it to the `sample()` function during evaluation sampling.
4. `ELBOLoss` stores the weights and applies them to edge/node positions during loss computation (Phase 3).

**Node class weights computation:** Same formula as edge class weights, but computed over node positions only (indices 0 through N_MAX-1, excluding PAD). Default: unweighted in v1 (pass `None` to `ELBOLoss`), but compute and log the statistics to inform whether weighting is needed.

This keeps `bd_gen/data/` and `bd_gen/diffusion/` fully decoupled — the training script is the only place that connects them.

### 3.2 `bd_gen/model/` — Transformer Denoiser

| File | Responsibility |
|------|---------------|
| `embeddings.py` | `NodeEmbedding`: `nn.Embedding(15, d_model)`. `EdgeEmbedding`: `nn.Embedding(13, d_model)`. `CompositePositionalEncoding`: uses `vocab_config.n_max` for embedding dimensions (see detailed specification below). `TimestepEmbedding`: see detailed specification below. |
| `transformer.py` | Custom transformer encoder blocks with adaptive LayerNorm (adaLN-Zero) for time conditioning, following the DiT pattern (Peebles & Xie, ICCV 2023). Standard PyTorch `nn.TransformerEncoder` cannot be used because it has no mechanism to inject time-dependent scale/shift parameters. With only SEQ_LEN tokens (36 for RPLAN), flash attention is unnecessary; use `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+). **PAD attention masking:** Apply a `key_padding_mask` derived from `pad_mask` (inverted: True=ignore) so that PAD positions neither attend to nor are attended by real positions. This prevents wasting attention capacity on uninformative PAD embeddings and avoids learning spurious correlations. See adaLN-Zero block pseudocode below. |
| `denoiser.py` | `BDDenoiser(nn.Module)` — the top-level model class. Constructor accepts `vocab_config: VocabConfig`. Composes all components: embeddings, transformer blocks, and two classification heads (`NodeClassificationHead`: `nn.Linear(d_model, 15)`, `EdgeClassificationHead`: `nn.Linear(d_model, 13)`). Forward signature: `forward(tokens, pad_mask, t, condition=None) → (node_logits, edge_logits)`. The `condition` argument is unused in v1; in v2 it will accept `(B, n_condition_tokens, d_condition)` house-boundary feature tensors integrated via cross-attention layers in the transformer blocks (no-op when `None`). Returns node logits `(B, N_MAX, NODE_VOCAB_SIZE)` and edge logits `(B, N_EDGES, EDGE_VOCAB_SIZE)`. Splits the transformer output into node `[:,:N_MAX]` and edge `[:,N_MAX:]` parts. Time input `t` must be handled flexibly: support scalar, 0D tensor, or 1D tensor of size `batch_size` (add a `_process_t()` helper — see specification below). `pad_mask` is inverted internally to create the attention mask for PAD exclusion. |

**CompositePositionalEncoding specification:**

```python
class CompositePositionalEncoding(nn.Module):
    entity_type_emb: nn.Embedding(2, d_model)     # 0 = node position, 1 = edge position
    node_index_emb: nn.Embedding(N_MAX, d_model)   # which room slot (from vocab_config.n_max)
    pair_index_emb: nn.Embedding(N_MAX, d_model)    # shared for both endpoints of an edge (from vocab_config.n_max)

    # For node position k (0 <= k < n_max):
    #   pos_encoding[k] = entity_type_emb(0) + node_index_emb(k)
    #
    # For edge position at (i, j) where 0 <= i < j < n_max:
    #   pos_encoding[n_max + flat_index] = entity_type_emb(1) + pair_index_emb(i) + pair_index_emb(j)
    #
    # The i, j for each edge position are computed via edge_position_to_pair() from vocab.py.
    # Output: added to the token embeddings before entering the transformer.
```

**TimestepEmbedding specification:**

```python
class TimestepEmbedding(nn.Module):
    # Step 1: Sinusoidal positional encoding
    #   Input: t as (B,) float32 tensor (continuous time in [0, 1])
    #   Output: (B, frequency_embedding_size) where frequency_embedding_size = 256 (default)
    #   Formula: standard sinusoidal from Vaswani et al. / GLIDE (same as DiT and MDLM repos)
    #     half = frequency_embedding_size // 2
    #     freqs = exp(-log(10000) * arange(0, half) / half)
    #     embedding = cat([cos(t * freqs), sin(t * freqs)], dim=-1)
    #
    # Step 2: MLP projection
    #   Linear(frequency_embedding_size, d_model) → SiLU → Linear(d_model, d_model)
    #
    # Step 3: In BDDenoiser.forward(), apply SiLU before passing to adaLN blocks:
    #   c = SiLU(self.timestep_embedding(t))   # (B, d_model)
    #   Then c is passed to each transformer block's adaLN modulation.
```

**adaLN-Zero block pseudocode (per transformer block):**

Each block receives the sequence `x: (B, SEQ_LEN, d_model)`, the conditioning `c: (B, d_model)` (from TimestepEmbedding + SiLU), and the `attn_mask: (B, SEQ_LEN)` (True=ignore, derived from inverting `pad_mask`).

```python
# 1. Compute 6 modulation parameters from conditioning:
#    adaLN_modulation = Linear(cond_dim, 6 * d_model, bias=True)
#    CRITICAL: zero-initialize both weights AND bias of this layer.
#    This ensures the block starts as a standard transformer (identity modulation).
(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
    adaLN_modulation(c).unsqueeze(1).chunk(6, dim=2)   # each: (B, 1, d_model)

# 2. Attention sub-block:
x_norm = LayerNorm(x)
x_modulated = x_norm * (1 + scale_msa) + shift_msa     # adaLN modulation formula
x_attn = MultiHeadSelfAttention(x_modulated, attn_mask) # using F.scaled_dot_product_attention
#   attn_mask excludes PAD positions from attention (passed as key_padding_mask)
x = x + gate_msa * Dropout(x_attn)                       # gated residual connection

# 3. FFN sub-block:
x_norm2 = LayerNorm(x)
x_modulated2 = x_norm2 * (1 + scale_mlp) + shift_mlp
x_ffn = FFN(x_modulated2)                                # Linear(d_model, mlp_ratio*d_model) → GELU → Linear(mlp_ratio*d_model, d_model)
x = x + gate_mlp * Dropout(x_ffn)
```

Reference: DiT paper Section 3 (Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023). The key insight is that zero-initialization of the modulation layer makes each block start as a standard transformer; the model gradually learns to use time conditioning during training.

**`_process_t()` helper specification:**

```python
def _process_t(self, t, batch_size: int, device: torch.device) -> Tensor:
    """Normalizes t to a 1D float32 tensor of shape (batch_size,).
    Accepts: Python float/int, 0D tensor, 1D tensor of size 1, or 1D tensor of size batch_size.
    Always returns: Tensor of shape (batch_size,), dtype=torch.float32, on the specified device.
    Raises ValueError for invalid shapes."""
```

### 3.3 `bd_gen/diffusion/` — Diffusion Core

> **Mathematical reference:** Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (MDLM). Reference implementation: `github.com/kuleshov-group/mdlm`. All noise schedule, loss, and sampling code is implemented directly in this module, tailored to our SEQ_LEN-token PAD-aware sequences.

| File | Responsibility |
|------|---------------|
| `noise_schedule.py` | Direct implementation of noise schedules. Provides a factory function `get_noise(config)` that returns a schedule object. Supports `LinearSchedule` and `CosineSchedule` initially, easily extensible via config (additional schedules like log-linear and geometric can be added by implementing the `NoiseSchedule` interface and adding a config YAML; see MDLM repo for reference). Each schedule exposes `sigma(t)`, `alpha(t)`, `alpha_prime(t)`, and optionally `importance_sampling_transformation(t)` methods (see formulas below). |
| `forward_process.py` | `forward_mask(x0, pad_mask, t, noise_schedule) → (x_t, mask_indicators)`. Computes masking probability from the schedule, applies stochastic masking to non-PAD positions only. **Critical invariant**: PAD positions are never masked, enforced by the `pad_mask` before any stochastic step. |
| `loss.py` | MDLM continuous-time ELBO loss. Loss mask = `mask_indicators AND NOT pad_mask`. The loss function **splits** `x0`, `pad_mask`, and `mask_indicators` along dim=1 into node ([:, :N_MAX]) and edge ([:, N_MAX:]) parts. Node targets index into NODE_VOCAB (0–14), edge targets index into EDGE_VOCAB (0–12). These are **separate cross-entropy computations** with separate vocabularies. Edge positions support class-weighted CE or focal loss. Optionally supports node class weights too (default: unweighted for v1). Class weights are **not computed internally** — they are received as constructor arguments, supplied by the training script from `BubbleDiagramDataset`. **Normalization:** Per-sample, divide by the number of loss-active positions (where `mask_indicators=True AND pad_mask=True`), then average across the batch. This ensures each sample contributes equally regardless of graph size (a 4-room graph should not be dominated by an N_MAX-room graph). |
| `sampling.py` | Reverse denoising loop: iterates from t=1 to t=0, predicting and unmasking tokens at each step. Supports pluggable `guidance_fn(logits, x_t, t, pad_mask, **kwargs) → modified_logits` for future guidance mechanisms (the `pad_mask` argument lets guidance functions know which positions are real; `**kwargs` allows passing additional context like noise_schedule or model reference for training-free guidance). Supports `fixed_tokens` for inpainting. PAD positions clamped to `[PAD]` throughout. Configurable sampling temperature: `temperature=0` → deterministic argmax, `temperature>0` → stochastic sampling via Gumbel-perturbed logits. Supports optional `remasking_fn(x_t, t) → x_t_remasked` for future ReMDM inference (default: `None` = standard MDLM). |

**Noise schedule convention (sigma → alpha):**

All schedules internally compute `sigma_t` (total noise) and derive `alpha_t` from it. This matches the convention in the MDLM reference repo (`github.com/kuleshov-group/mdlm`). The relationship is:

```
alpha_t = exp(-sigma_t)          # keeping probability (probability a token is NOT masked)
masking_prob = 1 - alpha_t       # probability a token IS masked at time t
```

**Noise schedule formulas:**

```
Linear schedule (config: sigma_min=0, sigma_max=10):
  sigma_t = sigma_min + t * (sigma_max - sigma_min)
  alpha_t = exp(-sigma_t)
  alpha'_t = -(sigma_max - sigma_min) * alpha_t       # derivative w.r.t. t

Cosine schedule (config: eps=1e-3):
  sigma_t = -log(eps + (1 - eps) * cos(t * pi / 2))
  alpha_t = eps + (1 - eps) * cos(t * pi / 2)
  alpha'_t = -(1 - eps) * (pi / 2) * sin(t * pi / 2)
```

At t=0: `alpha_0 ≈ 1` (nearly clean). At t=1: `alpha_1 ≈ 0` (nearly fully masked).

**ELBO loss formula (MDLM continuous-time):**

```
L = E_{t} [ w(t) * (1/N_active) * Σ_l CE(logits_l, x0_l) * loss_mask_l ]

where:
  w(t) = -alpha'_t / (1 - alpha_t + eps)     # per-timestep ELBO weight, eps=1e-8
  loss_mask_l = mask_indicators_l AND (NOT pad_mask_l)
  N_active = Σ_l loss_mask_l                  # number of loss-active positions per sample
  CE = cross-entropy loss (per-position, unreduced)
```

**ELBO weight clamping:** `w(t)` approaches infinity as `t → 0` (because `alpha_t → 1`, denominator → 0). In practice the loss mask has zero active positions near t=0, so the infinity is multiplied by zero, but the weight computation itself may produce NaN/Inf. **Clamp `t` to `[t_min, 1.0]` where `t_min = 1e-5` before computing `w(t)`.** Alternatively, clamp `w(t)` directly to a maximum value (e.g., 1000).

**Loss splitting:** The loss function receives `x0: (B, SEQ_LEN)` and `logits` as separate node/edge tensors. Internally, `x0`, `pad_mask`, and `mask_indicators` are **split along dim=1**: node part `[:, :N_MAX]` and edge part `[:, N_MAX:]`. Node targets use NODE_VOCAB indices; edge targets use EDGE_VOCAB indices. These are **separate CE computations**.

For edge positions (indices N_MAX through SEQ_LEN-1), the CE is class-weighted using `edge_class_weights`. For node positions (indices 0 through N_MAX-1), optionally class-weighted using `node_class_weights` (default: unweighted in v1, but the infrastructure is ready — node types can be imbalanced, e.g. every floor plan has 1 Entrance but may have 3+ SecondRooms).

**Normalization:** Per-sample, divide by `N_active` (the number of positions where `mask_indicators=True AND pad_mask=True`). Then average across the batch. This ensures a 4-room graph (with ~6 real edge positions) contributes equally to the gradient as an N_MAX-room graph (with N_EDGES real edge positions).

**Timestep sampling:** Support both uniform and importance-weighted sampling.
- Default: `t = torch.rand(batch_size)` (uniform on `[t_min, 1.0]`).
- Optional: importance sampling via the noise schedule's `importance_sampling_transformation(t)` method, which maps uniform samples to a distribution that reduces ELBO gradient variance (see MDLM paper Section 3 and MDLM repo). Add as a config flag: `training.importance_sampling: bool = false`.

**Sampling algorithm (MDLM ancestral sampling):**

```
Input: model, noise_schedule, num_steps N, batch_size B, device,
       num_rooms_distribution (optional), fixed_num_rooms (optional)
Output: generated tokens (B, SEQ_LEN)

1. Determine num_rooms for each sample:
   - If fixed_num_rooms is given: use that value for all samples
   - Otherwise: sample from num_rooms_distribution (empirical histogram from training set,
     stored as BubbleDiagramDataset.num_rooms_distribution: Tensor(vocab_config.n_max,), dtype=float32)
   - Default: match training distribution
   Compute pad_mask from num_rooms via compute_pad_mask()

2. Initialize x_T (all positions at max noise):
   - Node PAD positions (index >= num_rooms) → NODE_PAD_IDX
   - Edge PAD positions (either endpoint is PAD) → EDGE_PAD_IDX
   - Real node positions → NODE_MASK_IDX
   - Real edge positions → EDGE_MASK_IDX

3. For i = N-1 down to 0:
     t_now  = (i + 1) / N
     t_next = i / N

     a. Get model predictions:
        node_logits, edge_logits = model(x_t, pad_mask, t_now)

     b. For each position l where x_t[l] is MASK:
        alpha_now  = noise_schedule.alpha(t_now)
        alpha_next = noise_schedule.alpha(t_next)
        p_unmask = (alpha_next - alpha_now) / (1 - alpha_now + 1e-8)

        With probability p_unmask → unmask:
          If temperature == 0: x_next[l] = argmax(logits[l])
          If temperature > 0:  x_next[l] = sample from Gumbel-perturbed logits
                               (logits[l] / temperature + Gumbel noise)
        Otherwise → keep masked: x_next[l] = MASK

     c. Positions that are already unmasked: x_next[l] = x_t[l]

     d. Clamp all PAD positions to their PAD token index

     e. If guidance_fn is provided: modify logits before step (b)
        If fixed_tokens/fixed_mask provided: clamp fixed positions after step (d)

     f. x_t = x_next

4. Final cleanup: any remaining MASK tokens → argmax from model at t ≈ 0
5. Return x_t
```

**Numerical stability guidelines for the diffusion module:**
- Add epsilon guard `1e-8` to all denominators involving `(1 - alpha_t)`, e.g. `1.0 / (1.0 - alpha_t + 1e-8)`.
- **Clamp `t` to `[1e-5, 1.0]`** in the loss computation to prevent `w(t) → ∞` as `t → 0`.
- Use `torch.float64` for Gumbel noise generation in `sampling.py` to avoid underflow in `log(-log(u))`.
- Clamp `alpha_t` to `[eps, 1 - eps]` before log operations to prevent `-inf` values.
- In the sampling loop, the final step (`t_next = 0`) makes `p_unmask = (alpha_0 - alpha_{1/N}) / (1 - alpha_{1/N})`. Since `alpha_0 ≈ 1`, this gives `p_unmask ≈ 1`, which correctly unmasks remaining tokens. The `1e-8` epsilon guard prevents division by zero.
- These patterns are standard in masked diffusion implementations (see MDLM repo and related codebases).

### 3.4 `bd_gen/eval/` — Evaluation

| File | Responsibility |
|------|---------------|
| `validity.py` | `check_validity(tokens, pad_mask) → dict` — checks: graph connectivity (BFS/DFS), spatial relationship consistency, room-type constraints (at most one LivingRoom, etc.), no contradictions. Returns per-check booleans + overall validity. |
| `metrics.py` | `validity_rate(samples)`, `novelty(samples, training_set)` (graph edit distance to nearest training sample), `diversity(samples)` (unique valid graphs / total), `distribution_match(samples, training_set)` (KL divergence of node/edge histograms), `per_class_accuracy(predictions, targets, mask)`. |

### 3.5 `bd_gen/viz/` — Visualization

| File | Responsibility |
|------|---------------|
| `graph_viz.py` | Draws a BD as a labeled graph using matplotlib + networkx. Nodes colored by room type, edges labeled with spatial relationship. Used in notebooks and for wandb image logging during training. |

### 3.6 `bd_gen/utils/` — Utilities

| File | Responsibility |
|------|---------------|
| `logging_utils.py` | wandb initialization, metric logging helpers, git commit hash capture (see Section 9.3). |
| `checkpoint.py` | Save/load model checkpoints (model state dict, optimizer state, epoch, config). |
| `seed.py` | Set random seeds for torch, numpy, Python random for reproducibility. |

---

## 4. Data Type Definitions

All key data structures are defined here as typed references. These are the canonical definitions used throughout the codebase — all modules produce and consume these types.

> **Tensor dtype convention used throughout:** Token indices are always `torch.long` (int64). Masks are always `torch.bool`. Floating-point values (logits, weights, timesteps) are always `torch.float32`.

```python
from typing import TypedDict
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class VocabConfig:
    """Single source of truth for N_MAX-derived constants. Immutable after creation."""
    n_max: int   # max rooms per graph (8 for RPLAN, 14 for ResPlan)
    # Properties: n_edges = n_max*(n_max-1)//2, seq_len = n_max + n_edges
    # Methods: compute_pad_mask(), edge_position_to_pair(), pair_to_edge_position()

class GraphDict(TypedDict):
    """Intermediate format produced by all dataset loaders, consumed by tokenizer.
    This is the universal exchange format between data loading and tokenization."""
    node_types: list[int]              # length = num_rooms, each value in [0, 12] (room type index)
    edge_triples: list[tuple[int, int, int]]  # (i, j, rel_type), all 0-indexed
                                       # i < j (upper triangle only), rel_type in [0, 9]
    num_rooms: int                     # number of real rooms, 1 <= num_rooms <= N_MAX

class BatchDict(TypedDict):
    """Single sample returned by BubbleDiagramDataset.__getitem__."""
    tokens: torch.Tensor               # shape (SEQ_LEN,), dtype=torch.long
                                       # where SEQ_LEN = vocab_config.seq_len
                                       # positions [0:N_MAX] index into NODE_VOCAB (0–14)
                                       # positions [N_MAX:SEQ_LEN] index into EDGE_VOCAB (0–12)
    pad_mask: torch.Tensor             # shape (SEQ_LEN,), dtype=torch.bool
                                       # True = real position, False = PAD position
    num_rooms: int                     # 1 <= num_rooms <= N_MAX

class ForwardProcessOutput(TypedDict):
    """Output of forward_mask()."""
    x_t: torch.Tensor                  # shape (B, SEQ_LEN), dtype=torch.long — noised tokens
    mask_indicators: torch.Tensor      # shape (B, SEQ_LEN), dtype=torch.bool
                                       # True = position WAS masked, False = kept or PAD

class ModelOutput(TypedDict):
    """Structured view of BDDenoiser.forward output (returned as tuple)."""
    node_logits: torch.Tensor          # shape (B, N_MAX, NODE_VOCAB_SIZE), dtype=float32
    edge_logits: torch.Tensor          # shape (B, N_EDGES, EDGE_VOCAB_SIZE), dtype=float32
```

---

## 5. Interface Contracts

Precise typed signatures for all key interfaces. These enable parallel implementation — agents can independently implement modules knowing the exact input/output contracts.

### 5.1 Data Pipeline

```python
# vocab.py — Module-level constants (N_MAX-independent)
NODE_TYPES: list[str]                               # 13 room types (see Section 3.1 for exact indices)
EDGE_TYPES: list[str]                               # 10 spatial relationships
NODE_VOCAB_SIZE: int = 15                            # 13 + MASK + PAD
EDGE_VOCAB_SIZE: int = 13                            # 10 + no-edge + MASK + PAD
NODE_MASK_IDX: int = 13
NODE_PAD_IDX: int = 14
EDGE_NO_EDGE_IDX: int = 10
EDGE_MASK_IDX: int = 11
EDGE_PAD_IDX: int = 12

# vocab.py — VocabConfig (N_MAX-dependent sizing)
@dataclass(frozen=True)
class VocabConfig:
    n_max: int                                       # max rooms per graph (8 for RPLAN)
    # Properties: n_edges, seq_len
    def compute_pad_mask(self, num_rooms: int) -> Tensor:    # shape (SEQ_LEN,), dtype=torch.bool — True for real positions
    def edge_position_to_pair(self, pos: int) -> tuple[int, int]:   # edge position (0 to n_edges-1) → (i, j) node pair
    def pair_to_edge_position(self, i: int, j: int) -> int:         # (i, j) node pair → edge position (0 to n_edges-1)

RPLAN_VOCAB_CONFIG = VocabConfig(n_max=8)            # SEQ_LEN=36
RESPLAN_VOCAB_CONFIG = VocabConfig(n_max=14)         # SEQ_LEN=105

# graph2plan_loader.py
def load_graph2plan(mat_path: str, cache_path: str) -> list[dict]:
    """Each dict: {"node_types": list[int], "edge_triples": list[tuple[int,int,int]], "num_rooms": int}"""

# tokenizer.py
def tokenize(graph_dict: dict, vocab_config: VocabConfig) -> tuple[Tensor, Tensor]:
    """Returns (tokens: Tensor(SEQ_LEN, dtype=torch.long), pad_mask: Tensor(SEQ_LEN, dtype=torch.bool))"""
def detokenize(tokens: Tensor, pad_mask: Tensor, vocab_config: VocabConfig) -> dict:
    """Inverse of tokenize → intermediate-format dict"""

# dataset.py
class BubbleDiagramDataset(torch.utils.data.Dataset):
    def __init__(self, ..., vocab_config: VocabConfig): ...
    edge_class_weights: Tensor                       # shape (EDGE_VOCAB_SIZE,), dtype=torch.float32
    node_class_weights: Tensor                       # shape (NODE_VOCAB_SIZE,), dtype=torch.float32
    num_rooms_distribution: Tensor                   # shape (vocab_config.n_max,), dtype=torch.float32 — histogram of room counts
    def __getitem__(self, idx) -> BatchDict:
        """Returns BatchDict (see Section 4 Data Type Definitions)"""
```

### 5.2 Model

```python
# embeddings.py
class NodeEmbedding(nn.Module):                      # (B, n_nodes) → (B, n_nodes, d_model)
class EdgeEmbedding(nn.Module):                      # (B, n_edges) → (B, n_edges, d_model)
class CompositePositionalEncoding(nn.Module):        # (B, SEQ_LEN, d_model) → (B, SEQ_LEN, d_model)
class TimestepEmbedding(nn.Module):                  # (B,) or scalar → (B, d_model)

# denoiser.py
class BDDenoiser(nn.Module):
    def __init__(self, ..., vocab_config: VocabConfig): ...
    def forward(
        self,
        tokens: Tensor,        # (B, SEQ_LEN), dtype=torch.long — token indices
        pad_mask: Tensor,      # (B, SEQ_LEN), dtype=torch.bool — True = real position
        t: Tensor,             # scalar, 0D, or (B,), dtype=torch.float32 — diffusion timestep
        condition: Tensor | None = None
            # v1: None (unconditional)
            # v2 (planned): (B, n_condition_tokens, d_condition) house boundary features
            #   Integration: cross-attention layers in transformer blocks (no-op when None)
    ) -> tuple[Tensor, Tensor]:
        """Returns (node_logits: (B, N_MAX, NODE_VOCAB_SIZE, dtype=float32),
                    edge_logits: (B, N_EDGES, EDGE_VOCAB_SIZE, dtype=float32))"""
```

### 5.3 Diffusion

```python
# noise_schedule.py
def get_noise(config) -> NoiseSchedule:
class NoiseSchedule:
    def sigma(self, t: Tensor) -> Tensor:            # total noise at time t; t and output are float32
    def alpha(self, t: Tensor) -> Tensor:            # keeping probability: alpha_t = exp(-sigma_t). Masking prob = 1 - alpha_t; float32
    def alpha_prime(self, t: Tensor) -> Tensor:      # d(alpha_t)/dt — needed for ELBO loss weight; float32
    def importance_sampling_transformation(self, t: Tensor) -> Tensor:  # optional; maps uniform t to importance-weighted t

# forward_process.py
def forward_mask(
    x0: Tensor,              # (B, SEQ_LEN), dtype=torch.long
    pad_mask: Tensor,        # (B, SEQ_LEN), dtype=torch.bool
    t: Tensor,               # (B,), dtype=torch.float32
    noise_schedule: NoiseSchedule
) -> tuple[Tensor, Tensor]:  # (x_t: (B, SEQ_LEN) torch.long, mask_indicators: (B, SEQ_LEN) torch.bool — True where token WAS MASKED, False where kept or PAD)

# loss.py
class ELBOLoss(nn.Module):
    def __init__(
        self,
        edge_class_weights: Tensor,              # (EDGE_VOCAB_SIZE,), dtype=torch.float32
        node_class_weights: Tensor | None = None  # (NODE_VOCAB_SIZE,), dtype=torch.float32; None = unweighted
    ):
    def forward(
        self,
        node_logits: Tensor,   # (B, N_MAX, NODE_VOCAB_SIZE), dtype=torch.float32
        edge_logits: Tensor,   # (B, N_EDGES, EDGE_VOCAB_SIZE), dtype=torch.float32
        x0: Tensor,            # (B, SEQ_LEN), dtype=torch.long — splits into node [:,:N_MAX] and edge [:, N_MAX:]
        x_t: Tensor,           # (B, SEQ_LEN), dtype=torch.long
        pad_mask: Tensor,      # (B, SEQ_LEN), dtype=torch.bool — splits into node/edge parts internally
        mask_indicators: Tensor,  # (B, SEQ_LEN), dtype=torch.bool
        t: Tensor,             # (B,), dtype=torch.float32 — clamped to [1e-5, 1.0] internally
        noise_schedule: NoiseSchedule
    ) -> Tensor:               # scalar loss, dtype=torch.float32 — normalized per-sample, averaged over batch

# sampling.py
def sample(
    model: BDDenoiser,
    noise_schedule: NoiseSchedule,
    vocab_config: VocabConfig,
    batch_size: int,
    num_steps: int,
    temperature: float = 0.0,
    guidance_fn: Callable | None = None,         # signature: (logits, x_t, t, pad_mask, **kwargs) → logits
    fixed_tokens: Tensor | None = None,
    fixed_mask: Tensor | None = None,
    remasking_fn: Callable | None = None,        # signature: (x_t, t_now, t_next, pad_mask) → x_t_remasked; None = standard MDLM
    num_rooms_distribution: Tensor | None = None, # (vocab_config.n_max,) histogram; None = uniform over [1, n_max]
    fixed_num_rooms: int | None = None,           # override: use this value for all samples
    device: str = "cpu"
) -> Tensor:                   # (B, SEQ_LEN) — generated token sequences
```

### 5.4 Evaluation

```python
# validity.py
def check_validity(tokens: Tensor, pad_mask: Tensor) -> dict:
    """Returns {"connected": bool, "consistent": bool, "valid_types": bool, "overall": bool, ...}"""

# metrics.py
def validity_rate(samples: list[dict]) -> float:
def novelty(samples: list[dict], training_set: list[dict]) -> float:
def diversity(samples: list[dict]) -> float:
def distribution_match(samples: list[dict], training_set: list[dict]) -> dict:
def per_class_accuracy(predictions: Tensor, targets: Tensor, mask: Tensor) -> dict:
```

---

## 6. Development Phases, Milestones, and Parallelization

Each phase maps to a feature branch, a set of deliverables, and a version tag. Phases are sequential unless noted otherwise.

### 6.0 Cross-Phase Parallelism

Phase 2 (Model) does **not** depend on Phase 1 (Data Pipeline). The model only needs vocab sizes and `VocabConfig` from `vocab.py` (built in Phase 0). After Phase 0 completes, Phase 1 and Phase 2 can run **in parallel**:

```
Phase 0 (Scaffold)
       │
       ├──► Phase 1 (Data Pipeline)       ← parallel
       │
       └──► Phase 2 (Model Architecture)  ← parallel
                │
                ▼
           Phase 3 (Diffusion Core)        ← integration point, needs Phase 1 + 2
                │
                ▼
           Phase 4 (Training Loop)
                │
                ▼
           Phase 5 (Evaluation)
```

Within each phase, tasks are organized into **parallel workstreams** where possible. Tests for each module can be written by a separate agent in parallel with the implementation, as long as the interface contracts (Section 5) are respected.

### Phase 0: Repository Scaffold
**Branch:** `setup/repo-scaffold` → merge to `main` → **Tag: `v0.1.0`**

**Parallel workstreams:**
```
Workstream A: pyproject.toml + Makefile + project README.md
Workstream B: Package skeleton (__init__.py files) + vocab.py + tests/test_vocab.py + tests/conftest.py
Workstream C: Hydra config files + .gitignore updates
```

**Tasks:**
1. Create `BD_Generation/bd_gen/` package structure (all `__init__.py` files)
2. Write `pyproject.toml` with all dependencies
3. Create Hydra config directory with placeholder YAML files
4. Update `.gitignore` with ML-specific entries (Section 11)
5. Create `Makefile` with common commands (see Section 2.1)
6. Implement `bd_gen/data/vocab.py` (all constants, helpers, and VocabConfig dataclass)
7. Write `tests/test_vocab.py`
8. Create `tests/conftest.py` with shared test fixtures, reused across phases 1–5:
   - `sample_batch()` → dict with `"tokens": Tensor(4, SEQ_LEN, dtype=torch.long)`, `"pad_mask": Tensor(4, SEQ_LEN, dtype=torch.bool)`, `"num_rooms": [2, 4, 6, 8]`. Tokens must contain valid vocab indices with correct PAD placement (node PAD at `NODE_PAD_IDX`, edge PAD at `EDGE_PAD_IDX`, real positions use random valid indices). Uses `RPLAN_VOCAB_CONFIG`.
   - `vocab_constants()` → dict of all vocab constants for easy assertion in tests.
   - `dummy_model()` → small `BDDenoiser(d_model=32, n_layers=1, n_heads=2, vocab_config=RPLAN_VOCAB_CONFIG)` for fast test execution. (Only usable after Phase 2.)
9. Write project `README.md` with setup instructions

**Deliverable:** `pip install -e .` works. `python -c "from bd_gen.data.vocab import NODE_VOCAB_SIZE, RPLAN_VOCAB_CONFIG; print(NODE_VOCAB_SIZE, RPLAN_VOCAB_CONFIG)"` runs.

**CPU only.**

### Phase 1: Data Pipeline
**Branch:** `data/graph2plan-loader` → merge to `main` → **Tag: `v0.2.0`**

**Can run in parallel with Phase 2.**

**Parallel workstreams:**
```
Workstream A: graph2plan_loader.py + tests/test_loader.py + scripts/prepare_data.py
Workstream B: tokenizer.py + tests/test_tokenizer.py
                 │
                 ▼ (both complete)
Sequential: dataset.py + tests/test_dataset.py + scripts/inspect_data.py
```

**Tasks:**
1. Download Graph2Plan `Data.zip` from `https://github.com/HanHan55/Graph2plan/releases/download/data/Data.zip`, extract `data.mat`
2. Implement `bd_gen/data/graph2plan_loader.py` — parse .mat, produce intermediate dicts, cache as .pt
3. Implement `bd_gen/data/tokenizer.py` — convert to SEQ_LEN-token sequences (36 for RPLAN)
4. Implement `bd_gen/data/dataset.py` — `BubbleDiagramDataset` with train/val/test splits and class frequency stats
5. Write `scripts/prepare_data.py` — end-to-end data preparation
6. Write `scripts/inspect_data.py` — print dataset statistics (sample count, room count distribution, edge type distribution, PAD ratio)
7. Create `notebooks/01_data_exploration.ipynb`
8. Write tests: `tests/test_loader.py` (verify .mat parsing, intermediate dict format, 1-based→0-based indexing), `tests/test_tokenizer.py`, `tests/test_dataset.py`

**Deliverable:** `python scripts/prepare_data.py` produces a cached .pt file. `BubbleDiagramDataset` returns correct shapes. PAD positions verified correct. Edge class statistics documented.

**CPU only.**

### Phase 2: Model Architecture
**Branch:** `model/transformer-denoiser` → merge to `main` → **Tag: `v0.3.0`**

**Can run in parallel with Phase 1.**

**Parallel workstreams:**
```
Workstream A: embeddings.py + transformer.py + tests/test_embeddings.py
Workstream B: configs/model/small.yaml + configs/model/base.yaml
                 │
                 ▼ (Workstream A complete)
Sequential: denoiser.py + tests/test_denoiser.py
```

**Tasks:**
1. Implement `bd_gen/model/embeddings.py` — all embedding modules
2. Implement `bd_gen/model/transformer.py` — encoder blocks with adaLN time conditioning
3. Implement `bd_gen/model/denoiser.py` — `BDDenoiser` wrapping embeddings + transformer + classification heads
4. Write `configs/model/small.yaml` and `configs/model/base.yaml`
5. Write tests: `tests/test_embeddings.py`, `tests/test_denoiser.py` (include gradient flow test — verify all parameters receive gradients — and adaLN zero-init verification)
6. Validate: random input `(B=4, SEQ_LEN)` produces correct output shapes `(B, N_MAX, NODE_VOCAB_SIZE)` and `(B, N_EDGES, EDGE_VOCAB_SIZE)` (for RPLAN: `(4, 36)` → `(4, 8, 15)` and `(4, 28, 13)`)

**Deliverable:** `BDDenoiser` forward pass runs with correct shapes. Parameter count matches expectations (1–5M).

**CPU only.**

### Phase 3: Diffusion Core
**Branch:** `diffusion/noise-and-loss` → merge to `main` → **Tag: `v0.4.0`**

**Requires Phase 1 and Phase 2 to be complete.**

**Parallel workstreams:**
```
Workstream A: noise_schedule.py + forward_process.py + tests/test_forward_process.py
Workstream B: loss.py + tests/test_loss.py
                 │
                 ▼ (both complete)
Sequential: sampling.py + tests/test_sampling.py
```

**Tasks:**
1. Implement `bd_gen/diffusion/noise_schedule.py` — linear + cosine schedules (direct implementation)
2. Implement `bd_gen/diffusion/forward_process.py` — masking with PAD protection
3. Implement `bd_gen/diffusion/loss.py` — MDLM ELBO loss with class weighting
4. Implement `bd_gen/diffusion/sampling.py` — reverse sampling loop with guidance hooks
5. Write `configs/noise/linear.yaml`, `configs/noise/cosine.yaml`
6. Write tests: `tests/test_forward_process.py`, `tests/test_loss.py`, `tests/test_sampling.py`

**Critical tests (PAD correctness):**
- Forward process: verify PAD positions are NEVER masked (stress test with 10,000 random samples across various `num_rooms` values)
- Loss: verify PAD positions contribute exactly zero to loss
- Loss: verify class weighting applies correctly to edge positions
- Sampling: verify PAD positions remain `[PAD]` at every timestep throughout the reverse trajectory
- Sampling: verify no `[MASK]` tokens remain in non-PAD positions after full denoising

**Deliverable:** Full diffusion pipeline data → mask → predict → loss computes correctly. Sampling loop produces SEQ_LEN-token sequences (36 for RPLAN) with correct vocabulary constraints.

**CPU only.**

### Phase 4: Training Loop
**Branch:** `training/basic-loop` → merge to `main` → **Tag: `v0.5.0`**

**Parallel workstreams:**
```
Workstream A: scripts/train.py (main training loop)
Workstream B: utils/checkpoint.py + utils/seed.py + utils/logging_utils.py
Workstream C: configs/training/default.yaml + notebooks/03_training_monitoring.ipynb
                 │
                 ▼ (all complete)
Sequential: tests/test_integration.py
```

**Tasks:**
1. Write `scripts/train.py` — Hydra-based training entry point
2. Integrate wandb logging (loss, learning rate, sample quality at intervals)
3. Implement `bd_gen/utils/checkpoint.py` — save/load checkpoints
4. Implement `bd_gen/utils/seed.py` — reproducibility
5. Implement `bd_gen/utils/logging_utils.py` — wandb init, git commit hash logging
6. Write `configs/training/default.yaml`
7. Write `tests/test_integration.py` — one full training step (forward + backward + optimizer step) + checkpoint save/load roundtrip test (save, reload, verify identical model output)
8. Run a CPU-only debug training for 100 steps to verify everything works
9. Create `notebooks/03_training_monitoring.ipynb`

**Deliverable:** `python scripts/train.py model=small data=graph2plan noise=linear` runs for 100 steps on CPU, loss decreases, checkpoint saved, run logged to wandb.

**CPU for development, GPU for real training** (see Section 10).

### Phase 5: Evaluation and Metrics
**Branch:** `eval/metrics-and-validity` → merge to `main` → **Tag: `v0.6.0`**

**Parallel workstreams:**
```
Workstream A: eval/validity.py + eval/metrics.py + tests
Workstream B: viz/graph_viz.py + scripts/sample.py + notebooks
                 │
                 ▼ (both complete)
Sequential: scripts/evaluate.py + configs/eval/default.yaml
```

**Tasks:**
1. Implement `bd_gen/eval/validity.py` — validity checker
2. Implement `bd_gen/eval/metrics.py` — all evaluation metrics
3. Write `scripts/evaluate.py` — generate N samples, compute all metrics, log to wandb
4. Write `scripts/sample.py` — generate and save/visualize samples
5. Implement `bd_gen/viz/graph_viz.py` — BD visualization
6. Write tests: `tests/test_validity.py`, `tests/test_metrics.py` (verify metrics on known inputs with expected outputs)
7. Write `configs/eval/default.yaml`
8. Create `notebooks/04_sample_analysis.ipynb`

**Deliverable:** `python scripts/evaluate.py` generates samples, computes validity/novelty/diversity, produces visual output. Results compared with other methods on RPLAN.

**CPU + GPU** (sampling at scale needs GPU).

### Phase Summary

| Phase | Branch | Tag | CPU/GPU | Parallel with |
|-------|--------|-----|---------|---------------|
| 0 — Scaffold | `setup/repo-scaffold` | `v0.1.0` | CPU | — |
| 1 — Data | `data/graph2plan-loader` | `v0.2.0` | CPU | Phase 2 |
| 2 — Model | `model/transformer-denoiser` | `v0.3.0` | CPU | Phase 1 |
| 3 — Diffusion | `diffusion/noise-and-loss` | `v0.4.0` | CPU | — |
| 4 — Training | `training/basic-loop` | `v0.5.0` | CPU → GPU | — |
| 5 — Evaluation | `eval/metrics-and-validity` | `v0.6.0` | CPU + GPU | — |

**v1.0.0** is tagged when the full system produces paper-ready results on RPLAN.

---

## 7. Testing Strategy

### 7.1 Unit Tests (per module, fast, CPU-only)

| Test file | What it verifies |
|-----------|-----------------|
| `test_vocab.py` | Vocabulary sizes correct. `edge_position_to_pair` is bijective. `compute_pad_mask` correct for various `num_rooms`. VocabConfig properties correct for multiple n_max values. |
| `test_loader.py` | `.mat` parsing produces correct intermediate dict format. 1-based→0-based indexing conversion. Edge count and room count consistency. |
| `test_tokenizer.py` | Round-trip: tokenize then detokenize recovers original graph. Correct PAD placement. Correct `no-edge` vs `[PAD]` distinction. |
| `test_dataset.py` | Output shapes `(SEQ_LEN,)`, dtype `long`, PAD mask consistency, train/val/test split sizes. |
| `test_embeddings.py` | Output shape `(B, SEQ_LEN, d_model)` after all embeddings applied. |
| `test_denoiser.py` | Output shapes `(B, N_MAX, NODE_VOCAB_SIZE)` and `(B, N_EDGES, EDGE_VOCAB_SIZE)`. Parameter count in expected range. Gradient flow (all parameters receive gradients). adaLN zero-init verification. |
| `test_forward_process.py` | PAD never masked (stress test). Masking rate matches α_t statistically. Fully masked at t=1. |
| `test_loss.py` | PAD contributes zero loss. Loss is non-negative. Loss decreases when logits match targets. |
| `test_sampling.py` | Output shape `(B, SEQ_LEN)`. No `[MASK]` tokens in final output. PAD positions preserved. |
| `test_validity.py` | Known-valid graphs pass. Known-invalid graphs fail specific checks. |
| `test_metrics.py` | Metrics produce expected values on known inputs. Edge cases (empty samples, all-identical samples). |

### 7.2 Integration Test

`test_integration.py` runs one complete cycle:
1. Load a batch from `BubbleDiagramDataset`
2. Apply forward process (mask tokens)
3. Pass through `BDDenoiser` (get logits)
4. Compute ELBO loss
5. Call `loss.backward()` — verify gradients flow
6. Take one optimizer step
7. Verify loss is a valid scalar (not NaN, not Inf)
8. Checkpoint roundtrip: save checkpoint, reload into fresh model, verify identical output on same input

### 7.3 PAD Stress Tests (Critical Correctness)

PAD handling is the most important correctness requirement in this project. Dedicated stress tests:

- Generate 10,000 random masking operations with `num_rooms` from 1 to N_MAX. **Specifically include `num_rooms=1` and `num_rooms=2`** (most PAD, highest risk of edge-case bugs). Verify that **every** PAD position remains unmasked **every** time.
- Run 100 full sampling trajectories. Verify PAD positions are `[PAD]` at **every** timestep.
- Test forward process at `t=1e-6` and `t=1-1e-6`: verify masking rate matches `alpha_t` even at extremes.
- Minimal-graph loss test: create a batch with `num_rooms=1` where the single node is not masked (use a `t` value close to 0). Verify loss = 0 (no masked positions to contribute loss). Verify no NaN/Inf in any intermediate computation including `w(t)`.

### 7.4 Adversarial & Collapse-Detection Tests

These test **individual pathological examples** designed to trigger model failure modes. Each test targets a specific collapse scenario.

| Test | Input | What it catches | Expected behavior |
|------|-------|-----------------|-------------------|
| **Single-room graph** | `num_rooms=1`: 1 node, 0 real edges, SEQ_LEN - 1 PAD (35 for RPLAN) | PAD exclusion bugs. Forward process with 1 maskable position. Sampling must produce exactly 1 valid room type + all PAD. | Loss is finite and non-zero (from the 1 node). All SEQ_LEN - 1 edge+extra-node positions contribute zero to loss. Sampled output has exactly 1 non-PAD node. |
| **Two-room graph** | `num_rooms=2`: 2 nodes, 1 real edge, SEQ_LEN - 3 PAD (33 for RPLAN) | Minimal edge case. Only 1 real edge position — no redundancy. Tests loss computation when almost everything is PAD. | Loss computed only on 3 positions (2 nodes + 1 edge). |
| **Maximum-size graph** | `num_rooms=N_MAX`: 0 PAD positions, all SEQ_LEN real | Stress-tests the no-PAD path. At `t~0.99`, nearly all SEQ_LEN positions are masked — model has almost no context. | No NaN gradients. Forward pass completes. Loss is finite. |
| **All-same-node-type** | N_MAX nodes all set to `LivingRoom` (idx 2) | Detects if model collapses to predicting the modal class. Unrealistic graph but tests node prediction diversity. | Loss correctly computed. Model does not crash or produce NaN. |
| **Fully connected graph** | All N_EDGES edges are spatial relationships (zero `no-edge`) | Inverse of the sparsity problem. Tests whether class weights explode when the rare "all-edges-present" case appears. | Loss is finite. Gradient magnitudes comparable to a typical batch. |
| **All-no-edge graph** | All N_EDGES real edges are `no-edge` (idx 10) | Tests the dominant mode. If model collapsed to always predicting no-edge, this gets perfect accuracy — verifies loss is still meaningful. | Loss is small but non-zero (node positions contribute). |
| **Boundary timestep: t=0** | `t=1e-6` | `alpha_t -> 1`, nothing masked, `w(t) -> large`. Tests numerical stability. | Zero masked positions. Loss = 0. No NaN/Inf in w(t) computation. |
| **Boundary timestep: t=1** | `t=1-1e-6` | `alpha_t -> 0`, everything masked. Tests model with fully-masked input. | All non-PAD positions masked. Loss is finite. Model produces valid logits. |
| **Extreme batch mix** | Batch with `num_rooms=[1, 2, N_MAX-1, N_MAX]` | Variable PAD patterns cause shape mismatches, broadcasting errors, or loss domination by large graphs. | All 4 samples processed correctly. Per-sample loss is finite. After normalization, contributions are balanced. |
| **Contradictory edges** | Edge (A,B)=`left-of` but the reverse direction also set to `left-of` | Tokenizer/validity checker handles contradictions. Loss still computes (validity is the checker's job). | Loss succeeds. `check_validity()` returns `consistent=False`. |
| **Gradient magnitude comparison** | Compare `\|\|grad_node\|\|` vs `\|\|grad_edge\|\|` on typical batch, with and without class weights | Edge class weights cause edge gradients to overwhelm node gradients (or vice versa). A >10x imbalance indicates problems. | Gradient magnitudes within 10x. Log the ratio. |
| **Mode collapse canary** | After 100 training steps, sample 50 graphs. Compute: (a) entropy of predicted logits, (b) number of unique graphs. | Detects early mode collapse. If entropy drops to near-zero or all 50 samples are identical, model has collapsed. | Entropy > 0.5 * log(vocab_size). At least 10 unique graphs. |
| **Degenerate Gumbel noise** | Sampling with `temperature=1e-10` vs `temperature=0` (argmax) | Gumbel path must degrade gracefully to argmax at near-zero temperature. | Outputs match exactly. |

### 7.5 Running Tests

```bash
# All tests
make test
# which runs: pytest tests/ -v

# Specific module
pytest tests/test_forward_process.py -v

# Just the PAD stress test
pytest tests/test_forward_process.py::test_pad_never_masked_stress -v
```

Pytest configuration in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
```

---

## 8. Dependency Management

### 8.1 Python Version

Python 3.10+ (required for latest PyTorch compatibility). The project venv (`Davis/.venv/`) currently uses Python 3.14.2.

### 8.2 Virtual Environment

Use the existing repo-level venv at `Davis/.venv/`. Install the BD Generation package in editable mode:

```bash
pip install -e BD_Generation/
pip install -e "BD_Generation/[dev]"   # includes pytest, ruff, ipykernel
```

### 8.3 Dependencies (`pyproject.toml`)

```toml
[project]
name = "bd-gen"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "scipy>=1.11.0",            # loadmat for Graph2Plan .mat files
    "hydra-core>=1.3.0",        # Configuration management
    "omegaconf>=2.3.0",         # Config objects (Hydra dependency)
    "networkx>=3.1",            # Graph operations for evaluation
    "matplotlib>=3.7.0",        # Visualization
    "wandb>=0.16.0",            # Experiment tracking
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",              # Linting
    "ipykernel>=6.25.0",        # Jupyter notebooks
]
```

---

## 9. Configuration Management

Hydra externalizes all hyperparameters into composable YAML configs with CLI overrides. Resolved configs are auto-saved to `outputs/` for reproducibility.

### 9.1 Config Structure

**Main config** (`configs/config.yaml`):
```yaml
defaults:
  - model: small         # loads configs/model/small.yaml
  - data: graph2plan     # loads configs/data/graph2plan.yaml
  - noise: linear        # loads configs/noise/linear.yaml
  - training: default    # loads configs/training/default.yaml
  - eval: default        # loads configs/eval/default.yaml
  - _self_

seed: 42
experiment_name: "bd_gen_v1"
wandb:
  project: "bd-generation"
  entity: null            # set via env var or CLI override
  mode: "online"          # "disabled" for local debugging
```

**Sub-config file contents:**

```yaml
# configs/model/small.yaml
d_model: 128
n_layers: 4
n_heads: 4
cond_dim: 128              # timestep embedding output dim; MUST equal d_model in v1 (v2 may differ with projection layer)
mlp_ratio: 4               # FFN hidden dim = mlp_ratio * d_model = 512
dropout: 0.1
frequency_embedding_size: 256  # sinusoidal timestep embedding input dim

# configs/model/base.yaml
d_model: 256
n_layers: 6
n_heads: 8
cond_dim: 256
mlp_ratio: 4
dropout: 0.1
frequency_embedding_size: 256

# configs/data/graph2plan.yaml
n_max: 8                               # max rooms per graph — used to instantiate VocabConfig(n_max=8)
mat_url: "https://github.com/HanHan55/Graph2plan/releases/download/data/Data.zip"
mat_path: "data/data.mat"              # relative to BD_Generation/
cache_path: "data_cache/graph2plan_nmax8.pt"  # cached tokenized dataset (encodes N_MAX in filename)
batch_size: 256
num_workers: 4
splits:
  train: 0.8
  val: 0.1
  test: 0.1

# configs/noise/linear.yaml
type: linear
sigma_min: 0.0
sigma_max: 10.0

# configs/noise/cosine.yaml
type: cosine
eps: 1e-3

# configs/training/default.yaml
lr: 3e-4
weight_decay: 0.01
warmup_steps: 1000
epochs: 500
optimizer: adamw
grad_clip: 1.0
checkpoint_every: 50       # epochs
sample_every: 25           # epochs — generate samples and log to wandb
val_every: 5               # epochs
ema: false                 # v1: no EMA; add in v2 if needed
importance_sampling: false  # v1: uniform t-sampling; set true for MDLM importance sampling

# configs/eval/default.yaml
num_samples: 1000
sampling_steps: 100
temperature: 0.0            # deterministic argmax (0.0); set > 0 for stochastic
metrics: [validity, novelty, diversity, distribution_match]
```

**Running experiments via CLI overrides:**
```bash
# Default config
python scripts/train.py

# Switch model size
python scripts/train.py model=base

# Tune learning rate
python scripts/train.py training.lr=1e-4 training.epochs=100

# Debug locally without wandb
python scripts/train.py wandb.mode=disabled training.epochs=5

# Combine overrides freely
python scripts/train.py model=base noise=cosine training.lr=5e-4 seed=123
```

### 9.2 Experiment Tracking with wandb

**Why wandb:** Free for personal and academic use, excellent for comparing ML experiment runs across hyperparameters, integrates trivially with PyTorch and Hydra. Superior to TensorBoard for managing dozens of experiment runs.

**What we log:**
- Training loss (per step, per epoch)
- Validation loss (per epoch)
- Per-class accuracy: node accuracy, no-edge accuracy, spatial-relationship accuracy (per epoch) — critical for detecting the edge sparsity problem
- Learning rate (per step)
- Sample quality metrics: validity rate, diversity (every N epochs)
- Generated sample visualizations as wandb Images (every N epochs)
- Full resolved Hydra config (logged once at init)
- **Git commit hash** — links each wandb run to the exact code version:

```python
import subprocess
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
wandb.config.update({"git_commit": commit_hash})
```

This allows tracing any wandb run back to the exact code that produced it.

---

## 10. Google Cloud GPU Strategy

**Phases 0–3:** CPU only. All coding, testing, and debug training runs happen locally.

**Phases 4–5:** GPU needed. A single T4 or V100 is sufficient (1–5M parameters, 36-token sequences (RPLAN)). Estimated full training: a few hours for 500 epochs on 80K samples.

**Workflow:** Develop + test locally → `git push` → `git clone` on cloud VM → `pip install -e .` → train → monitor via wandb → stop VM.

See `google_cloud_usage_efficiency.md` for detailed cost-saving practices (spot VMs, checkpointing, billing alerts).

---

## 11. Version Control Best Practices for ML

### 11.1 .gitignore Strategy

The Davis repository tracks **only** the `BD_Generation/` directory. All other directories at the root level (`Others/`, `DiDAPS_COPY/`, `Papers/`, etc.) are excluded from version control.

The `.gitignore` at the Davis root should contain:

```gitignore
# ---- Exclude everything except BD_Generation/ ----
/*
!/BD_Generation/
!/.gitignore
!/README.md

# ---- ML Data & Artifacts (within BD_Generation/) ----
BD_Generation/*.mat
BD_Generation/*.pt
BD_Generation/*.pth
BD_Generation/*.ckpt
BD_Generation/*.safetensors
BD_Generation/*.npz
BD_Generation/*.npy
BD_Generation/*.h5
BD_Generation/*.hdf5
BD_Generation/Data/
BD_Generation/data/
BD_Generation/data_cache/

# ---- Experiment outputs ----
BD_Generation/outputs/      # Hydra output directories
BD_Generation/wandb/        # wandb local logs
BD_Generation/runs/         # TensorBoard runs (if used)

# ---- Large binary files ----
BD_Generation/*.zip
BD_Generation/*.tar.gz
BD_Generation/*.tar.bz2

# ---- Environment ----
BD_Generation/.env
BD_Generation/*.env

# ---- IDE / tooling ----
.claude/
.tldr/
.tldrignore
```

### 11.2 What to Commit vs What to Ignore

**Commit:** All Python source code, config YAML files, tests, notebooks (with outputs cleared), documentation, Makefile, pyproject.toml, README files.

**Do NOT commit:** Training data (.mat, .pt caches), model checkpoints, wandb logs, Hydra output directories, large binary files, generated images/plots (unless for documentation), virtual environment directories.

### 11.3 Separation of Concerns: Git vs wandb

- **Git tracks code and configs**: what model architecture, what loss function, what data pipeline.
- **wandb tracks runs**: what hyperparameters were used, what the loss curves look like, what the sample quality is.
- **Hydra outputs track reproducibility**: the exact resolved config for each run, auto-saved.
- **Link**: each wandb run logs its git commit hash, allowing any run to be traced to its exact code version.

---

## 12. Documentation Standards

Docstrings on all public classes and functions (what it does, parameters, returns). Inline comments only where the logic is non-obvious — specifically:
- PAD handling logic (why certain positions are excluded, why PAD != no-edge)
- ELBO weighting formula (the `1 / (1 - alpha_t)` factor and where it comes from)
- Edge sparsity mitigation (why class weights are computed this way)
- Composite positional encoding (which embedding applies to which positions and why)

The rule: explain the **why**, not the **what**. `# exclude PAD from loss` is unnecessary. `# PAD positions carry no information — including them would reward trivial predictions and dominate the gradient` is useful.

The project README (`BD_Generation/README.md`) must contain: what the project does, architecture overview, getting started instructions, project structure, how to run experiments, and links to `research_T1.md`.

Notebook outputs are **cleared before committing** (only source cells are tracked).

---

## 13. Key Architectural Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package structure | `bd_gen/` installable via pyproject.toml | Proper Python packaging; importable from scripts, tests, notebooks |
| Config system | Hydra | Composable configs, CLI overrides, auto-saved resolved configs for reproducibility |
| Experiment tracking | wandb | Free for solo/academic use; ML-optimized dashboards; Hydra integration |
| MDLM code | Direct implementation; MDLM paper/repo as mathematical reference only | Our variable-length PAD-aware sequences (SEQ_LEN depends on dataset N_MAX) need tailored logic; adapter wrapping adds complexity without benefit; all future extensions (ReMDM, guidance) modify our sampling loop, not MDLM's |
| Transformer impl | Custom encoder blocks with adaLN (not `nn.TransformerEncoder`) | adaLN requires custom blocks for time-conditioned modulation; SEQ_LEN tokens (36 for RPLAN, 105 for ResPlan) makes flash attention unnecessary |
| Branching | Trunk-based + feature branches | Simple for solo; scales to team with PR reviews; preserves history |
| Data caching | Parse .mat once → save as .pt | Avoids repeated heavy I/O; required by cloud cost-saving strategy |
| Noise schedule | Factory pattern, config-driven | Easy swapping via config; no code changes for new schedules |
| PAD handling | Explicit `pad_mask` propagated through entire pipeline + PAD attention masking in transformer | Most critical correctness requirement; extensively stress-tested including adversarial cases |
| Loss normalization | Per-sample normalization by N_active positions | Prevents large graphs from dominating gradient over small graphs |
| Timestep sampling | Uniform (default) with optional importance sampling | Reduces ELBO gradient variance; toggle via config flag |
| N_MAX configurability | VocabConfig frozen dataclass, n_max from Hydra data config | Right-sizes to dataset; RPLAN N_MAX=8 gives 14x attention reduction vs N_MAX=16; single source of truth prevents model/data mismatch |
| Parallelization | Interface contracts enable parallel agent workstreams | Phase 1 ∥ Phase 2; within-phase workstreams per dependency DAG |

---

## Appendix A: Extensibility Map (Post-v1)

This implementation framework is designed from the start to support future extensions without rewriting the core. The evaluation pipeline is complete and all hooks listed below are in place. **Blocker analysis (2026-02-18): nothing in the codebase is blocking any of the five planned extensions.**

### A.1 Hook Inventory

| Future capability | Current hook | Where it lives | Status / Concrete interface |
|---|---|---|---|
| **ReMDM max-capped remasking** | `remasking_fn(x_t, t_now, t_next, pad_mask)` | `bd_gen/diffusion/sampling.py`, `bd_gen/diffusion/remasking.py` | **Implemented.** `RemaskingSchedule` with cap + rescale strategies. eta=0.1 evaluated (see `eval_results/comparison.md`). No retraining. |
| **ReMDM confidence-based remasking** | `remasking_fn` + `unmasking_mode="confidence"` | `bd_gen/diffusion/sampling.py`, `bd_gen/diffusion/remasking.py` | **Implemented.** Confidence-based unmasking (LLaDA-inspired, Nie et al., arXiv:2502.09992) already in sampling.py. Combine with `RemaskingSchedule` to re-mask low-confidence decoded tokens. No retraining. |
| **MELD — learned forward process** | Per-position masking rates | `bd_gen/diffusion/noise_schedule.py`, `forward_process.py`, `loss.py`, `sampling.py` | **Blocker: MEDIUM.** Learned per-position rates β_l(t) require a parallel path through the entire diffusion pipeline (see Roadmap item 3 for details). Existing MDLM code paths remain intact. **Requires retraining.** |
| **Constrained generation via guidance** | `guidance_fn` + `fixed_tokens`/`fixed_mask` | `bd_gen/diffusion/sampling.py` | **Ready.** `guidance_fn(logits, x_t, t, pad_mask, **kwargs) → modified_logits` already threaded. SMC-based derivative-free guidance with GRPO-learned proposals (see `Presentation_BPI.pdf` Section 1.3). Architectural constraints in `constrains_examples.md`. No retraining for graph-level constraints. |
| **Conditioned generation** (house boundary / latent z_t) | `BDDenoiser.forward(... condition=None)` | `bd_gen/model/denoiser.py` | **Ready.** v2: `condition: Tensor(B, n_cond_tokens, d_cond)`. Integration via cross-attention layers in transformer blocks (no-op when `None`). Adding cross-attention is a parameter change, not a rewrite. |
| **Joint BD + continuous co-generation** | Modular embeddings + classification heads + `condition` placeholder | `bd_gen/model/`, `bd_gen/diffusion/` | **Ready for extension.** Vocab system (`VocabConfig`) is modular, embedding layer is modular. Current BDDenoiser becomes the discrete head; continuous geometry head and shared latent z_t require new modules but no rewrites. See Roadmap item 5. |
| **New noise schedules** | Factory pattern `get_noise(config)` | `bd_gen/diffusion/noise_schedule.py`, `configs/noise/` | Implement `NoiseSchedule` interface (sigma, alpha, alpha_prime, optionally importance_sampling_transformation). |

### A.2 Post-v1 Experiment Roadmap

All experiments below require the evaluation pipeline to be complete (it is). Listed in recommended order:

1. **ReMDM max-capped eta sweep** — Try eta = [0.1, 0.2, 0.3, 0.4, 0.5] with cap strategy. **DONE** — all 5 etas evaluated (see `eval_results/comparison.md`). Sweet spot at eta=0.2–0.3. No retraining.

2. **ReMDM confidence-based remasking** (ReMDM paper Section 4.1) — A **remasking strategy** where the per-position remasking probability depends on model confidence: low-confidence decoded tokens are more likely to be re-masked, high-confidence ones are kept. This is distinct from LLaDA confidence-based *unmasking* (which decides which masked tokens to reveal). Implementation requires the remasking function to receive model logits so it can compute per-position confidence scores. No retraining.
   - **NOT** `unmasking_mode="confidence"` — that is LLaDA-style unmasking (already implemented and evaluated separately as `remdm_cap_eta*_confidence` in comparison.md, but is a different experiment).
   - Needs: new `strategy="confidence"` in `RemaskingSchedule`, modified `__call__` signature to accept logits, updated `sampling.py` to pass logits to remasking_fn.

3. **MELD — learned forward process** — Per-element learned rates β_l(t) to avoid state-clashing in edges (Sahoo et al., arXiv:2410.02940). Symmetric edge pairs (e.g., left-of/right-of) can conflict when masked/unmasked at the same rate. MELD learns per-position corruption rates so correlated positions can be scheduled differently. **Requires retraining + touches every diffusion module:**
   - `noise_schedule.py`: new `LearnedRateSchedule` with trainable rate network β_l(t) → per-position α_l(t) = exp(−∫β_l(s)ds) and α'_l(t)
   - `forward_process.py`: masking probability becomes per-position `1 − α_l(t)` instead of scalar broadcast
   - `loss.py`: ELBO weight becomes per-position `w_l(t) = β_l(t) · α_l(t) / (1 − α_l(t))` instead of global w(t)
   - `sampling.py`: unmasking probability becomes per-position `(α_l(t_next) − α_l(t_now)) / (1 − α_l(t_now))`
   - New trainable component: rate network mapping (position features, t) → β_l(t), jointly trained with the denoiser
   - The existing code paths remain valid for the standard MDLM schedule — MELD adds a parallel per-position path. No existing code needs rewriting, but every module needs a MELD-aware branch.

4. **Constrained generation via guidance** — SMC-based derivative-free guidance (see `Presentation_BPI.pdf` Section 1.3) with non-differentiable reward functions encoding architectural constraints (see `constrains_examples.md`). Two constraint categories:
   - **Graph-level constraints** (applicable to BD-only v1): required program (1 living room, 1 kitchen), connectivity, kitchen adjacency, bedroom access, bathroom-kitchen separation. Reward = binary constraint satisfaction.
   - **Geometry-level constraints** (requires joint model): room sizes, aspect ratios, exterior exposure. Deferred to after joint architecture.
   - Approach: SMC particles + GRPO-learned proposals. `guidance_fn` hook already threaded. No retraining for graph-level constraints.

5. **Joint BD + continuous co-generation** — The final goal. Pair discrete graph generation (BD) with continuous diffusion over room geometry (bounding boxes or wall-junction vertices). Communication through shared latent space z_t (see `Presentation_BPI.pdf` Section 2):
   - Factorized reverse step: `p(x_{t-1}, y_{t-1}, z_{t-1} | x_t, y_t, z_t)` — discrete and continuous updates are conditionally independent given z_t
   - Discrete head: current BDDenoiser conditioned on z_t (via `condition` argument → cross-attention)
   - Continuous head: DDPM-style geometry denoiser conditioned on z_t
   - Latent update: z_{t-1} from encoder over (x_t, y_t, z_t)
   - Training: L_total = λ_disc·L_disc + λ_y·L_y + λ_z·L_z + λ_geo·L_geo-aux (CCDD recipe, Zhou et al. 2025)
   - Requires new modules, new training loop, new dataset loader for geometry. Current BD code becomes the discrete head.

---

## References

| Formula source | Paper/Repo |
|---------------|------------|
| Noise schedules, ELBO loss, sampling | MDLM (Sahoo et al., arXiv:2406.07524), repo: `github.com/kuleshov-group/mdlm` |
| adaLN-Zero conditioning | DiT (Peebles & Xie, ICCV 2023), repo: `github.com/facebookresearch/DiT` |
| Evaluation metrics | DiGress (Vignac et al.), repo: `github.com/cvignac/DiGress` |
| Dataset (RPLAN BDs) | Graph2Plan (Han et al.), repo: `github.com/HanHan55/Graph2plan` |
| Remasking (inference-time scaling) | ReMDM (Shi et al., arXiv:2503.00307), repo: `github.com/liuqidong07/ReMDM` |
| Learned forward process | MELD (Sahoo et al., arXiv:2410.02940), repo: `github.com/kuleshov-group/meld` |
| Training-free guidance | Nisonoff et al. (arXiv:2409.07359), repo: `github.com/nisonoff/discrete_guidance` |
| Confidence-based unmasking | LLaDA (Nie et al., arXiv:2502.09992), repo: `github.com/ML-GSAI/LLaDA` |
| Coevolutionary continuous-discrete diffusion | CCDD (Zhou et al., 2025) |
| Diffusion-based vector floorplan generation | GSDiff (Hu et al., 2025) |
