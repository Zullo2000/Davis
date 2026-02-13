# Model Architecture Reference (`bd_gen.model`)

## 1. Purpose

The model module implements the **transformer denoiser** for the MDLM-based bubble diagram generator. In the diffusion pipeline, the denoiser receives:

- **Noised token sequence** `(B, SEQ_LEN)`: partially masked bubble diagram tokens (nodes + edges)
- **PAD mask** `(B, SEQ_LEN)`: which positions are real (True) vs padding (False)
- **Timestep** `t ∈ [0, 1]`: how much noise was applied (0 = clean, 1 = fully masked)

And produces:

- **Node logits** `(B, N_MAX, NODE_VOCAB_SIZE)`: per-node probability over 15 room types
- **Edge logits** `(B, N_EDGES, EDGE_VOCAB_SIZE)`: per-edge probability over 13 edge types

The diffusion loss and sampling loop (Phase 3) use these logits to train and generate bubble diagrams.

### Why a custom transformer (not `nn.TransformerEncoder`)

PyTorch's built-in `nn.TransformerEncoder` has no mechanism to inject time-dependent conditioning into each block. Our adaLN-Zero design requires predicting 6 modulation parameters (shift, scale, gate for both attention and FFN) from the timestep embedding at each block. This is impossible with the standard API — we need custom transformer blocks.

---

## 2. Architecture Overview

```
Input: tokens (B, 36)    pad_mask (B, 36)    t (scalar or (B,))
         │                     │                     │
         ▼                     │                     ▼
  ┌─────────────┐              │           ┌──────────────────┐
  │ Split into  │              │           │ _process_t()     │
  │ nodes (B,8) │              │           │ normalize to (B,)│
  │ edges (B,28)│              │           └────────┬─────────┘
  └──────┬──────┘              │                    ▼
         │                     │           ┌──────────────────┐
    ┌────┴────┐                │           │ TimestepEmbedding│
    ▼         ▼                │           │ sinusoidal + MLP │
┌────────┐ ┌────────┐         │           └────────┬─────────┘
│NodeEmb │ │EdgeEmb │         │                    ▼
│(B,8,D) │ │(B,28,D)│         │           ┌──────────────────┐
└───┬────┘ └───┬────┘         │           │ SiLU activation  │
    │          │               │           │ → c: (B, D)      │
    ▼          ▼               │           └────────┬─────────┘
┌───────────────────┐          │                    │
│ cat → (B, 36, D)  │          │                    │
└────────┬──────────┘          │                    │
         ▼                     │                    │
┌───────────────────┐          │                    │
│ + CompositePos    │          │                    │
│   Encoding (36,D) │          │                    │
└────────┬──────────┘          │                    │
         │                     ▼                    │
         │            ┌──────────────┐              │
         │            │ ~pad_mask    │              │
         │            │ → attn_mask  │              │
         │            │ True=IGNORE  │              │
         │            └──────┬───────┘              │
         │                   │                      │
         ▼                   ▼                      ▼
  ┌──────────────────────────────────────────────────┐
  │            AdaLN Block × n_layers                │
  │  (each block receives x, c, and attn_mask)       │
  └────────────────────┬─────────────────────────────┘
                       ▼
              ┌──────────────────┐
              │ Final adaLN      │
              │ (shift + scale)  │
              │ + LayerNorm      │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Split into       │
              │ nodes (B,8,D)    │
              │ edges (B,28,D)   │
              └───────┬──────────┘
                 ┌────┴─────┐
                 ▼          ▼
           ┌──────────┐ ┌──────────┐
           │ NodeHead  │ │ EdgeHead │
           │ (B,8,15)  │ │(B,28,13) │
           └──────────┘ └──────────┘
```

All dimension values shown are for RPLAN (n_max=8, seq_len=36) with `D = d_model`.

---

## 3. Embedding Layer Design (`embeddings.py`)

### 3.1 Separate Node and Edge Embedding Tables

**What:** Two independent `nn.Embedding` modules.
- `NodeEmbedding`: maps indices 0–14 → d_model vectors (15 classes: 13 room types + MASK + PAD)
- `EdgeEmbedding`: maps indices 0–12 → d_model vectors (13 classes: 10 relationships + no-edge + MASK + PAD)

**Why separate tables, not a single shared embedding:** Nodes and edges have fundamentally different vocabularies with different semantics. Room types (LivingRoom, Kitchen, etc.) and spatial relationships (left-of, above, etc.) occupy completely different semantic spaces. A shared embedding would force these into the same representation space, conflating "Bathroom" with "left-of" at the embedding level. Separate tables allow each to learn domain-specific representations. The transformer's self-attention then learns cross-domain relationships between node and edge representations.

**Usage:**
```python
from bd_gen.model.embeddings import NodeEmbedding, EdgeEmbedding

node_emb = NodeEmbedding(d_model=128)       # nn.Embedding(15, 128)
edge_emb = EdgeEmbedding(d_model=128)       # nn.Embedding(13, 128)

node_vectors = node_emb(node_tokens)        # (B, 8, 128)
edge_vectors = edge_emb(edge_tokens)        # (B, 28, 128)
x = torch.cat([node_vectors, edge_vectors], dim=1)  # (B, 36, 128)
```

### 3.2 CompositePositionalEncoding — Full Breakdown

**What:** A learned positional encoding with three components that capture both entity type (node vs edge) and graph structure (which room slot, which room pair).

**Why not standard sinusoidal or learned positional encoding:** Our token sequence is a flattened graph — the first 8 positions are room nodes and the next 28 are edge tokens, laid out in upper-triangle row-major order. Standard position indices (0, 1, 2, ..., 35) would be meaningless here: position 8 (first edge) is not "adjacent" to position 7 (last node) in any structural sense. Instead, we encode what each position *represents* in the graph.

**The three embedding components:**

#### entity_type_emb: `nn.Embedding(2, d_model)`

- Index 0 = node position, index 1 = edge position
- **Why:** The transformer needs to know whether a position represents a room or a spatial relationship. Without this, the model would have to infer entity type purely from the token embedding, which wastes capacity. With this signal, the model can immediately apply different processing strategies (e.g., attend differently to node-node vs node-edge vs edge-edge pairs).

#### node_index_emb: `nn.Embedding(n_max, d_model)`

- One embedding per room slot (0 through n_max-1)
- Applied only to node positions
- **Why:** Without this, all node positions would be indistinguishable — the model would see 8 identical "node-type" positions and couldn't learn that room 0's relationships differ from room 5's. This embedding gives each room slot a unique identity.

#### pair_index_emb: `nn.Embedding(n_max, d_model)`

- Shared across both endpoints of an edge
- For edge connecting rooms (i, j): encoding = `pair_index_emb(i) + pair_index_emb(j)`
- Applied only to edge positions
- **Why shared (not separate source/target tables):** The edge type token already encodes directionality (e.g., "left-of" vs "right-of"). The positional encoding only needs to say *which rooms* are involved, not which is source/target. Using a single shared table and adding the embeddings makes the positional encoding order-invariant for the pair, which is correct since `edge(2,5)` and the hypothetical `edge(5,2)` refer to the same physical relationship. The addition is commutative: `pair_emb(2) + pair_emb(5) = pair_emb(5) + pair_emb(2)`.

#### Buffer precomputation

All index tensors (`node_indices`, `entity_types`, `edge_i`, `edge_j`) are precomputed in `__init__` and registered as buffers via `register_buffer()`. This ensures:
- They automatically move to the correct device with `.to(device)` or `.cuda()`
- They are not treated as parameters (no gradients, not in `model.parameters()`)
- The edge endpoint loop (28 iterations for RPLAN) runs only once at construction

#### Concrete example (n_max=4, seq_len=10)

```
Positions 0-3 (nodes):
  pos[0] = entity_type_emb(0) + node_index_emb(0)   ← room 0
  pos[1] = entity_type_emb(0) + node_index_emb(1)   ← room 1
  pos[2] = entity_type_emb(0) + node_index_emb(2)   ← room 2
  pos[3] = entity_type_emb(0) + node_index_emb(3)   ← room 3

Positions 4-9 (edges, upper-triangle row-major):
  pos[4] = entity_type_emb(1) + pair_index_emb(0) + pair_index_emb(1)  ← edge (0,1)
  pos[5] = entity_type_emb(1) + pair_index_emb(0) + pair_index_emb(2)  ← edge (0,2)
  pos[6] = entity_type_emb(1) + pair_index_emb(0) + pair_index_emb(3)  ← edge (0,3)
  pos[7] = entity_type_emb(1) + pair_index_emb(1) + pair_index_emb(2)  ← edge (1,2)
  pos[8] = entity_type_emb(1) + pair_index_emb(1) + pair_index_emb(3)  ← edge (1,3)
  pos[9] = entity_type_emb(1) + pair_index_emb(2) + pair_index_emb(3)  ← edge (2,3)
```

Notice how edges connecting the same room share components: edge (0,1) and edge (0,2) both contain `pair_index_emb(0)`, encoding that room 0 is involved in both relationships. This lets the model learn structural patterns like "rooms adjacent to room 0 tend to share properties".

**Usage:**
```python
from bd_gen.model.embeddings import CompositePositionalEncoding
from bd_gen.data.vocab import RPLAN_VOCAB_CONFIG

pe = CompositePositionalEncoding(RPLAN_VOCAB_CONFIG, d_model=128)
pos_encoding = pe()          # (36, 128)
x = x + pos_encoding         # broadcasts to (B, 36, 128)
```

### 3.3 TimestepEmbedding — Full Breakdown

**What:** Converts a continuous timestep `t ∈ [0, 1]` into a d_model-dimensional conditioning vector.

**Two-stage pipeline:**

#### Stage 1: Sinusoidal encoding

- Input: `t` as `(B,)` float32 tensor
- Output: `(B, frequency_embedding_size)` where default = 256
- Formula:
  ```
  half = 128
  freqs = exp(-log(10000) * arange(128) / 128)    → 128 frequencies
  embedding = cat([cos(t * freqs), sin(t * freqs)])  → 256-dimensional
  ```

**Why sinusoidal basis:** The timestep `t` is a single scalar, but the model needs to distinguish nearby timesteps (e.g., t=0.10 vs t=0.11) while also capturing the global noise level. The sinusoidal basis provides a multi-frequency representation: low-frequency components capture the broad noise level (clean vs noisy), while high-frequency components encode fine-grained timestep differences. This is the same encoding used in the original Transformer (Vaswani et al.), GLIDE, and DiT — it's a well-proven approach for mapping scalars to high-dimensional vectors.

#### Stage 2: MLP projection

- `Linear(256, d_model) → SiLU → Linear(d_model, d_model)`

**Why MLP after sinusoidal:** The sinusoidal encoding is a fixed function — it provides a rich input representation but cannot be learned. The MLP learns to project this into the model's representation space, weighting different frequency components as needed for the task. For instance, the model might learn that certain frequency bands are more informative for distinguishing "almost clean" from "mostly noisy" states.

#### The outer SiLU in BDDenoiser.forward()

After the TimestepEmbedding module, an additional `F.silu()` is applied:
```python
c = F.silu(self.timestep_embedding(t))   # (B, d_model)
```

**Why this extra SiLU:** Following the DiT/DiDAPS convention (`c = F.silu(self.sigma_map(sigma))`). The MLP's internal SiLU and this outer SiLU serve different purposes:
- **Internal SiLU** (inside the MLP): Part of the learned projection — it adds nonlinearity between the two linear layers so the MLP can learn complex mappings from sinusoidal space to model space.
- **Outer SiLU**: Shapes the conditioning signal *after* it's been projected into model space, *before* it enters the adaLN modulation blocks. This adds a final nonlinearity that has been empirically shown to stabilize training in diffusion transformers. SiLU (Sigmoid Linear Unit, also called Swish) is preferred over ReLU because it's smooth everywhere (no dead neurons) and has a slight negative region that helps with gradient flow.

**Usage:**
```python
from bd_gen.model.embeddings import TimestepEmbedding

te = TimestepEmbedding(d_model=128, frequency_embedding_size=256)
t = torch.tensor([0.1, 0.5, 0.9])       # (3,)
t_emb = te(t)                             # (3, 128)
c = F.silu(t_emb)                         # (3, 128) — the conditioning vector
```

---

## 4. Transformer Block Design (`transformer.py`)

### 4.1 MultiHeadSelfAttention

**What:** Standard multi-head self-attention using PyTorch 2.0+ `F.scaled_dot_product_attention`.

**Combined QKV projection:** Uses a single `Linear(d_model, 3*d_model)` instead of three separate Q, K, V projections. This is more efficient (one matrix multiply instead of three) and is standard practice in modern transformers (GPT-2, DiT, etc.).

**Attention mask conversion chain:**

The pad_mask flows through several transformations before reaching the attention function:

```
pad_mask: (B, SEQ_LEN) bool
  └─ Convention: True = real position, False = PAD
  └─ Set by: BubbleDiagramDataset / compute_pad_mask()

     ▼ (inverted in BDDenoiser.forward)

attn_mask: (B, SEQ_LEN) bool
  └─ Convention: True = IGNORE (PAD position)
  └─ This is the logical inverse of pad_mask

     ▼ (converted in MultiHeadSelfAttention.forward)

sdpa_mask: (B, 1, 1, SEQ_LEN) float32
  └─ Convention: 0.0 = attend, -inf = do not attend
  └─ Shape explanation:
     - B: one mask per sample in the batch
     - 1: broadcasts across all attention heads
     - 1: broadcasts across all query positions
     - SEQ_LEN: one value per key position
```

**Why this chain:** `F.scaled_dot_product_attention` requires an additive float mask, not a boolean mask. The `(B, 1, 1, S)` shape broadcasts correctly: every head and every query position uses the same key-masking pattern. This ensures no real token attends to PAD keys.

**What happens to PAD query positions:** They still compute attention outputs (they attend to non-PAD keys), but these outputs are harmless:
1. The adaLN gate modulates the residual contribution (at init, gate=0 for all positions)
2. The final classification heads produce logits at PAD positions, but these logits are never used — the loss function excludes PAD positions, and the sampling loop clamps PAD positions to PAD tokens

We only need to prevent PAD keys from contaminating real position representations via attention, which the key-side masking achieves.

**Why `F.scaled_dot_product_attention`, not flash_attn:** RPLAN sequences are only 36 tokens. Flash attention is optimized for long sequences (1K+) where the quadratic attention cost becomes significant. At 36 tokens, the standard SDPA is perfectly efficient and avoids adding `flash_attn` as a dependency (which requires specific CUDA versions and hardware).

### 4.2 AdaLN-Zero Block — The Core Innovation

**What is adaLN:** Adaptive Layer Normalization. Instead of the fixed shift and scale parameters in standard LayerNorm, adaLN predicts them dynamically from a conditioning signal. In our case, the conditioning signal is the timestep embedding — this lets the transformer adjust its behavior based on the noise level.

**Why time conditioning matters:** At low noise (t ≈ 0), most tokens are already correct and the model should make small refinements. At high noise (t ≈ 1), most tokens are masked and the model must generate from scratch. The model needs to know which regime it's in to produce appropriate predictions. adaLN injects this information into every layer.

#### The 6 modulation parameters

Each block predicts 6 vectors from the conditioning signal `c`:

```python
adaLN_modulation = Linear(cond_dim, 6 * d_model)  # zero-initialized
(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
    adaLN_modulation(c).unsqueeze(1).chunk(6, dim=2)
```

| Parameter | Applied to | Role |
|-----------|-----------|------|
| `shift_msa` | Attention LN | Additive shift of normalized activations |
| `scale_msa` | Attention LN | Multiplicative scale of normalized activations |
| `gate_msa` | Attention residual | Controls how much the attention output contributes |
| `shift_mlp` | FFN LN | Additive shift of normalized activations |
| `scale_mlp` | FFN LN | Multiplicative scale of normalized activations |
| `gate_mlp` | FFN residual | Controls how much the FFN output contributes |

#### Why ZERO initialization

This is the "Zero" in "adaLN-Zero" (from the DiT paper, Peebles & Xie, ICCV 2023).

At initialization:
- `shift = 0, scale = 0, gate = 0` (all modulation weights and biases are zero)

What this means:
- **Modulation formula:** `x_mod = LayerNorm(x) * (1 + scale) + shift = LayerNorm(x) * 1 + 0 = LayerNorm(x)` → identity modulation
- **Gated residual:** `x = x + gate * dropout(output) = x + 0 = x` → no residual contribution

**Effect:** Each transformer block starts as if it doesn't exist. The model output at initialization is just the positional encoding passed through LayerNorm and the zero-init final layer, producing all-zero logits. During training, the model gradually learns to activate each block by moving the modulation parameters away from zero. This has been empirically shown to significantly stabilize diffusion transformer training compared to standard initialization.

**The (1 + scale) formulation is crucial:** If the formula were `x * scale + shift`, then `scale=0` would zero out the input entirely, destroying information. With `x * (1 + scale)`, `scale=0` means no scaling (identity), preserving the input. This is what makes zero-initialization work.

#### Gated residual connections

```python
x = x + gate * dropout(attn_output)     # attention sub-block
x = x + gate * dropout(ffn_output)       # FFN sub-block
```

The gate multiplies the entire sub-block output before adding to the residual stream. With `gate=0` at init, the residual stream passes through unchanged. As training progresses, the model learns non-zero gates, gradually incorporating the attention and FFN contributions. This prevents the large random initial perturbations that standard residual connections would produce.

#### Dropout placement

Dropout is applied to the attention and FFN outputs individually, before the gated residual add. This matches the standard transformer dropout pattern and the DiDAPS implementation. The dropout is not applied to the conditioning signal or the modulation parameters.

---

## 5. Denoiser Composition (`denoiser.py`)

### 5.1 Forward Pass Walkthrough

Concrete example: B=4 samples, RPLAN (n_max=8, seq_len=36), d_model=128.

| Step | Operation | Input shape(s) | Output shape |
|------|-----------|----------------|--------------|
| 1 | Split tokens | `(4, 36) long` | `(4, 8)` nodes + `(4, 28)` edges |
| 2 | Embed tokens | `(4, 8)` + `(4, 28)` | `(4, 8, 128)` + `(4, 28, 128)` |
| 3 | Concatenate | two tensors | `(4, 36, 128)` |
| 4 | Add positional encoding | `(4, 36, 128)` + `(36, 128)` | `(4, 36, 128)` |
| 5 | Process timestep | `t` (various) → `(4,)` → MLP → SiLU | `c: (4, 128)` |
| 6 | Invert pad mask | `(4, 36)` bool True=real | `(4, 36)` bool True=ignore |
| 7 | Transformer blocks (×L) | `x: (4, 36, 128)`, `c: (4, 128)`, mask | `(4, 36, 128)` |
| 8 | Final adaLN + LayerNorm | `(4, 36, 128)` | `(4, 36, 128)` |
| 9 | Split features | `(4, 36, 128)` | `(4, 8, 128)` + `(4, 28, 128)` |
| 10 | Classification heads | `(4, 8, 128)` + `(4, 28, 128)` | `(4, 8, 15)` + `(4, 28, 13)` |
| 11 | Return tuple | — | `(node_logits, edge_logits)` |

### 5.2 Final adaLN Layer

The final layer applies a last adaLN modulation with only 2 parameters (shift + scale), not 6:

```python
final_shift, final_scale = final_adaLN(c).unsqueeze(1).chunk(2, dim=2)
x = final_norm(x) * (1 + final_scale) + final_shift
```

**Why no gate:** The gate mechanism controls the residual connection strength. The final layer doesn't have a residual connection — it directly produces the input to the classification heads. There's nothing to gate. This follows the DiDAPS `DDitFinalLayer` pattern exactly.

**Why zero-init:** Same reason as the block adaLN — at initialization, `shift=0, scale=0` means the final layer is just a LayerNorm, which combined with zero-init classification heads produces all-zero logits.

### 5.3 Zero-Initialized Classification Heads

Both `node_head` and `edge_head` are `nn.Linear` with all weights and biases initialized to zero:

```python
self.node_head = nn.Linear(d_model, NODE_VOCAB_SIZE)   # d_model → 15
self.node_head.weight.data.zero_()
self.node_head.bias.data.zero_()
```

**Why zero-init heads:** Combined with the zero-init final adaLN, this ensures the model initially produces all-zero logits for every position. All-zero logits correspond to a **uniform distribution** over classes after softmax:
- For nodes: `softmax([0, 0, ..., 0]) = [1/15, 1/15, ..., 1/15]`
- For edges: `softmax([0, 0, ..., 0]) = [1/13, 1/13, ..., 1/13]`

This means:
- **Initial loss = log(vocab_size)** for each position — the theoretically maximum-entropy starting point
- **No bias toward any class** at initialization — the model doesn't accidentally start by predicting the most common class
- **Clean gradient signal** from the start — every parameter receives meaningful gradients pointing toward the correct class

### 5.4 `_process_t()` Helper

**Motivation:** The diffusion framework passes timesteps in various forms depending on context:
- **Training:** `t` is a 1D tensor `(B,)` with different timesteps per sample (sampled from [0, 1])
- **Sampling:** `t` is often a scalar (same timestep for the entire batch at each denoising step)
- **Testing:** Convenience of passing Python floats or ints

Rather than requiring every caller to prepare a correctly-shaped tensor, the model handles normalization internally via `_process_t()`.

**Accepted inputs and behavior:**

| Input type | Example | Behavior |
|-----------|---------|----------|
| Python float | `0.5` | Broadcast to `(B,)` via `torch.full` |
| Python int | `1` | Cast to float, broadcast to `(B,)` |
| 0D tensor | `torch.tensor(0.7)` | Expand to `(B,)` |
| 1D tensor, size 1 | `torch.tensor([0.3])` | Expand to `(B,)` |
| 1D tensor, size B | `torch.tensor([0.1, 0.5, 0.9, 0.2])` | Used as-is |
| Invalid: 1D wrong size | `torch.tensor([0.1, 0.2])` with B=4 | `ValueError` |
| Invalid: 2D+ tensor | `torch.tensor([[0.5]])` | `ValueError` |

### 5.5 `condition=None` Placeholder

The `condition` argument is unused in v1 (unconditional generation). In v2, it will accept `(B, n_cond_tokens, d_condition)` house boundary features. These will be integrated via cross-attention layers added to the transformer blocks. The current interface is future-proof: the only change needed in v2 is to process the `condition` tensor when it's not `None`, without changing the function signature.

---

## 6. Configuration Reference

### Config parameters

| Parameter | small.yaml | base.yaml | Description |
|-----------|-----------|-----------|-------------|
| `d_model` | 128 | 256 | Hidden dimension throughout the transformer |
| `n_layers` | 4 | 6 | Number of adaLN transformer blocks |
| `n_heads` | 4 | 8 | Attention heads per block (d_model must be divisible by n_heads) |
| `cond_dim` | 128 | 256 | Conditioning dimension (must equal d_model in v1) |
| `mlp_ratio` | 4 | 4 | FFN hidden dim = mlp_ratio × d_model |
| `dropout` | 0.1 | 0.1 | Dropout probability for attention and FFN |
| `frequency_embedding_size` | 256 | 256 | Sinusoidal timestep encoding dimension |

### Config → Constructor mapping

```python
model = BDDenoiser(
    d_model=cfg.model.d_model,
    n_layers=cfg.model.n_layers,
    n_heads=cfg.model.n_heads,
    vocab_config=VocabConfig(n_max=cfg.data.n_max),
    cond_dim=cfg.model.cond_dim,
    mlp_ratio=cfg.model.mlp_ratio,
    dropout=cfg.model.dropout,
    frequency_embedding_size=cfg.model.frequency_embedding_size,
)
```

### Parameter count estimates

**Small config (d_model=128, L=4, H=4):**

| Component | Parameters |
|-----------|-----------|
| NodeEmbedding (15 × 128) | 1,920 |
| EdgeEmbedding (13 × 128) | 1,664 |
| CompositePositionalEncoding (2+8+8) × 128 | 2,304 |
| TimestepEmbedding (256→128 MLP) | 49,408 |
| AdaLN Block × 4 (QKV, out, FFN, norms, adaLN) | ~1,189,376 |
| Final LayerNorm + adaLN | ~33,280 |
| NodeHead (128 → 15) | 1,935 |
| EdgeHead (128 → 13) | 1,677 |
| **Total** | **~1.28M** |

**Base config (d_model=256, L=6, H=8):** ~5.0M parameters

Both are well within the 1–5M target range for this problem (36-token sequences on 80K training samples).

---

## 7. Usage Examples

### Creating a model

```python
from bd_gen.model import BDDenoiser
from bd_gen.data.vocab import RPLAN_VOCAB_CONFIG

# Small model (matches configs/model/small.yaml)
model = BDDenoiser(
    d_model=128, n_layers=4, n_heads=4,
    vocab_config=RPLAN_VOCAB_CONFIG,
)

# Check parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")  # ~1.28M
```

### Running a forward pass

```python
import torch

# Simulated batch (4 samples, RPLAN seq_len=36)
B = 4
tokens = torch.randint(0, 10, (B, 36))          # token indices
pad_mask = torch.ones(B, 36, dtype=torch.bool)   # all real positions
t = 0.5                                           # scalar timestep

model.eval()
with torch.no_grad():
    node_logits, edge_logits = model(tokens, pad_mask, t)

print(node_logits.shape)   # torch.Size([4, 8, 15])
print(edge_logits.shape)   # torch.Size([4, 28, 13])
```

### Using with real data

```python
from bd_gen.data.dataset import BubbleDiagramDataset
from torch.utils.data import DataLoader

dataset = BubbleDiagramDataset(...)
loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    tokens = batch["tokens"]       # (32, 36)
    pad_mask = batch["pad_mask"]   # (32, 36)
    t = torch.rand(32)            # random timesteps per sample

    node_logits, edge_logits = model(tokens, pad_mask, t)
    # → node_logits: (32, 8, 15), edge_logits: (32, 28, 13)
```

### Moving to GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# All registered buffers (positional encoding indices) move automatically
tokens = tokens.to(device)
pad_mask = pad_mask.to(device)
node_logits, edge_logits = model(tokens, pad_mask, t=0.5)
```
