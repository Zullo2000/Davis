# How Our Model Differs from MDLM

This document describes how the masked diffusion model used in the BD Generation pipeline relates to and differs from the original **MDLM** (Masked Diffusion Language Model) by Sahoo et al. (arXiv:2406.07524, [project page](https://s-sahoo.com/mdlm/)).

Our model is built on the MDLM framework — same forward process, same continuous-time ELBO loss, same ancestral sampling structure — but adapted for **graph-structured floorplan generation** rather than natural language. The differences stem from the fact that we are generating small discrete graphs (bubble diagrams with 4–8 rooms and their spatial relationships), not sequences of text tokens.

---

## 1. What We Share with MDLM

These core components are identical in formulation:

| Component | Formulation |
|-----------|-------------|
| **Forward process** | Each non-special position is independently masked with probability `1 − α(t)`, where `α(t) = exp(−σ(t))`. At `t = 0` the sequence is clean; at `t = 1` it is fully masked. |
| **ELBO loss** | `L = E_t [ w(t) · (1/N_active) · Σ_l CE(logits_l, x0_l) · loss_mask_l ]`, where `w(t) = −α'(t) / (1 − α(t))` is the continuous-time ELBO weight. |
| **SUBS parameterization** | The model predicts `p(x0 | x_t)` directly (not the noise). Carry-over unmasking: already-unmasked tokens are kept as-is during sampling. |
| **Noise schedules** | Linear (`σ(t) = σ_min + t(σ_max − σ_min)`) and cosine (`α(t) = ε + (1−ε)cos(tπ/2)`) options, same functional forms as MDLM. |
| **Ancestral sampling** | Reverse loop from `t = 1` to `t = 0` in N discrete steps, unmasking positions based on the schedule's `p_unmask = (α_next − α_now) / (1 − α_now)`. |

---

## 2. Key Differences

### 2.1 Domain: Graphs, Not Text

MDLM generates **1D text sequences** (up to thousands of tokens from a single vocabulary). Our model generates **small graphs** represented as flat token sequences:

```
[ room_0, room_1, ..., room_{n_max-1}, edge_(0,1), edge_(0,2), ..., edge_(i,j), ... ]
     ← 8 node tokens →                ← 28 edge tokens (upper triangle) →
```

A bubble diagram with `n` rooms is tokenized into `n` room-type tokens + `n(n−1)/2` edge tokens arranged in upper-triangle row-major order, for a fixed sequence length of 36 (with `n_max = 8`). This means:
- The sequence has inherent **graph structure** (nodes + edges), not linear structure.
- Positions have meaning beyond order — position 10 is specifically the edge between rooms 0 and 2.
- The sequence is short (36 tokens vs. hundreds/thousands in language).

### 2.2 SUBS Parameterization

MDLM's SUBS parameterization has two properties:

1. **Zero masking probabilities** — the logit corresponding to the MASK token is set to `−∞` before softmax, so the model can *never* predict MASK as an output token.
2. **Carry-over unmasking** — tokens that have already been unmasked are kept as-is in subsequent sampling steps; the model's predictions for those positions are ignored.

**We implement both.**

**Carry-over unmasking** is enforced in [sampling.py:260-261](BD_Generation/bd_gen/diffusion/sampling.py#L260-L261):

```python
x_t = torch.where(should_unmask, pred_tokens, x_t)
```

`should_unmask` is always restricted to currently-MASK positions (via `& is_mask`), so already-unmasked tokens are never overwritten.

**Zero masking probabilities** is enforced at the end of `BDDenoiser.forward()` ([denoiser.py:214-221](BD_Generation/bd_gen/model/denoiser.py#L214-L221)): MASK and PAD logits are clamped to `−∞` for both node and edge vocabularies, so `softmax` assigns them exactly zero probability. This is a hard architectural constraint that applies to both training and inference.

```python
node_logits[:, :, NODE_MASK_IDX] = float('-inf')
node_logits[:, :, NODE_PAD_IDX] = float('-inf')
edge_logits[:, :, EDGE_MASK_IDX] = float('-inf')
edge_logits[:, :, EDGE_PAD_IDX] = float('-inf')
```

The final cleanup step in sampling (lines 278–300) is retained as a safety net but is expected to be a no-op given the architectural constraint.

### 2.3 Dual Vocabulary

MDLM uses a **single vocabulary** (e.g., byte-pair encoding tokens for text, all from one embedding table).

We use **two separate vocabularies** with independent embedding tables and classification heads:

| | Node vocabulary | Edge vocabulary |
|---|---|---|
| **Real tokens** | 13 room types (LivingRoom, Kitchen, Bathroom, ...) | 10 spatial relationships (left-of, above, ...) + no-edge |
| **Special tokens** | MASK (idx 13), PAD (idx 14) | MASK (idx 11), PAD (idx 12) |
| **Vocab size** | 15 | 13 |
| **Embedding** | `NodeEmbedding(15, d_model)` | `EdgeEmbedding(13, d_model)` |
| **Classification head** | `Linear(d_model, 15)` | `Linear(d_model, 13)` |

The model outputs two sets of logits — `(B, 8, 15)` for nodes and `(B, 28, 13)` for edges — and the loss is the sum of two separate cross-entropy terms. This is necessary because room types and spatial relationships are semantically unrelated; a shared vocabulary would conflate "Bathroom" with "left-of".

### 2.4 PAD Handling for Variable-Size Graphs

MDLM operates on fixed-length sequences (or uses standard sequence padding for batching). The PAD tokens are a minor batching detail.

In our model, **PAD is a first-class structural concept**. A 5-room graph uses only 5 node slots and 10 edge slots out of the 36 available; the remaining 26 positions are PAD. This creates strict invariants:

- **PAD positions are never masked** during the forward process (`should_mask & pad_mask`).
- **PAD positions are never unmasked** during sampling (masked indicators exclude PAD).
- **PAD positions contribute zero to the loss** (`loss_mask = mask_indicators & pad_mask`).
- **PAD is re-clamped every sampling step** to prevent drift.
- The PAD mask is an **attention mask** in the transformer: PAD keys receive `−∞` additive bias so no real token attends to them.

This is more elaborate than standard sequence padding because the graph *structure* depends on which positions are PAD (e.g., edge (3,5) is PAD when `num_rooms ≤ 5`).

### 2.5 Graph-Aware Positional Encoding

MDLM uses **rotary positional embeddings (RoPE)**, which encode sequential position (token 1, token 2, ...). This makes sense for text where adjacency in the sequence corresponds to adjacency in meaning.

Our sequence is a flattened graph, so sequential position is meaningless — position 8 (first edge) is not "adjacent" to position 7 (last node) in any structural sense. We use a **composite learned positional encoding** with three components:

| Component | What it encodes |
|-----------|----------------|
| `entity_type_emb(2, d)` | Whether the position is a node (0) or edge (1) |
| `node_index_emb(n_max, d)` | Which room slot this node occupies (node positions only) |
| `pair_index_emb(n_max, d)` | Which two rooms this edge connects: `emb(i) + emb(j)` (edge positions only) |

The pair encoding is additive and commutative, capturing that edge (2, 5) involves rooms 2 and 5 without imposing directionality (the edge token itself encodes the spatial direction, e.g., "left-of" vs "right-of").

### 2.6 Timestep Conditioning: adaLN-Zero (DiT-style)

MDLM integrates the timestep via a **DiT-style** approach, but the original MDLM paper uses a relatively standard transformer encoder.

Our model uses **adaLN-Zero** (Peebles & Xie, ICCV 2023) in every transformer block. For each block, the timestep embedding predicts 6 modulation parameters:

```
c = SiLU(TimestepEmbedding(t))                    # (B, d_model)
shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = Linear(c)  # 6 × d_model

# Attention sub-block
x = x + gate_1 * Attention(LayerNorm(x) * (1 + scale_1) + shift_1)

# FFN sub-block
x = x + gate_2 * FFN(LayerNorm(x) * (1 + scale_2) + shift_2)
```

The gate and modulation weights are **zero-initialized**, meaning each block starts as an identity function and gradually learns to incorporate time conditioning. This is particularly important for our small model (~5M parameters), providing stable training from the start.

### 2.7 Per-Sample Loss Normalization

MDLM computes the loss over masked positions and averages across the batch. Since text sequences are typically the same length, per-position averaging works naturally.

Our graphs vary in size (4–8 rooms), so the number of loss-active positions varies significantly across samples within a batch (a 4-room graph has ~10 masked positions while an 8-room graph has ~36). We **normalize per-sample by `N_active`** (the count of masked non-PAD positions, clamped to ≥ 1) before averaging across the batch. This ensures small and large graphs contribute equally to the gradient, preventing the model from biasing toward larger floorplans.

### 2.8 Edge Class Weighting

MDLM does not use class-weighted cross-entropy — with large text vocabularies and relatively uniform token distributions, this isn't necessary.

Our edge vocabulary has a highly imbalanced distribution: "no-edge" (index 10) dominates because most room pairs in a floorplan are not adjacent. Without weighting, the model would learn to predict "no-edge" everywhere and achieve low loss. We apply **inverse-frequency class weights** to the edge cross-entropy, ensuring rare but important spatial relationships (e.g., "inside-of") are not drowned out. Node CE is unweighted in v1.

### 2.9 Confidence-Based Unmasking (Post-v1)

Standard MDLM sampling uses **random unmasking**: at each step, each masked position is independently unmasked with probability `p_unmask` — a coin-flip per position.

Our sampler additionally supports **confidence-based unmasking** (inspired by LLaDA, Nie et al.), where positions are unmasked in order of model confidence. At each step:
1. The model predicts tokens for all masked positions.
2. The confidence `P(predicted_token)` is computed per position.
3. A budget of `p_unmask × num_remaining_masked` positions is unmasked, selecting those with the highest confidence.

This lets the model resolve easy structural decisions first (obvious room types, dominant spatial relationships) and defer ambiguous positions to later steps when more context is available. The default remains `"random"` for backward compatibility.

### 2.10 Extension Hooks (Not in MDLM)

Our sampling function includes hooks that MDLM's standard sampler does not provide:

| Hook | Purpose |
|------|---------|
| `guidance_fn` | Modify logits each step (e.g., classifier-free guidance for conditional generation) |
| `fixed_tokens` + `fixed_mask` | Inpainting: keep certain positions fixed while generating the rest |
| `remasking_fn` | ReMDM-style remasking after each unmasking step |
| `num_rooms_distribution` | Control the room-count distribution of generated samples |

These are scaffolded for future extensions (conditional generation, constrained editing) and are no-ops by default.

---

## 3. Summary Table

| Aspect | MDLM (Sahoo et al.) | Our Model (BD Generation) |
|--------|---------------------|---------------------------|
| **Domain** | Natural language (text) | Bubble diagrams (graphs) |
| **Sequence** | Variable-length text tokens | Fixed 36 tokens: 8 nodes + 28 edges |
| **Vocabulary** | Single (BPE, ~32K–50K) | Dual: nodes (15) + edges (13) |
| **Embedding** | Single shared table | Separate NodeEmbedding + EdgeEmbedding |
| **Positional encoding** | RoPE (sequential) | Composite learned (entity type + node index + pair index) |
| **Timestep conditioning** | DiT-style | adaLN-Zero with zero-init gates |
| **PAD handling** | Standard sequence padding | Structural: never masked, never unmasked, zero loss, attention-masked |
| **Loss normalization** | Per-position average | Per-sample by N_active (graph-size invariant) |
| **Class weighting** | None | Inverse-frequency on edge CE |
| **SUBS: zero masking probs** | MASK logit set to −∞ (hard constraint) | Same: MASK + PAD logits clamped to −∞ in denoiser forward() |
| **SUBS: carry-over** | Unmasked tokens kept as-is | Same (`torch.where(should_unmask, pred, x_t)`) |
| **Unmasking** | Random coin-flip | Random (default) or confidence-based (LLaDA-style) |
| **Scale** | Hundreds of millions of params | ~5M parameters |
| **Sequence length** | Hundreds to thousands | 36 |

---

## References

- Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (MDLM), NeurIPS 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524), [project page](https://s-sahoo.com/mdlm/).
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), ICCV 2023.
- Nie et al., "Large Language Diffusion Models" (LLaDA), 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992).
