# Planning T1: Learned Forward Process (v2)

**Version:** 2.0
**Date:** 2026-02-20
**Status:** REVISED (user review incorporated 2026-02-20)
**Reference:** MELD — Seo et al., "Learning Flexible Forward Trajectories for Masked Molecular Diffusion" (arXiv:2505.16790)
**Predecessor:** `planning_T1_with_fixed_forward_process.md` (v1)
**Best v1 model:** `llada_topp0.9_remdm_confidence_tsw0.5` (log-linear schedule)

---

## Table of Contents

1. [Overview & Motivation](#1-overview--motivation) (incl. 1.5 v2 Removal Checklist)
2. [Design Decisions & Justifications](#2-design-decisions--justifications) (incl. 2.6 Element-Wise vs Class-Wise, 2.7 v1 Invariants Inherited)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Type Definitions](#4-data-type-definitions)
5. [Rate Network Module](#5-rate-network-module) (incl. 5.4 Embedding Design Rationale)
6. [Forward Process v2](#6-forward-process-v2)
7. [ELBO Loss v2](#7-elbo-loss-v2) (separate node/edge normalization)
8. [Denoiser Changes](#8-denoiser-changes)
9. [Sampling v2](#9-sampling-v2)
10. [Training Script v2](#10-training-script-v2)
11. [Config System](#11-config-system)
12. [Evaluation Plan](#12-evaluation-plan) (incl. remasking comparison context)
13. [Implementation Phases](#13-implementation-phases) (parallelization annotations)
14. [Test Cases](#14-test-cases)
15. [Computational Advantages of v2 over v1](#15-computational-advantages-of-v2-over-v1)

---

## 1. Overview & Motivation

### 1.1 The State Clashing Problem

In v1, the forward process applies **uniform masking**: every non-PAD position is independently masked with the same probability `(1 - α(t))` at timestep `t`. This means:

- At intermediate timesteps, structurally distinct bubble diagrams can collapse to **identical masked states**.
- The true posterior `p(g | g_t)` becomes **highly multimodal**: many different clean graphs are consistent with the same partially-masked observation.
- Our denoiser has a **factorized output** (independent per-position predictions), making it **unimodal**. It cannot represent the multimodal posterior.
- The mode-covering KL loss forces the model to spread probability mass broadly, producing high-entropy predictions that degrade generation quality.

This is the **state clashing problem** (Seo et al., 2025). It is particularly acute for structured data like graphs where positions (nodes and edges) have strong correlations. In our domain, correlated edge pairs (e.g., left-of / right-of between the same room pair) and rooms with similar connectivity patterns are especially prone to clashing.

### 1.2 The MELD Solution

MELD proposes learning **per-position masking rates** `β_l(t)` via a small neural network (the **rate network**), jointly trained with the denoiser. Instead of a single `α(t)` broadcast to all positions:

- Each position `l` has its own keeping probability `α_l(t, φ)`, parameterized by the rate network weights `φ`.
- Positions with high structural correlation can be assigned different masking trajectories, reducing the probability that distinct graphs produce identical masked states.
- The rate network learns to orchestrate per-position corruption: some positions are masked early, others late, breaking symmetry and avoiding collisions.

### 1.3 Goals

1. Implement the learned forward process (MELD) for bubble diagram generation.
2. Jointly train the rate network and denoiser.
3. Compare v2 against the v1 best model on the same evaluation metrics.
4. Preserve v1 code paths — all existing code must continue to work unchanged.

### 1.4 Scope

**In scope:**
- Rate network module
- STGS (Straight-Through Gumbel-Softmax) for gradient flow
- Per-position ELBO loss
- Per-position sampling
- v2 training script
- Evaluation comparison with v1

**Out of scope (deferred to post-v2):**
- Remasking with learned rates (v2 trains and evaluates without remasking)
- Class-wise rate schedules (MELD ablation; we use element-wise from the start)
- Importance sampling of timesteps with learned rates (v2 uses uniform t)

### 1.5 v2 Removal Checklist

If v2 is abandoned, v1 continues to work exactly as-is. To fully remove v2:

**Delete these new files (v2-only):**
- `bd_gen/diffusion/rate_network.py`
- `scripts/train_v2.py`
- `configs/noise/learned.yaml`
- `configs/training/v2.yaml`
- Any v2 test files (e.g., `tests/test_rate_network.py`, `tests/test_v2_*.py`)

**Optionally revert additive changes in existing files:**
| File | Change | Safe to leave? |
|------|--------|:-:|
| `bd_gen/model/denoiser.py` | `pre_embedded=None` param in `forward()` | Yes — default `None` is never used by v1 callers |
| `bd_gen/diffusion/sampling.py` | `rate_network=None` param in `sample()` | Yes — default `None` triggers v1 path |
| `bd_gen/diffusion/forward_process.py` | New functions `forward_mask_learned()`, `stgs_sample()`, `forward_mask_eval_learned()` | Yes — unused by v1 code, dead code |
| `bd_gen/diffusion/loss.py` | New class `ELBOLossV2` | Yes — never imported by v1 code |

**Not modified at all:** `scripts/train.py`, `scripts/evaluate.py`, `bd_gen/model/embeddings.py`, `bd_gen/model/transformer.py`, `bd_gen/diffusion/noise_schedule.py`, `bd_gen/diffusion/remasking.py`, `bd_gen/data/*`, `bd_gen/eval/*`.

All optional params default to `None` and all new code is additive. Removing v2 = deleting 4 files + optionally cleaning 4 optional params.

---

## 2. Design Decisions & Justifications

### 2.1 STGS vs Weight-Only Gradient Flow

**Decision:** Implement full Straight-Through Gumbel-Softmax (STGS).

**Context:** The training loss `L(θ,φ) = E[Σ_l w_l(t,φ) · CE(p_θ(·|g_t), x_0^l)]` provides two gradient pathways to the rate network `φ`:

- **Path 1 (through weights):** The per-position ELBO weight `w_l(t,φ) = -α̇_l/(1-α_l)` is a smooth function of `φ`, providing direct gradient signal about how well each position is reconstructed at each timestep.
- **Path 2 (through masking):** The masked state `g_t` depends on which positions are masked, which depends on `α_l(t,φ)`. Changing `φ` changes the denoiser's input, which changes the loss. But masking is discrete (Bernoulli sampling), blocking gradient flow.

**Why not weight-only?** Path 1 alone tells the rate network "this position is hard/easy to reconstruct at this timestep" but misses the crucial inter-position coupling: "if I mask position A instead of B, does the denoiser get better context for predicting position C?" For bubble diagrams where nodes and edges are structurally coupled (a node's type constrains its edges' types), this inter-position signal is important.

**Why STGS?** STGS relaxes the discrete masking step by mixing clean and MASK embeddings with soft Gumbel-Softmax weights during training. In the forward pass, hard one-hot decisions are used (matching inference); in the backward pass, gradients flow through the soft probabilities. This captures both pathways without high-variance estimators (unlike REINFORCE).

**Backward compatibility guarantee:** The STGS code path activates ONLY when `pre_embedded` is passed to the denoiser. When `pre_embedded is None` (the default, used by all v1 code), the denoiser behaves identically to v1. No existing code is touched.

### 2.2 Independent Rate Embeddings (Clean Decoupling)

**Decision:** The rate network uses its own learnable embeddings (`d_emb=32`), independent from the denoiser's `CompositePositionalEncoding` (`d_model=128`).

**What is "clean decoupling"?** Clean decoupling means the rate network and denoiser have **zero shared parameters** and **zero gradient cross-talk**:

- **No shared parameters:** The rate network's embeddings are separate `nn.Parameter` tensors. Updating them during training does not affect the denoiser's positional encoding, and vice versa.
- **No gradient interference:** During joint optimization, gradients from the rate-learning objective flow only into rate network parameters; gradients from the denoising objective flow only into denoiser parameters. There is no hidden coupling where optimizing one network degrades the other.
- **Independent evolution:** The rate embeddings can specialize for the rate-learning task (learning which positions need different masking schedules) without being constrained by what the denoiser needs for self-attention positioning.
- **Safe v1 reuse:** When loading a v1 checkpoint for the denoiser (transfer learning or comparison), the denoiser's weights are untouched by the rate network. The rate network initializes from scratch.

**Overhead:** 8 node embeddings × 32 dims = 256 params. Edge embeddings are derived as `h_i + h_j` (no extra params). Total rate network: ~5K params vs ~1.28M denoiser params (0.4% overhead).

#### Training Approaches: Joint vs Freeze-Denoiser

Clean decoupling enables two training strategies:

**Approach A: Joint training (chosen default).** Both denoiser and rate network train together from scratch (or with v1 weight initialization for the denoiser). The single optimizer updates all parameters. STGS provides full gradient signal: Path 1 (through ELBO weights `w_l`) tells the rate network which positions are hard/easy; Path 2 (through soft embeddings) captures inter-position coupling ("masking node A helps the denoiser predict edge A-B"). This is the strongest setup and is what v2 implements.

**Approach B: Freeze denoiser (documented for reference, not default).** Load the already-trained v1 denoiser checkpoint and freeze its weights (`requires_grad=False`). Train only the rate network parameters. The rate network learns per-position rates that are optimal for the frozen v1 denoiser, receiving gradient signal only through Path 1 (the ELBO weights). Advantages: reuses the existing trained model, faster training (~5K trainable params vs ~1.28M). Disadvantages: loses Path 2 inter-position coupling signal (the denoiser never sees STGS soft embeddings, so masking decisions don't flow gradient to the rate network through the denoiser's input); the denoiser cannot co-adapt to the learned rates. This approach could serve as a quick baseline or warm-start: train a rate network with a frozen denoiser, then unfreeze and fine-tune both jointly.

**Why joint training is chosen:** The primary motivation for v2 is to reduce state clashing through inter-position coupling — nodes constrain edges and vice versa. Path 2 (STGS) is the mechanism that captures this coupling. Freezing the denoiser eliminates Path 2, losing the core benefit.

### 2.3 Separate Training Script

**Decision:** Create `scripts/train_v2.py` separate from `scripts/train.py`.

**Motivation:** The v2 training loop differs substantially from v1:
1. Rate network is a jointly-optimized model (v1 has only the denoiser)
2. Forward masking uses STGS soft embeddings (v1 uses discrete masking)
3. Loss uses per-position ELBO weights (v1 uses scalar weight)
4. Gumbel temperature annealing schedule (v1 has no temperature concept)
5. Optimizer includes rate network parameters (v1 only has denoiser params)

These differences are not configuration toggles — they represent a fundamentally different training algorithm. Merging them into one script with branching would make both harder to read and maintain.

Shared utilities (config loading, validation, checkpointing) are imported from the same modules. Only the training loop is duplicated.

### 2.4 Evaluation Without Remasking (Initial)

**Decision:** v2 initially evaluates WITHOUT remasking. Remasking with learned rates is a post-v2 enhancement.

**Motivation:**
1. The primary hypothesis is that learned rates improve **base generation quality** (validity, distribution match, state clashing reduction) independently of remasking.
2. Remasking currently uses `NoiseSchedule.alpha(t)` (scalar). Adapting it to per-position alpha requires interface changes to `RemaskingSchedule`.
3. Evaluating v2 without remasking isolates the effect of learned rates from remasking effects, giving a cleaner comparison.
4. If v2 base performance is strong, remasking can be layered on top as a simple follow-up.

### 2.5 Uniform Timestep Sampling

**Decision:** v2 uses uniform `t ~ U[0, 1]` instead of importance sampling.

**Motivation:** With learned per-position rates, each position has a different `α_l(t)`, so the optimal importance sampling distribution depends on all 36 position schedules simultaneously. The v1 importance sampling transformation is derived for a single global schedule and doesn't generalize directly. Uniform sampling is simpler and sufficient for training — the per-position weights `w_l(t)` already provide adaptive emphasis.

### 2.6 Element-Wise vs Class-Wise Rate Schedules

**Decision:** v2 uses **element-wise** rate schedules (one rate per sequence position). Class-wise schedules are out of scope.

**Class-wise:** One masking rate per vocabulary class. All positions with the same token type (e.g., all "Kitchen" nodes, all "left-of" edges) share a single `α_class(t)`. The rate depends on *what* the token is, not *where* it sits. This reduces state collisions among different token types but still allows collisions among same-type tokens in different positions.

**Element-wise:** One masking rate per sequence position. Position 0 gets `α_0(t)`, position 1 gets `α_1(t)`, independent of the token value. Each position has its own learned corruption trajectory.

**Why element-wise is better (MELD ablation):** Class-wise schedules cannot distinguish between two "Kitchen" nodes in slots 2 and 5 — they get identical masking trajectories and can still produce colliding intermediate states. Element-wise schedules assign each position a unique trajectory, maximally reducing state collision probability. MELD's ablation on ZINC250K shows element-wise achieving 93% validity vs substantially lower with class-wise. For bubble diagrams, where multiple rooms can share a type (e.g., two Bedrooms), element-wise is critical.

**Implementation:** The rate network takes position indices as input (via learned embeddings), not token values. This is naturally element-wise: each of the 36 positions has independent polynomial coefficients.

### 2.7 v1 Invariants Inherited by v2

v2 builds ON TOP of v1's proven codebase — it does NOT rebuild from scratch. The following v1 improvements are inherited, either through reusing unmodified v1 code or by replicating the same patterns in new v2 code:

**Inherited via unmodified v1 code (zero effort):**

| Invariant | Where | How v2 inherits |
|-----------|-------|-----------------|
| **SUBS zero masking probabilities** | `denoiser.py:218-222` | Denoiser clamps MASK/PAD logits to `-inf` AFTER the embedding step. The `pre_embedded` branch only changes embedding input; SUBS clamping runs identically on both v1 and v2 paths. |
| **Top-p (nucleus) sampling** | `sampling.py:52-84` | `_top_p_sample()` is called in the sampling loop regardless of `rate_network` presence. v2 inherits this entirely. |
| **Gumbel temperature sampling** | `sampling.py:30-49` | `_gumbel_sample()` with float64 precision. v2 uses the same stochastic sampling. |
| **LLaDA confidence unmasking** | `sampling.py:273-313` | `unmasking_mode="llada"` code path is shared. v2 only changes how `p_unmask` is computed (per-position vs scalar), not the confidence ranking logic. |
| **Carry-over unmasking** | `sampling.py:264-316` | `should_unmask & is_mask` ensures only currently-masked positions are unmasked. v2 uses the same `is_mask` check. Already-decoded tokens are preserved across steps. |
| **ReMDM remasking** | `remasking.py` + `sampling.py:328-332` | Deferred for v2, but code is untouched. Will work once `RemaskingSchedule` is adapted to per-position alpha. |
| **Evaluation metrics** | `bd_gen/eval/*` | All metric functions (validity, coverage, distribution, structure, conditional) are unchanged. |

**Must be replicated in new v2 code:**

| Invariant | v1 location | v2 location | What to replicate |
|-----------|-------------|-------------|-------------------|
| **Safe CE targets** | `loss.py:145-146` | `ELBOLossV2.forward()` | Clamp non-loss-position targets to class 0 before indexing logits, preventing access to `-inf` SUBS logits. `safe_node_x0 = torch.where(node_loss_mask, node_x0, torch.zeros_like(node_x0))` |
| **Edge class weights** | `loss.py:42-69` | `ELBOLossV2.__init__()` | Pass `weight=self.edge_class_weights` to `F.cross_entropy` for edge positions. Node weights remain `None`. |
| **Float64 ELBO weights** | `loss.py:84-92` | `ELBOLossV2.forward()` | Compute `w_l = -alpha_prime / (1 - alpha)` in float64 before casting back to float32. Already specified in Section 7.6. |
| **Float64 p_unmask** | `sampling.py:240-245` | `sample()` v2 branch | Compute `p_unmask_l = (alpha_next - alpha_now) / (1 - alpha_now)` in float64 for per-position alpha. |
| **Per-sample N_active normalization** | `loss.py:169-174` | `ELBOLossV2.forward()` | Separate `N_active_nodes` and `N_active_edges` per sample, clamped to `min=1.0`. Already specified in Section 7.2. |
| **Denoising eval** | `bd_gen/eval/denoising_eval.py` | `scripts/evaluate.py` v2 path | When evaluating v2, denoising accuracy uses per-position `α_l(t)` from rate network (different masking fractions per position). Threshold interpretation changes. |

**Not applicable to v2 (replaced by learned mechanism):**

| v1 Feature | Why not applicable |
|------------|-------------------|
| Noise schedule (log-linear) | Replaced by rate network's per-position `α_l(t)` |
| Importance sampling of timesteps | Per-position `w_l(t)` provides adaptive emphasis natively (Section 2.5) |
| Schedule selection (linear vs log-linear) | Rate network learns the optimal schedule per position |

---

## 3. Architecture Overview

### 3.1 v1 vs v2 Pipeline Comparison

```
v1 (fixed forward process):
  x_0 → forward_mask(α(t)) → x_t → denoiser(x_t, t) → logits → ELBO(w(t))
         scalar α(t)                discrete tokens        scalar w(t)

v2 (learned forward process):
  x_0 → rate_network(t) → α_l(t) → STGS → soft_emb → denoiser(soft_emb, t) → logits → ELBO(w_l(t))
         per-position α_l(t)         Gumbel-Softmax   pre_embedded path       per-pos w_l(t)
```

### 3.2 Component Classification

**New modules:**
| Component | File | Description |
|-----------|------|-------------|
| RateNetwork | `bd_gen/diffusion/rate_network.py` | Per-position learned rates |
| train_v2 | `scripts/train_v2.py` | Joint training loop |
| learned.yaml | `configs/noise/learned.yaml` | Rate network hyperparams |
| v2.yaml | `configs/training/v2.yaml` | v2 training hyperparams |

**Modified modules (backward-compatible):**
| Component | File | Change |
|-----------|------|--------|
| forward_process | `bd_gen/diffusion/forward_process.py` | Add `forward_mask_learned()` |
| loss | `bd_gen/diffusion/loss.py` | Add `ELBOLossV2` class |
| sampling | `bd_gen/diffusion/sampling.py` | Per-position alpha in `sample()` |
| denoiser | `bd_gen/model/denoiser.py` | Add `pre_embedded` parameter |

**Unchanged modules:** `bd_gen/data/*`, `bd_gen/eval/*`, `bd_gen/model/embeddings.py`, `bd_gen/model/transformer.py`, `bd_gen/diffusion/noise_schedule.py`, `bd_gen/diffusion/remasking.py`, `scripts/train.py`, `scripts/evaluate.py`

---

## 4. Data Type Definitions

```python
from typing import TypedDict
from torch import Tensor

class RateNetworkOutput(TypedDict):
    """Output of the rate network for a batch."""
    alpha: Tensor           # (B, SEQ_LEN) per-position keeping probability
    alpha_prime: Tensor     # (B, SEQ_LEN) per-position d(alpha)/dt
    gamma: Tensor           # (B, SEQ_LEN) pre-sigmoid log-odds (for debugging)

class STGSOutput(TypedDict):
    """Output of STGS masking step."""
    soft_embeddings: Tensor   # (B, SEQ_LEN, d_model) mixed clean+mask embeddings
    x_t: Tensor              # (B, SEQ_LEN) discrete masked tokens (hard decisions)
    mask_indicators: Tensor   # (B, SEQ_LEN) bool, True where masked
    alpha_per_pos: Tensor     # (B, SEQ_LEN) per-position alpha from rate network
    gumbel_weights: Tensor    # (B, SEQ_LEN, 2) soft keep/mask weights
```

---

## 5. Rate Network Module

**File:** `bd_gen/diffusion/rate_network.py`

### 5.1 Mathematical Formulation

For each position `l` in the sequence (36 positions: 8 nodes + 28 edges):

1. **Learnable element embeddings:**
   - Node embeddings: `H_node ∈ R^{n_max × d_emb}`, where `h_node^i = H_node[i]`
   - Edge embeddings (derived): `h_edge^{ij} = h_node^i + h_node^j`
   - This follows MELD: edge embeddings are sums of endpoint node embeddings

2. **Type-specific projection:**
   - `proj_l = W_node · h_node^i` for node positions
   - `proj_l = W_edge · h_edge^{ij}` for edge positions
   - `W_node, W_edge ∈ R^{d_emb × d_emb}` are separate linear layers

3. **MLP → polynomial coefficients:**
   - `w^l = softplus(MLP(proj_l)) ∈ R^K` (K positive coefficients per position)
   - MLP: `Linear(d_emb, hidden_dim) → SiLU → Linear(hidden_dim, K)`

4. **Polynomial evaluation:**
   - `γ̂_l(t) = Σ_{k=1}^{K} w_k^l · t^k / Σ_{k=1}^{K} w_k^l`
   - This is a weighted average of monomials `t, t², ..., t^K`
   - Properties: `γ̂_l(0) = 0`, `γ̂_l(1) = 1`, monotonically increasing

5. **Scale to gamma range:**
   - `γ_l(t) = γ̂_l(t) · (γ_max - γ_min) + γ_min`
   - Default: `γ_min = -13`, `γ_max = 5` (from MELD)

6. **Sigmoid to alpha:**
   - `α_l(t) = σ(-γ_l(t))` where `σ` is the sigmoid function
   - At `t=0`: `γ = γ_min = -13` → `α = σ(13) ≈ 1.0` (clean)
   - At `t=1`: `γ = γ_max = 5` → `α = σ(-5) ≈ 0.007` (masked)

7. **Analytical derivative:**
   - `dγ̂_l/dt = Σ_k w_k^l · k · t^{k-1} / Σ_k w_k^l`
   - `dγ_l/dt = (γ_max - γ_min) · dγ̂_l/dt`
   - `dα_l/dt = -α_l · (1 - α_l) · dγ_l/dt`

### 5.2 Interface

```python
class RateNetwork(nn.Module):
    """Per-position learned masking rates.

    Args:
        vocab_config: VocabConfig providing n_max, n_edges, seq_len.
        d_emb: Embedding dimension for position embeddings. Default 32.
        K: Number of polynomial basis functions. Default 4.
        gamma_min: Minimum gamma (at t=0). Default -13.0.
        gamma_max: Maximum gamma (at t=1). Default 5.0.
        hidden_dim: MLP hidden dimension. Default 64.
    """

    def __init__(
        self,
        vocab_config: VocabConfig,
        d_emb: int = 32,
        K: int = 4,
        gamma_min: float = -13.0,
        gamma_max: float = 5.0,
        hidden_dim: int = 64,
    ) -> None: ...

    def forward(self, t: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Compute per-position keeping probability α_l(t).

        Args:
            t: (B,) timestep in [0, 1].
            pad_mask: (B, SEQ_LEN) bool, True=real position.
                PAD positions get α=1.0 (never masked).

        Returns:
            (B, SEQ_LEN) float32 per-position alpha values.
        """

    def alpha_prime(self, t: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Compute per-position dα_l(t)/dt analytically.

        Args:
            t: (B,) timestep.
            pad_mask: (B, SEQ_LEN) bool.

        Returns:
            (B, SEQ_LEN) float32 per-position alpha derivatives.
            PAD positions return 0.0.
        """

    def forward_with_derivative(
        self, t: Tensor, pad_mask: Tensor | None = None,
    ) -> RateNetworkOutput:
        """Compute alpha and alpha_prime in a single forward pass (efficient).

        Returns:
            RateNetworkOutput with alpha, alpha_prime, gamma tensors.
        """
```

### 5.3 Implementation Notes

- **Softplus for positive coefficients:** `w^l = softplus(MLP(...))` ensures all polynomial coefficients are strictly positive. This guarantees `γ̂_l(t)` is monotonically increasing (weighted sum of increasing monomials with positive weights), and therefore `α_l(t)` is monotonically decreasing.
- **PAD handling:** PAD positions are forced to `α = 1.0` (never masked) and `α' = 0.0` regardless of rate network output. This preserves the critical PAD invariant.
- **Edge endpoint index buffers:** Precompute `edge_i, edge_j` tensors from `vocab_config.edge_position_to_pair()` and register as buffers. Same pattern as `CompositePositionalEncoding`.
- **Parameter count:** With `d_emb=32, K=4, hidden_dim=64`: 8×32 (node emb) + 32×32 (W_node) + 32×32 (W_edge) + 32×64+64 (MLP layer 1) + 64×4+4 (MLP layer 2) ≈ 5K params.

### 5.4 Embedding Design Rationale (d_emb=32 vs CompositePositionalEncoding)

The denoiser's `CompositePositionalEncoding` uses `d_model=128` and encodes three components: entity type (node vs edge via `entity_type_emb(2, 128)`), node slot identity (`node_index_emb(8, 128)`), and edge endpoint pair (`pair_index_emb(i) + pair_index_emb(j)` at `d=128`). These embeddings feed into self-attention and must be compatible with the full transformer width.

The rate network's embeddings serve a fundamentally simpler purpose: they need only to distinguish positions and output `K=4` polynomial coefficients per position. They do NOT participate in self-attention, represent token semantics, or support `d_model`-dimensional computations. Specifically:

- **Sufficient dimensionality:** 8 node vectors in R^32 have ample room to be orthogonal (32 >> 8). Each node gets a unique learned identity, and edge embeddings `h_i + h_j` distinguish all 28 pairs.
- **Entity type distinction:** Implicit via separate `W_node` and `W_edge` projections. Node and edge positions pass through different linear layers, providing the same type-awareness as the denoiser's explicit `entity_type_emb` but through the network architecture rather than a dedicated embedding.
- **Graph structure capture:** `h_edge^{ij} = h_node^i + h_node^j` captures the same structural information as `CompositePositionalEncoding.pair_index_emb(i) + pair_index_emb(j)`. Edges sharing a node share an embedding component, enabling the rate network to learn correlated masking rates for structurally related positions.

**Key difference from MELD's molecular domain:** MELD works with SMILES strings, where positions are sequential tokens with a separate "SMILES-to-graph" transformation. Our bubble diagrams skip this step — positions in the flattened sequence ARE graph elements directly (8 room nodes + 28 edge slots in upper-triangle layout). The `h_i + h_j` construction is particularly natural here because the edge position indices directly encode which room pair they connect (via `vocab_config.edge_position_to_pair()`). No additional graph-structure encoding is needed.

**Configurability:** `d_emb` is a constructor argument. If experiments show 32 is insufficient, it can be increased to 64 (~12K params, still negligible vs ~1.28M denoiser params) without any architectural changes.

---

## 6. Forward Process v2

**File:** `bd_gen/diffusion/forward_process.py` (new function added)

### 6.1 STGS Implementation

```python
def stgs_sample(
    alpha: Tensor,
    gumbel_temperature: float = 1.0,
) -> Tensor:
    """Straight-Through Gumbel-Softmax for discrete masking.

    Args:
        alpha: (B, SEQ_LEN) per-position keeping probability.
        gumbel_temperature: Softmax temperature. Lower → harder.

    Returns:
        (B, SEQ_LEN, 2) tensor. Channel 0 = keep weight, channel 1 = mask weight.
        Forward: hard one-hot. Backward: soft Gumbel-Softmax gradient.
    """
    # Logits for 2-class categorical: [log(alpha), log(1-alpha)]
    logits = torch.stack([
        torch.log(alpha.clamp(min=1e-8)),
        torch.log((1 - alpha).clamp(min=1e-8)),
    ], dim=-1)  # (B, SEQ_LEN, 2)

    # Gumbel noise (float64 for precision)
    u = torch.rand_like(logits, dtype=torch.float64)
    u = u.clamp(min=1e-10, max=1 - 1e-10)
    gumbel = -torch.log(-torch.log(u))

    # Soft sample
    p_soft = F.softmax((logits.double() + gumbel) / gumbel_temperature, dim=-1).float()

    # Hard sample (argmax → one-hot)
    hard_idx = p_soft.argmax(dim=-1)  # (B, SEQ_LEN)
    p_hard = F.one_hot(hard_idx, num_classes=2).float()  # (B, SEQ_LEN, 2)

    # Straight-through: forward uses hard, backward uses soft gradient
    return p_hard - p_soft.detach() + p_soft


def forward_mask_learned(
    x0: Tensor,
    pad_mask: Tensor,
    t: Tensor,
    rate_network: nn.Module,
    denoiser: nn.Module,
    vocab_config: VocabConfig,
    gumbel_temperature: float = 1.0,
) -> STGSOutput:
    """Forward masking with learned per-position rates and STGS.

    Used during v2 training. For v2 inference, use forward_mask_eval_learned().

    Args:
        x0: (B, SEQ_LEN) long tensor of clean token indices.
        pad_mask: (B, SEQ_LEN) bool tensor, True=real position.
        t: (B,) float32 timestep in [0, 1].
        rate_network: RateNetwork providing per-position alpha.
        denoiser: BDDenoiser (for embedding layers).
        vocab_config: VocabConfig for n_max.
        gumbel_temperature: STGS temperature. Annealed during training.

    Returns:
        STGSOutput dict with soft_embeddings, x_t, mask_indicators,
        alpha_per_pos, gumbel_weights.
    """
```

### 6.2 Soft Embedding Construction

The core of STGS: mix clean and MASK embeddings using soft Gumbel-Softmax weights.

```
For node positions (0..n_max-1):
    clean_emb = denoiser.node_embedding(x0[:, :n_max])        # (B, n_max, d)
    mask_emb  = denoiser.node_embedding(NODE_MASK_IDX tensor)  # (B, n_max, d)
    soft_emb  = w_keep * clean_emb + w_mask * mask_emb         # (B, n_max, d)

For edge positions (n_max..seq_len-1):
    clean_emb = denoiser.edge_embedding(x0[:, n_max:])         # (B, n_edges, d)
    mask_emb  = denoiser.edge_embedding(EDGE_MASK_IDX tensor)  # (B, n_edges, d)
    soft_emb  = w_keep * clean_emb + w_mask * mask_emb         # (B, n_edges, d)

PAD positions: force w_keep=1, w_mask=0 (override STGS output).
```

The soft embeddings then go into the denoiser via `pre_embedded` (see Section 8).

### 6.3 Inference-Time Forward Masking

During inference and evaluation, STGS is not used. Standard discrete masking applies:

```python
def forward_mask_eval_learned(
    x0: Tensor,
    pad_mask: Tensor,
    t: Tensor,
    rate_network: nn.Module,
    vocab_config: VocabConfig,
) -> tuple[Tensor, Tensor]:
    """Forward masking for eval with learned rates (discrete, no STGS).

    Same as v1 forward_mask but with per-position alpha_l(t) instead of
    scalar alpha(t). Returns (x_t, mask_indicators) identical to v1 signature.
    """
    alpha = rate_network(t, pad_mask)  # (B, SEQ_LEN)
    rand = torch.rand_like(alpha)
    should_mask = (rand >= alpha) & pad_mask
    # ... apply MASK tokens exactly as v1 forward_mask
```

### 6.4 PAD Invariant

**CRITICAL:** PAD positions must NEVER be masked, in both training (STGS) and eval paths.

- In `rate_network.forward()`: PAD positions forced to `α = 1.0`.
- In `stgs_sample()`: PAD-position STGS weights overridden to `[1, 0]` (keep=1, mask=0).
- In `forward_mask_eval_learned()`: `should_mask &= pad_mask` (same as v1).

---

## 7. ELBO Loss v2

**File:** `bd_gen/diffusion/loss.py` (new class added)

### 7.1 Per-Position ELBO Weight

v1: `w(t) = -α'(t) / (1 - α(t))` — scalar, same for all positions.

v2: `w_l(t) = -α̇_l(t) / (1 - α_l(t))` — per-position, from rate network.

### 7.2 Loss Formulation (Separate Node/Edge Normalization)

Following MELD's factorized reverse distribution `p(g|g_t) = ∏_i p(x_i|g_t) · ∏_{ij} p(e_ij|g_t)`, the loss separates node and edge contributions with independent normalization:

```
L(θ,φ) = E_{t~U[0,1], g~data, g_t~q_φ} [
    L_nodes + λ_edge · L_edges
]

where:
    L_nodes = (1/N_active_nodes) · Σ_{l ∈ nodes} w_l(t) · CE(logits_l, x_0^l) · loss_mask_l
    L_edges = (1/N_active_edges) · Σ_{l ∈ edges} w_l(t) · CE(logits_l, x_0^l) · loss_mask_l
```

Where:
- `w_l(t) = -α̇_l(t,φ) / (1 - α_l(t,φ) + ε)` — per-position ELBO weight
- `loss_mask_l = mask_indicators_l AND pad_mask_l` — only compute loss on masked real positions
- `N_active_nodes = Σ_{l ∈ nodes} loss_mask_l` — count of masked node positions per sample
- `N_active_edges = Σ_{l ∈ edges} loss_mask_l` — count of masked edge positions per sample
- `CE` is cross-entropy with class weights (node_class_weights for nodes, edge_class_weights for edges)

### 7.3 Separate Normalization Rationale

**Why not joint normalization?** With joint `N_active = N_active_nodes + N_active_edges`, edges (up to 28 positions) dominate nodes (up to 8 positions) by ~3.5×. The node loss signal is diluted, making it harder for the model to learn accurate room type predictions.

**Separate normalization** computes `L_nodes / N_active_nodes` and `L_edges / N_active_edges` independently. This ensures each node position contributes proportionally the same as each edge position, before `λ_edge` scaling is applied. Default `λ_edge = 1.0` (equal weighting). Can be tuned to emphasize edge reconstruction if needed.

**Difference from v1:** v1 uses joint `N_active` normalization. This change is specific to `ELBOLossV2` and does not affect v1's `ELBOLoss`.

### 7.4 Interface

```python
class ELBOLossV2(nn.Module):
    """ELBO loss with per-position weights for learned forward process.

    Args:
        edge_class_weights: (EDGE_VOCAB_SIZE,) inverse-frequency weights.
        node_class_weights: (NODE_VOCAB_SIZE,) or None.
        vocab_config: VocabConfig for n_max.
        lambda_edge: Relative weight for edge loss. Default 1.0.
        eps: Numerical stability epsilon. Default 1e-8.
        t_min: Minimum t for clamping. Default 1e-5.
        w_max: Maximum per-position weight clamp. Default 1000.0.
    """

    def forward(
        self,
        node_logits: Tensor,      # (B, n_max, NODE_VOCAB_SIZE)
        edge_logits: Tensor,      # (B, n_edges, EDGE_VOCAB_SIZE)
        x0: Tensor,               # (B, SEQ_LEN) clean tokens
        pad_mask: Tensor,         # (B, SEQ_LEN) bool
        mask_indicators: Tensor,  # (B, SEQ_LEN) bool
        alpha_per_pos: Tensor,    # (B, SEQ_LEN) from rate network
        alpha_prime_per_pos: Tensor,  # (B, SEQ_LEN) from rate network
    ) -> Tensor:
        """Compute per-position ELBO loss.

        Returns:
            Scalar float32 loss, averaged across the batch.
        """
```

### 7.5 Key Differences from ELBOLoss (v1)

| Aspect | v1 (ELBOLoss) | v2 (ELBOLossV2) |
|--------|---------------|------------------|
| Weight | `w(t)` scalar per sample | `w_l(t)` per position |
| Weight source | `noise_schedule.alpha(t)` | `rate_network.forward_with_derivative(t)` |
| Noise schedule | Passed as argument | Not needed (rates from rate_network) |
| Normalization | Joint `N_active` (nodes + edges) | Separate `N_active_nodes`, `N_active_edges` |
| Lambda | N/A | `lambda_edge` config param |

### 7.6 Numerical Stability

Same float64 precision as v1 for the weight computation:

```python
alpha_64 = alpha_per_pos.double()          # (B, SEQ_LEN)
alpha_prime_64 = alpha_prime_per_pos.double()
denominator = 1.0 - alpha_64 + eps
w_per_pos = (-alpha_prime_64 / denominator).float()  # (B, SEQ_LEN)
w_per_pos = torch.clamp(w_per_pos, max=w_max)
```

### 7.7 Safe CE Targets (from v1)

Same pattern as v1 `ELBOLoss`: at non-loss positions, clamp target indices to class 0 before computing cross-entropy. This prevents indexing into `-inf` logits (MASK/PAD indices clamped by SUBS):

```python
safe_node_x0 = torch.where(node_loss_mask, node_x0, torch.zeros_like(node_x0))
safe_edge_x0 = torch.where(edge_loss_mask, edge_x0, torch.zeros_like(edge_x0))
```

Combined with `loss_mask` multiplication, these positions contribute zero to the final loss. This is defense-in-depth: even though `loss_mask` zeros them out, accessing `-inf` logits can produce NaN gradients.

---

## 8. Denoiser Changes

**File:** `bd_gen/model/denoiser.py` (minimal modification)

### 8.1 Change: `pre_embedded` Parameter

Add ONE optional parameter to `BDDenoiser.forward()`:

```python
def forward(
    self,
    tokens: Tensor,
    pad_mask: Tensor,
    t: Tensor | float | int,
    condition: Tensor | None = None,
    pre_embedded: Tensor | None = None,   # NEW (v2 only)
) -> tuple[Tensor, Tensor]:
    """
    Args:
        ...
        pre_embedded: (B, seq_len, d_model) float tensor. If provided,
            skip token embedding and use these directly. Used by v2
            STGS training where soft embeddings replace discrete tokens.
            When None (default), embeds tokens as in v1.
    """
    if pre_embedded is not None:
        x = pre_embedded
    else:
        # v1 path — UNCHANGED
        node_emb = self.node_embedding(tokens[:, :n_max])
        edge_emb = self.edge_embedding(tokens[:, n_max:])
        x = torch.cat([node_emb, edge_emb], dim=1)

    # Everything below is UNCHANGED from v1
    x = x + self.positional_encoding()
    # ... timestep embedding, transformer blocks, heads, SUBS masking
```

### 8.2 Backward Compatibility

- Default `pre_embedded=None` → 100% identical to v1 behavior.
- All v1 callers (train.py, evaluate.py, sampling.py, tests) pass no `pre_embedded` → no change needed.
- The v2 training loop is the ONLY caller that passes `pre_embedded`.
- v1 checkpoint loading works unchanged (no new parameters in denoiser).

---

## 9. Sampling v2

**File:** `bd_gen/diffusion/sampling.py` (modified `sample()` function)

### 9.1 Per-Position Unmasking Probability

v1 (scalar):
```
p_unmask = (α(t_next) - α(t_now)) / (1 - α(t_now))   →   (B, 1) broadcast to (B, SEQ_LEN)
```

v2 (per-position):
```
p_unmask_l = (α_l(t_next) - α_l(t_now)) / (1 - α_l(t_now))   →   (B, SEQ_LEN)
```

**Float64 precision (from v1):** Compute `p_unmask_l` in float64, same as v1's scalar `p_unmask`. Rate network outputs are cast to double before the division, then clamped to `[0, 1]` and cast back to float32. This prevents catastrophic cancellation when `α_l(t_next) ≈ α_l(t_now)` (high num_steps).

### 9.2 Modified `sample()` Signature

Add `rate_network` parameter to `sample()`:

```python
def sample(
    model, noise_schedule, vocab_config, batch_size, num_steps,
    ...,
    rate_network: nn.Module | None = None,  # NEW: if provided, use learned rates
    ...,
) -> Tensor:
```

When `rate_network is None` (default): v1 behavior (scalar alpha from noise_schedule).
When `rate_network is not None`: v2 behavior (per-position alpha from rate_network).

### 9.3 Unmasking Modes with Per-Position Alpha

**"random" mode:** Per-position Bernoulli sampling with `p_unmask_l`:
```python
unmask_rand = torch.rand(B, SEQ_LEN, device=device)
should_unmask = (unmask_rand < p_unmask_per_pos) & is_mask
```
Straightforward — each position has its own probability.

**"llada" mode:** Budget-based confidence ranking:
```python
# Budget = expected number of positions to unmask
budget = p_unmask_per_pos[is_mask].sum()  # total across masked positions
# Or per-sample: budget_b = (p_unmask_per_pos * is_mask).sum(dim=1)
num_to_unmask = budget.round().long().clamp(min=1)

# Rank by confidence (same as v1), unmask top-k
confidence = ...  # same as v1
_, topk_idx = torch.topk(confidence, num_to_unmask)
```

For llada mode with per-position alpha, the "budget" is the sum of per-position unmasking probabilities across masked positions. This naturally gives more unmasking at timesteps where more positions are scheduled to be revealed.

### 9.4 Initialization with Learned Rates

At `t=1`, the rate network gives `α_l(1) ≈ 0.007` for all positions (by construction: `γ_max=5` → `σ(-5) ≈ 0.007`). The initial state is still fully masked, same as v1.

At `t=0`, `α_l(0) ≈ 1.0`, so all positions should be unmasked. Same endpoint as v1.

### 9.5 Remasking Compatibility (Deferred)

When `rate_network` is provided and `remasking_fn` is also provided, log a warning and skip remasking. Remasking with learned rates requires adapting `RemaskingSchedule` to use per-position sigma_max values. This is deferred to post-v2.

---

## 10. Training Script v2

**File:** `scripts/train_v2.py`

### 10.1 Training Loop Pseudocode

```python
def train_v2(cfg):
    # 1. Setup (same as v1)
    model = BDDenoiser(...).to(device)
    rate_network = RateNetwork(...).to(device)

    # 2. Single optimizer for both networks
    optimizer = AdamW(
        list(model.parameters()) + list(rate_network.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # 3. Loss
    criterion = ELBOLossV2(edge_class_weights, lambda_edge=cfg.training.lambda_edge)

    # 4. Gumbel temperature schedule
    gumbel_temp_start = cfg.training.gumbel_temperature_start  # e.g., 1.0
    gumbel_temp_end = cfg.training.gumbel_temperature_end      # e.g., 0.1
    gumbel_temp_decay = cfg.training.gumbel_temperature_decay  # 'linear' or 'cosine'

    for epoch in range(epochs):
        # Anneal Gumbel temperature
        progress = epoch / max(epochs - 1, 1)
        if gumbel_temp_decay == 'linear':
            gumbel_temp = gumbel_temp_start + progress * (gumbel_temp_end - gumbel_temp_start)
        elif gumbel_temp_decay == 'cosine':
            gumbel_temp = gumbel_temp_end + 0.5 * (gumbel_temp_start - gumbel_temp_end) * (1 + cos(pi * progress))

        for batch in train_loader:
            tokens, pad_mask = batch['tokens'], batch['pad_mask']
            t = torch.rand(B, device=device).clamp(min=1e-5)

            # Forward masking with STGS
            stgs_out = forward_mask_learned(
                tokens, pad_mask, t, rate_network, model,
                vocab_config, gumbel_temperature=gumbel_temp,
            )

            # Denoiser with soft embeddings
            node_logits, edge_logits = model(
                tokens, pad_mask, t,
                pre_embedded=stgs_out['soft_embeddings'],
            )

            # Per-position ELBO loss
            rate_out = rate_network.forward_with_derivative(t, pad_mask)
            loss = criterion(
                node_logits, edge_logits, tokens,
                pad_mask, stgs_out['mask_indicators'],
                rate_out['alpha'], rate_out['alpha_prime'],
            )

            loss.backward()
            clip_grad_norm_(
                list(model.parameters()) + list(rate_network.parameters()),
                max_norm=cfg.training.grad_clip,
            )
            optimizer.step()
            optimizer.zero_grad()
```

### 10.2 Gumbel Temperature Annealing

- **Start:** `η = 1.0` (soft, maximum gradient flow)
- **End:** `η = 0.1` (near-hard, matches inference behavior)
- **Schedule:** Linear or cosine decay over training epochs
- **Rationale:** High temperature early allows the rate network to explore; low temperature later reduces the train-inference distribution gap.

### 10.3 Validation

Validation uses discrete masking (no STGS):
```python
# Validation: use eval path (discrete, like inference)
x_t, mask_ind = forward_mask_eval_learned(tokens, pad_mask, t, rate_network, vocab_config)
node_logits, edge_logits = model(x_t, pad_mask, t)  # pre_embedded=None → v1 path
loss = criterion(node_logits, edge_logits, tokens, pad_mask, mask_ind, alpha, alpha_prime)
```

### 10.4 Sampling During Training

Use `sample()` with `rate_network=rate_network`:
```python
samples = sample(
    model=model, noise_schedule=None, vocab_config=vocab_config,
    rate_network=rate_network, batch_size=8, num_steps=100,
    unmasking_mode='llada', top_p=0.9, device=device,
)
```

### 10.5 Checkpointing

Save both model and rate network:
```python
save_checkpoint_v2(model, rate_network, optimizer, epoch, cfg, path)
```

The checkpoint dict includes:
- `model_state_dict` — denoiser weights
- `rate_network_state_dict` — rate network weights
- `optimizer_state_dict`
- `epoch`, `config`

### 10.6 Shared Utilities

Import from existing modules (no duplication):
- `_load_config()` from train.py pattern (copy the function, not import)
- `_build_lr_lambda()` — same warmup schedule
- `_validate()` — adapted for v2 (per-position weights)
- `BubbleDiagramDataset`, `VocabConfig`, `save_checkpoint`, `set_seed`, etc.

---

## 11. Config System

### 11.1 Rate Network Config (`configs/noise/learned.yaml`)

```yaml
type: learned
d_emb: 32
K: 4
gamma_min: -13.0
gamma_max: 5.0
hidden_dim: 64
```

### 11.2 Training Config (`configs/training/v2.yaml`)

```yaml
lr: 3e-4
weight_decay: 0.01
warmup_steps: 1000
epochs: 500
optimizer: adamw
grad_clip: 1.0
checkpoint_every: 50
sample_every: 25
val_every: 5
ema: false
importance_sampling: false    # Disabled for v2 (uniform t)
lambda_edge: 1.0              # Node/edge loss balance
gumbel_temperature_start: 1.0
gumbel_temperature_end: 0.1
gumbel_temperature_decay: linear  # 'linear' or 'cosine'
```

### 11.3 Main Config Override

```bash
python scripts/train_v2.py noise=learned training=v2
```

---

## 12. Evaluation Plan

### 12.1 Metrics (Same as v1)

All metrics from `bd_gen/eval/metrics.py` and `bd_gen/eval/validity.py` are reused unchanged:

| Category | Metrics |
|----------|---------|
| Validity | no_mask, no_oor, connected, consistent, valid_types |
| Coverage | novelty, diversity, mode_coverage (weighted) |
| Distribution | node_TV, edge_TV, num_rooms_W1 |
| Structure | graph_structure_mmd, spatial_transitivity |
| Conditional | cond_edge_TV (weighted), type_cond_degree_TV (weighted) |
| Model quality | denoising_accuracy, denoising_val_elbo |

### 12.2 Comparison Methodology

1. Train v2 model for 500 epochs (same as v1)
2. Evaluate with **same settings** as v1 best model:
   - Unmasking mode: `llada`
   - Sampling: `top_p=0.9`, `temperature=1.0`
   - Steps: 100
   - Samples: 1000 per seed
   - Seeds: `[42, 123, 456, 789, 1337]`
   - **No remasking** (to isolate learned-rate effect)
3. Compare against:
   - `llada_topp0.9_no_remask` (v1, fixed rates, no remasking) — **primary comparison** (same inference setup, isolates learned-rate effect)
   - `llada_topp0.9_remdm_confidence_tsw0.5` (v1, fixed rates, with remasking) — **secondary/aspirational reference** (if v2 without remasking beats v1 with remasking, learned rates subsume remasking benefits)

#### Remasking Context

**v1 remasking is inference-only.** The v1 denoiser was trained with standard MDLM (no remasking awareness). Remasking is applied only during `sample()` at inference time. No retraining was needed to add remasking to v1.

**v2 does NOT use remasking** in this initial implementation. The primary hypothesis is that learned rates improve base generation quality independently of remasking. Evaluating without remasking isolates the learned-rate effect from remasking effects.

**Could remasking be added to v2 later?** Yes, as an inference-only enhancement — no retraining required. However, the current `RemaskingSchedule` uses scalar `α(t)` from `NoiseSchedule`. Adapting it to per-position `α_l(t)` from the rate network requires interface changes (per-position `sigma_max` values). This is deferred to post-v2.

**Why v2 vs `llada_topp0.9_remdm_confidence_tsw0.5` is NOT a controlled comparison:** It conflates two independent changes (learned rates vs remasking). The primary comparison (v2 no-remasking vs v1 no-remasking) is the scientifically valid one. The secondary comparison is aspirational: beating v1-with-remasking without remasking would be strong evidence for the learned forward process.

### 12.3 Expected Improvements

Based on MELD results (93% vs 15% validity on ZINC250K) and our domain:

| Metric | v1 Baseline | Expected v2 | Rationale |
|--------|-------------|-------------|-----------|
| Validity | 99.7-99.9% | ~same | Already near-ceiling |
| Mode coverage | 69.6% (no remask) | Higher | Less state clashing → more diverse trajectories |
| Spatial transitivity | 99.9% (no remask) | ~same or better | Per-position rates can learn to avoid edge conflicts |
| Cond. edge TV | 0.472 (no remask) | Lower (better) | Less confusion from state clashing |
| Diversity | ~0.63 | Higher | Different masking trajectories → different samples |

### 12.4 Evaluation Script

Use existing `scripts/evaluate.py` with `rate_network` loaded from v2 checkpoint:
```bash
python scripts/evaluate.py \
    eval.checkpoint_path=outputs/<v2_run>/checkpoints/checkpoint_final.pt \
    eval.forward_process=learned \
    eval.unmasking_mode=llada \
    eval.top_p=0.9
```

This requires a small addition to `evaluate.py`: load rate network from checkpoint when `eval.forward_process=learned`.

---

## 13. Implementation Phases

> **Tracking:** Each phase will be tracked in `BD_Generation/implementation_state_T1.md` following the same per-phase compact summary format used for v1 phases.

### Phase 1: Rate Network Module
**Files:** `bd_gen/diffusion/rate_network.py`
**Parallelization:** Single agent — self-contained new file, no dependencies.
**Scope:**
- Implement `RateNetwork` class with polynomial parameterization
- `forward()`, `alpha_prime()`, `forward_with_derivative()`
- PAD handling
- Unit tests: shape checks, boundary conditions (t=0→α≈1, t=1→α≈0), monotonicity, PAD invariant

### Phase 2: STGS and Forward Process v2
**Files:** `bd_gen/diffusion/forward_process.py`
**Parallelization:** Single agent — depends on Phase 1 (uses `RateNetwork`).
**Scope:**
- Implement `stgs_sample()` function
- Implement `forward_mask_learned()` function (training path)
- Implement `forward_mask_eval_learned()` function (eval path)
- Verify PAD invariant in both paths
- Unit tests: STGS gradient flow, soft embedding shapes, PAD protection

### Phase 3: Loss v2
**Files:** `bd_gen/diffusion/loss.py`
**Parallelization:** **Can run in parallel with Phase 2.** Only needs `RateNetworkOutput` type from Phase 1, not the STGS code.
**Scope:**
- Implement `ELBOLossV2` class
- Separate node/edge normalization (`N_active_nodes`, `N_active_edges`)
- Per-position weight computation with float64 precision
- Lambda edge weighting
- Unit tests: weight shapes, gradient flow to rate network, comparison with v1 loss at uniform alpha

### Phase 4: Denoiser Change
**Files:** `bd_gen/model/denoiser.py`
**Parallelization:** **Can run in parallel with Phases 2/3.** Trivial change (one optional param), independent of other v2 modules.
**Scope:**
- Add `pre_embedded` parameter to `forward()`
- Verify backward compatibility (all existing tests pass)
- Unit test: soft embedding input produces same-shape logits

### Phase 5: Sampling v2
**Files:** `bd_gen/diffusion/sampling.py`
**Parallelization:** Depends on Phase 1 (uses `RateNetwork.forward()` for per-position alpha). Can start once Phase 1 is complete, in parallel with Phase 3 if Phase 1 is done.
**Scope:**
- Add `rate_network` parameter to `sample()`
- Per-position unmasking probability
- Adapt both "random" and "llada" modes
- Unit tests: per-position p_unmask, generated samples have no MASK tokens

### Phase 6: Training Script v2
**Files:** `scripts/train_v2.py`, `configs/noise/learned.yaml`, `configs/training/v2.yaml`
**Parallelization:** Depends on all prior phases (1-5). Single agent — integrates all components.
**Scope:**
- Full training loop with joint optimization
- Gumbel temperature annealing
- Validation with discrete masking
- Checkpoint saving (both model and rate_network)
- Smoke test: 2 epochs, loss decreases

### Phase 7: Evaluation Integration
**Files:** `scripts/evaluate.py` (minor addition)
**Parallelization:** Depends on Phase 6. Single agent.
**Scope:**
- Load rate_network from v2 checkpoint
- Pass to `sample()` during evaluation
- Run full evaluation on v2 model
- Generate comparison.md against v1 models

### Parallelization Summary

```
Phase 1 ──→ Phase 2 ──→┐
    │                    │
    ├──→ Phase 3 ──────→├──→ Phase 6 ──→ Phase 7
    │                    │
    └──→ Phase 4 ──────→┘
    │
    └──→ Phase 5 ──────→┘
```

Phases 2, 3, 4 can run in parallel once Phase 1 completes. Phase 5 can start after Phase 1 but is lighter and can overlap with 3/4. Phase 6 blocks on all five.

---

## 14. Test Cases

### 14.1 Rate Network Tests

| # | Test | Assertion |
|---|------|-----------|
| 1 | Shape check | `rate_network(t, pad_mask).shape == (B, SEQ_LEN)` |
| 2 | Boundary t=0 | All `alpha > 0.99` (positions nearly clean) |
| 3 | Boundary t=1 | All real-position `alpha < 0.02` (nearly masked) |
| 4 | Monotonicity | For t1 < t2: `alpha(t1) > alpha(t2)` element-wise |
| 5 | PAD invariant | `alpha[~pad_mask] == 1.0` exactly |
| 6 | alpha_prime sign | `alpha_prime <= 0` everywhere (decreasing alpha) |
| 7 | alpha_prime PAD | `alpha_prime[~pad_mask] == 0.0` exactly |
| 8 | Gradient flow | `rate_network.node_embeddings.grad is not None` after backward |
| 9 | Different positions | Not all positions have identical alpha (learned diversity) after 100 gradient steps |
| 10 | Consistency | `forward_with_derivative()` matches `forward()` + `alpha_prime()` |

### 14.2 STGS Tests

| # | Test | Assertion |
|---|------|-----------|
| 11 | Shape | `stgs_sample(alpha).shape == (B, SEQ_LEN, 2)` |
| 12 | Sum to 1 | `weights.sum(dim=-1) ≈ 1.0` |
| 13 | Hard in forward | `(weights == 0) | (weights == 1)` for each channel |
| 14 | Gradient flow | `alpha.grad is not None` after `stgs_sample(alpha).sum().backward()` |
| 15 | PAD override | `weights[~pad_mask, 0] == 1.0` (keep=1 for PAD) |
| 16 | Temperature effect | Lower temp → weights closer to one-hot |

### 14.3 Forward Process v2 Tests

| # | Test | Assertion |
|---|------|-----------|
| 17 | Soft emb shape | `soft_emb.shape == (B, SEQ_LEN, d_model)` |
| 18 | Discrete x_t shape | `x_t.shape == (B, SEQ_LEN)` and dtype long |
| 19 | mask_indicators bool | `mask_indicators.dtype == bool` |
| 20 | PAD never masked | `mask_indicators[~pad_mask].all() == False` |
| 21 | t=0 almost clean | `mask_indicators.sum() < 0.05 * pad_mask.sum()` |
| 22 | t=1 almost masked | `mask_indicators.sum() > 0.95 * pad_mask.sum()` |
| 23 | Eval discrete path | `forward_mask_eval_learned()` returns same signature as v1 `forward_mask()` |

### 14.4 Loss v2 Tests

| # | Test | Assertion |
|---|------|-----------|
| 24 | Scalar output | `loss.shape == ()` |
| 25 | Positive loss | `loss > 0` for random logits |
| 26 | Gradient to denoiser | `model.node_head.weight.grad is not None` |
| 27 | Gradient to rate net | `rate_network.node_embeddings.grad is not None` |
| 28 | PAD exclusion | Loss unchanged when PAD-position logits are randomized |
| 29 | Lambda=0 edges | `lambda_edge=0` → loss depends only on node positions |
| 30 | Separate normalization | `N_active_nodes` and `N_active_edges` computed independently; node loss not diluted by edge count |

### 14.5 Denoiser Backward Compatibility Tests

| # | Test | Assertion |
|---|------|-----------|
| 31 | v1 path | `model(x_t, pad_mask, t)` works identically (no pre_embedded) |
| 32 | v2 path | `model(x0, pad_mask, t, pre_embedded=soft_emb)` produces valid logits |
| 33 | Same shapes | Both paths produce `(B, n_max, 15)` and `(B, n_edges, 13)` |
| 34 | Existing tests pass | All tests in `tests/` pass without modification |

### 14.6 Sampling v2 Tests

| # | Test | Assertion |
|---|------|-----------|
| 35 | No MASK tokens | Generated samples have no NODE_MASK_IDX or EDGE_MASK_IDX in real positions |
| 36 | PAD correct | PAD positions contain NODE_PAD_IDX / EDGE_PAD_IDX |
| 37 | Shapes | Output `(B, SEQ_LEN)` long |
| 38 | rate_network=None | Behaves identically to v1 (backward compatible) |

### 14.7 Integration Tests

| # | Test | Assertion |
|---|------|-----------|
| 39 | 2-epoch training | train_v2.py runs 2 epochs without error, loss decreases |
| 40 | Checkpoint roundtrip | Save + load v2 checkpoint, produce same samples |
| 41 | v1 checkpoint unaffected | Loading v1 checkpoint into denoiser still works |

---

## 15. Computational Advantages of v2 over v1

| Advantage | Detail |
|-----------|--------|
| **Reduced state clashing** | Per-position masking rates ensure structurally distinct bubble diagrams maintain distinct intermediate states. The denoiser gets a clearer, less multimodal training signal, avoiding the mode-covering KL problem. |
| **Per-position adaptive emphasis** | ELBO weights `w_l(t)` automatically focus training on positions that are hard to reconstruct at each timestep. Unlike v1's scalar `w(t)`, this provides fine-grained learning signal per position. |
| **Learnable corruption ordering** | The rate network discovers an optimal masking hierarchy (e.g., mask edges before nodes, or mask peripheral rooms before core rooms). This emergent ordering reflects structural dependencies in bubble diagrams without hand-engineering. |
| **No noise schedule tuning** | v1 required manual selection among linear, log-linear, and cosine schedules (linear + IS broke the model; log-linear was needed). v2 learns the optimal schedule per-position, eliminating this design decision. |
| **No importance sampling needed** | v1 uses importance sampling to reduce gradient variance from the global `w(t)`. v2's per-position weights `w_l(t)` already provide position-adaptive emphasis, making uniform `t ~ U[0,1]` sufficient. |
| **Tiny parameter overhead** | ~5K rate network params vs ~1.28M denoiser params (0.4%). The rate network is negligible in memory and compute. |
| **Same inference cost** | At sampling time, one rate network forward pass computes all 36 `α_l(t)` values — comparable cost to the scalar `α(t)` lookup in v1. No STGS at inference. |
| **Separate node/edge normalization** | v2's `ELBOLossV2` normalizes node and edge losses independently, preventing the 28 edge positions from drowning out the 8 node positions. |
| **Post-hoc remasking compatible** | Remasking can be added as inference-only enhancement later (requires `RemaskingSchedule` adaptation to per-position `α_l`), potentially combining learned rates and remasking benefits. |

---

## 16. Initial Results (500 epochs, jabiru GPU)

**Training:** 500 epochs, lr=3e-4, AdamW, Gumbel temp 1.0→0.1 linear, lambda_edge=1.0, uniform t.
**Evaluation:** llada + top-p 0.9, 100 steps, 1000 samples, 5 seeds, no remasking.
**Checkpoint:** `outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt`

### 16.1 Primary Comparison: v2 vs v1 (llada_topp0.9_no_remask)

This is the controlled comparison — same inference setup, isolating the effect of learned rates.

| Metric | v1 | v2 | Change |
|---|:---:|:---:|---|
| Validity | 100.0% | 100.0% | Same |
| Spatial transitivity | 99.9% | 100.0% | Same |
| **Edge JS** | 0.106 | **0.035** | **3x better** |
| **Node JS** | 0.023 | **0.013** | **44% better** |
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

### 16.2 Secondary Comparison: v2 vs v1 best (llada_topp0.9_remdm_confidence_tsw0.5)

This is aspirational — comparing v2 without remasking to v1 with remasking (not a controlled comparison; conflates learned rates vs remasking).

| Metric | v1 + remasking | v2 (no remask) | Change |
|---|:---:|:---:|---|
| Validity | 99.8% | **100.0%** | Better |
| Edge JS | 0.207 | **0.035** | **6x better** |
| Node JS | 0.046 | **0.013** | **3.5x better** |
| Cond. edge JS (wt) | 0.254 | **0.155** | 39% better |
| Mode coverage (wt) | 72.7% | **78.3%** | +5.6% |
| Denoising acc_edge@0.5 | 0.54 | **0.60** | +11% |
| Diversity | **0.983** | 0.671 | Much worse |
| Unique archetypes | **120.2** | 26.0 | Much worse |
| MMD-Degree | **0.035** | 0.104 | 3x worse |

v2 without remasking already beats v1's best remasking configuration on distribution fidelity by a wide margin. The diversity gap is the main deficit.

### 16.3 Interpretation

**What worked (MELD hypothesis confirmed):**
- Per-position learned rates reduce state clashing, producing a significantly better denoiser. Denoising accuracy improved 11-18% at medium masking rates — the model reconstructs masked positions far more accurately.
- Distribution fidelity improved dramatically: Edge JS dropped from 0.106 to 0.035, matching `random_topp` levels (best v1 distribution matcher) while retaining llada's perfect validity and spatial transitivity.
- Mode coverage increased from 69.6% to 78.3%, confirming that reduced state clashing allows the model to capture more of the data distribution.

**What regressed (diversity-accuracy tradeoff):**
- Diversity dropped from 0.945 to 0.671. The learned masking trajectories are more structured and deterministic than uniform random masking, reducing the stochastic variation between sampling runs that drives diversity.
- MMD graph structure metrics (degree, clustering) regressed 2-3x, suggesting the generated graphs cluster around a narrower set of structural patterns.
- Novelty dropped from 0.975 to 0.864 — more generated samples match training data exactly.

**Key insight:** v2 trades diversity for accuracy. The model generates *more correct* but *less varied* outputs. This is a fundamentally different Pareto point from v1.

### 16.4 Next Steps

1. **Add remasking on top of v2** — remasking was the primary diversity driver in v1 (archetypes: 29 → 120 with confidence remasking). Adapting `RemaskingSchedule` to use per-position `α_l(t)` from the rate network could recover diversity while preserving v2's distribution fidelity gains. This requires modifying the remasking sigma computation to use per-position alpha values.

2. **Try `random` unmasking mode with v2** — in v1, random unmasking had the best diversity (1.0) and distribution match. Combined with v2's better denoiser, random unmasking may produce a strong balance of accuracy and variety.

3. **Increase top-p** (e.g., 0.95 or 1.0) — higher nucleus sampling threshold injects more stochasticity at token selection time, which may compensate for the reduced trajectory-level stochasticity from learned rates.
