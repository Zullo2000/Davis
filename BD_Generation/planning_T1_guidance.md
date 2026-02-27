# Planning T1: Inference-Time Guidance (SVDD)

**Version:** 1.0
**Date:** 2026-02-27
**Status:** DRAFT
**Predecessor:** `planning_T1_with_learned_forward_process.md` (v2)
**Reference:** SVDD — Li et al., "Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding" (arXiv:2408.08252)
**Best models:** `v2_llada_topp0.9_no_remask` (quality), `llada_topp0.9_remdm_confidence_tsw0.5` (diversity)

---

## Table of Contents

1. [Overview & Motivation](#1-overview--motivation)
2. [Design Decisions & Justifications](#2-design-decisions--justifications)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Type Definitions](#4-data-type-definitions)
5. [Constraint Primitives Module](#5-constraint-primitives-module)
6. [Soft Violation Computation](#6-soft-violation-computation)
7. [Reward Composition](#7-reward-composition)
8. [Guided Sampler (SVDD Reweighting)](#8-guided-sampler-svdd-reweighting)
9. [Constraint Schema & Configuration](#9-constraint-schema--configuration)
10. [Calibration Protocol](#10-calibration-protocol)
11. [Guidance Diagnostics & Evaluation](#11-guidance-diagnostics--evaluation)
12. [Scripts & CLI](#12-scripts--cli)
13. [Implementation Phases](#13-implementation-phases)
14. [Test Cases](#14-test-cases)

---

## 1. Overview & Motivation

### 1.1 The Constraint Problem

The bubble diagram generation model produces structurally valid floorplan graphs (rooms and spatial relationships) but has **no mechanism to enforce architectural constraints**. Generated diagrams may have wrong room counts (3 kitchens, 0 living rooms), forbidden adjacencies (bathroom directly connected to kitchen), or missing required relationships (kitchen not adjacent to living room).

Constraints are the most important feature of the model. Without them, the generated diagrams are architecturally meaningless — they are statistically plausible but not controllable. The constraint set will grow over time as the model matures.

### 1.2 The SVDD Approach

SVDD (Soft Value-Based Decoding) provides **inference-time guidance** — no retraining of the denoiser or rate network is needed. At each denoising step:

1. The pre-trained model predicts logits (one forward pass, shared).
2. K candidate transitions are generated from the base distribution.
3. Each candidate is scored by a reward function `r(x) = -E(x)` derived from constraint violations.
4. Candidates are resampled with importance weights `w_k ∝ exp(r / α)`.
5. The selected candidate proceeds to the next step.

The reward function uses **graded (non-binary) violation magnitudes** to provide dense signal. A diagram with 2 kitchens scores better than one with 3 kitchens, even though both violate "exactly 1 kitchen." This avoids the sparse-reward problem that plagues binary feasibility checks.

### 1.3 Two Reward Modes

The reward at intermediate timesteps (when some positions are still masked) can be computed in two ways:

**Hard mode (standard SVDD posterior mean):** Hard-decode `x̂_0` from logits (committed positions keep their token, masked positions use argmax), evaluate constraints on the decoded graph. Simple but **discontinuous** — small logit changes can flip argmax and jump the violation by integer amounts.

**Soft mode (posterior expectation refinement):** Use the full posterior distributions (softmax of logits) to compute expected constraint violations. **Smooth** — small logit changes produce small violation changes. Lower variance because it uses the full distribution, not a point estimate. Reduces to hard mode when all positions are committed.

Both modes are implemented; the better one is chosen empirically (see Section 11).

### 1.4 Goals

1. Implement SVDD-style inference-time guidance with K-candidate reweighting.
2. Build a modular constraint library (4 initial primitives, extensible to more).
3. Support both hard and soft violation modes, with comprehensive diagnostics.
4. Work with both v1 (fixed schedule) and v2 (learned per-position rates).
5. Establish calibration and evaluation methodology for guided generation.

### 1.5 Scope

**In scope:**
- Constraint primitives: ExactCount, CountRange, RequireAdj, ForbidAdj
- Soft violation computation from logits
- RewardComposer with shaping functions and calibration
- Guided sampler with SVDD K-candidate reweighting
- Constraint schema (Pydantic, JSON/YAML)
- Calibration protocol (P90 normalization)
- Full trajectory diagnostics (ESS, reward, violation curves per step)
- Generation script and config system

**Out of scope (deferred):**
- Connectivity constraint (requires graph-level computation, non-decomposable)
- AccessRule constraint (bedroom access, depends on connectivity)
- ConditionalRequirement (Poisson-binomial DP, complex)
- LLM natural language constraint interface
- Interactive CLI constraint selector
- Guidance-aware remasking (protect constraint-satisfying positions from re-masking)

### 1.6 Removal Checklist

If guidance is abandoned, all existing code continues unchanged. To remove:

**Delete these new files (guidance-only):**
- `bd_gen/guidance/` (entire package: `__init__.py`, `constraints.py`, `reward.py`, `soft_violations.py`, `constraint_schema.py`, `calibration.py`, `guided_sampler.py`)
- `scripts/generate_guided.py`, `scripts/calibrate_constraints.py`
- `configs/guidance/` (all YAML files)
- `tests/test_constraints.py`, `tests/test_soft_violations.py`, `tests/test_constraint_schema.py`, `tests/test_guided_sampler.py`

**Revert additive changes in existing files:**

| File | Change | Safe to leave? |
|------|--------|:-:|
| `bd_gen/diffusion/sampling.py` | `_single_step_unmask()` and `_single_step_remask()` helper extraction | Yes — `sample()` calls both sequentially, behavior identical |

**Not modified at all:** `bd_gen/model/*`, `bd_gen/data/*`, `bd_gen/eval/*`, `bd_gen/diffusion/noise_schedule.py`, `bd_gen/diffusion/remasking.py`, `bd_gen/diffusion/forward_process.py`, `bd_gen/diffusion/loss.py`, `scripts/train.py`, `scripts/train_v2.py`, `scripts/evaluate.py`, `scripts/generate_samples.py`.

---

## 2. Design Decisions & Justifications

### 2.1 SVDD K-Candidate Reweighting (Not Logit Modification)

**Decision:** Implement SVDD-style K-candidate importance reweighting.

**Alternative rejected:** Logit modification (add reward gradients directly to logits before token selection). This is simpler and cheaper but less principled — it distorts the base distribution in ways that are hard to characterize, and has no theoretical convergence guarantees.

**Why SVDD:** K-candidate reweighting preserves the base model's learned distribution as the proposal and only tilts it toward constraint satisfaction via importance weights. The model call is shared across K candidates (1x cost), and the K-fold expansion only applies to the lightweight stochastic transition and scoring operations. With K=8 and batch_size=64, the expanded batch is 512 — well within GPU capacity.

### 2.2 Graded Violations (Not Binary Feasibility)

**Decision:** Use continuous, graded violation magnitudes for all constraints.

**Why not binary?** A binary reward (1 if all constraints satisfied, 0 otherwise) is extremely sparse at intermediate timesteps: nearly all candidates score 0, and SVDD's importance weights become uniform (all equally bad). Graded violations provide dense signal — a diagram with 2 kitchens (violation=1) scores better than one with 3 kitchens (violation=2), giving SVDD meaningful differentiation between candidates.

**Shaping functions** `φ(v)` transform raw violations before weighting:
- `linear`: `φ(v) = v` — default, stable, interpretable
- `quadratic`: `φ(v) = v²` — penalizes large violations more heavily
- `log1p`: `φ(v) = log(1+v)` — compresses large violations, prevents domination

### 2.3 Score Before Remasking (Unmask-Score-Remask Ordering)

**Decision:** Split the sampling loop into `_single_step_unmask` and `_single_step_remask`. The guided sampler scores K candidates **after unmasking but before remasking**, then applies remasking only to the selected winner.

**Why?** Remasking re-masks low-confidence positions — which may be exactly the positions where guidance steered the model toward constraint satisfaction (constraint-satisfying tokens in unusual positions tend to have low base-model confidence). If remasking runs before scoring:

1. Many positions are re-masked back to MASK across all K candidates.
2. MASK positions contribute the same logits (shared model call) — no differentiation.
3. The only reward signal comes from positions that survived remasking (high-confidence from the base model's perspective — exactly where guidance was least needed).
4. ESS stays near K, not because guidance is uninformative, but because remasking erased the evidence.

By scoring pre-remask and remasking only the winner, ESS reflects pure guidance effectiveness. The `reward_remasking_delta[t]` diagnostic (Section 11) tracks whether remasking cooperates with or fights guidance.

### 2.4 Soft Violations from Full Posterior (Not Argmax Decode)

**Decision:** Implement both soft (posterior expectation) and hard (argmax decode) reward modes. The soft mode uses the full softmax(logits) distribution rather than a single `x̂_0 = argmax(logits)`.

**Why both?** The soft mode is theoretically superior (smooth, lower variance), but we need empirical validation. The hard mode serves as a baseline and may be simpler to debug. Both are implemented with a `reward_mode` config flag.

**Soft mode details:** For each position, the effective probability distribution is:
- PAD: all zeros (excluded)
- Committed (not MASK): one-hot on the committed token
- MASK: `softmax(logits)` from the model

Expected structural quantities (counts, adjacency probabilities) are computed from these distributions. E.g., expected kitchen count: `n̂_K = Σ_i q_i(Kitchen)` where `q_i` is the effective probability at node position `i`.

### 2.5 Separate Guidance Package (Not Inline in Sampling)

**Decision:** Create a new `bd_gen/guidance/` package, not add guidance logic inline to `sampling.py`.

**Motivation:**
1. Guidance is conceptually separate from base diffusion — constraints, rewards, and diagnostics are orthogonal to the sampling algorithm.
2. The constraint library will grow over time; it needs its own module structure.
3. The schema/validation system (Pydantic) is a distinct concern.
4. Testing is cleaner with isolated modules.

The only change to `sampling.py` is extracting `_single_step_unmask` and `_single_step_remask` — a pure refactoring with no behavior change.

### 2.6 v1 and v2 Compatibility from the Start

**Decision:** Guidance works with both v1 (fixed schedule) and v2 (learned per-position rates) from day one.

**Why not v2-only?** v1 with remasking produces higher diversity (121 archetypes vs 40 for v2). Depending on the use case, users may prefer v1+guidance (max diversity under constraints) or v2+guidance (max quality under constraints). The guided sampler accepts `rate_network=None` (v1 path) or a rate network (v2 path), reusing the existing branching in `sampling.py`.

### 2.7 Existing Invariants Inherited

Guidance inherits all v1/v2 invariants through reusing unmodified code:

| Invariant | Where | How guidance inherits |
|-----------|-------|----------------------|
| **SUBS zero masking** | `denoiser.py:218-222` | Model call is shared; MASK/PAD logits already `-inf` |
| **Top-p sampling** | `sampling.py:55-87` | Called inside `_single_step_unmask` unchanged |
| **LLaDA confidence unmasking** | `sampling.py:285-325` | Called inside `_single_step_unmask` unchanged |
| **PAD invariant** | `sampling.py:90-116` | `_clamp_pad` runs inside `_single_step_unmask` |
| **Carry-over unmasking** | `sampling.py:332` | `should_unmask & is_mask` logic unchanged |
| **v2 per-position alpha** | `sampling.py:235-246` | p_unmask computation unchanged, passed to `_single_step_unmask` |
| **Remasking** | `remasking.py` | Called inside `_single_step_remask`, applied only to selected winner |

### 2.8 Floating-Point Precision

**Rule:** All violation values, rewards, energies, importance weights, and probabilities must be `float64` . Never use integer arithmetic for violations — even "integer-looking" quantities like ExactCount violations (`|count - target|`) must be float tensors to ensure smooth flow through shaping functions, calibration normalization, and the `softmax(reward / α)` importance weight computation.

Specifically:
- `soft_violation()` returns `Tensor` (float), not `int`.
- `hard_violation()` returns `ConstraintResult` with `violation: float`, not `int`.
- `build_effective_probs()` returns float64 probability tensors.
- RewardComposer energy/reward are float64 tensors (soft mode) or Python `float` (hard mode).
- Importance weights `w_k = softmax(rewards / α)` are float64.

---

## 3. Architecture Overview

### 3.1 Guided Generation Pipeline

```
Constraint YAML ──→ Pydantic validation ──→ compile_constraints() ──→ [Constraint, ...]
                                                                          │
                                                                          ▼
                                                                    RewardComposer
                                                                          │
    ┌─────────────────── guided_sample() loop ────────────────────────────┤
    │                                                                     │
    │  For each step i = N-1 down to 0:                                   │
    │    1. model(x_t, pad_mask, t) → node_logits, edge_logits           │
    │    2. Compute p_unmask (v1 scalar or v2 per-position)               │
    │    3. Expand x_t to K*B candidates                                  │
    │    4. _single_step_unmask on K*B (NO remasking)                     │
    │    5. Reshape to (K, B, SEQ)                                        │
    │    6. Score each candidate via RewardComposer  ◄────────────────────┘
    │    7. w_k = softmax(reward / α) per sample
    │    8. Resample: select winner per sample
    │    9. _single_step_remask on winner only
    │    10. Record diagnostics (ESS, rewards, violations)
    │
    └──→ final tokens (B, SEQ_LEN) + GuidanceStats
```

### 3.2 Component Classification

**New modules:**

| Component | File | Description |
|-----------|------|-------------|
| constraints | `bd_gen/guidance/constraints.py` | Constraint ABC + 4 primitives |
| soft_violations | `bd_gen/guidance/soft_violations.py` | `build_effective_probs` utility |
| reward | `bd_gen/guidance/reward.py` | RewardComposer |
| guided_sampler | `bd_gen/guidance/guided_sampler.py` | SVDD reweighting loop |
| constraint_schema | `bd_gen/guidance/constraint_schema.py` | Pydantic models + compilation |
| calibration | `bd_gen/guidance/calibration.py` | P90 normalization utilities |
| generate_guided | `scripts/generate_guided.py` | CLI generation script |
| calibrate_constraints | `scripts/calibrate_constraints.py` | Calibration script |

**Modified modules (backward-compatible):**

| Component | File | Change |
|-----------|------|--------|
| sampling | `bd_gen/diffusion/sampling.py` | Extract `_single_step_unmask()` + `_single_step_remask()` from loop body. `sample()` calls both sequentially — zero behavior change. |

**Unchanged modules:** `bd_gen/model/*`, `bd_gen/data/*`, `bd_gen/eval/*`, `bd_gen/diffusion/noise_schedule.py`, `bd_gen/diffusion/remasking.py`, `bd_gen/diffusion/forward_process.py`, `bd_gen/diffusion/loss.py`, `bd_gen/diffusion/rate_network.py`, `scripts/train.py`, `scripts/train_v2.py`, `scripts/evaluate.py`, `scripts/generate_samples.py`.

---

## 4. Data Type Definitions

```python
from dataclasses import dataclass, field
from typing import Any, TypedDict

from torch import Tensor


@dataclass(frozen=True)
class ConstraintResult:
    """Result of evaluating one constraint on one decoded graph."""
    name: str
    violation: float          # >= 0, 0 = satisfied
    satisfied: bool           # violation == 0
    details: dict[str, Any]   # constraint-specific diagnostic info


class GuidanceStatsStep(TypedDict):
    """Diagnostics for a single denoising step."""
    # Weight distribution
    ess: float                          # effective sample size = 1/sum(w^2), batch-averaged
    max_weight: float                   # max(w) across K candidates, batch-averaged
    weight_entropy: float               # H(w)/log(K) normalized to [0,1]
    # Reward trajectories
    reward_selected: float              # mean reward of selected candidate
    reward_all_candidates: float        # mean reward across all K candidates
    reward_gap: float                   # selected - all_candidates
    # Remasking interaction
    reward_pre_remask: float            # reward of winner before remasking
    reward_post_remask: float           # reward of winner after remasking
    reward_remasking_delta: float       # post - pre
    positions_remasked: float           # mean positions re-masked per sample
    # Per-constraint soft violations
    violations: dict[str, float]        # constraint_name -> batch-averaged soft violation


class GuidanceStatsStepPerSample(TypedDict):
    """Per-sample diagnostics for a single step (shape B)."""
    ess: Tensor                         # (B,)
    reward_selected: Tensor             # (B,)
    reward_all_candidates: Tensor       # (B,)
    reward_pre_remask: Tensor           # (B,)
    reward_post_remask: Tensor          # (B,)
    violations: dict[str, Tensor]       # constraint_name -> (B,)


@dataclass
class GuidanceStats:
    """Complete diagnostics for a guided generation run."""
    # Per-step scalar trajectories (length N_steps)
    steps: list[GuidanceStatsStep] = field(default_factory=list)
    # Per-step per-sample arrays (length N_steps, each entry has shape B)
    steps_per_sample: list[GuidanceStatsStepPerSample] = field(default_factory=list)
    # Final per-constraint hard evaluation (on decoded output)
    final_satisfaction: dict[str, float] = field(default_factory=dict)       # name -> rate
    final_mean_violation: dict[str, float] = field(default_factory=dict)     # name -> mean
    final_mean_violation_when_failed: dict[str, float] = field(default_factory=dict)
    final_violation_histograms: dict[str, dict[str, int]] = field(default_factory=dict)
    satisfaction_overall: float = 0.0   # fraction where ALL constraints satisfied
```

---

## 5. Constraint Primitives Module

**File:** `bd_gen/guidance/constraints.py`

### 5.1 Base Class

```python
class Constraint(ABC):
    name: str
    weight: float = 1.0
    p90_normalizer: float = 1.0  # set by calibration

    @abstractmethod
    def hard_violation(self, graph_dict: dict) -> ConstraintResult

    @abstractmethod
    def soft_violation(
        self,
        node_probs: Tensor,      # (n_max, NODE_VOCAB_SIZE)
        edge_probs: Tensor,      # (n_edges, EDGE_VOCAB_SIZE)
        pad_mask: Tensor,        # (seq_len,) bool
        vocab_config: VocabConfig,
    ) -> Tensor  # scalar, >= 0
```

### 5.2 ExactCount

Require exactly `target` rooms of a given type.

**Hard:** `v = |count(type) - target|`

**Soft:** `v = |n̂ - target|` where `n̂ = Σ_{i: active} q_i(type)`

The expected count `n̂` is the sum of P(room_type) across active (non-PAD) node positions. Active positions are identified via `pad_mask[:n_max]`.

### 5.3 CountRange

Require room count of a given type to be within `[lo, hi]`.

**Hard:** `v = max(0, lo - count) + max(0, count - hi)`

**Soft:** Same formula with `n̂ = Σ_{i: active} q_i(type)`:
- `v = max(0, lo - n̂) + max(0, n̂ - hi)`

### 5.4 RequireAdj

Require at least one adjacency between rooms of `type_a` and `type_b`.

**Hard:** `v = 0` if any edge triple exists between rooms of the required types, else `v = 1`.

**Soft:** For each edge position `(i,j)` where both are active:

```
p_types_ij = q_i(a) * q_j(b) + q_i(b) * q_j(a)    [if a ≠ b]
p_types_ij = q_i(a) * q_j(a)                         [if a = b]

P_adj_ij = Σ_{r=0}^{9} edge_probs[edge_idx, r]      [sum over 10 spatial types]
         = 1 - P(NO_EDGE)                             [since MASK/PAD are 0]

p_ij = p_types_ij * P_adj_ij
```

Then `P(exists) = 1 - Π_{(i,j)} (1 - p_ij)`, accumulated in log-space for numerical stability:

```
log_complement = Σ_{(i,j)} log(1 - p_ij)
P(exists) = 1 - exp(log_complement)
v = 1 - P(exists)
```

The violation is in `[0, 1]`: 0 when the adjacency certainly exists, 1 when it certainly doesn't.

### 5.5 ForbidAdj

Forbid any adjacency between rooms of `type_a` and `type_b`.

**Hard:** `v = count of forbidden adjacency pairs in edge_triples`

**Soft:** `v = Σ_{(i,j)} p_types_ij * P_adj_ij` — the expected count of forbidden adjacencies. Uses the same `p_types_ij` and `P_adj_ij` formulas as RequireAdj. The violation is >= 0, naturally graded (3 forbidden adjacencies is worse than 1).

### 5.6 Helper: P(adjacency at edge position)

```python
def _p_adj_at_edge(edge_probs: Tensor, edge_idx: int) -> Tensor:
    """P(any spatial relationship at edge position), excluding NO_EDGE."""
    return edge_probs[edge_idx, :EDGE_NO_EDGE_IDX].sum()
```

Since the denoiser clamps MASK (idx 11) and PAD (idx 12) logits to `-inf`, softmax gives them 0 probability. So `P(adj) = sum(probs[0:10]) = 1 - P(NO_EDGE at idx 10)`.

---

## 6. Soft Violation Computation

**File:** `bd_gen/guidance/soft_violations.py`

### 6.1 build_effective_probs (Single Sample)

Constructs the effective probability distributions for scoring a single candidate.

```python
def build_effective_probs(
    x_t: Tensor,              # (SEQ_LEN,) current tokens
    node_logits: Tensor,      # (n_max, NODE_VOCAB_SIZE)
    edge_logits: Tensor,      # (n_edges, EDGE_VOCAB_SIZE)
    pad_mask: Tensor,         # (SEQ_LEN,) bool
    vocab_config: VocabConfig,
) -> tuple[Tensor, Tensor]:
    """Build per-position probability distributions.

    For each position:
    - PAD: all zeros
    - Committed (not MASK): one-hot on current token
    - MASK: softmax(logits)

    Returns: (node_probs, edge_probs) matching constraint interface.
    """
```

**Implementation:** Fully vectorized using `torch.where` and `scatter_`:
1. Compute `softmax_probs = softmax(logits, dim=-1)` for all positions.
2. Build `one_hot = F.one_hot(tokens, num_classes=vocab_size).float()`.
3. Create `is_mask` boolean: `x_t[:n_max] == NODE_MASK_IDX` for nodes, `x_t[n_max:] == EDGE_MASK_IDX` for edges.
4. Create `is_pad = ~pad_mask`.
5. `effective = where(is_mask, softmax_probs, one_hot)` then zero out PAD positions.

No Python loops over positions.

### 6.2 build_effective_probs_batch (K*B Candidates)

Batched version for scoring all candidates at once.

```python
def build_effective_probs_batch(
    x_t: Tensor,              # (KB, SEQ_LEN)
    node_logits: Tensor,      # (KB, n_max, NODE_VOCAB_SIZE)
    edge_logits: Tensor,      # (KB, n_edges, EDGE_VOCAB_SIZE)
    pad_mask: Tensor,         # (KB, SEQ_LEN)
    vocab_config: VocabConfig,
) -> tuple[Tensor, Tensor]:
    """Batched build_effective_probs for K*B candidates.

    Note: node_logits/edge_logits are expanded from (B,...) by repeating
    K times, since the model call is shared across candidates. The x_t
    tensors differ across candidates (different stochastic transitions).
    """
```

Same vectorized logic, with an extra leading batch dimension.

### 6.3 Hard Decode for Hard Mode

For `reward_mode="hard"`, we need to decode `x̂_0` from each candidate:

```python
def hard_decode_x0(
    x_t: Tensor,              # (SEQ_LEN,) or (KB, SEQ_LEN)
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
) -> Tensor:
    """Hard-decode x̂_0: committed positions keep token, masked use argmax.

    Returns tensor of same shape as x_t with valid token indices at all
    real positions (no MASK tokens). PAD positions unchanged.
    """
```

The decoded tokens are then detokenized to `graph_dict` and evaluated via `hard_violation()`.

---

## 7. Reward Composition

**File:** `bd_gen/guidance/reward.py`

### 7.1 Interface

```python
class RewardComposer:
    def __init__(
        self,
        constraints: list[Constraint],
        phi: Literal["linear", "quadratic", "log1p"] = "linear",
        reward_mode: Literal["soft", "hard"] = "soft",
    ):
        ...

    def compute_energy_soft(
        self, node_probs, edge_probs, pad_mask, vocab_config,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """E(x) = Σ_i (w_i / p90_i) * φ(v_i(x)). Returns (energy, per_constraint_violations)."""

    def compute_energy_hard(
        self, graph_dict,
    ) -> tuple[float, dict[str, ConstraintResult]]:
        """Same formula on decoded graph. Returns (energy, per_constraint_results)."""

    def compute_reward_soft(...) -> tuple[Tensor, dict[str, Tensor]]:
        """r(x) = -E(x) on posterior distributions."""

    def compute_reward_hard(...) -> tuple[float, dict[str, ConstraintResult]]:
        """r(x) = -E(x) on decoded graph."""

    def load_calibration(self, calibration: dict[str, float]) -> None:
        """Set P90 normalizers from calibration dict."""
```

### 7.2 Energy Formula

```
E(x) = Σ_i (λ_i / p90_i) * φ(v_i(x))
r(x) = -E(x)
w_k ∝ exp(r(x_k) / α)
```

Where:
- `λ_i` = `constraint.weight` (user-specified relative importance)
- `p90_i` = `constraint.p90_normalizer` (from calibration, makes violations comparable)
- `φ` = shaping function (linear by default)
- `v_i` = hard or soft violation
- `α` = guidance temperature (higher = weaker guidance)

### 7.3 Reward Mode Dispatch

In `guided_sampler.py`, the `reward_mode` controls which code path runs:

```python
if composer.reward_mode == "soft":
    node_probs, edge_probs = build_effective_probs(x_t_candidate, ...)
    reward, violations = composer.compute_reward_soft(node_probs, edge_probs, ...)
elif composer.reward_mode == "hard":
    x0_decoded = hard_decode_x0(x_t_candidate, ...)
    graph_dict = detokenize(x0_decoded, pad_mask, vocab_config)
    reward, results = composer.compute_reward_hard(graph_dict)
```

---

## 8. Guided Sampler (SVDD Reweighting)

**File:** `bd_gen/guidance/guided_sampler.py`

### 8.1 Refactoring sampling.py

Extract the loop body (steps 4c-4i in `sample()`) into two functions:

```python
def _single_step_unmask(
    x_t: Tensor,              # (B, SEQ_LEN)
    node_logits: Tensor,      # (B, n_max, NODE_VOCAB_SIZE)
    edge_logits: Tensor,      # (B, n_edges, EDGE_VOCAB_SIZE)
    pad_mask: Tensor,         # (B, SEQ_LEN)
    p_unmask: Tensor,         # (B, 1) or (B, SEQ_LEN)
    i: int,                   # step index
    num_steps: int,
    n_max: int,
    top_p: float | None,
    temperature: float,
    unmasking_mode: str,
    device: torch.device,
    fixed_tokens: Tensor | None,
    fixed_mask: Tensor | None,
) -> Tensor:
    """Steps 4c-4h: token selection + unmasking + PAD clamp + inpainting.
    Model call NOT included. No remasking."""

def _single_step_remask(
    x_t: Tensor,
    remasking_fn: Callable | None,
    t_now: float,
    t_next: float,
    t_switch: float,
    i: int,
    pad_mask: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
) -> Tensor:
    """Step 4i: apply remasking_fn if applicable."""
```

The existing `sample()` loop body becomes:
```python
x_t = _single_step_unmask(x_t, node_logits, edge_logits, ...)
x_t = _single_step_remask(x_t, remasking_fn, ...)
```

**Zero behavior change.** All existing tests must pass.

### 8.2 guided_sample() Interface

```python
@torch.no_grad()
def guided_sample(
    model: torch.nn.Module,
    noise_schedule: NoiseSchedule,
    vocab_config: VocabConfig,
    reward_composer: RewardComposer,
    batch_size: int,
    num_steps: int,
    num_candidates: int = 8,
    guidance_alpha: float = 1.0,
    temperature: float = 0.0,
    top_p: float | None = None,
    unmasking_mode: str = "random",
    t_switch: float = 1.0,
    fixed_tokens: Tensor | None = None,
    fixed_mask: Tensor | None = None,
    remasking_fn: Callable | None = None,
    rate_network: torch.nn.Module | None = None,
    num_rooms_distribution: Tensor | None = None,
    fixed_num_rooms: int | None = None,
    device: str = "cpu",
) -> tuple[Tensor, GuidanceStats]:
    """SVDD-style guided sampling with K-candidate reweighting."""
```

### 8.3 Algorithm (Pseudocode)

```
GUIDED_SAMPLE(model, K, α, ...):

  1. Initialize x_t, pad_mask (same as sample())

  FOR i = N-1 down to 0:
    t_now = (i+1)/N, t_next = i/N

    # --- Model call (shared across K candidates) ---
    node_logits, edge_logits = model(x_t, pad_mask, t_now)
    p_unmask = compute_p_unmask(...)            # v1 or v2 path

    # --- Generate K candidates ---
    x_t_expanded = x_t.repeat_interleave(K, dim=0)           # (K*B, SEQ)
    logits_expanded = node_logits.repeat_interleave(K, dim=0) # (K*B, n_max, V)
    pad_expanded = pad_mask.repeat_interleave(K, dim=0)       # (K*B, SEQ)
    p_expanded = p_unmask.repeat_interleave(K, dim=0)         # (K*B, ...)

    candidates = _single_step_unmask(
        x_t_expanded, logits_expanded, ..., p_expanded, ...)  # (K*B, SEQ)
    candidates = candidates.view(K, B, SEQ)                   # (K, B, SEQ)

    # --- Score candidates ---
    rewards = torch.zeros(K, B)
    per_constraint_violations = {name: torch.zeros(K, B) for name in constraint_names}

    FOR k in range(K):
      FOR b in range(B):       # vectorized in practice
        if reward_mode == "soft":
          node_probs, edge_probs = build_effective_probs(
              candidates[k, b], node_logits[b], edge_logits[b], pad_mask[b], vc)
          r, violations = composer.compute_reward_soft(
              node_probs, edge_probs, pad_mask[b], vc)
        else:  # hard
          x0 = hard_decode_x0(candidates[k, b], node_logits[b], edge_logits[b], pad_mask[b], vc)
          graph = detokenize(x0, pad_mask[b], vc)
          r, results = composer.compute_reward_hard(graph)
        rewards[k, b] = r

    # --- Importance weights ---
    log_weights = rewards / α                         # (K, B)
    weights = softmax(log_weights, dim=0)             # (K, B), sums to 1 per sample

    # --- Resample ---
    selected_k = torch.multinomial(weights.T, 1).squeeze(-1)  # (B,)
    x_t = candidates[selected_k, torch.arange(B)]             # (B, SEQ)

    # --- Record pre-remask reward ---
    reward_pre_remask = rewards[selected_k, torch.arange(B)]  # (B,)

    # --- Apply remasking to winner only ---
    x_t = _single_step_remask(x_t, remasking_fn, t_now, t_next, t_switch, i,
                               pad_mask, node_logits, edge_logits)

    # --- Record post-remask reward ---
    # (recompute reward on x_t after remasking)
    reward_post_remask = ...

    # --- Diagnostics ---
    ess = 1.0 / (weights ** 2).sum(dim=0)                  # (B,)
    max_w = weights.max(dim=0).values                        # (B,)
    w_entropy = -(weights * weights.log().clamp(min=-30)).sum(dim=0) / math.log(K)

    record_step(ess, max_w, w_entropy, rewards, reward_pre_remask,
                reward_post_remask, per_constraint_violations, ...)

  RETURN x_t, guidance_stats
```

### 8.4 Vectorization Notes

The inner `FOR k, FOR b` loop is written for clarity but must be vectorized:

- **Candidate generation:** `repeat_interleave(K, dim=0)` expands `(B,...) → (K*B,...)`. Run `_single_step_unmask` once on the full `K*B` batch.
- **Scoring (soft mode):** `build_effective_probs_batch` operates on `(K*B, ...)` tensors. Constraint `soft_violation` methods are called on reshaped `(K, B, ...)` tensors with batched operations.
- **Scoring (hard mode):** Requires per-sample `detokenize` → Python loop. For K*B=512, this is ~512 detokenize calls. Acceptable for K=8, B=64; may need optimization for larger K.
- **Resampling:** `torch.multinomial` on `weights.T` (shape `(B, K)`).

### 8.5 Compute Cost

| Operation | v1 (unguided) | Guided (K=8) | Overhead |
|-----------|:---:|:---:|---|
| Model call | 1 per step | 1 per step | 0x |
| Stochastic transition | B | K*B = 8B | 8x |
| Reward computation | 0 | K*B = 8B | new |
| Remasking | B | B (winner only) | 0x |
| Total per step | O(B) | O(K*B) | ~8x for non-model ops |

The model call (the expensive part) is shared. Overhead is in the K-fold transition and scoring, which are dominated by tensor ops on the expanded batch.

---

## 9. Constraint Schema & Configuration

**File:** `bd_gen/guidance/constraint_schema.py`

### 9.1 Pydantic Models

```python
class ExactCountSpec(BaseModel):
    type: Literal["ExactCount"]
    name: str
    room_type: str   # validated against NODE_TYPES
    target: int      # >= 0
    weight: float = 1.0

class CountRangeSpec(BaseModel):
    type: Literal["CountRange"]
    name: str
    room_type: str   # validated against NODE_TYPES
    lo: int          # >= 0
    hi: int          # >= 0, >= lo
    weight: float = 1.0

class RequireAdjSpec(BaseModel):
    type: Literal["RequireAdj"]
    name: str
    type_a: str      # validated against NODE_TYPES
    type_b: str      # validated against NODE_TYPES
    weight: float = 1.0

class ForbidAdjSpec(BaseModel):
    type: Literal["ForbidAdj"]
    name: str
    type_a: str
    type_b: str
    weight: float = 1.0

ConstraintSpec = Union[ExactCountSpec, CountRangeSpec, RequireAdjSpec, ForbidAdjSpec]

class GuidanceConfig(BaseModel):
    constraints: list[ConstraintSpec]
    num_candidates: int = 8          # K
    alpha: float = 1.0               # guidance temperature
    phi: Literal["linear", "quadratic", "log1p"] = "linear"
    reward_mode: Literal["soft", "hard"] = "soft"
    calibration_file: str | None = None
```

### 9.2 Compilation

```python
def compile_constraints(config: GuidanceConfig) -> list[Constraint]:
    """Convert specs to executable Constraint objects.
    Maps room type strings to indices via NODE_TYPES."""

def load_guidance_config(path: str | Path) -> GuidanceConfig:
    """Load from .json or .yaml file. Validates against Pydantic schema."""
```

### 9.3 Example YAML

```yaml
# configs/guidance/example_basic.yaml
constraints:
  - type: ExactCount
    name: one_kitchen
    room_type: Kitchen
    target: 1
  - type: ExactCount
    name: one_living
    room_type: LivingRoom
    target: 1
  - type: RequireAdj
    name: kitchen_near_living
    type_a: Kitchen
    type_b: LivingRoom
  - type: ForbidAdj
    name: no_bath_kitchen
    type_a: Bathroom
    type_b: Kitchen
num_candidates: 8
alpha: 1.0
phi: linear
reward_mode: soft
```

---

## 10. Calibration Protocol

**File:** `bd_gen/guidance/calibration.py`

### 10.1 Purpose

Different constraints produce violations at different scales: ExactCount may range 0-7, while RequireAdj is always in {0, 1}. Without normalization, high-scale constraints dominate the energy and SVDD's importance weights are insensitive to low-scale constraints.

Calibration normalizes all violations to comparable scales by dividing each by its 90th percentile on unguided samples.

### 10.2 Protocol

1. Load unguided samples from existing `*_samples.pt` files.
2. Detokenize all samples to graph dicts.
3. For each constraint, compute `hard_violation()` on all samples.
4. For each constraint: `P90 = 90th_percentile(violations[violations > 0])`.
   - If all violations are 0 (constraint always satisfied in unguided generation), set `P90 = 1.0` (no normalization needed).
5. Save calibration to JSON: `{constraint_name: p90_value}`.
6. `RewardComposer.load_calibration()` sets each `constraint.p90_normalizer`.

### 10.3 Interface

```python
def calibrate_from_samples(
    graph_dicts: list[dict],
    constraints: list[Constraint],
) -> dict[str, float]:
    """Compute P90 normalizers. Returns {name: p90}."""

def save_calibration(path: Path, calibration: dict[str, float]) -> None:
def load_calibration(path: Path) -> dict[str, float]:
```

### 10.4 Script

```bash
python scripts/calibrate_constraints.py \
    --schedule loglinear_noise_sc \
    --model llada_topp0.9_no_remask \
    --constraints configs/guidance/example_basic.yaml \
    --output configs/guidance/calibration_basic.json
```

---

## 11. Guidance Diagnostics & Evaluation

### 11.1 Per-Step Trajectory Tracking

`guided_sample()` returns a `GuidanceStats` object with two storage tiers:

- **Scalars:** batch-averaged, one value per step — for quick analysis and cross-run comparison.
- **Per-sample arrays:** shape `(N_steps, B)` — for debugging individual sample trajectories. Stored with `_per_sample` suffix.

**A) Weight distribution diagnostics (per step):**

| Metric | What it reveals |
|--------|-----------------|
| `ess[t]` | Effective sample size = `1/Σ(w²)`. Healthy: close to K early (constraints uninformative on masked tokens), dropping mid-denoising (guidance bites), recovering slightly at end (candidates converge). Pathological: collapses to 1 (α too low) or stays at K (guidance has no effect). |
| `max_weight[t]` | Max weight across K candidates. If ~1.0, one candidate dominates — no diversity. |
| `weight_entropy[t]` | `H(w)/log(K)` normalized to [0,1]. 1.0 = uniform, 0.0 = degenerate. Complements ESS: entropy captures the full weight shape. |

**B) Reward and violation trajectories (per step):**

| Metric | What it reveals |
|--------|-----------------|
| `reward_selected[t]` | Mean reward of the selected (winning) candidate. Actual reward trajectory. |
| `reward_all_candidates[t]` | Mean reward across all K candidates. Baseline without guidance. |
| `reward_gap[t]` | `selected - all_candidates`. Measures how much guidance biases selection. Large gap = guidance is steering. Zero = candidates score identically. |
| `violation_{name}[t]` | Per-constraint mean soft violation. Reveals when each constraint resolves and which is the bottleneck. |

**C) Remasking-guidance interaction (per step):**

| Metric | What it reveals |
|--------|-----------------|
| `reward_pre_remask[t]` | Reward of winner right after unmasking, before remasking. |
| `reward_post_remask[t]` | Reward of same winner after remasking. |
| `reward_remasking_delta[t]` | `post - pre`. Persistently negative = remasking fights guidance. Near zero = compatible. |
| `positions_remasked[t]` | Mean positions re-masked per sample. Context for interpreting the delta. |

### 11.2 Final Per-Constraint Statistics (on decoded output)

| Metric | Description |
|--------|-------------|
| `satisfaction_{name}` | Fraction of samples where `hard_violation == 0` |
| `satisfaction_overall` | Fraction where ALL constraints simultaneously satisfied |
| `mean_violation_{name}` | Mean hard violation (over all samples, including satisfied) |
| `mean_violation_when_failed_{name}` | Mean hard violation conditioned on `violation > 0` |
| `violation_histogram_{name}` | Histogram of hard violation values (bins: 0, 1, 2, 3+) |

### 11.3 Soft vs Hard Reward Mode Comparison

The first experiment (fixed α=1.0, K=8) runs each model variant with both reward modes to empirically compare:

| Comparison dimension | Expected difference |
|---|---|
| Reward variance across K candidates at same step | Soft should be lower (smoother) |
| ESS trajectories | Soft should be more stable (fewer extreme weight ratios) |
| Final constraint satisfaction | Primary outcome — which mode satisfies more constraints? |
| Reward noise early in denoising | Hard may produce misleading signals when most positions are masked |
| Computational cost | Hard requires detokenization per candidate; soft uses tensor ops |

Findings are documented. One mode is selected for remaining experiments.

### 11.4 Pathology Detection

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| ESS < 2 at any step | α too aggressive | Increase α |
| ESS = K throughout | Guidance has no effect | Decrease α, check constraint violations are nonzero |
| max_weight ~1.0 | Single candidate dominates | Increase α |
| reward_gap = 0 | All candidates score identically | Check soft violations are informative, increase K |
| reward_remasking_delta persistently negative | Remasking fights guidance | Consider disabling remasking under guidance |

### 11.5 Storage

All trajectories and final stats are saved in the `.pt` file under a `guidance_stats` key per `(α, K, constraint_set, model_variant)` configuration. This enables overlaying curves across configurations.

---

## 12. Scripts & CLI

### 12.1 generate_guided.py

```bash
python scripts/generate_guided.py \
    --checkpoint checkpoints/best.pt \
    --guidance-config configs/guidance/example_basic.yaml \
    --schedule loglinear_noise_sc \
    --num-samples 1000 \
    --seeds 42 123 456 789 2024 \
    --device cuda
```

Follows the same pattern as `generate_samples.py`:
- Load model, noise schedule, rate network (if v2)
- Load guidance config, compile constraints, build RewardComposer
- Load calibration if specified
- Call `guided_sample()` in batches
- Save results with method name: `{unmasking}_{sampling}_guided_{tag}_K{K}_a{alpha}`
- Save `GuidanceStats` alongside samples

### 12.2 calibrate_constraints.py

See Section 10.4.

---

## 13. Implementation Phases

### Phase G1: Constraint Primitives + Hard Violations

**Goal:** Build the constraint library, reward composer (hard mode only), and constraint configuration system. After this phase, we can evaluate hard violations on any decoded graph — both hand-crafted test graphs and existing unguided samples.

**Files created:**

| File | Purpose |
|------|---------|
| `bd_gen/guidance/__init__.py` | Package init, re-exports `Constraint`, `RewardComposer`, `compile_constraints` |
| `bd_gen/guidance/constraints.py` | `Constraint` ABC + 4 concrete primitives (ExactCount, CountRange, RequireAdj, ForbidAdj) |
| `bd_gen/guidance/reward.py` | `RewardComposer` — combines constraint violations into a single energy/reward score |
| `bd_gen/guidance/constraint_schema.py` | Pydantic models for JSON/YAML constraint DSL + `compile_constraints()` + `load_guidance_config()` |
| `tests/test_constraints.py` | Hard violation tests for all 4 primitives (tests 1–11 from Section 14.1) |
| `tests/test_constraint_schema.py` | Schema validation tests (tests 33–39 from Section 14.5) |

**What gets built:**

1. **Constraint ABC** (Section 5.1): Base class with `hard_violation(graph_dict) → ConstraintResult` and `soft_violation(node_probs, edge_probs, ...) → Tensor` abstract methods. The `soft_violation()` methods are declared but raise `NotImplementedError` — they get implemented in G2. Each constraint also carries `weight` (user-specified importance) and `p90_normalizer` (set later by calibration).

2. **Four constraint primitives** (Sections 5.2–5.5): Only `hard_violation()` is implemented in this phase. Each primitive operates on a decoded `graph_dict` (room types + edge triples) and returns a `ConstraintResult` with a graded violation magnitude:
   - **ExactCount**: `v = |count(type) - target|` — counts rooms of a specific type.
   - **CountRange**: `v = max(0, lo - count) + max(0, count - hi)` — range constraint on room count.
   - **RequireAdj**: `v = 0` if any adjacency exists between the required room types, else `v = 1`.
   - **ForbidAdj**: `v = count of forbidden adjacency pairs` — graded, 2 violations worse than 1.

3. **RewardComposer** (Section 7): Combines multiple constraint violations into a single energy score: `E(x) = Σ_i (λ_i / p90_i) * φ(v_i)` where `φ` is a shaping function (linear/quadratic/log1p). Reward is `r = -E`. In this phase, only `compute_energy_hard()` and `compute_reward_hard()` work (they call `hard_violation()` on decoded graphs). The soft-mode methods exist but delegate to G2's `soft_violation()`.

4. **Pydantic schema** (Section 9): Validates constraint configs from YAML/JSON. Room type names (e.g., "Kitchen") are validated against `NODE_TYPES` from `vocab.py`. `compile_constraints()` converts spec objects into executable `Constraint` instances with correct room type indices.

**Verification:**
- All 11 hard violation tests pass (satisfied, violated, correct magnitude for each primitive).
- Schema validation tests pass (valid specs parse, invalid room types / negative targets / lo > hi rejected).
- Run hard violations on existing unguided samples as a sanity check: verify violation distributions look reasonable (e.g., ExactCount(Kitchen=1) violations are not all 0 or all 7).
- All pre-existing project tests still pass.

**Tests:** 1–11 (Section 14.1), 26–32 (Section 14.4, hard mode subset), 33–39 (Section 14.5).

---

### Phase G2: Soft Violations from Logits

**Goal:** Enable scoring constraint violations on partially-masked sequences using the full posterior distribution (softmax of logits), not just decoded graphs. After this phase, we can compute smooth, differentiable violation scores at any point during denoising — the foundation for SVDD guidance.

**Files created:**

| File | Purpose |
|------|---------|
| `bd_gen/guidance/soft_violations.py` | `build_effective_probs()`, batched version, `hard_decode_x0()` |
| `tests/test_soft_violations.py` | Soft violation + effective probs tests (tests 12–25 from Section 14.2–14.3) |

**Files modified:**

| File | Change |
|------|--------|
| `bd_gen/guidance/constraints.py` | Fill in `soft_violation()` for all 4 primitives (replacing `NotImplementedError` stubs from G1) |
| `bd_gen/guidance/reward.py` | Enable `compute_energy_soft()` / `compute_reward_soft()` methods |

**What gets built:**

1. **`build_effective_probs()`** (Section 6.1): Given the current token sequence `x_t`, model logits, and a pad mask, constructs per-position probability distributions:
   - **PAD positions** → all-zero probabilities (excluded from scoring).
   - **Committed positions** (already unmasked, not MASK) → one-hot on the current token.
   - **MASK positions** (still masked) → `softmax(logits)` from the model.

   Returns `(node_probs, edge_probs)` tensors that the constraint `soft_violation()` methods consume. Implementation is fully vectorized using `torch.where` and `scatter_` — no Python loops over positions.

2. **`build_effective_probs_batch()`** (Section 6.2): Same as above but for `(K*B, SEQ_LEN)` tensors — used in G3 when scoring K candidates simultaneously. The logits are shared across K candidates (expanded via `repeat_interleave`), but the `x_t` tokens differ (different stochastic transitions).

3. **`hard_decode_x0()`** (Section 6.3): For hard reward mode — argmax-decodes masked positions to get `x̂_0`, which is then detokenized to a graph_dict and scored via `hard_violation()`. Committed positions keep their tokens; masked positions use `argmax(logits)`.

4. **Soft violation implementations** (Sections 5.2–5.5): Each constraint's `soft_violation()` method is filled in:
   - **ExactCount soft**: `v = |n̂ - target|` where `n̂ = Σ_i q_i(type)` is the expected count from node probabilities.
   - **CountRange soft**: `v = max(0, lo - n̂) + max(0, n̂ - hi)` — same formula, expected count.
   - **RequireAdj soft**: Computes per-edge-position probability `p_ij = p_types_ij * P_adj_ij`, then `v = 1 - P(exists)` where `P(exists) = 1 - Π(1 - p_ij)` accumulated in log-space for numerical stability.
   - **ForbidAdj soft**: `v = Σ p_types_ij * P_adj_ij` — expected count of forbidden adjacencies.

   All soft violations are smooth: small logit perturbations → small violation changes. They converge exactly to hard violations when all positions are committed (one-hot distributions = deterministic tokens).

5. **RewardComposer soft mode** (Section 7.3): `compute_reward_soft()` calls `build_effective_probs()` then evaluates each constraint's `soft_violation()`, applies shaping and calibration normalization.

**Verification:**
- Soft-converges-to-hard tests: for each primitive, set all positions to one-hot (committed) probabilities and verify `soft_violation == hard_violation` exactly.
- Smoothness tests: perturb logits by `ε = 0.01` and verify violation changes by `O(ε)`.
- Range tests: RequireAdj soft violation always in `[0, 1]`, ForbidAdj always `>= 0`.
- All-masked test: when all positions are MASK (early denoising), violations are finite and non-NaN.
- PAD exclusion test: zeroing PAD positions doesn't change violations.
- Batch consistency: batched version matches per-sample version.
- All pre-existing project tests still pass.

**Tests:** 12–25 (Sections 14.2–14.3), 26–32 (Section 14.4, now including soft mode).

**Depends on:** G1

---

### Phase G3: Guided Sampler

**Goal:** Implement the SVDD K-candidate reweighting loop and the generation script. This is the core guidance mechanism: at each denoising step, generate K candidate transitions, score them via the reward composer, and resample toward constraint satisfaction. After this phase, we can generate constrained samples.

**Files modified:**

| File | Change |
|------|--------|
| `bd_gen/diffusion/sampling.py` | Extract the loop body into `_single_step_unmask()` and `_single_step_remask()`. The existing `sample()` calls both sequentially — **zero behavior change**. |

**Files created:**

| File | Purpose |
|------|---------|
| `bd_gen/guidance/guided_sampler.py` | `guided_sample()` function — the SVDD reweighting loop |
| `scripts/generate_guided.py` | CLI script for guided generation (follows `generate_samples.py` pattern) |
| `configs/guidance/default.yaml` | Default guidance config (K=8, α=1.0, linear shaping, soft mode) |
| `configs/guidance/example_basic.yaml` | Example: Kitchen=1, LivingRoom=1, Kitchen-adj-LivingRoom, no Bathroom-Kitchen |
| `tests/test_guided_sampler.py` | Integration tests (tests 40–49 from Section 14.6) |

**What gets built:**

1. **Refactoring `sampling.py`** (Section 8.1): The loop body of `sample()` (currently ~90 lines doing token selection, unmasking, PAD clamping, inpainting, and remasking) is split into two helper functions:

   - **`_single_step_unmask(x_t, node_logits, edge_logits, pad_mask, p_unmask, ...)`**: Performs steps 4c–4h of the sampling loop — token selection (argmax or top-p sampling), unmasking (random or LLaDA confidence-based position selection), PAD clamping, and inpainting. The model call is NOT included — logits are passed in. No remasking.

   - **`_single_step_remask(x_t, remasking_fn, t_now, t_next, t_switch, ...)`**: Performs step 4i — applies the remasking function if applicable. No-op if `remasking_fn` is None.

   **Why two functions (not one)?** SVDD guidance must score K candidates *after unmasking but before remasking*. If remasking runs before scoring, it re-masks low-confidence positions across all K candidates — positions where guidance may have steered toward constraint satisfaction. Since all K candidates share the same logits (single model call), re-masked positions produce identical distributions, destroying the K-way differentiation that ESS depends on. By scoring post-unmask and re-masking only the winner, ESS reflects pure guidance effectiveness.

   The existing `sample()` loop body becomes: `x_t = _single_step_unmask(...)` then `x_t = _single_step_remask(...)`. All existing sampling tests must pass unchanged — this is a pure refactoring.

2. **`guided_sample()` function** (Sections 8.2–8.3): The SVDD reweighting loop. For each denoising step:

   ```
   a. Single model call (shared): node_logits, edge_logits = model(x_t, pad_mask, t)
   b. Compute p_unmask (v1 scalar or v2 per-position rates)
   c. Expand x_t from (B, SEQ) to (K*B, SEQ) via repeat_interleave
   d. Expand logits, p_unmask, pad_mask similarly
   e. Run _single_step_unmask on expanded batch → K*B candidate transitions
   f. Reshape to (K, B, SEQ)
   g. Score each candidate: build_effective_probs → compute_reward (soft or hard mode)
   h. Importance weights: w_k = softmax(reward / α, dim=0) per sample over K
   i. Resample: for each sample in batch, select one winner via multinomial(weights)
   j. Record pre-remask reward of winner
   k. Apply _single_step_remask ONLY to the selected winner (B samples, not K*B)
   l. Record post-remask reward of winner
   m. Record full diagnostics: ESS, max_weight, weight_entropy, reward trajectories,
      per-constraint violations, remasking delta, positions_remasked
   ```

   The function returns `(final_tokens, GuidanceStats)` where `GuidanceStats` contains per-step scalar trajectories (batch-averaged) and per-step per-sample arrays for deep debugging.

3. **`scripts/generate_guided.py`** (Section 12.1): Follows the pattern of `generate_samples.py`:
   - Load model checkpoint, noise schedule, rate network (if v2).
   - Load constraint config from YAML, compile constraints, build RewardComposer.
   - Load calibration file if specified in config.
   - Call `guided_sample()` in batches across multiple seeds.
   - Save results with method name `{unmasking}_{sampling}_guided_{tag}_K{K}_a{alpha}`.
   - Save `GuidanceStats` alongside samples in the `.pt` file under `guidance_stats` key.

4. **Config files**: `default.yaml` (K=8, α=1.0, linear, soft) and `example_basic.yaml` (the 4-constraint example from Section 9.3).

**Verification:**
- **Refactoring regression test**: With the same seed, `sample()` produces bit-identical output before and after the refactoring (test 50).
- **All existing sampling tests pass** (test 51) — this is the critical non-regression check.
- **K=1 no constraints = unguided**: With K=1 and empty constraint list, `guided_sample()` produces identical output to `sample()` for the same seed (test 40). This validates the plumbing.
- **PAD invariant**: PAD positions unchanged (test 41).
- **No remaining MASKs**: No MASK tokens in final output (test 42).
- **Constraint effectiveness**: Guided ExactCount(Kitchen=1) yields higher kitchen-count satisfaction rate than unguided (test 44).
- **ESS sanity**: ESS > 1.0 at all steps with α=1.0, K=8 (test 45) — guidance is active but not degenerate.
- **GuidanceStats completeness**: `len(stats.steps) == num_steps` (test 46).
- **v1 and v2 compatibility**: Works with `rate_network=None` (v1) and with a rate network (v2) (tests 47–48).
- **Remasking compatibility**: Works with `remasking_fn`, `reward_remasking_delta` is recorded (test 49).

**Tests:** 40–53 (Sections 14.6–14.7).

**Depends on:** G1, G2

---

### Phase G4: Calibration + Evaluation

**Goal:** Build the calibration pipeline (P90 normalization) and extend evaluation to report constraint satisfaction metrics. After this phase, constraints are on comparable scales and we can systematically compare guided vs unguided generation.

**Files created:**

| File | Purpose |
|------|---------|
| `bd_gen/guidance/calibration.py` | `calibrate_from_samples()`, `save_calibration()`, `load_calibration()` |
| `scripts/calibrate_constraints.py` | CLI script to run calibration on unguided samples |

**Files modified:**

| File | Change |
|------|--------|
| `scripts/evaluate.py` | Add constraint satisfaction metrics to `compute_all_metrics()` when a `--guidance-config` is provided. New metrics: `satisfaction_{name}`, `satisfaction_overall`, `mean_violation_{name}`, `mean_violation_when_failed_{name}`, `violation_histogram_{name}`. |

**What gets built:**

1. **Calibration protocol** (Section 10): Different constraints produce violations at different scales — ExactCount ranges 0–7 while RequireAdj is always in {0, 1}. Without normalization, high-scale constraints dominate the energy and SVDD's weights are insensitive to low-scale ones.

   The calibration pipeline:
   - Load unguided samples from existing `*_samples.pt` files.
   - Detokenize all samples to graph dicts.
   - For each constraint, compute `hard_violation()` on all samples.
   - For each constraint: `P90 = 90th_percentile(violations[violations > 0])`. If all violations are 0 (constraint always satisfied), set P90 = 1.0 (no normalization needed).
   - Save normalizers to JSON: `{constraint_name: p90_value}`.
   - `RewardComposer.load_calibration()` sets each `constraint.p90_normalizer`.

2. **Calibration script** (Section 10.4): `scripts/calibrate_constraints.py` — loads specified model's unguided samples, constraint config, runs the calibration, and saves the JSON.

3. **Evaluation extension**: When `evaluate.py` receives a `--guidance-config` flag, it additionally computes per-constraint satisfaction metrics on the decoded samples (Section 11.2):
   - `satisfaction_{name}`: fraction of samples where `hard_violation == 0`.
   - `satisfaction_overall`: fraction where ALL constraints are simultaneously satisfied.
   - `mean_violation_{name}`: mean hard violation across all samples.
   - `mean_violation_when_failed_{name}`: mean violation conditioned on failure.
   - `violation_histogram_{name}`: distribution of violation values (bins: 0, 1, 2, 3+).

   These metrics appear alongside the standard generation metrics (edge_tv, inside_validity, diversity, etc.) in the comparison tables, enabling apples-to-apples comparison of guided vs unguided variants.

**Verification:**
- P90 computation on known distribution gives correct value (test 54).
- All-zero violations → P90 = 1.0 (test 55).
- Save/load roundtrip produces identical dict (test 56).
- Run calibration on actual unguided samples and verify normalizers are reasonable (not 0, not enormous).
- Evaluate existing unguided samples with constraint metrics — verify they report non-trivial violation rates.
- All pre-existing project tests still pass.

**Tests:** 54–56 (Section 14.8).

**Depends on:** G1 (only needs hard violations on decoded graphs — can run in parallel with G2/G3)

---

### Phase G5: End-to-End Integration + Tuning

**Goal:** Run the full guided generation pipeline end-to-end, compare soft vs hard reward modes, sweep hyperparameters (α, K), and produce systematic comparison tables. This is experiments only — no new code.

**No new files. No code changes.**

**Steps:**

1. **Calibration**: Run `calibrate_constraints.py` on unguided v1 and v2 samples with the `example_basic.yaml` constraints. Save P90 normalizers.

2. **Reward mode comparison** (first experiment, fixed α=1.0, K=8): Run each model variant with both `reward_mode="soft"` and `reward_mode="hard"`. Compare across 5 dimensions using the trajectory diagnostics from Section 11:
   - Reward variance across K candidates at the same step (soft should be lower).
   - ESS trajectories (soft should be more stable — fewer extreme weight ratios).
   - Final constraint satisfaction rates (the actual goal — which mode wins?).
   - Reward noise early in denoising (hard may produce misleading signals when most positions are masked).
   - Computational cost (hard requires detokenization per candidate; soft uses tensor ops).

   Document findings → select one mode for remaining experiments.

3. **Generate guided samples** across model variants:
   - v1 + llada + top-p=0.9 + no remasking
   - v1 + llada + top-p=0.9 + confidence remasking tsw=1.0
   - v2 + llada + top-p=0.9 + no remasking
   - v2 + llada + top-p=0.9 + confidence remasking tsw=1.0

4. **α grid search**: [0.1, 0.5, 1.0, 2.0, 5.0] — identifies the guidance temperature range where constraints are enforced without destroying sample quality/diversity.

5. **K grid**: [4, 8, 16] — determines the number of candidates needed for reliable guidance.

6. **Full evaluation**: Run `evaluate.py` with `--guidance-config` on all guided variants. Produces comparison tables with both standard generation metrics AND constraint satisfaction columns.

7. **Generate comparison tables** with soft/hard comparison writeup.

**What to monitor** (using trajectory diagnostics from Section 11):

- **Guidance effectiveness:**
  - ESS(t) curves overlaid across α values — identify the α range where guidance is active but not degenerate.
  - Per-constraint `violation_{name}(t)` — when does each constraint resolve? Which is the bottleneck?
  - `reward_gap(t)` — is guidance actually steering, or are all K candidates scoring similarly?
  - `weight_entropy(t)` — complement to ESS, captures full weight distribution shape.

- **Quality preservation:**
  - Inside validity, edge TV, diversity — compared to unguided baseline (any degradation?).
  - Per-constraint satisfaction rates and violation histograms — bimodal (all-or-nothing) vs gradual?
  - `mean_violation_when_failed` — when guidance fails on a constraint, how badly?

- **Pathology detection** (Section 11.4):
  - ESS < 2 at any step → α too aggressive, increase it.
  - ESS = K throughout → guidance has no effect, decrease α or check violations.
  - max_weight ~1.0 → single candidate dominates, diversity destroyed.
  - reward_gap = 0 → all candidates score identically, soft violations may be uninformative.
  - `reward_remasking_delta` persistently negative → remasking fights guidance, consider disabling remasking under guidance.

**Depends on:** G1, G2, G3, G4

---

### Dependency Graph

```
G1 ──→ G2 ──→ G3 ──→ G5
 │                  ↗
 └──→ G4 ──────────┘
```

G4 (calibration) can run in parallel with G2/G3 since it only needs G1 (hard violations on decoded graphs).

### Subagent Parallelization Strategy

During implementation, independent work items within each phase and across parallel phases can be dispatched to separate subagents. This section maps out what can run concurrently.

**Within G1 (3 parallel tracks):**

| Track | Subagent task | Output |
|-------|--------------|--------|
| G1-a | Implement `Constraint` ABC + 4 primitives (`constraints.py`) with `hard_violation()` only. `soft_violation()` raises `NotImplementedError`. | `constraints.py` |
| G1-b | Implement `RewardComposer` (`reward.py`) with hard-mode energy/reward. Soft-mode methods stub out pending G2. | `reward.py` |
| G1-c | Implement Pydantic schema (`constraint_schema.py`) + `compile_constraints()` + `load_guidance_config()`. | `constraint_schema.py` |

G1-b depends on G1-a's `Constraint` interface (but only the ABC signature, not the implementations). G1-c depends on G1-a's concrete class names. In practice, define the ABC first (small task), then launch all three.

**Tests after G1 merge:** A single subagent writes `test_constraints.py` + `test_constraint_schema.py` and runs all tests.

**G2 + G4 in parallel (after G1):**

| Subagent | Task | Dependencies |
|----------|------|-------------|
| G2 agent | `soft_violations.py` + fill in `soft_violation()` for all 4 primitives + enable RewardComposer soft mode + `test_soft_violations.py` | G1 complete |
| G4 agent | `calibration.py` + `calibrate_constraints.py` + evaluate.py extension | G1 complete (only needs `hard_violation()`) |

These are fully independent — G4 uses hard violations on decoded graphs while G2 works on soft violations from logits.

**Within G3 (2 sequential steps, then parallel):**

| Step | Task | Notes |
|------|------|-------|
| G3-step1 | Refactor `sampling.py`: extract `_single_step_unmask` + `_single_step_remask`. Run all existing sampling tests. | Must be done first — regression-critical. |
| G3-step2a | Implement `guided_sampler.py` with `guided_sample()` function. | Depends on G3-step1 + G2 |
| G3-step2b | Implement `generate_guided.py` script + config YAML files. | Can start from template while G3-step2a is in progress, finalize after. |

**G5 parallelization (experiments):**

Each model variant x hyperparameter combination is an independent generation + evaluation run. With 4 model variants x 5 alpha values x 3 K values = 60 configurations, these can be batched across GPU runs. The soft-vs-hard comparison (step 2) should complete first since it determines which reward mode is used for the remaining grid.

---

## 14. Test Cases

### 14.1 Constraint Hard Violation Tests

| # | Test | Assertion |
|---|------|-----------|
| 1 | ExactCount satisfied | Graph with 1 kitchen → violation = 0, satisfied = True |
| 2 | ExactCount over | Graph with 3 kitchens, target=1 → violation = 2 |
| 3 | ExactCount under | Graph with 0 kitchens, target=1 → violation = 1 |
| 4 | CountRange in range | Graph with 2 bedrooms, [1,4] → violation = 0 |
| 5 | CountRange under | Graph with 0 bedrooms, [1,4] → violation = 1 |
| 6 | CountRange over | Graph with 5 bedrooms, [1,4] → violation = 1 |
| 7 | RequireAdj satisfied | Graph with Kitchen-LivingRoom edge → violation = 0 |
| 8 | RequireAdj missing | Graph with Kitchen and LivingRoom but no edge → violation = 1 |
| 9 | ForbidAdj satisfied | No Bathroom-Kitchen edges → violation = 0 |
| 10 | ForbidAdj one pair | One Bathroom-Kitchen edge → violation = 1 |
| 11 | ForbidAdj multiple | Two Bathroom-Kitchen edges → violation = 2 |

### 14.2 Constraint Soft Violation Tests

| # | Test | Assertion |
|---|------|-----------|
| 12 | ExactCount soft converges | All one-hot probs → soft violation == hard violation |
| 13 | ExactCount soft smooth | Perturb logits by ε → violation changes by O(ε) |
| 14 | CountRange soft converges | All one-hot probs → soft == hard |
| 15 | RequireAdj soft converges | All one-hot probs → soft == hard |
| 16 | RequireAdj soft range | With probabilities → violation in [0, 1] |
| 17 | ForbidAdj soft converges | All one-hot probs → soft == hard |
| 18 | ForbidAdj soft non-negative | With any probabilities → violation >= 0 |
| 19 | Soft violation all masked | When all positions are MASK → violations are sensible (not NaN, not all-zero for non-trivial constraints) |
| 20 | PAD positions excluded | Zeroing PAD probs doesn't change violations |

### 14.3 build_effective_probs Tests

| # | Test | Assertion |
|---|------|-----------|
| 21 | Committed position | Non-MASK token → one-hot probability vector |
| 22 | Masked position | MASK token → softmax(logits) probability vector |
| 23 | PAD position | PAD token → all-zero probability vector |
| 24 | Sum to 1 | Active non-PAD positions sum to 1.0 (or 0 for PAD) |
| 25 | Batch consistency | Batched version matches per-sample version |

### 14.4 RewardComposer Tests

| # | Test | Assertion |
|---|------|-----------|
| 26 | Energy non-negative | E(x) >= 0 for any input |
| 27 | Reward non-positive | r(x) <= 0 for any input |
| 28 | Perfect graph | All constraints satisfied → E(x) = 0, r(x) = 0 |
| 29 | Phi quadratic | φ(2) = 4 for quadratic shaping |
| 30 | Phi log1p | φ(2) = log(3) for log1p shaping |
| 31 | Calibration loading | After load_calibration(), p90_normalizers updated |
| 32 | Calibration effect | With P90=2, violation=4 → normalized contribution = 4/2 = 2 |

### 14.5 Schema Validation Tests

| # | Test | Assertion |
|---|------|-----------|
| 33 | Valid ExactCount spec | Parses without error |
| 34 | Invalid room type | `room_type="InvalidRoom"` → ValidationError |
| 35 | Unknown constraint type | `type="MaxDistance"` → ValidationError |
| 36 | Negative target | `target=-1` → ValidationError |
| 37 | lo > hi | `lo=5, hi=3` → ValueError in CountRange constructor |
| 38 | Compilation roundtrip | Spec → compile → Constraint object with correct room type index |
| 39 | YAML load | Load example_basic.yaml → valid GuidanceConfig |

### 14.6 Guided Sampler Tests

| # | Test | Assertion |
|---|------|-----------|
| 40 | K=1 no constraints matches unguided | Same seed → identical output tokens |
| 41 | PAD invariant | PAD positions contain NODE_PAD_IDX / EDGE_PAD_IDX |
| 42 | No MASK tokens | No NODE_MASK_IDX or EDGE_MASK_IDX in real positions |
| 43 | Output shapes | `(B, SEQ_LEN)` long tensor |
| 44 | Constraint improves satisfaction | Guided ExactCount(Kitchen=1) yields higher satisfaction than unguided |
| 45 | ESS not degenerate | ESS > 1.0 at all steps with α=1.0, K=8 |
| 46 | GuidanceStats length | `len(stats.steps) == num_steps` |
| 47 | Works with v1 | `rate_network=None` → runs without error |
| 48 | Works with v2 | `rate_network=model` → runs without error |
| 49 | Works with remasking | `remasking_fn=RemaskingSchedule(...)` → runs, remasking_delta recorded |

### 14.7 Sampling Refactoring Tests

| # | Test | Assertion |
|---|------|-----------|
| 50 | sample() unchanged | After refactoring, `sample()` produces identical output for same seed |
| 51 | All existing sampling tests pass | No regressions |
| 52 | _single_step_unmask returns valid tokens | No MASK in unmasked positions, PAD correct |
| 53 | _single_step_remask no-op when remasking_fn=None | Output == input |

### 14.8 Calibration Tests

| # | Test | Assertion |
|---|------|-----------|
| 54 | P90 computation | Known violation distribution → correct P90 |
| 55 | All-zero violations | P90 = 1.0 (default, no normalization needed) |
| 56 | Save/load roundtrip | Save calibration JSON, reload → identical dict |
