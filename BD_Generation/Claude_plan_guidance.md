# Guidance Implementation Plan for Discrete Masked Diffusion

## Context

The bubble diagram generation model (discrete masked diffusion, 36-position flat sequences: 8 nodes + 28 edges) can generate unconstrained samples but lacks any mechanism to enforce architectural constraints (room counts, adjacency requirements, forbidden adjacencies). Guidance is the most important feature of the entire model. This plan implements SVDD-style inference-time guidance with soft violations computed from model logits, enabling constraint-driven generation without retraining.

**Decisions confirmed:**
- Method: SVDD K-candidate reweighting (K=8 default)
- Model variants: Both v1 (fixed schedule) and v2 (learned per-position)
- Initial constraints: ExactCount, CountRange, RequireAdj, ForbidAdj (4 of 7 primitives)
- UX: YAML/JSON config files with Pydantic validation

---

## Phase G1: Constraint Primitives + Hard Violations

**Goal**: Build and test the constraint library with decoded-graph evaluation.

### New files

| File | Purpose |
|------|---------|
| `bd_gen/guidance/__init__.py` | Package init, re-export key classes |
| `bd_gen/guidance/constraints.py` | `Constraint` ABC + 4 concrete primitives |
| `bd_gen/guidance/reward.py` | `RewardComposer` — combines constraints into energy |
| `bd_gen/guidance/constraint_schema.py` | Pydantic models for JSON/YAML constraint DSL |
| `tests/test_constraints.py` | Unit tests per primitive |
| `tests/test_constraint_schema.py` | Schema validation tests |

### Constraint base class (`constraints.py`)

```python
class Constraint(ABC):
    name: str
    weight: float = 1.0
    _p90_normalizer: float = 1.0  # set by calibration

    @abstractmethod
    def hard_violation(self, graph_dict: dict) -> ConstraintResult

    @abstractmethod
    def soft_violation(self, node_probs, edge_probs, pad_mask, vocab_config) -> float
```

`ConstraintResult` is a `@dataclass(frozen=True)` with `name`, `violation` (float >= 0), `satisfied` (bool), `details` (dict).

### Four initial primitives

1. **ExactCount** — `v = |count(type) - target|`
   - Soft: `v = |sum_i q_i(type) - target|` over active node positions

2. **CountRange** — `v = max(0, lo - count) + max(0, count - hi)`
   - Soft: same formula with expected count `n_hat = sum_i q_i(type)`

3. **RequireAdj** — `v = 1 - P(exists required adjacency pair)`
   - Soft: for each edge pos (i,j), compute `p_ij = (q_i(a)*q_j(b) + q_i(b)*q_j(a)) * P_adj(e_ij)` where `P_adj = sum(edge_probs[0:10])` (10 spatial types, excluding NO_EDGE). Then `P(exists) = 1 - prod(1 - p_ij)`, `v = 1 - P(exists)`.

4. **ForbidAdj** — `v = expected count of forbidden adjacency pairs`
   - Soft: `v = sum_{(i,j)} (q_i(a)*q_j(b) + q_i(b)*q_j(a)) * P_adj(e_ij)`

### RewardComposer (`reward.py`)

```python
class RewardComposer:
    constraints: list[Constraint]
    phi: str  # "linear" | "quadratic" | "log1p"

    def compute_energy(self, *, node_probs, edge_probs, pad_mask, vocab_config,
                       graph_dict=None, mode="soft") -> (float, dict)
    def compute_reward(self, **kwargs) -> (float, dict)  # r = -E
```

- `phi` shaping functions: `linear(v)=v`, `quadratic(v)=v^2`, `log1p(v)=log(1+v)`
- `reward_mode`: `"soft"` or `"hard"` — controls how intermediate rewards are computed (see below)
- Energy: `E(x) = sum_i (weight_i / p90_i) * phi(v_i(x))`

### Reward mode: soft vs hard (both implemented, pick one empirically)

Both modes use SVDD-style importance weights: `w_k ∝ exp(r(·) / α)`. They differ in **how the reward is evaluated at intermediate timesteps** when some positions are still masked.

**Hard mode** (`reward_mode="hard"`) — standard SVDD posterior mean approximation:
- For each candidate, hard-decode x̂_0: committed positions keep their token, masked positions use argmax(logits)
- Evaluate `r(x̂_0)` using `hard_violation()` on the decoded graph
- Properties: simplest to implement and debug. But reward is **discontinuous** — small logit changes can flip argmax and jump the violation by integer amounts. Higher variance across K candidates at the same step.

**Soft mode** (`reward_mode="soft"`) — posterior expectation refinement:
- For each candidate, build effective probability distributions: committed positions → one-hot, masked positions → softmax(logits)
- Evaluate `r_soft(x_t)` using `soft_violation()` with the probability distributions
- Properties: **smooth** — small logit changes produce small violation changes. Lower variance because it uses the full distribution, not a point estimate. Theoretically: `E[r_soft] ≈ E[r(x̂_0)]` but with lower variance. Reduces to hard mode when all positions are committed (one-hot = delta = argmax).

**Key differences to document in experiments:**
1. Reward variance across K candidates at the same step (soft should be lower)
2. ESS trajectories (soft should be more stable — fewer extreme weight ratios)
3. Final constraint satisfaction (the actual goal — which mode satisfies more constraints?)
4. Reward noise early in denoising (hard may produce misleading signals when most positions are masked; soft averages over uncertainty)
5. Computational cost (hard requires detokenization per candidate; soft uses tensor ops on logits)

### Constraint DSL (`constraint_schema.py`)

Pydantic models that validate JSON/YAML configs. Room type names (e.g., "Kitchen") are validated against `NODE_TYPES` from `vocab.py`. A `compile_constraints()` function converts specs into `Constraint` objects.

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
```

### Testing (`test_constraints.py`)

For each primitive:
- `test_satisfied` — hand-crafted graph with v=0
- `test_violated` — graph with v>0, verify correct magnitude
- `test_soft_converges_to_hard` — all-committed probs (one-hot) match hard violation

---

## Phase G2: Soft Violations from Logits

**Goal**: Implement the `build_effective_probs` utility and all `soft_violation()` methods.

### New file

| File | Purpose |
|------|---------|
| `bd_gen/guidance/soft_violations.py` | `build_effective_probs` + batched version |
| `tests/test_soft_violations.py` | Soft violation tests |

### `build_effective_probs(x_t, node_logits, edge_logits, pad_mask, vocab_config)`

For each position in the sequence:
- **PAD**: all-zero probs (excluded from scoring)
- **Committed** (not MASK): one-hot on the current token
- **MASK**: `softmax(logits)` from the model

Returns `(node_probs, edge_probs)` tensors. Implementation must be fully vectorized using `torch.where` and `scatter_` — no Python loops.

### Batched version for K*B candidates

`build_effective_probs_batch(x_t, node_logits, edge_logits, pad_mask, vocab_config)` operates on `(KB, SEQ_LEN)` tensors. The node_logits/edge_logits are expanded from (B,...) to (KB,...) since the model call is shared across K candidates.

### Key detail: P(adjacency)

`P_adj(e) = 1 - P(NO_EDGE at e) = edge_probs[e, 0:10].sum()`. Since the denoiser clamps MASK/PAD logits to -inf, softmax gives 0 to indices 11 and 12 automatically.

### Testing

- Verify soft violations are smooth (small logit perturbation -> small violation change)
- Verify soft violations at various masking ratios (100% masked, 50%, 0%) give sensible values
- Verify the batched version matches the per-sample version

---

## Phase G3: Guided Sampler (SVDD Reweighting)

**Goal**: Implement the guided sampling loop with K-candidate reweighting.

### Files to modify

| File | Change |
|------|--------|
| `bd_gen/diffusion/sampling.py` | Extract `_single_step_transition()` from loop body (lines 260-349). Existing `sample()` calls it once. No behavior change. |

### New files

| File | Purpose |
|------|---------|
| `bd_gen/guidance/guided_sampler.py` | `guided_sample()` function |
| `scripts/generate_guided.py` | CLI script for guided generation |
| `configs/guidance/default.yaml` | Default guidance config |
| `configs/guidance/example_basic.yaml` | Room counts + adjacency example |
| `tests/test_guided_sampler.py` | Integration tests |

### Refactoring `sampling.py`

Extract the loop body into **two** helpers:

```python
def _single_step_unmask(
    x_t, node_logits, edge_logits, pad_mask, p_unmask, i, num_steps, n_max,
    top_p, temperature, unmasking_mode, device,
    fixed_tokens, fixed_mask,
) -> Tensor:
    """Steps 4c-4h: token selection + unmasking + PAD clamp + inpainting.
    No remasking. Model call NOT included — logits are passed in."""

def _single_step_remask(
    x_t, remasking_fn, t_now, t_next, t_switch, i, pad_mask,
    node_logits, edge_logits,
) -> Tensor:
    """Step 4i: apply remasking_fn if applicable.
    Separated so guided sampler can score BETWEEN unmask and remask."""
```

The existing `sample()` loop body calls both sequentially: `x_t = _single_step_unmask(...)` then `x_t = _single_step_remask(...)`. **All existing tests must pass unchanged.**

**Why two functions:** SVDD guidance must score candidates *after* unmasking but *before* remasking. If remasking runs before scoring, it re-masks low-confidence positions (potentially the ones guidance steered toward constraint satisfaction), and all K candidates collapse to similar MASK patterns — destroying the reward differentiation that ESS depends on. See "Remasking-guidance interaction" below.

### SVDD algorithm in `guided_sample()`

```
For each step i = N-1 down to 0:
  1. Single model call: node_logits, edge_logits = model(x_t, pad_mask, t)
  2. Compute p_unmask (v1 scalar or v2 per-position)
  3. Expand x_t from (B, SEQ) to (K*B, SEQ) via repeat_interleave
  4. Expand logits, p_unmask, pad_mask similarly
  5. Run _single_step_unmask on expanded batch (K*B samples, NO remasking yet)
  6. Reshape to (K, B, SEQ)
  7. For each candidate k,b: build_effective_probs, compute reward
  8. Weights = softmax(rewards / alpha, dim=0)  — per-sample over K
  9. Resample: for each b, select one candidate via multinomial(weights[:, b])
  10. Record reward_pre_remask for selected candidate
  11. Apply _single_step_remask ONLY to selected candidates (B samples)
  12. Record reward_post_remask for selected candidate
  13. Record full diagnostics per step:
      - ESS = 1 / sum(w^2), max_weight, weight_entropy = H(w)/log(K)
      - reward_selected (pre-remask), reward_all_candidates, reward_gap
      - reward_remasking_delta = post_remask - pre_remask
      - Per-constraint soft violation (mean over batch)
```

**Remasking-guidance interaction:** Scoring happens on post-unmask, pre-remask candidates. This ensures ESS reflects pure guidance effectiveness — how well K candidates differentiate on constraint satisfaction. Remasking runs only on the winner, preserving its error-correction benefit without blurring the K-way comparison. The `reward_remasking_delta[t]` diagnostic tracks whether remasking is cooperating with or fighting guidance: consistently negative means remasking is undoing constraint-satisfying decisions and may need to be disabled or made guidance-aware.

**Compute cost**: Model call is 1x (shared). The K-fold expansion only affects the stochastic transition ops + reward computation, which are lightweight tensor operations.

### `scripts/generate_guided.py`

Follows the pattern of `generate_samples.py`:
- Load model, noise schedule, rate network (if v2)
- Load constraint config from YAML
- Compile constraints via `constraint_schema.compile_constraints()`
- Build `RewardComposer`
- Call `guided_sample()` in batches
- Save results with method name `{unmasking}_{sampling}_guided_{tag}_K{k}_a{alpha}`
- Save full `GuidanceStats` (all per-step trajectories + per-constraint final stats + violation histograms) alongside samples in the `.pt` file under `guidance_stats` key

### Testing

- `test_k1_no_constraints_matches_unguided` — with K=1 and no constraints, output is statistically identical to `sample()` (set same seed)
- `test_pad_invariant` — PAD positions unchanged
- `test_no_remaining_masks` — no MASK tokens in final output
- `test_constraint_improves_satisfaction` — guided ExactCount(Kitchen=1) yields higher kitchen-count satisfaction than unguided
- `test_ess_not_degenerate` — ESS > 1.0 at all steps
- `test_works_with_v1_and_v2` — compatible with both rate_network=None and rate_network=model
- `test_works_with_remasking` — compatible with remasking_fn

---

## Phase G4: Calibration + Evaluation

**Goal**: Calibration protocol and constraint satisfaction metrics.

### New files

| File | Purpose |
|------|---------|
| `bd_gen/guidance/calibration.py` | `calibrate_from_samples()`, save/load P90 normalizers |
| `scripts/calibrate_constraints.py` | Run calibration on unguided samples |

### Files to modify

| File | Change |
|------|--------|
| `scripts/evaluate.py` | Add constraint satisfaction metrics to `compute_all_metrics()` when a constraint config is provided |

### Calibration protocol

1. Load unguided samples (from existing `*_samples.pt`)
2. Detokenize to graph dicts
3. Compute `hard_violation()` for each constraint on all samples
4. For each constraint: P90 = 90th percentile of non-zero violations
5. Save normalizers to JSON: `{constraint_name: p90_value}`
6. `RewardComposer` loads calibration and sets `constraint._p90_normalizer`

### Guidance diagnostics: full trajectory tracking

The `guided_sample()` function returns a `GuidanceStats` object containing **per-step trajectories** (length N=num_steps), not averages. Two storage tiers:

- **Scalars**: batch-averaged, one value per step — for quick analysis and cross-run comparison.
- **Per-sample arrays**: shape `(N_steps, B)` — for deep debugging individual sample trajectories when aggregate behavior looks off.

Both are stored in the `.pt` file. The tables below show the scalar (batch-averaged) versions; the per-sample arrays use the same names with a `_per_sample` suffix.

**A) Weight distribution diagnostics (per step):**

| Metric | Shape | What it reveals |
|--------|-------|-----------------|
| `ess[t]` | scalar | Effective sample size = 1/sum(w^2). Healthy: close to K early (constraints uninformative on masked tokens), dropping mid-denoising (guidance bites), recovering slightly at end (candidates converge). Pathological: collapses to 1 (alpha too low) or stays at K (guidance has no effect). |
| `max_weight[t]` | scalar | Maximum weight across K candidates, batch-averaged. If ~1.0, one candidate dominates — guidance is deterministic, no diversity. |
| `weight_entropy[t]` | scalar | `H(w) / log(K)` normalized to [0,1]. 1.0 = uniform (no guidance), 0.0 = degenerate. Complements ESS: entropy captures the full weight shape, ESS mainly reacts to the dominant weight. |

**B) Reward and violation trajectories (per step):**

| Metric | Shape | What it reveals |
|--------|-------|-----------------|
| `reward_selected[t]` | scalar | Mean reward of the *selected* candidate (the one that wins resampling). This is the actual reward trajectory of the generated samples. |
| `reward_all_candidates[t]` | scalar | Mean reward across all K candidates. |
| `reward_gap[t]` | scalar | `reward_selected - reward_all_candidates`. Directly measures how much guidance biases selection at each step. Large gap = guidance is actively steering. Zero gap = all candidates score similarly (guidance is idle). |
| `violation_{name}[t]` | scalar | Per-constraint mean soft violation across the batch at each step. Reveals *when* each constraint gets resolved and *which* is hardest. E.g., ExactCount might resolve by step 30, while RequireAdj stays nonzero until step 80. |

**C) Remasking-guidance interaction (per step):**

| Metric | Shape | What it reveals |
|--------|-------|-----------------|
| `reward_pre_remask[t]` | scalar | Reward of selected candidate right after unmasking, before remasking touches it. |
| `reward_post_remask[t]` | scalar | Reward of same candidate after remasking. |
| `reward_remasking_delta[t]` | scalar | `post - pre`. Persistently negative = remasking fights guidance (re-masks constraint-satisfying tokens). Near zero = compatible. Positive = remasking accidentally helps (unlikely). |
| `positions_remasked[t]` | scalar | Mean number of positions re-masked per sample at this step. Context for interpreting the delta. |

**D) Per-constraint final statistics (on decoded samples):**

| Metric | Per-constraint |
|--------|---------------|
| `satisfaction_{name}` | Fraction of samples where `hard_violation == 0` |
| `satisfaction_overall` | Fraction where ALL constraints simultaneously satisfied |
| `mean_violation_{name}` | Mean hard violation (over all samples, including satisfied ones) |
| `mean_violation_when_failed_{name}` | Mean hard violation conditioned on `violation > 0` — how badly does it fail when it fails? |
| `violation_histogram_{name}` | Histogram of hard violation values (bins: 0, 1, 2, 3+) — reveals bimodal patterns (either perfect or catastrophic) vs gradual degradation |

**E) Stored for cross-run comparison:**

All trajectories and final stats are saved per `(alpha, K, constraint_set, model_variant)` configuration in the samples `.pt` file under a `guidance_stats` key. This enables overlaying ESS(t) curves across alpha values, plotting constraint resolution timing across K values, etc.

### Comparison methodology

Guided methods evaluated alongside unguided using the same `compare_selected.py` infrastructure. The guided method name encodes all parameters for reproducibility. The comparison table includes both standard generation metrics AND the constraint satisfaction columns.

---

## Phase G5: End-to-End Integration + Tuning

**Goal**: Full pipeline test and systematic alpha tuning.

### Steps

1. Run calibration on unguided v1 and v2 samples
2. Generate guided samples with the `example_basic.yaml` constraints (kitchen=1, living=1, kitchen-adj-living, no bath-kitchen) across:
   - v1 + llada + top-p=0.9 + no remasking
   - v2 + llada + top-p=0.9 + no remasking
   - v2 + llada + top-p=0.9 + confidence remasking tsw=1.0
3. **Reward mode comparison** (first experiment, fixed alpha=1.0, K=8):
   - Run each model variant with `reward_mode="soft"` and `reward_mode="hard"`
   - Compare: reward variance across K, ESS trajectories, final constraint satisfaction
   - Document findings → pick one mode for remaining experiments
4. Alpha grid search: [0.1, 0.5, 1.0, 2.0, 5.0]
5. K grid: [4, 8, 16]
6. Evaluate all guided variants with full metric suite + constraint satisfaction
7. Generate comparison table with soft/hard comparison writeup

### What to monitor (using full trajectory diagnostics)

**Guidance effectiveness:**
- ESS(t) curves overlaid across alpha values — identify the alpha range where guidance is active but not degenerate
- Per-constraint violation(t) trajectories — when does each constraint resolve? Which is the bottleneck?
- reward_gap(t) — is guidance actually steering, or are all K candidates scoring similarly?
- weight_entropy(t) — complement to ESS, captures full weight distribution shape

**Quality preservation:**
- Per-constraint satisfaction rates and violation histograms — bimodal (all-or-nothing) vs gradual?
- Inside validity, edge TV, diversity — compared to unguided baseline
- `mean_violation_when_failed` — when guidance fails on a constraint, how badly?

**Pathology detection:**
- ESS collapse (< 2) at any step → alpha too aggressive
- ESS = K throughout → guidance has no effect (constraints too weak or alpha too high)
- max_weight ~1.0 → single candidate dominates, diversity destroyed
- reward_gap = 0 → all candidates score identically (soft violations uninformative at that step)

---

## Key Files Reference

| Existing file | Role in guidance |
|---|---|
| [sampling.py](BD_Generation/bd_gen/diffusion/sampling.py) | Refactor: extract `_single_step_transition()` |
| [vocab.py](BD_Generation/bd_gen/data/vocab.py) | `NODE_TYPES`, `EDGE_NO_EDGE_IDX`, `VocabConfig.edge_position_to_pair()` |
| [remasking.py](BD_Generation/bd_gen/diffusion/remasking.py) | Pattern to follow: callable class + factory function |
| [generate_samples.py](BD_Generation/scripts/generate_samples.py) | Template for `generate_guided.py` |
| [evaluate.py](BD_Generation/scripts/evaluate.py) | Extend with constraint satisfaction metrics |
| [tokenizer.py](BD_Generation/bd_gen/data/tokenizer.py) | `detokenize()` for hard violation computation |
| [test_sampling.py](BD_Generation/tests/test_sampling.py) | Testing patterns, fixtures to reuse |

## Verification

After each phase:
- **G1**: Run hard violations on existing unguided samples, verify expected violation distributions
- **G2**: Compute soft violations at various timesteps on real denoising trajectories, verify they're informative and converge to hard violations when fully decoded
- **G3**: Run `python scripts/generate_guided.py` with basic constraints, verify improved constraint satisfaction. Run all existing tests to confirm no regression.
- **G4**: Run calibration, verify P90 normalizers are reasonable. Evaluate guided samples with full metrics.
- **G5**: Compare guided vs unguided systematically. Identify best (alpha, K) configuration.
