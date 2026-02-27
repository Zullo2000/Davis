# Guidance Module (`bd_gen/guidance/`)

Inference-time guidance for bubble diagram generation using SVDD (Soft Value-Based Decoding). Enforces architectural constraints during sampling via K-candidate importance reweighting — no retraining required.

## Module Structure

| File | Purpose | Phase |
|------|---------|-------|
| `constraints.py` | `Constraint` ABC + 4 primitives (ExactCount, CountRange, RequireAdj, ForbidAdj) | G1 |
| `reward.py` | `RewardComposer` — energy/reward from constraint violations | G1 |
| `constraint_schema.py` | Pydantic validation + `compile_constraints()` + `load_guidance_config()` | G1 |
| `soft_violations.py` | `build_effective_probs()`, `hard_decode_x0()` | G2 |
| `guided_sampler.py` | `guided_sample()` — SVDD K-candidate reweighting loop | G3 |
| `calibration.py` | P90 normalization utilities | G4 |

## Constraint Primitives

All constraints compute graded (non-binary) violations:

- **ExactCount** — `|count(type) - target|`. Example: require exactly 1 Kitchen.
- **CountRange** — `max(0, lo - count) + max(0, count - hi)`. Example: 1–4 bedrooms.
- **RequireAdj** — 0 if adjacency exists, 1 otherwise. Example: Kitchen adjacent to LivingRoom.
- **ForbidAdj** — count of forbidden adjacency pairs. Example: no Bathroom–Kitchen edges.

Each has `hard_violation(graph_dict)` (on decoded graphs) and `soft_violation(probs)` (on posterior distributions).

## Soft Violations (Phase G2)

Soft violations use the full posterior distribution `softmax(logits)` to compute smooth, differentiable constraint scores at any denoising step.

### Effective Probability Distributions

`build_effective_probs(x_t, node_logits, edge_logits, pad_mask, vocab_config)` constructs per-position distributions:

- **PAD positions** → all zeros (excluded from scoring)
- **Committed positions** (not MASK) → one-hot on the current token
- **MASK positions** → `softmax(logits)` from the model

All outputs are float64. Fully vectorized via `torch.where` — no Python loops.

`build_effective_probs_batch(...)` is the batched version for `(K*B, SEQ)` tensors.

### Soft Violation Formulas

| Constraint | Soft formula |
|-----------|-------------|
| **ExactCount** | `v = \|n̂ - target\|` where `n̂ = Σ q_i(type)` over active positions |
| **CountRange** | `v = max(0, lo - n̂) + max(0, n̂ - hi)` |
| **RequireAdj** | `v = 1 - P(exists)` via log-space: `Σ log(1 - p_ij)` |
| **ForbidAdj** | `v = Σ p_ij` (expected count of forbidden adjacencies) |

Where `p_ij = p_types_ij * P_adj_ij` is the joint probability that edge position (i,j) carries the relevant adjacency.

### Key Properties

- **Convergence**: soft violation == hard violation when all positions are committed (one-hot)
- **Smoothness**: small logit perturbations → small violation changes (no discrete jumps)
- **Range**: RequireAdj ∈ [0, 1], ForbidAdj ≥ 0, ExactCount ≥ 0, CountRange ≥ 0

### Hard Decode

`hard_decode_x0(x_t, logits, pad_mask, vocab_config)` — for hard reward mode, argmax-decodes MASK positions while keeping committed tokens.

## RewardComposer

Combines violations into energy/reward:

```
E(x) = Σ_i (λ_i / p90_i) * φ(v_i)
r(x) = -E(x)
w_k ∝ exp(r / α)
```

- `λ_i` = constraint weight, `p90_i` = calibration normalizer
- `φ` = shaping function: `linear` (default), `quadratic`, `log1p`
- `α` = guidance temperature (higher = weaker guidance)

## Configuration

Constraints are specified in YAML/JSON via Pydantic-validated schemas:

```yaml
constraints:
  - type: ExactCount
    name: one_kitchen
    room_type: Kitchen
    target: 1
  - type: RequireAdj
    name: kitchen_near_living
    type_a: Kitchen
    type_b: LivingRoom
num_candidates: 8
alpha: 1.0
phi: linear
reward_mode: soft
```

Load and compile: `config = load_guidance_config("path.yaml")` → `constraints = compile_constraints(config)`.

## Tests

- `tests/test_constraints.py` — 32 tests: hard violations (spec 1–11), RewardComposer hard mode (26–32), edge cases, soft violation returns tensor.
- `tests/test_constraint_schema.py` — 30 tests: Pydantic validation (33–39), compilation, YAML/JSON loading.
- `tests/test_soft_violations.py` — 36 tests: build_effective_probs (21–25), soft convergence (12,14,15,17), smoothness (13), ranges (16,18), edge cases (19,20), hard_decode_x0, RewardComposer soft mode (26–32 extended), triu_indices ordering.
- `tests/test_guided_sampler.py` — 16 tests: K=1 unguided match (40), PAD/MASK invariants (41–42), output shapes (43), constraint effectiveness (44), ESS sanity (45), stats completeness (46), v1/v2 compatibility (47–48), remasking (49), sampling refactoring (50, 52–53).

## Guided Sampler (Phase G3)

SVDD-style K-candidate importance reweighting at each denoising step. No retraining — works with any trained denoiser (v1 or v2).

### Sampling Refactoring

The `sample()` loop body in `bd_gen/diffusion/sampling.py` was refactored into two extracted helpers:

| Helper | Steps | Responsibility |
|--------|-------|---------------|
| `_single_step_unmask()` | 4c–4h | Token selection (argmax/top-p) + unmasking + PAD clamp + inpainting |
| `_single_step_remask()` | 4i | ReMDM remasking (no-op when `remasking_fn=None`) |

`sample()` calls both sequentially — zero behavior change. All 38 pre-existing sampling tests pass unchanged.

### SVDD Reweighting Loop

`guided_sample()` in `bd_gen/guidance/guided_sampler.py`:

```
for each denoising step t:
  1. Model call (shared, 1x cost)        → node_logits, edge_logits
  2. Expand x_t to K*B candidates        → repeat_interleave(K)
  3. _single_step_unmask on K*B           → K candidate transitions
  4. Score candidates via RewardComposer  → rewards (K, B)
     - soft mode: build_effective_probs → soft violations
     - hard mode: hard_decode_x0 → detokenize → hard violations
  5. Importance weights: softmax(r / α)   → weights (K, B)
  6. Resample: multinomial(weights)       → winner per sample
  7. _single_step_remask on winner only
  8. Record diagnostics
```

### Per-Step Diagnostics

`GuidanceStats` records both batch-averaged and per-sample diagnostics at each step:

| Metric | Description | Healthy range |
|--------|-------------|---------------|
| `ess` | Effective sample size (1/Σw²) | 1 < ESS ≤ K |
| `max_weight` | Largest importance weight | < 0.9 (not degenerate) |
| `weight_entropy` | H(w)/log(K), normalized | > 0.3 |
| `reward_selected` | Reward of resampled winner | Increases over steps |
| `reward_all_candidates` | Mean reward across K | Increases over steps |
| `reward_gap` | selected - mean | > 0 means guidance steers |
| `reward_pre_remask` | Winner reward before remasking | — |
| `reward_post_remask` | Winner reward after remasking | — |
| `reward_remasking_delta` | post - pre (remasking cooperation) | ≥ 0 ideal |
| `positions_remasked` | Count of positions re-masked | Decreases over steps |
| `violations` | Per-constraint violation of winner | → 0 over steps |

Final output includes `final_satisfaction`, `final_mean_violation`, `final_mean_violation_when_failed`, `final_violation_histograms`, and `satisfaction_overall`.

### Generation Script

`scripts/generate_guided.py` — CLI for guided sample generation:

```bash
# Basic usage
python scripts/generate_guided.py \
  eval.checkpoint_path=path/to/ckpt.pt \
  --guidance-config configs/guidance/example_basic.yaml

# Override alpha and K
python scripts/generate_guided.py \
  eval.checkpoint_path=path/to/ckpt.pt \
  --guidance-config configs/guidance/example_basic.yaml \
  --alpha 0.5 --K 16 --guidance-tag my_experiment
```

Guidance-specific args (`--guidance-config`, `--alpha`, `--K`, `--guidance-tag`) are parsed via argparse; remaining args pass through to Hydra. Output is saved to `eval_results/{schedule}_noise_sc/` with `guidance_stats` key alongside tokens.

### Compute Cost

| K | Model calls per step | Scoring per step | Total overhead vs unguided |
|---|---------------------|-----------------|---------------------------|
| 1 | 1 (shared) | B scores | ~1x (negligible) |
| 8 | 1 (shared) | 8×B scores | ~1x model + 8x scoring |
| 16 | 1 (shared) | 16×B scores | ~1x model + 16x scoring |

The model call is shared across all K candidates (the bottleneck). Scoring is a Python loop over K×B — acceptable for K≤16, B≤64. Vectorized scoring is a future optimization.
