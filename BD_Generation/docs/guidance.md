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

### Soft vs Hard Reward Mode

The `reward_mode` parameter controls how each candidate is scored at every denoising step:

- **Soft mode**: uses the model's probability distributions (`softmax(logits)`) at masked positions. A position that's 70% likely to be Kitchen contributes 0.7 to the expected kitchen count. Produces smooth, continuous violation values that change gradually as the model's beliefs evolve.

- **Hard mode**: applies argmax on logits to get a single predicted token per masked position, then counts discrete violations on the decoded graph. Binary — a position is either Kitchen or it isn't. More accurate reflection of the final decoded output, but discontinuous (small logit changes can flip the argmax and cause violation jumps).

Both modes produce a reward `r(candidate)` that feeds into the same importance weighting: `w_k = softmax(r / α)`. The α temperature then controls how aggressively those rewards translate into selection pressure. All pilot results below use soft mode.

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

## Gradual Discoveries — Design Choices & Tuning

> Updated as new experiment rounds bring insights. Each round appends findings; older rounds are kept for context.

### Design choices under exploration

| Hyperparameter | What it controls | Range explored so far |
|----------------|-----------------|----------------------|
| **α** (guidance temperature) | Selection sharpness: `w_k = softmax(r / α)`. Smaller α = sharper softmax = stronger guidance. | 0.1, 1.0, 5.0 |
| **K** (num candidates) | Pool size per denoising step. More candidates = more options to select from. | 4, 16 |
| **Reward mode** | How candidates are scored: soft (probability-based) vs hard (argmax-based). See §Soft vs Hard above. | soft |
| **Constraint set** | Which architectural rules to enforce. | See below |

### Comparison metrics

- **Satisfaction (all)**: % of samples where ALL constraints are simultaneously met. Primary metric — this is what guidance exists to improve.
- **Diversity**: pairwise distance between generated samples. Measures whether guidance collapses output variety.
- **Cond. edge TV**: total variation distance of edge distributions conditioned on node types. Lower = closer to training data distribution. Measures quality degradation from guidance.
- **Validity**: structural validity rate. Must stay at 100% — guidance should never break the model.
- **Guidance diagnostics** (trajectory-based, via `--plot-analysis`): ESS, reward gap, per-constraint violations. Used to understand *how* guidance steers, not just the outcome. Outlier analysis (P1 cutoff) separates clean samples from failure cases.

### Round 1 — Coarse α/K sweep (2026-02-28)

**Setup**: v1 + llada + top-p=0.9 + no remasking, soft reward, 5000 samples (5 seeds × 1000). Constraints: one_kitchen, one_living, kitchen_near_living, no_bath_kitchen.

**Findings**:

1. **α=0.1 is the only effective temperature.** α=1.0 and α=5.0 barely improve over the unguided baseline (~43% → ~48% and ~44% respectively). At α=0.1, satisfaction jumps to 68–77%. The softmax at α≥1.0 is too flat to meaningfully differentiate candidates — ESS stays near K (uniform weights), and the reward gap fluctuates around zero.

2. **K=16 > K=4** — more candidates helps. At α=0.1: 77% (K=16) vs 68.5% (K=4). The larger pool gives guidance more options to select from at each step.

3. **Quality tradeoff is mild at α=0.1**: diversity drops ~4%, cond. edge TV worsens by +0.045. Mode coverage, spatial transitivity, and node TV are essentially unchanged. Validity stays at 100%.

4. **one_living was trivially satisfied** (100% in all configs including baseline) — provided no signal. Replaced with `between_2_and_3_bathrooms` (CountRange) for Round 2.

**Timing** (from file timestamps on jabiru, under GPU contention with another user's training job sharing the A5000):

| K | Steady-state per config (5000 samples) | Per sample |
|---|----------------------------------------|------------|
| 4 | ~45 min | ~0.54s |
| 16 | ~2h 50min | ~2.0s |

K=16 is ~3.8x slower than K=4 (close to the theoretical 4x from scoring 4x more candidates per step). The first K=16 run took ~4h due to CUDA warmup / varying contention — subsequent K=16 runs were consistent at ~2h 50min. With a free GPU, generation should be ~60x faster (estimated ~3s per config, ~0.03s per sample for K=16).

**Design decisions taken**: fix α near 0.1, fix K=16, proceed with finer α sweep and revised constraint set.

### Understanding the guidance dynamics

The following observations were established during Round 1 and are expected to hold generally. They explain *why* the metrics behave as they do.

#### ESS and α

ESS = 1 / Σ(w_k²). At low α (sharp softmax), even moderate reward differences among K candidates cause one or two to absorb most of the weight, crashing ESS toward 1–2. At other timesteps — especially early when most tokens are masked — candidates are nearly identical, so ESS recovers toward K. This creates high variability in the ESS trajectory, which is a signature of **active steering**. At high α (flat softmax), ESS stays near K, meaning the guidance is not differentiating between candidates.

#### Reward trajectory

The reward of the selected candidate increases at a similar pace for all α values. This is because the bulk of the reward increase comes from the **base model denoising** (tokens getting unmasked, structure emerging), not from guidance selection. What guidance does is **cumulative**: consistently picking the slightly-better candidate at each of 100 steps compounds into the large final-satisfaction gap, even though the per-step advantage is small. The guidance gets its leverage in the middle regime of denoising, where candidates meaningfully differ.

#### Reward gap and stochastic selection

The reward gap (reward_selected - mean_reward_all_K) can be negative because selection is **stochastic multinomial sampling**, not deterministic argmax. Any candidate with non-zero weight can be selected, including below-average ones. At α=0.1 negative gaps are rare (sharp softmax); at α=1.0 they are common (~50%, consistent with near-random selection). This stochasticity is by design — deterministic argmax would destroy diversity.

#### Violation trajectories and candidate switching

Per-constraint violation trajectories under low α appear jumpier (non-monotone) than under high α. This is because a *different* candidate often wins at adjacent steps, so the trajectory stitches together snapshots from different candidates. A candidate that minimizes total energy may have higher violation on one specific constraint than the previous step's winner. With high α, selection is near-random, so the trajectory follows the model's natural (smooth, monotone) denoising curve — but this smoothness reflects the absence of steering, not superior guidance. **Trajectory smoothness is not a proxy for guidance quality.**

#### Why not α→0?

With α→0, the softmax becomes a hard argmax: the highest-reward candidate is deterministically selected at every step. This causes **diversity collapse** — greedy selection at each of 100 steps funnels all samples down the same narrow path, producing near-identical outputs and potentially trapping the model in locally optimal but globally poor trajectories. The stochastic selection at α=0.1 strikes a balance: strong enough to meaningfully steer (ESS drops to 2–4 when candidates differ), soft enough to preserve diversity.

### Round 2 — K* sweep (2026-03-02)

#### Motivation

Round 1 established α=0.1 as the effective guidance temperature and K=16 as better than K=4. However, the Round 2 fine α sweep (α ∈ {0.01, 0.05, 0.15, 0.3} × K ∈ {16, 24}, 5000 samples each) proved too slow under GPU contention (~2h50min per K=16 config at 5000 samples). The objective shifted to a more targeted question: **what is the minimal K\* that gives good constraint satisfaction?** Beyond K\*, more candidates yield diminishing returns — finding K\* allows future experiments to run faster without sacrificing quality.

A secondary question: does K\* differ between no-remasking and confidence remasking? Remasking undoes some committed tokens each step, which could either fight guidance (requiring higher K\*) or provide self-correction (allowing lower K\*).

#### Setup

- **Fixed**: α=0.1, soft reward, v1 loglinear checkpoint, revised 4-constraint set (one_kitchen, kitchen_near_living, no_bath_kitchen, between_2_and_3_bathrooms)
- **Sweep**: K ∈ {4, 8, 10, 12, 14, 16, 20, 24}
- **Variants**: no-remasking + confidence remasking (tsw=1.0)
- **Reduced samples**: 2 seeds (42, 123) × 100 samples = 200 per config. CI ≈ ±6% on Satisfaction (all) — sufficient to identify the plateau region, not to distinguish adjacent K values
- **Script**: `run_g5_kstar.sh noremask all && run_g5_kstar.sh confidence all`
- **Constraint set note**: the revised set includes `between_2_and_3_bathrooms` (CountRange, Bathroom ∈ [2,3]), which replaced the trivially-satisfied `one_living`. This makes the baseline much harder (13% vs 43% in Round 1).

#### Timing (on jabiru A5000, under GPU contention)

| K | No-remasking (min) | Confidence (min) |
|---|---|---|
| 4 | ~9 | ~5 |
| 8 | ~9 | ~5 |
| 10 | ~7 | ~4 |
| 12 | ~7 | ~6 |
| 14 | ~6 | ~6 |
| 16 | ~7 | ~7 |
| 20 | ~8 | ~9 |
| 24 | ~9 | ~11 |
| **Total** | **~53 min** | **~48 min** |

Total wall-clock for both variants (generation only): ~1h 41min. Per-sample cost scales roughly linearly with K, but the 200-sample batches are small enough that overhead dominates at low K.

#### Results — Satisfaction (all)

| K | No-remasking | Confidence remasking |
|---|---|---|
| baseline | 13.3 ± 1.2% | 16.7 ± 1.1% |
| 4 | 39.0 ± 1.0% | 32.5 ± 2.5% |
| 8 | 51.5 ± 2.5% | 39.0 ± 4.0% |
| 10 | 49.5 ± 1.5% | 48.0 ± 3.0% |
| 12 | **56.5 ± 4.5%** | 49.0 ± 1.0% |
| 14 | 56.5 ± 4.5% | 53.0 ± 4.0% |
| 16 | 56.5 ± 2.5% | 55.5 ± 2.5% |
| 20 | 53.0 ± 6.0% | 56.0 ± 4.0% |
| 24 | 57.5 ± 2.5% | **61.0 ± 2.0%** |

#### Findings

1. **No-remasking: K\* ≈ 12.** Satisfaction plateaus at ~56% for K=12–24 (all within the ±6% CI). No meaningful improvement beyond K=12. The big gains come from K=4→8 (+12.5pp) and K=4→12 (+17.5pp).

2. **Confidence remasking: no clear plateau — K\* > 24.** The curve keeps climbing: K=12 (49%) → K=16 (55.5%) → K=24 (61%). The K=20→24 jump of +5pp suggests gains continue beyond K=24. Remasking fights guidance by undoing committed tokens, so more candidates are needed to compensate.

3. **Remasking shifts K\* UP, not down.** The "self-correction" hypothesis (remasking fixes guidance mistakes → fewer candidates needed) is not supported. Instead, remasking destroys some guidance progress each step, requiring a larger candidate pool to maintain steering pressure.

4. **Confidence remasking may ultimately achieve higher satisfaction** — at K=24 it reaches 61% vs no-remasking's 57.5%. The remasking adds exploration that may help escape local optima, but only if K is large enough to overcome the per-step destruction.

5. **The bottleneck constraint is `no_bath_kitchen`** (ForbidAdj). Even at K=24: 72% (no-remask), 79.5% (confidence). The other 3 constraints are near-saturated by K=10:
   - `one_kitchen`: 99–100% by K=10
   - `kitchen_near_living`: 99–100% by K=10
   - `between_2_and_3_bathrooms`: ~84% (no-remask) / ~80% (confidence) by K=10

6. **Quality tradeoff is notable.** Guidance degrades distribution fidelity:
   - Cond. edge TV: +0.10 (no-remask K=12) / +0.07 (confidence K=16) vs baselines
   - Mode coverage (weighted): drops from 70%→51% (no-remask) / 73%→45% (confidence)
   - Unique archetypes: halved (no-remask) / quartered (confidence)
   - Validity, spatial transitivity: unaffected (100%)

#### Per-constraint breakdown — no-remasking

| K | one_kitchen | kitchen_near_living | no_bath_kitchen | between_2_and_3_bathrooms |
|---|---|---|---|---|
| baseline | 91.3% | 91.3% | 52.0% | 49.2% |
| 4 | 97.5% | 97.5% | 57.5% | 75.5% |
| 8 | 99.0% | 99.0% | 63.0% | 84.5% |
| 12 | 99.5% | 99.5% | 70.0% | 83.5% |
| 16 | 100% | 100% | 68.0% | 85.0% |
| 24 | 99.5% | 99.5% | 72.0% | 84.5% |

`one_kitchen` and `kitchen_near_living` saturate by K=8. `between_2_and_3_bathrooms` saturates by K=8 (~84%). The only constraint that keeps improving with K is `no_bath_kitchen` (52% → 72%), but even this has diminishing returns after K=12.

#### Per-constraint breakdown — confidence remasking

| K | one_kitchen | kitchen_near_living | no_bath_kitchen | between_2_and_3_bathrooms |
|---|---|---|---|---|
| baseline | 81.3% | 92.0% | 46.6% | 58.9% |
| 4 | 96.0% | 97.0% | 62.5% | 67.0% |
| 8 | 98.0% | 98.5% | 68.0% | 68.0% |
| 12 | 98.5% | 99.0% | 68.5% | 77.0% |
| 16 | 98.5% | 99.5% | 74.0% | 81.5% |
| 24 | 99.5% | 99.5% | 79.5% | 79.5% |

Same pattern: `one_kitchen` and `kitchen_near_living` saturate early. `no_bath_kitchen` keeps improving steadily even to K=24 (46.6% → 79.5%). `between_2_and_3_bathrooms` reaches ~80% by K=12–16.

#### Practical recommendations

- **No-remasking: use K=12** for future experiments. The plateau is clear and this saves ~25% compute vs K=16.
- **Confidence remasking: use K=16 minimum, K=20–24 if compute budget allows.** The curve hasn't flattened.
- **α fine-tuning** should be revisited at the chosen K\* (next experiment round). The current α=0.1 was established on the old constraint set; the harder `between_2_and_3_bathrooms` may benefit from a different α.
