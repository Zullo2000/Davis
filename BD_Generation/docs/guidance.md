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

Both modes produce a reward `r(candidate)` that feeds into the same importance weighting: `w_k = softmax(r / α)`. The α temperature then controls how aggressively those rewards translate into selection pressure.

#### Why we use soft reward mode (Round 4 finding)

Round 4 compared soft vs hard reward on the no-remasking variant (K=16, α=0.01, 600 samples each). Results:

| Metric | Soft | Hard |
|--------|------|------|
| Overall constraint satisfaction | **69.0%** ± 4.1 | 35.0% ± 2.3 |
| `one_kitchen` satisfaction | **100%** | 97.8% |
| `kitchen_near_living` satisfaction | **100%** | 97.8% |
| `no_bath_kitchen` satisfaction | **74.7%** | 70.2% |
| `between_2_and_3_bathrooms` satisfaction | **93.8%** | 61.3% |
| Diversity | 90.5% | **98.3%** |
| Edge TV (↓ better) | 0.565 | **0.363** |
| Mode coverage (weighted) | 0.412 | **0.630** |

Soft reward mode nearly doubles overall satisfaction (69% vs 35%). The gap is largest on `between_2_and_3_bathrooms` (94% vs 61%), and soft achieves 100% on the two "easy" constraints where hard still fails ~2% of the time.

Hard mode does produce better distributional quality (higher diversity, lower edge TV, better mode coverage), because argmax decoding gives sharper per-candidate scores that preserve more of the model's natural distribution. However, this sharpness is also its weakness for guidance: the reward landscape is spikier and harder for importance reweighting to exploit. Soft mode's smooth probability-based violations provide continuous gradients that guidance steers much more effectively.

**Decision:** all experiments use soft reward mode. The constraint satisfaction advantage is too large to trade for distributional quality gains — satisfying constraints is the primary goal of guidance.

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

### Round 3 — α fine-tuning at K=16 (2026-03-03)

#### Motivation

Round 1 established α=0.1 as the sweet spot on the old (easy) constraint set. The K* sweep (Round 2) used α=0.1 throughout but focused on K, not α. With the revised constraint set — which includes the hard `between_2_and_3_bathrooms` replacing the trivially-satisfied `one_living` — the optimal α may have shifted. This round sweeps α finely at the practical K=16 to find the best temperature for each variant.

#### Setup

- **Fixed**: K=16, soft reward, v1 loglinear checkpoint, revised 4-constraint set
- **Sweep**: α ∈ {0.01, 0.05, 0.1, 0.15, 0.3, 0.5}
- **Variants**: no-remasking + confidence remasking (tsw=1.0)
- **Samples**: 3 seeds (42, 123, 456) × 200 samples = 600 per config
- **Script**: `run_g5_alpha.sh noremask all && run_g5_alpha.sh confidence all`

#### Timing (on jabiru A5000, under GPU contention)

| Phase | Per config | Total (6 configs) |
|-------|-----------|-------------------|
| No-remasking generation | ~6 min | ~36 min |
| Confidence generation | ~21 min | ~125 min |
| **Full pipeline** (calibrate + generate + evaluate + compare + analyze) | | |
| No-remasking | | ~61 min |
| Confidence | | ~2h 18min |
| **Grand total** | | **~3h 19min** |

Confidence is ~3.5× slower per config than no-remasking at the same K=16 — the remasking step adds overhead at every denoising step.

#### Results — Satisfaction (all)

| α | No-remasking | Confidence remasking |
|---|---|---|
| baseline | 13.3 ± 1.2% | 16.7 ± 1.1% |
| 0.01 | **69.0 ± 4.1%** | 55.3 ± 1.6% |
| 0.05 | 62.2 ± 4.1% | 56.5 ± 3.3% |
| 0.1 | 53.5 ± 2.9% | **57.3 ± 4.3%** |
| 0.15 | 50.2 ± 1.6% | 55.7 ± 3.1% |
| 0.3 | 32.7 ± 2.3% | 38.3 ± 3.0% |
| 0.5 | 23.5 ± 1.4% | 30.8 ± 2.1% |

#### Per-constraint breakdown — no-remasking

| α | one_kitchen | kitchen_near_living | no_bath_kitchen | between_2_and_3_bathrooms |
|---|---|---|---|---|
| baseline | 91.3% | 91.3% | 52.0% | 49.2% |
| 0.01 | 100% | 100% | 74.7% | 93.8% |
| 0.05 | 100% | 100% | 71.2% | 89.2% |
| 0.1 | 99.5% | 99.5% | 65.5% | 84.5% |
| 0.15 | 99.2% | 99.2% | 65.2% | 80.8% |
| 0.3 | 97.3% | 97.3% | 57.5% | 68.0% |
| 0.5 | 96.3% | 96.3% | 52.7% | 60.8% |

#### Per-constraint breakdown — confidence remasking

| α | one_kitchen | kitchen_near_living | no_bath_kitchen | between_2_and_3_bathrooms |
|---|---|---|---|---|
| baseline | 81.3% | 92.0% | 46.6% | 58.9% |
| 0.01 | 97.7% | 98.7% | 77.3% | 78.2% |
| 0.05 | 97.7% | 99.3% | 74.3% | 81.0% |
| 0.1 | 99.0% | 99.2% | 73.5% | 81.3% |
| 0.15 | 97.2% | 98.2% | 73.5% | 82.0% |
| 0.3 | 94.7% | 96.7% | 60.8% | 74.5% |
| 0.5 | 92.7% | 96.7% | 53.5% | 69.2% |

#### Quality tradeoff — no-remasking

| α | Diversity | Mode cov. (w) | Cond. edge TV | Type-cond. degree TV | Spatial trans. | Unique archetypes |
|---|---|---|---|---|---|---|
| baseline | 0.945 | 69.6% | 0.472 | 0.159 | 99.9% | 28.6 |
| 0.01 | 0.905 (-4.0pp) | 41.2% (-28.4pp) | 0.626 (+0.154) | 0.203 (+0.044) | 100% | 15.0 |
| 0.05 | 0.932 (-1.4pp) | 42.7% (-26.9pp) | 0.604 (+0.132) | 0.214 (+0.056) | 100% | 13.7 |
| 0.1 | 0.953 (+0.8pp) | 60.6% (-9.0pp) | 0.573 (+0.101) | 0.215 (+0.056) | 100% | 14.0 |
| 0.15 | 0.972 (+2.6pp) | 61.0% (-8.6pp) | 0.551 (+0.079) | 0.209 (+0.050) | 100% | 14.7 |
| 0.3 | 0.975 (+3.0pp) | 61.7% (-7.9pp) | 0.508 (+0.036) | 0.169 (+0.010) | 99.7% | 14.7 |
| 0.5 | 0.982 (+3.6pp) | 61.9% (-7.7pp) | 0.484 (+0.012) | 0.167 (+0.007) | 99.8% | 15.0 |

#### Quality tradeoff — confidence remasking

| α | Diversity | Mode cov. (w) | Cond. edge TV | Type-cond. degree TV | Spatial trans. | Unique archetypes |
|---|---|---|---|---|---|---|
| baseline | 0.982 | 73.3% | 0.571 | 0.169 | 98.7% | 120.8 |
| 0.01 | 0.988 (+0.6pp) | 49.8% (-23.5pp) | 0.607 (+0.037) | 0.172 (+0.003) | 99.0% | 33.7 |
| 0.05 | 0.988 (+0.6pp) | 54.1% (-19.2pp) | 0.619 (+0.048) | 0.154 (-0.015) | 99.2% | 36.7 |
| 0.1 | 0.985 (+0.3pp) | 51.2% (-22.1pp) | 0.620 (+0.050) | 0.150 (-0.019) | 99.0% | 34.3 |
| 0.15 | 0.983 (+0.1pp) | 49.9% (-23.4pp) | 0.635 (+0.064) | 0.153 (-0.017) | 99.5% | 37.7 |
| 0.3 | 0.990 (+0.8pp) | 58.9% (-14.4pp) | 0.629 (+0.058) | 0.194 (+0.025) | 98.7% | 41.7 |
| 0.5 | 0.995 (+1.3pp) | 52.0% (-21.3pp) | 0.625 (+0.054) | 0.162 (-0.007) | 99.2% | 41.3 |

#### Findings

1. **α=0.01 is the new optimal for no-remasking** — 69.0% satisfaction, up from 53.5% at α=0.1. The harder constraint set requires a much smaller (sharper) temperature to maintain effective guidance. Satisfaction decreases monotonically with increasing α: 69% → 62% → 54% → 50% → 33% → 24%.

2. **Confidence remasking is insensitive to α in the 0.01–0.15 range** — satisfaction is flat at 55–57% for α ∈ {0.01, 0.05, 0.1, 0.15}, then drops at α ≥ 0.3. The plateau suggests that remasking is the dominant effect, washing out α differences. Best is α=0.1 (57.3%) but within noise of α=0.01–0.15.

3. **No-remasking beats confidence at the same K=16** — best no-remasking (69.0% at α=0.01) vs best confidence (57.3% at α=0.1). This reverses the K* sweep finding where confidence at K=24 (61%) exceeded no-remasking at K=24 (57.5%). At equal K, no-remasking wins because guidance gains persist.

4. **Quality cost scales with guidance strength** — at α=0.01 (strongest guidance), no-remasking pays: diversity -4pp, mode coverage -28pp, cond. edge TV +0.15. At α=0.1 (moderate): diversity +0.8pp, mode coverage -9pp, cond. edge TV +0.10. The satisfaction-quality Pareto front has a clear knee around α=0.05.

5. **`no_bath_kitchen` remains the bottleneck** — even at the best α=0.01: 74.7% (no-remask), 77.3% (confidence). `between_2_and_3_bathrooms` reaches 93.8% at α=0.01 (no-remask), and `one_kitchen`/`kitchen_near_living` are fully saturated at 100%.

6. **Confidence remasking has milder quality impact** — cond. edge TV only worsens by +0.04–0.06 across all α (vs +0.15 for no-remasking at α=0.01). But this comes at the cost of lower satisfaction. The remasking "smooths" the distribution but also smooths away the constraint satisfaction gains.

7. **The α=0.1 sweet spot from Round 1 was specific to the easy constraint set.** With the revised (harder) constraints, the optimal α shifted to 0.01 — a 10× decrease. **This suggests α should be re-tuned whenever the constraint set changes significantly.**

#### Per-step guidance dynamics — no-remasking

| α | Mean ESS | Min ESS (step) | Mean reward gap | Gap ≈ 0 steps | Guidance active? |
|---|----------|----------------|-----------------|---------------|------------------|
| 0.01 | 4.28 | 2.60 (step 0) | 0.319 | — | YES, strongly |
| 0.05 | 6.36 | 3.46 (step 0) | 0.263 | — | YES |
| 0.1 | 9.47 | 5.61 (step 0) | 0.020 | 64/100 | Barely — ESS=16 by step 40 |
| 0.15 | 15.25 | 7.04 (step 0) | 0.015 | 64/100 | NO |
| 0.3 | 15.65 | 10.28 (step 0) | 0.010 | 64/100 | NO |
| 0.5 | 15.84 | 13.04 (step 0) | 0.005 | 64/100 | NO |

At α ≥ 0.1, ESS reaches 16 (= K, uniform weights) by step ~40 and stays there for the remaining 60 steps — guidance is effectively off for 60% of the denoising process. At α=0.01, ESS stays low (2.6–4.3) throughout, meaning the importance weights are concentrated and guidance is actively selecting among candidates at every step.

Per-constraint violation trajectories (no-remasking, trimmed means excluding P1 outliers):

| α | `one_kitchen` | `kitchen_near_living` | `no_bath_kitchen` | `between_2_3_bath` |
|---|---|---|---|---|
| 0.01 | 0.000 | 0.000 | 0.200 | 0.060 |
| 0.05 | 0.000 | 0.000 | 0.258 | 0.107 |
| 0.1 | 0.000 | 0.000 | 0.369 | 0.154 |
| 0.15 | 0.019 | 0.000 | 0.373 | 0.192 |
| 0.3 | 0.019 | 0.019 | 0.466 | 0.314 |
| 0.5 | 0.032 | 0.032 | 0.523 | 0.389 |

`one_kitchen` and `kitchen_near_living` resolve by step 32–33 for all α values — these are easy constraints that the base model nearly satisfies. The discriminating constraints are `no_bath_kitchen` (0.20 at α=0.01 vs 0.52 at α=0.5) and `between_2_and_3_bathrooms` (0.06 vs 0.39).

#### Per-step guidance dynamics — confidence remasking

| α | Mean ESS | Min ESS (step) | Mean reward gap | Mean remasking Δ | Negative Δ steps |
|---|----------|----------------|-----------------|------------------|------------------|
| 0.01 | 4.70 | 3.20 (step 51) | 0.307 | **-0.227** | 98/100 |
| 0.05 | 6.50 | 4.06 (step 47) | 0.293 | **-0.215** | 98/100 |
| 0.1 | 7.51 | 5.07 (step 47) | 0.274 | **-0.198** | 98/100 |
| 0.15 | 8.66 | 6.45 (step 47) | 0.246 | **-0.173** | 98/100 |
| 0.3 | 11.39 | 10.25 (step 53) | 0.183 | **-0.121** | 98/100 |
| 0.5 | 13.62 | 12.16 (step 53) | 0.108 | **-0.077** | 98/100 |

**Key observation: remasking fights guidance at 98/100 steps for ALL α values.** The remasking delta is consistently negative, meaning the reward drops after remasking at nearly every step. The magnitude scales with guidance strength: at α=0.01 (strongest), remasking erases -0.23 reward units per step; at α=0.5 (weakest), -0.08. This is the mechanism behind the K* shift — guidance builds up constraint satisfaction, remasking tears it down.

Contrast with no-remasking, where the remasking delta is exactly 0 (no remasking occurs) and guidance gains accumulate monotonically.

The reward gap stays much higher for confidence (0.11–0.31) than for no-remasking at the same α (0.005–0.32). This is not because guidance is "working harder" — it's because remasking keeps re-introducing disorder, so candidates remain diverse and the gap stays large. But this diversity is counter-productive: the guidance is running on a treadmill.

#### ESS trajectory shape comparison

**No-remasking**: ESS is lowest at step 0 (high noise, candidates most diverse) and increases monotonically as tokens get committed. At α=0.01, ESS goes from 2.6 → ~16 by step ~80 (guidance active for most of the process). At α ≥ 0.1, ESS reaches 16 by step ~40 (guidance active only in the noisy early phase).

**Confidence remasking**: ESS is lowest at steps 47–53 (mid-denoising), not at step 0. This is because remasking re-introduces masked positions in the middle of the process, creating candidate diversity that keeps ESS low. ESS then recovers toward the end as remasking stops (t_switch). This "mid-dip" in ESS is a signature of the guidance-remasking conflict — guidance keeps trying to select, remasking keeps undoing.

Trajectory snapshots at α=0.01 (strongest guidance) illustrate the difference:

**No-remasking** (α=0.01):
| Step | ESS | MaxW | R_gap | R_sel | R_delta |
|------|-----|------|-------|-------|---------|
| 0 | 2.60 | 0.460 | 0.335 | -1.607 | 0.000 |
| 20 | 3.06 | 0.421 | 0.315 | -1.564 | 0.000 |
| 40 | 5.07 | 0.375 | 0.350 | -0.660 | 0.000 |
| 60 | 13.77 | 0.159 | 0.044 | -0.305 | 0.000 |
| 80 | 16.00 | 0.062 | 0.000 | -0.198 | 0.000 |
| 99 | 16.00 | 0.062 | 0.000 | -0.198 | 0.000 |

**Confidence** (α=0.01):
| Step | ESS | MaxW | R_gap | R_sel | R_delta |
|------|-----|------|-------|-------|---------|
| 0 | 4.32 | 0.357 | 0.324 | -1.620 | 0.000 |
| 20 | 3.90 | 0.383 | 0.306 | -1.574 | -0.241 |
| 40 | 3.60 | 0.417 | 0.330 | -1.513 | -0.255 |
| 60 | 4.12 | 0.499 | 0.361 | -1.506 | -0.290 |
| 80 | 6.72 | 0.385 | 0.231 | -1.653 | -0.118 |
| 99 | 9.50 | 0.269 | 0.470 | -0.410 | 0.000 |

In no-remasking, R_sel improves steadily from -1.61 → -0.20 (higher = better). In confidence, R_sel barely moves from -1.62 → -0.41 — the remasking delta is eating the gains. By step 60, no-remasking has R_sel = -0.31 vs confidence R_sel = -1.51, a 5× gap in reward.

#### Per-step trajectory analysis — individual samples (2026-03-05)

The tables above report **batch-averaged** diagnostics. To understand the actual per-step dynamics, we examine trajectory plots of **individual floorplan samples** through the 100-step guided denoising process. Each line in a trajectory plot is one sample's full 100-step history for a given metric (ESS, reward, violations, etc.).

##### Methodology: clean vs outlier samples

The `--plot-analysis` pipeline in `analyze_guidance_stats.py` classifies samples based on their **final reward** (reward at step 99):

1. Compute the P1 threshold (1st percentile of final reward across all 600 samples in a config).
2. **Outlier samples** = bottom 1% by final reward (typically ~6 out of 600). These are the samples where guidance failed to improve constraint satisfaction.
3. **Clean samples** = the remaining 99%.
4. Two random samples are picked from each group and their full 100-step trajectories are plotted.

Each plot title shows the sample identity and its final reward, e.g. `seed=42 idx=41 (r=0.500)` means seed 42, sample index 41, with final reward 0.5. The P1 threshold is also shown (e.g. `threshold=-1.500`).

The trajectory PNG files are in `eval_results/loglinear_noise_sc/` with naming convention:
- `{model}_trajectories_clean.png` — 2 clean (typical) samples
- `{model}_trajectories_outliers.png` — 2 outlier (failure) samples

##### No-remasking: ESS has a sharp phase transition

The averaged tables suggested a smooth monotonic ESS increase. The individual trajectories reveal a **binary phase transition** instead:

**α=0.01** (strongest guidance):
ESS oscillates noisily between 2 and 14 for the first ~30–40 steps, then jumps sharply to 16 (= K) within ~5 steps and stays flat. Guidance is either fully active (low ESS, concentrated weights) or completely off (ESS = K, uniform weights). There is no gradual transition.
→ See: `llada_topp0.9_no_remask_guided_alpha_K16_a0.01_trajectories_clean.png`

**α=0.05**: Similar pattern, transition slightly earlier (~step 25–35).
→ See: `llada_topp0.9_no_remask_guided_alpha_K16_a0.05_trajectories_clean.png`

**α=0.1**: ESS floor is higher (5–8), transition to K happens by step ~15–20. Guidance is active for only the first ~15% of denoising.
→ See: `llada_topp0.9_no_remask_guided_alpha_K16_a0.1_trajectories_clean.png`

**α=0.3–0.5**: ESS starts near K and reaches it within ~5–10 steps. Guidance is essentially off for >90% of the process.
→ See: `llada_topp0.9_no_remask_guided_alpha_K16_a0.3_trajectories_clean.png`, `..._a0.5_trajectories_clean.png`

Summary — the transition step moves earlier as α increases:

| α | ESS floor (early) | Transition to K | Guidance active window |
|---|---|---|---|
| 0.01 | 2–5 | step ~30–40 | ~35% of denoising |
| 0.05 | 2–5 | step ~25–35 | ~30% |
| 0.1 | 5–8 | step ~15–20 | ~15% |
| 0.15 | 5–14 | step ~10–20 | ~10% |
| 0.3 | 7–14 | step ~10 | ~5% |
| 0.5 | 10–16 | step ~5 | ~3% |

The mechanism: once enough positions are committed, all K candidates receive nearly identical tokens from the shared model logits. The remaining stochasticity affects too few positions to create meaningful reward differences, so importance weights become uniform regardless of α. The sharpness of the transition (ESS jumps from ~5 to 16 in ~5 steps) reflects the discrete nature of token commitment — there is a critical number of committed positions beyond which candidate diversity collapses.

**Deviation from spec prediction (Section 11.3):** The spec predicted an inverted-U (high ESS early → dip mid-denoising → partial recovery). The actual shape is a step function (low → sharp jump → flat). The spec assumed "constraints uninformative on masked tokens," but our soft violations compute meaningful reward differences even on masked positions (via `softmax(logits)` effective probabilities). Combined with 1/α amplification (particularly at small α), this makes guidance active from the very first step rather than "waking up" mid-denoising.

##### Confidence remasking: chaotic ESS, no clean phase transition

The averaged table showed smooth mean ESS values (e.g. 4.70 at α=0.01) with a minimum around step 47–53. The individual trajectories reveal this is an artifact of averaging over **chaotic oscillation**.

**α=0.01**: ESS bounces between 2 and 15 at every step throughout the entire 100-step process. There is no trend, no phase transition, no stable regime. Each remasking step re-introduces stochasticity, destroying whatever convergence the previous guidance step achieved.
→ See: `llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01_trajectories_clean.png`

**α=0.1**: Same chaotic pattern with a slightly higher ESS floor (3–16 oscillation range).
→ See: `..._a0.1_trajectories_clean.png`

**α=0.3**: Oscillation dampens (range 8–16) but ESS still drops below 10 regularly.
→ See: `..._a0.3_trajectories_clean.png`

**α=0.5**: ESS mostly 12–16 with occasional dips to 8–10. Closest to stable, but remasking effect still visible.
→ See: `..._a0.5_trajectories_clean.png`

The "mid-dip" described in the ESS shape comparison section above (min ESS at step 47–53) is an artifact of averaging: the chaotic oscillations happen to have a slightly lower mean mid-process because remasking activity peaks mid-denoising (before t_switch). Individual samples show no coherent dip.

##### Violation resolution: monotonic staircase vs treadmill

**No-remasking (clean samples):** Constraints resolve in order of difficulty and **stay resolved** once satisfied. The reward trajectory has a staircase shape — discrete jumps as each constraint flips from violated to satisfied:

1. `between_2_and_3_bathrooms` — resolves ~step 15–25 (count constraint, resolved once enough room types are committed)
2. `one_kitchen` — resolves ~step 25–35 (easy, base model nearly always satisfies this)
3. `kitchen_near_living` — co-resolves with `one_kitchen` ~step 25–35
4. `no_bath_kitchen` — resolves last (~step 25–40) or **never** (the bottleneck constraint)

→ See violation subplots in: `..._a0.01_trajectories_clean.png` through `..._a0.5_trajectories_clean.png`

**Confidence (clean samples):** Violations **oscillate** between 0 and 1 repeatedly. Even the easy constraints (`one_kitchen`) flip back and forth as remasking un-commits positions that guidance had steered. The reward trajectory is noisy and non-monotonic. By step 99, violations may resolve (t_switch stops remasking), but the final reward is lower than no-remasking because the late-stage resolution has less room to maneuver.
→ See violation subplots in: `..._confidence_..._a0.01_trajectories_clean.png`

The remasking delta subplots confirm this mechanism: for confidence remasking, the delta shows **large negative spikes** (down to -1 to -5 in individual samples) at most steps, far worse than the batch-averaged mean of -0.2. The averaging smooths over these catastrophic individual-step regressions.

##### Outlier mechanism: early topology lock-in (no-remasking)

Outlier samples (bottom 1% by final reward) reveal why guidance fails on certain samples.

**No-remasking outliers:** The `no_bath_kitchen` violation stays near 1.0 for the entire 100 steps — guidance never manages to eliminate the bathroom-kitchen adjacency. All other constraints (`one_kitchen`, `kitchen_near_living`, `between_2_and_3_bathrooms`) resolve normally by step ~30. The outlier samples are "stuck" because the early committed positions (before step ~20) locked in a graph topology where bathroom and kitchen are adjacent, and subsequent guidance steps cannot undo committed tokens.
→ See: `llada_topp0.9_no_remask_guided_alpha_K16_a0.01_trajectories_outliers.png`

This reveals that **early-step guidance is critical for no-remasking**: if the wrong topology is committed in the first ~20 steps (when ESS is low and guidance is most active), the damage is permanent. The ForbidAdj constraint (`no_bath_kitchen`) is the hardest because it requires a specific adjacency to NOT exist — once committed, adjacency tokens cannot change.

**Confidence outliers:** Show the same chaotic oscillation as clean samples, but with higher violation amplitudes and worse final reward. The distinction between clean and outlier is less sharp for confidence because the remasking keeps all samples in a high-variance regime — both clean and outlier samples have noisy, non-converging trajectories, and the final outcome depends heavily on what happens in the last few steps.
→ See: `llada_topp0.9_remdm_confidence_tsw1.0_guided_alpha_K16_a0.01_trajectories_outliers.png`

##### Summary of trajectory analysis

| Aspect | No-remasking | Confidence remasking |
|---|---|---|
| ESS shape | Step function: low → sharp jump to K | Chaotic oscillation, no stable regime |
| Transition mechanism | Token commitment collapses candidate diversity | Remasking prevents convergence |
| Violation resolution | Monotonic staircase, constraints stay resolved | Treadmill: constraints oscillate, resolve only at t_switch |
| Remasking delta | Exactly 0 (no remasking) | Large negative spikes (-1 to -5), averaged to -0.2 |
| Outlier cause | Early topology lock-in (`no_bath_kitchen` stuck) | High variance regime, poor final-step luck |
| Averaged tables vs reality | Smooth increase is an artifact of bimodal ESS | Smooth U-shape is an artifact of averaging chaos |

### Option C — Reward-Attributed Confidence Boosting

> Status: **implemented** in `bd_gen/guidance/guided_sampler.py` (`_compute_attribution_boost` helper + integration in the SVDD loop). Enable via `attribution_boost=True` in `guided_sample()` or `--attribution-boost` CLI flag in `scripts/generate_guided.py`. Only effective when confidence remasking is active.

#### Problem

Confidence remasking (ReMDM) selects positions to re-mask based on model confidence: lowest-confidence tokens are most likely to be remasked. SVDD guidance selects the winning candidate based on reward, but the winning tokens at just-unmasked positions may have **low model confidence** — the model is uncertain about them, which is precisely why they differed across candidates and why guidance could steer them. Remasking then preferentially destroys these guided positions, undoing the guidance gain.

The trajectory analysis confirms this: remasking delta is negative at 98/100 steps for all α values, with individual-sample spikes reaching -1 to -5 reward units.

#### Mechanism

At each denoising step, after SVDD selects a winner from K candidates:

1. **Identify the affected set** $\mathcal{U}_t$ = positions that were masked before this step and are now unmasked (i.e., positions where the winner placed a token). Only these positions have meaningful attributions — already-committed positions carry the same token across all K candidates, giving $a_l = 0$ trivially.

2. **Compute per-position reward attribution.** For each position $l \in \mathcal{U}_t$:
   - Partition the K candidates into those that **match** the winner's token at position $l$ vs those that **don't**.
   - $a_l = \bar{r}_{\text{match}}^l - \bar{r}_{\text{all}}$

   If $a_l > 0$: candidates sharing the winner's token at this position tend to have higher rewards — the token is **reward-aligned**, and SVDD likely selected the winner partly because of it. If $a_l \approx 0$: the token doesn't correlate with reward and ended up in the winner incidentally.

3. **Boost effective confidence** at reward-attributed positions:

$$\text{conf}_{\text{eff}}^l = \begin{cases} \text{conf}^l + \beta \cdot \max(a_l, 0) & \text{if } l \in \mathcal{U}_t \\ \text{conf}^l & \text{otherwise} \end{cases}$$

Use $\text{conf}_{\text{eff}}$ in place of $\text{conf}$ when computing remasking probabilities ($\text{softmax}(-\text{confidence})$). Positions with high reward attribution become less likely to be remasked.

#### Self-calibrating β

The attribution $a_l$ lives on the **reward scale** while confidence lives on **[0, 1]**. A fixed β cannot bridge these scales across steps because reward spread varies step-to-step (early steps: lots of masked tokens → large reward variance; late steps: small variance) and the statistical reliability of the attribution depends on K.

The formula:

$$\beta(K, \{r_k\}) = \frac{K}{K + K_0} \cdot \frac{1}{\sigma_r + \epsilon}$$

where:
- $\sigma_r = \text{std}(r_1, \ldots, r_K)$ — reward standard deviation across the K candidates, computed per batch element per step
- $K_0 = 4$ — a constant (not a tuning knob; justified below)
- $\epsilon$ — numerical stability constant (e.g. $10^{-8}$)

**$1 / \sigma_r$ — scale normalization.** The maximum meaningful attribution is $a_l \sim O(\sigma_r)$ (a position whose token perfectly predicts the reward). Dividing by $\sigma_r$ converts the attribution into a z-score-like quantity on the [0, ~1] scale, commensurate with confidence values. When guidance has little effect (all candidates score similarly, $\sigma_r \to 0$), attributions also go to zero, so the product $\beta \cdot a_l = a_l / \sigma_r$ remains well-behaved. Guard: if $\sigma_r < \epsilon$, set $\beta = 0$ (no guidance discrimination → nothing to protect).

**$K / (K + K_0)$ — statistical trust ramp.** The attribution is a difference-of-means estimator with ~K samples split into two groups. Its standard error is $\sim 2\sigma_r / \sqrt{K}$ in the balanced case. This factor is a sigmoid in K that shrinks the boost when K is too small for the estimate to be reliable:

| K | $K/(K+K_0)$ | Interpretation |
|---|---|---|
| 1 | 0.20 | Almost no trust (can't split into groups) |
| 4 | 0.50 | Moderate — 2-vs-2 split is barely informative |
| 8 | 0.67 | Reasonable |
| 16 | 0.80 | High confidence |
| 64 | 0.94 | Near-full trust |

**Why $K_0 = 4$.** It is the minimum K at which a binary partition (match vs non-match) can have at least 2 samples in each group, giving the difference-of-means test minimal statistical power. This is a combinatorial floor, not a tuning parameter.

#### Resulting behavior

The effective boost at position $l$ is:

$$\beta \cdot \max(a_l, 0) = \frac{K}{K + 4} \cdot \frac{\max(a_l, 0)}{\sigma_r}$$

This is the **standardized attribution, attenuated by a K-dependent trust factor**:

- A position with $a_l = \sigma_r$ (strong reward correlation, ~1 std above mean) gets a boost of $K/(K+4)$ — from 0.5 at K=4 to 0.8 at K=16.
- A position with $a_l \approx 0$ (incidental) gets essentially zero — remasking proceeds normally.

**Verification against the worked example** (§15.4.3 in the overview): $\sigma_r \approx 0.41$, K=3, $a_4 = 0.5$:

$$\beta \cdot a_4 = \frac{3}{7} \cdot \frac{0.5}{0.41} \approx 0.52$$

So $\text{conf}_{\text{eff}}^4 = 0.35 + 0.52 = 0.87$ — comparable to committed-node confidence, no longer the lowest-confidence position. This matches the manually-chosen result ($\beta=1$ giving 0.85) but is now derived automatically.

#### Edge cases

- **$\sigma_r = 0$** (all candidates score identically): $a_l = 0$ for all positions. Guard: set $\beta = 0$, skip boosting.
- **K < 2**: attribution undefined (can't form groups). Guard: skip boosting.
- **Outlier reward**: $\sigma_r$ absorbs it naturally, preventing one extreme candidate from causing outsized boosts.

#### Properties

- **Zero extra model calls.** Computation is $O(K \times B \times \text{SEQ\_LEN})$ — comparison and averaging over candidates, negligible vs the model call.
- **Fully adaptive.** $\sigma_r$ is computed from actual rewards at each step, so the boost scales down when guidance has little effect and scales up when guidance is discriminative.
- **No hyperparameter sweep.** The only constant ($K_0 = 4$) is a combinatorial fact about minimum group sizes.
- **Graceful degradation at low K.** The trust ramp ensures noisy attributions from small K are attenuated rather than amplified.
- **Compatible with Option B** (protect all just-unmasked positions for 1 step). Option B is a blunt safety net; Option C adds surgical precision on top.

### Round 4 — Remasking × Reward-mode grid at K=16, α=0.01 (2026-03-05)

#### Motivation

Round 3 established α=0.01 as optimal for no-remasking (69% satisfaction) and showed confidence remasking fighting guidance at 98/100 steps. Option C (Reward-Attributed Confidence Boosting) was implemented to mitigate this conflict by boosting effective confidence of reward-aligned positions before remasking. Round 4 tests whether Option C rescues confidence remasking, and also whether hard reward mode (argmax-decoded scoring) outperforms soft reward mode (probability-based scoring).

#### Setup

- **Fixed**: K=16, α=0.01, v1 loglinear checkpoint, revised 4-constraint set, Option C (RACB) enabled for confidence configs
- **Grid**: {no-remasking, confidence+RACB} × {soft reward, hard reward} = 4 configs
- **Samples**: 3 seeds (42, 123, 456) × 200 samples = 600 per config
- **Tags**: `r4soft` / `r4hard` to differentiate reward modes
- **Script**: `run_g5_round4.sh`

#### Results — Satisfaction (all)

| Remasking | Soft reward | Hard reward |
|-----------|-----------|-----------|
| No-remasking | **69%** | 35% |
| Confidence + RACB | 56% | 36% |

#### Per-constraint breakdown

| Config | one_kitchen | kitchen_near_living | no_bath_kitchen | between_2_3_bath |
|--------|------------|-------------------|----------------|-----------------|
| no-remask soft | 100% | 100% | 74.7% | 93.0% |
| no-remask hard | ~97% | ~97% | ~50% | ~73% |
| confidence+RACB soft | ~98% | ~99% | ~79% | ~78% |
| confidence+RACB hard | ~96% | ~97% | ~54% | ~72% |

#### Findings

1. **Option C (RACB) failed to rescue confidence remasking.** Overall satisfaction dropped from 69% (no-remask) to 56% (confidence+RACB), a 13pp regression. The confidence boost was insufficient to prevent the guidance-remasking conflict.

2. **Soft reward mode dominates hard reward mode.** Consistent ~20pp gap: 69% vs 35% (no-remask), 56% vs 36% (confidence+RACB). Hard reward mode collapses because argmax decoding at intermediate steps produces noisy, discrete scores that don't differentiate candidates well. Soft mode uses the full posterior distribution, providing smoother gradients for the importance weights.

3. **ForbidAdj is worst case for Option C.** The `no_bath_kitchen` constraint actually improved slightly with RACB (+2.3pp vs the un-boosted confidence baseline from Round 3), but `between_2_and_3_bathrooms` degraded (-15pp vs no-remask). The net effect was negative because RACB's boost wasn't strong enough to prevent remasking of guided positions.

#### Why Option C failed — mechanism analysis

The root cause is that **additive confidence boosting before `softmax(-confidence)` gets washed out by softmax normalization**.

The remasking probability for position $l$ is:

$$p_{\text{remask}}^l = \frac{\exp(-\text{conf}_{\text{eff}}^l)}{\sum_j \exp(-\text{conf}_{\text{eff}}^j)}$$

Adding a boost $\Delta$ to position $l$'s confidence reduces its numerator by a factor of $\exp(-\Delta)$, but the **denominator** (sum over all positions) barely changes because the boost only applies to the few just-unmasked positions in $\mathcal{U}_t$. For a typical sequence of 120 positions where $|\mathcal{U}_t| \approx 2$–5, boosting 2–5 positions leaves 115+ positions unchanged. The remasking probability for the boosted position decreases, but not to zero — it is merely *reduced*, not eliminated.

The trajectory diagnostics confirm undamped oscillation:
- **Remasking delta**: oscillates wildly between -2 and +1 (soft) / -3 and +2 (hard) across all 100 steps, never narrowing
- **ESS**: spikes to K (degenerate) intermittently — same chaotic pattern as un-boosted confidence
- **Per-constraint violations**: never converge; constraints flip between satisfied and violated throughout denoising
- **Reward trajectory**: chaotic, non-monotonic — no sustained improvement

In contrast, the no-remasking baseline shows monotonic reward improvement, clean ESS phase transition, and permanent constraint resolution.

#### Verdict

**Option C is insufficient as a standalone mitigation.** The additive boost mechanism cannot prevent remasking of guided positions — it can only reduce the probability. Because remasking happens at every step (98/100 steps show negative delta), even a small probability of remasking a guided position compounds over 100 steps into near-certain destruction of guidance gains.

**Soft reward mode is confirmed as the only viable scoring approach.** Hard reward mode should not be used.

**No-remasking at α=0.01, K=16 remains the best configuration** at 69% overall satisfaction. To improve further, either:
- **(a) Option B** — protect just-unmasked positions from remasking for 1 step (zero cost, directly breaks the same-step feedback loop)
- **(b) Option B+C** — combine protection with surgical boosting for multi-step protection
- **(c) Accept no-remasking** as the production configuration and move to v2 variants

### Option B — Protect Just-Unmasked Positions

> Status: **implemented** in `bd_gen/guidance/guided_sampler.py` and `bd_gen/diffusion/remasking.py`. Enable via `protect_just_unmasked=True` in `guided_sample()` or `--protect-just-unmasked` CLI flag.

#### Problem

At each denoising step, SVDD selects a winner from K candidates, placing tokens at just-unmasked positions. Confidence remasking then immediately evaluates these freshly placed tokens using logits from the *pre-unmask* state — logits that don't account for the new token context. This same-step feedback loop means guided tokens are evaluated with stale confidence and systematically remasked.

#### Mechanism

After SVDD selects the winning candidate at step $t$:

1. **Identify just-unmasked positions**: $\mathcal{P}_t = \{l : \text{was MASK before step } t, \text{now decoded}\}$
2. **Exclude from remasking candidates**: Pass `protect_mask` to the remasking schedule. Positions in $\mathcal{P}_t$ are excluded from the candidate set for this step only — they become eligible for remasking at step $t+1$.

This directly breaks the same-step feedback loop: positions placed by guidance cannot be immediately undone by remasking using stale confidence values.

#### Properties

- **Zero cost.** No extra model calls, no extra computation beyond a boolean mask comparison.
- **No hyperparameters.** Binary: either protect for 1 step or don't.
- **Strategy-agnostic.** Works with cap, rescale, and confidence remasking strategies.
- **Composable.** Orthogonal to Option A (fresh logits) and Option C (confidence boosting). Can be combined with either or both.
- **Conservative.** Only prevents same-step remasking. If a position is truly wrong, it will be remasked at the next step when confidence is computed from the updated context.

#### CLI usage

```bash
python scripts/generate_guided.py \
    eval.checkpoint_path=path/to/ckpt.pt \
    --guidance-config configs/guidance/example.yaml \
    --protect-just-unmasked
```

### Option A — Fresh Logits for Remasking

> Status: **implemented** in `bd_gen/guidance/guided_sampler.py`. Enable via `fresh_logits_for_remask=True` in `guided_sample()` or `--fresh-logits-remask` CLI flag.

#### Problem

Confidence remasking computes per-position confidence from model logits. In the standard pipeline, these logits come from the model call *before* unmasking — they reflect the pre-transition state. After SVDD selects a winner and places new tokens, the context has changed, but confidence values are stale. This mismatch causes remasking to poorly target truly incorrect positions.

#### Mechanism

After SVDD selects the winning candidate at step $t$:

1. **Re-run the model** on the post-unmask winner: `fresh_logits = model(x_t_winner, pad_mask, t)`
2. **Use fresh logits** for the remasking decision instead of the stale pre-unmask logits.

The fresh logits reflect the actual post-transition context, so confidence values are accurate. Positions the model is genuinely uncertain about (given the current context) are remasked; positions that are confident in context — including those placed by guidance — are more likely to be preserved.

#### Properties

- **2× model cost per guided step.** One call for SVDD candidate generation, one for fresh logits before remasking. This doubles the neural network inference cost.
- **No hyperparameters.** The fresh logits are used as-is; no tuning needed.
- **Gold standard for confidence accuracy.** The remasking decision uses the most up-to-date information available.
- **Strategy-agnostic.** Works with any remasking strategy that uses logits (confidence, and potentially future strategies).
- **Composable.** Orthogonal to Option B (protect mask) and Option C (confidence boosting). Can be combined with either or both.

#### CLI usage

```bash
python scripts/generate_guided.py \
    eval.checkpoint_path=path/to/ckpt.pt \
    --guidance-config configs/guidance/example.yaml \
    --fresh-logits-remask
```

### Combining Options A, B, and C

All three remasking mitigation options are orthogonal and can be combined:

| Option | What it changes | Cost | CLI flag |
|---|---|---|---|
| A (Fresh logits) | Which logits remasking uses | 2× model calls | `--fresh-logits-remask` |
| B (Protect mask) | Which positions are eligible for remasking | Zero | `--protect-just-unmasked` |
| C (RACB) | Confidence scores used for remasking | Negligible | `--attribution-boost` |

Possible combinations for Round 5 experiments:
- **B alone**: Zero cost, breaks same-step feedback loop
- **A alone**: 2× cost, gold standard confidence accuracy
- **B+C**: Zero cost protection + surgical boosting
- **A+B**: Fresh logits + same-step protection (should be strongest)
- **A+B+C**: All three (maximum protection, 2× cost)

### Round 5 — Option A vs Option B at K=16, α=0.01 (2026-03-05)

#### Motivation

Round 4 showed that Option C (RACB) alone is insufficient to rescue confidence remasking: 56% overall satisfaction vs 69% for no-remasking. The additive confidence boost gets washed out by softmax normalization, and the guidance-remasking conflict persists at 98/100 steps.

Round 5 tests the two remaining mitigation options individually:
- **Option A** (fresh logits): Re-run the model on the post-unmask winner to get accurate confidence values for remasking. 2× model cost per step.
- **Option B** (protect just-unmasked): Exclude just-unmasked positions from remasking for 1 step. Zero cost.

Both use confidence remasking + soft reward (hard reward was conclusively ruled out in Round 4).

#### Setup

- **Fixed**: K=16, α=0.01, v1 loglinear checkpoint, revised 4-constraint set, soft reward only
- **Grid**: 2 configs (confidence + Option A, confidence + Option B)
- **Samples**: 3 seeds (42, 123, 456) × 200 samples = 600 per config
- **Tags**: `r5optA` / `r5optB`
- **Script**: `run_g5_round5.sh`

#### Comparison targets (from Round 4, not re-generated)

| Config | Overall satisfaction | Source |
|--------|---------------------|--------|
| No-remasking + soft | 69% | Round 4 |
| Confidence + Option C + soft | 56% | Round 4 |

#### Expected outcomes

- **Option B**: Should break the same-step feedback loop (protect guided tokens for 1 step). If the conflict is primarily same-step, this should close most of the gap to no-remasking. If multi-step compounding dominates, improvement will be modest.
- **Option A**: Gold standard — fresh logits give the model a chance to evaluate guided tokens in context. Should be the strongest mitigation, but at 2× cost. Provides an upper bound on what informed remasking can achieve.

#### Results

> Pending — run `bash scripts/run_g5_round5.sh all` on jabiru.
