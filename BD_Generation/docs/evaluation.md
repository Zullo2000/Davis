# Evaluation Module

Phase 5 of the BD Generation pipeline. Provides validity checking, generation quality metrics, visualization, and evaluation scripts.

## Architecture

```
bd_gen/eval/
├── validity.py         # Graph validity checker (connectivity, constraints, sanity)
├── metrics.py          # Quality metrics (validity rate, novelty, diversity, distribution match, JS/TV/W1)
└── denoising_eval.py   # Sampler-independent model quality metrics (accuracy, cross-entropy, ELBO)

bd_gen/viz/
└── graph_viz.py    # Bubble diagram visualization (networkx + matplotlib)

scripts/
├── sample.py       # Generate and visualize samples from checkpoint
└── evaluate.py     # Full evaluation pipeline with multi-seed, metrics, and wandb logging
```

## Validity Checks

`check_validity(tokens, pad_mask, vocab_config)` runs five checks:

| Check | What it verifies |
|-------|-----------------|
| `no_mask_tokens` | No MASK tokens remain in real (non-PAD) positions |
| `no_out_of_range` | Node values in [0,12], edge values in [0,10] |
| `connected` | BFS from node 0 reaches all rooms via spatial edges |
| `consistent` | All edge types are valid spatial relationships |
| `valid_types` | At most 1 LivingRoom, at most 1 Entrance |

`overall` is the AND of all checks. Single-room graphs are trivially connected.

### Spec Deviations

- **`vocab_config` parameter added** to `check_validity` signature (spec omitted it, but `detokenize` requires it — same precedent as Phase 3).
- **`consistent` check simplified** — the upper-triangle token format stores each (i,j) pair once, preventing directional contradictions. Cross-edge spatial consistency is now covered by the `spatial_transitivity` metric (see below).

## Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Validity rate | `validity_rate(results)` | Fraction of samples passing all checks |
| Diversity | `diversity(graph_dicts)` | Unique graphs / total (hash-based) |
| Novelty | `novelty(samples, training_set)` | Fraction not in training set (exact match) |
| Distribution match | `distribution_match(samples, training_set)` | KL/JS/TV/W1 of node/edge/num_rooms histograms |
| Per-class accuracy | `per_class_accuracy(preds, targets, mask)` | Training-time accuracy per class |
| Conditional edge KL/JS/TV | `conditional_edge_kl(samples, training_set)` | KL/JS/TV of edge types conditioned on room-type pair |
| Conditional edge top-N | `conditional_edge_distances_topN(samples, training_set)` | KL/JS/TV for top-N most frequent canonical pairs |
| Graph structure MMD | `graph_structure_mmd(samples, reference)` | MMD on degree, clustering, spectral features |
| Spatial transitivity | `spatial_transitivity(graph_dicts)` | Fraction of graphs with 2D-consistent spatial relationships |
| Type-conditioned degree KL/JS/TV | `type_conditioned_degree_kl(samples, reference)` | KL/JS/TV of node degree per room type |
| Mode coverage | `mode_coverage(samples, training_set)` | Fraction of training room-type archetypes covered |
| Denoising accuracy | `denoising_eval(model, dataloader, ...)` | Sampler-independent accuracy and CE at noise levels |
| Validity by num_rooms | `validity_by_num_rooms(validity_results, graph_dicts)` | Stratified validity breakdown |
| Transitivity by num_rooms | `spatial_transitivity_by_num_rooms(graph_dicts)` | Stratified spatial transitivity |
| Edge-present rate by num_rooms | `edge_present_rate_by_num_rooms(graph_dicts)` | Stratified edge density |

### Conditional Edge KL

`conditional_edge_kl(samples, training_set, min_pair_count=5)` measures whether the model uses appropriate spatial relationships for each specific combination of room types. Unlike the marginal `edge_kl` in `distribution_match` (which counts all relationships globally), this metric computes the edge-type distribution separately for each canonical room-type pair — e.g., (LivingRoom, Bathroom) or (Kitchen, Balcony) — and then computes KL(sample || training) per pair.

**Canonicalization.** Edges are stored as `(i, j, rel)` with `i < j` in node-index space, but the same spatial fact can appear with different node orderings across graphs. To compare by room type, we canonicalize: if `type_i <= type_j`, the canonical pair is `(type_i, type_j)` with `rel` unchanged; if `type_i > type_j`, the pair is `(type_j, type_i)` with `rel = 9 - rel` (applying the spatial inverse).

**Return values:**

| Key | Meaning |
|-----|---------|
| `conditional_edge_kl_mean` | Unweighted mean KL across all eligible room-type pairs. Treats rare and common pairs equally. |
| `conditional_edge_kl_weighted` | Training-frequency-weighted mean KL. Common pairs (e.g., LivingRoom–Bathroom) contribute more. |
| `num_pairs_evaluated` | Number of canonical pairs included in the comparison. |

**What it catches that `edge_kl` misses.** A model could have a correct marginal edge distribution while consistently applying the wrong relationship for specific room pairs (e.g., "Bathroom surrounding LivingRoom" instead of "inside"). `conditional_edge_kl` detects this.

**Sparse pair filtering.** Pairs with fewer than `min_pair_count` edges in training are excluded — their empirical histogram is too noisy to serve as a reliable reference.

```python
from bd_gen.eval import conditional_edge_kl

cekl = conditional_edge_kl(graph_dicts, training_dicts, min_pair_count=5)
print(cekl["conditional_edge_kl_mean"])      # lower is better
print(cekl["conditional_edge_kl_weighted"])  # lower is better
print(int(cekl["num_pairs_evaluated"]))      # how many pairs were compared
```

### Graph Structure MMD (DiGress / MELD / GraphARM Protocol)

`graph_structure_mmd(samples, reference, n_max=8)` measures whether generated graphs have realistic *topology* — i.e., whether the connectivity patterns (who is connected to whom) resemble the training data, ignoring what the edge types are. This follows the standard evaluation protocol from DiGress, MELD, and GraphARM.

**How it works.** Each graph is converted to an undirected networkx graph (edge types discarded, only adjacency kept). Three structural features are extracted per graph, then compared between the generated set and the reference set using Maximum Mean Discrepancy (MMD) with a Gaussian RBF kernel. MMD measures the distance between two distributions of feature vectors: 0 means the distributions are identical, higher means more different.

**The three sub-metrics:**

| Sub-metric | How it's computed | What it measures | Example of what it catches |
|------------|-------------------|------------------|---------------------------|
| `mmd_degree` | For each graph, compute a histogram of node degrees (how many connections each room has), normalized by the number of rooms and zero-padded to length `n_max`. | Whether rooms have realistic numbers of connections. | A model that generates star-shaped layouts (one central room connected to all others) would have a very different degree distribution than real floorplans where rooms typically connect to 2–4 neighbors. |
| `mmd_clustering` | For each node, compute the clustering coefficient (fraction of its neighbors that are also connected to each other), then bin all node coefficients into a 10-bin histogram over [0, 1]. | Whether rooms form realistic local groupings — e.g., in real floorplans, a bathroom, bedroom, and hallway might all be mutually adjacent, forming a triangle. | A model that generates tree-like layouts (no triangles) would score 0.0 clustering everywhere, while real floorplans have clusters of mutually-adjacent rooms. |
| `mmd_spectral` | Compute the eigenvalues of the normalized graph Laplacian, sorted in ascending order and zero-padded to length `n_max`. | The overall shape and connectivity structure of the graph. The smallest eigenvalues capture large-scale properties like how many connected components exist, while larger eigenvalues capture finer structural details. | A model that always generates fully-connected graphs or disconnected subgraphs would have a very different spectral signature from real floorplans. |

**Interpretation.** Lower values = generated graphs are structurally more similar to real ones. These metrics are **topology-only** (edge types are ignored), which complements `distribution_match` that captures the *type* distributions (which room types appear, which spatial relationships occur) but not the connectivity structure.

**Kernel bandwidth.** If `sigma` is not specified, the median heuristic is used: sigma is set to `sqrt(median(pairwise_distances) / 2)` over the combined sample+reference feature vectors. This adaptive choice avoids manual tuning.

### Novelty Implementation

Uses exact-match novelty (hash-based, O(N+M)) instead of graph edit distance. GED is NP-hard and impractical at scale (1000 samples x 64K training set). Each graph is hashed as `(num_rooms, tuple(node_types), tuple(sorted(edge_triples)))`.

### Spatial Transitivity

`spatial_transitivity(graph_dicts)` detects bubble diagrams that are structurally valid yet **physically impossible** — graphs where the spatial relationships cannot coexist in any 2D layout.

**Why this matters.** A graph can pass all five validity checks (connected, correct types, no MASK residue, in-range values, no duplicates) and still be unrealizable. Consider three rooms where A is left-of B, B is left-of C, and C is left-of A. Each individual edge is a valid spatial relationship, the graph is connected, and all types are legal — but no 2D arrangement satisfies all three constraints simultaneously. This is the most dangerous class of generation failures: outputs that *look* valid by every existing metric but can never become a real floorplan.

**How it works.** Each of the 10 spatial relationship types is decomposed into horizontal (H) and vertical (V) ordering constraints:

| Rel | Name | H-constraint (x-axis) | V-constraint (y-axis) |
|-----|------|-----------------------|-----------------------|
| 0 | left-above | A.x < B.x | A.y > B.y |
| 1 | left-below | A.x < B.x | A.y < B.y |
| 2 | left-of | A.x < B.x | — |
| 3 | above | — | A.y > B.y |
| 4 | inside | — | — |
| 5 | surrounding | — | — |
| 6 | below | — | A.y < B.y |
| 7 | right-of | A.x > B.x | — |
| 8 | right-above | A.x > B.x | A.y > B.y |
| 9 | right-below | A.x > B.x | A.y < B.y |

Two directed graphs are built — one for horizontal ordering, one for vertical. An edge A→B in the H-graph means "A must be left of B." A directed cycle in either graph means no valid placement exists on that axis. Cycle detection uses DFS in O(N + E) time.

`inside` (4) and `surrounding` (5) are containment relationships. They do not impose strict positional ordering (a room can be inside another at any position), so they contribute no constraints to either axis.

**Return values:**

| Key | Meaning |
|-----|---------|
| `transitivity_score` | Fraction of graphs with no contradictions on either axis (higher is better). |
| `h_consistent` | Fraction with no horizontal contradictions. |
| `v_consistent` | Fraction with no vertical contradictions. |

**What it catches.** Any graph where spatial relationships form a contradictory ordering cycle. This is orthogonal to all other metrics: `connected` checks reachability (topology), `consistent` checks individual edge validity, `conditional_edge_kl` checks distributional accuracy — none detect cross-edge contradictions.

```python
from bd_gen.eval import spatial_transitivity

st = spatial_transitivity(graph_dicts)
print(f"Spatially realizable: {st['transitivity_score']:.1%}")
print(f"H-consistent: {st['h_consistent']:.1%}")
print(f"V-consistent: {st['v_consistent']:.1%}")
```

### Inside Validity

`inside_validity(graph_dicts)` checks whether generated bubble diagrams contain architecturally implausible containment relationships — e.g., a LivingRoom inside a Bathroom, or a Kitchen inside a Storage room.

**How it works.** For each graph, every edge with `rel=4` (inside) or `rel=5` (surrounding) is checked against a set of 69 forbidden `(A_type, B_type)` pairs. If `rel=4`, the pair is `(node_types[u], node_types[v])` — "u is inside v". If `rel=5`, the pair is `(node_types[v], node_types[u])` — "v is inside u". A single forbidden match flags the sample as invalid.

**Formula:** `inside_validity = count(samples with zero violations) / total_samples`

**The forbidden pairs** are defined in [inside_surrounding_rules.md](../inside_surrounding_rules.md). They encode size and functional constraints: large rooms should not be inside small rooms (LivingRoom inside Storage), and functionally incompatible containment should not occur (Kitchen inside Bathroom). The pair "MasterRoom inside LivingRoom" is **not** forbidden — it appears in 6.6% of RPLAN samples and represents a legitimate open-plan annotation pattern (see [vocab.md](vocab.md#note-on-masterroom-inside-livingroom)).

**RPLAN baseline:** 99.78% (176 violations out of 80,788 samples).

```python
from bd_gen.eval.metrics import inside_validity

score = inside_validity(graph_dicts)
print(f"Inside validity: {score:.1%}")  # higher is better
```

### Type-Conditioned Degree KL

`type_conditioned_degree_kl(samples, reference, n_max=8, min_type_count=20)` measures whether each room type has a realistic number of connections.

**Why this matters.** The global `mmd_degree` metric pools all room types into one histogram. A model could match the overall degree distribution while systematically giving bathrooms too many connections (e.g., 4-5 neighbors instead of the typical 1-2) and compensating by under-connecting living rooms. Since different room types have distinct connectivity patterns in real floorplans, this per-type decomposition catches errors that global metrics miss.

**How it works.** For each room type (LivingRoom, Bathroom, Kitchen, etc.):
1. Collect the degree of every node of that type across all graphs in the set
2. Build a normalized histogram over degree values 0 to n_max-1
3. Compute KL(samples || reference) per type

Room types with fewer than `min_type_count` nodes in the reference set are excluded — their histograms are too noisy to be reliable.

**Return values:**

| Key | Meaning |
|-----|---------|
| `degree_kl_per_type_mean` | Unweighted mean KL across eligible room types (lower is better). |
| `degree_kl_per_type_weighted` | Reference-frequency-weighted mean KL. Common types (e.g., SecondRoom) contribute more. |
| `num_types_evaluated` | Number of room types included in the comparison. |

**What it catches.** A model that produces structurally plausible graphs (correct global degree distribution) but assigns unrealistic connectivity to specific room types. For example, bathrooms that connect to 5 rooms, or balconies that serve as central hubs.

```python
from bd_gen.eval import type_conditioned_degree_kl

tcdkl = type_conditioned_degree_kl(graph_dicts, training_dicts, n_max=8)
print(f"Per-type degree KL (mean): {tcdkl['degree_kl_per_type_mean']:.4f}")
print(f"Per-type degree KL (weighted): {tcdkl['degree_kl_per_type_weighted']:.4f}")
print(f"Types evaluated: {int(tcdkl['num_types_evaluated'])}")
```

### Mode Coverage

`mode_coverage(samples, training_set)` measures what fraction of the training set's room-type *archetypes* the model reproduces.

**Why this matters.** `diversity` and `novelty` measure whether generated samples differ from each other and from training data, respectively. But neither detects **mode collapse** — a model could generate 1000 unique, novel graphs that all have the same basic room composition (e.g., always LivingRoom + Kitchen + 2 Bedrooms) while ignoring studios, large apartments, or configurations with balconies and guest rooms. Mode coverage directly measures whether the model has learned to produce the full variety of floorplan compositions.

**How it works.** An "archetype" is the sorted multiset of room types in a graph — it captures the *composition* of a floorplan regardless of how rooms are connected. For example, `(LivingRoom, Kitchen, Bathroom, MasterRoom, Balcony)` is one archetype.

1. Collect all distinct archetypes in the training set and their frequencies
2. Collect all distinct archetypes in the generated set
3. Compute the overlap

**Return values:**

| Key | Meaning |
|-----|---------|
| `mode_coverage` | Fraction of distinct training archetypes that appear at least once in samples (higher is better). |
| `mode_coverage_weighted` | Weighted by training frequency — measures what fraction of training *mass* is covered. A model covering only the 10 most common archetypes may still score high here. |
| `num_training_modes` | Total distinct archetypes in training (indicates how diverse the dataset is). |
| `num_sample_modes` | Total distinct archetypes in generated samples. |

**Interpreting the two variants.** Unweighted coverage is stricter — it penalizes missing any archetype, even rare ones. Weighted coverage tells you whether the model covers the "bulk" of the distribution. A large gap between weighted and unweighted coverage means the model generates common configurations well but misses rare ones.

```python
from bd_gen.eval import mode_coverage

mc = mode_coverage(graph_dicts, training_dicts)
print(f"Mode coverage: {mc['mode_coverage']:.1%}")
print(f"Mode coverage (weighted): {mc['mode_coverage_weighted']:.1%}")
print(f"Training has {int(mc['num_training_modes'])} distinct archetypes")
print(f"Samples produced {int(mc['num_sample_modes'])} distinct archetypes")
```

### JS/TV/W1 Distance Metrics

In addition to KL divergence (retained as a diagnostic), all distribution comparisons now include:

| Distance | Formula | Properties | Used for |
|----------|---------|------------|----------|
| **Jensen-Shannon (JS)** | `JS(p,q) = 0.5*KL(p\|\|m) + 0.5*KL(q\|\|m)`, `m=(p+q)/2` | Symmetric, bounded [0, ln(2)], stable on sparse histograms | Node/edge/conditional distributions |
| **Total Variation (TV)** | `TV(p,q) = 0.5 * sum\|p_k - q_k\|` | No logs, very stable, interpretable as "mass difference" | Node/edge/conditional distributions |
| **Wasserstein-1 (W1)** | `W1(p,q) = sum\|CDF_p(k) - CDF_q(k)\|` | Respects ordinal structure, penalizes "distance" between bins | num_rooms (ordinal) |

KL remains computed but is no longer the headline distribution metric. JS and TV are preferred for stability; W1 is used specifically for `num_rooms` where bin ordering matters.

### Denoising Evaluation (Model Quality)

`denoising_eval()` and `denoising_val_elbo()` evaluate the denoiser on held-out validation data, independent of the sampling procedure. See [denoising_eval.md](denoising_eval.md) for details.

### Multi-Seed Evaluation

The evaluation pipeline runs generation with multiple seeds (default: `[42, 123, 456, 789, 1337]`) and reports mean +/- std for each scalar metric. This replaces the previous single-seed approach and provides uncertainty quantification without bootstrapping.

Output JSON structure:
```json
{
  "meta": { "checkpoint": "...", "num_samples": 1000, "seeds": [42, 123, 456, 789, 1337] },
  "per_seed": { "42": {...}, "123": {...}, ... },
  "summary": { "eval/validity_rate": {"mean": 0.99, "std": 0.005}, ... },
  "denoising": { "denoise/acc_node@t=0.5": 0.85, ... }
}
```

### Stratified Drill-Down Metrics

Three metrics are stratified by `num_rooms`:

1. **Validity by num_rooms** — reveals if validity degrades for larger graphs
2. **Spatial transitivity by num_rooms** — checks if spatial contradictions increase with complexity
3. **Edge-present rate by num_rooms** — measures edge density per graph size

### Scoreboard Prefixes (wandb)

Metrics are logged with organized prefixes for wandb grouping:

| Prefix | Category |
|--------|----------|
| `denoise/*` | Sampler-independent model quality |
| `sampler/validity/*` | Validity checks |
| `sampler/coverage/*` | Diversity, novelty, mode coverage |
| `sampler/distribution/*` | JS, TV, W1, KL distances |
| `sampler/structure/*` | MMD, transitivity |
| `sampler/conditional/*` | Conditional edge distances, degree distances |

## Visualization

`draw_bubble_diagram(graph_dict)` renders a graph using networkx spring layout:
- Nodes colored by room type (13-color palette)
- Edges labeled with spatial relationship names
- Short labels for compact display (Liv, Mstr, Kit, Bath, etc.)

`draw_bubble_diagram_grid(graph_dicts, ncols=4)` renders a grid of diagrams.

## Usage

### Generate Samples

```bash
python scripts/sample.py eval.checkpoint_path=outputs/<timestamp>/checkpoints/checkpoint_final.pt

# With options
python scripts/sample.py \
    eval.checkpoint_path=path/to/checkpoint.pt \
    eval.num_samples=16 \
    eval.temperature=0.5 \
    eval.unmasking_mode=llada \
    wandb.mode=disabled
```

The `unmasking_mode` parameter controls how MASK positions are selected for unmasking at each step:
- `"random"` (default) — independent coin-flip per position (standard MDLM).
- `"llada"` — unmask highest-confidence positions first (LLaDA-style, Nie et al.). See [diffusion.md](diffusion.md) for details.

### Full Evaluation

```bash
python scripts/evaluate.py eval.checkpoint_path=path/to/checkpoint.pt

# Quick local test
python scripts/evaluate.py \
    eval.checkpoint_path=path/to/checkpoint.pt \
    eval.num_samples=100 \
    eval.batch_size=32 \
    wandb.mode=disabled
```

### Python API

```python
from bd_gen.eval import (
    check_validity, validity_rate, novelty, diversity,
    graph_structure_mmd, spatial_transitivity,
    type_conditioned_degree_kl, mode_coverage,
)
from bd_gen.viz import draw_bubble_diagram

# Check a single sample
result = check_validity(tokens, pad_mask, vocab_config)
print(result["overall"], result["connected"], result["valid_types"])

# Compute metrics on a list of graph dicts
v_rate = validity_rate(validity_results)
div = diversity(graph_dicts)
nov = novelty(graph_dicts, training_dicts)

# Graph structure MMD (topology-based, lower = better)
mmd = graph_structure_mmd(graph_dicts, training_dicts, n_max=8)
print(mmd["mmd_degree"], mmd["mmd_clustering"], mmd["mmd_spectral"])

# Spatial transitivity (higher = better)
st = spatial_transitivity(graph_dicts)
print(f"Realizable: {st['transitivity_score']:.1%}")

# Type-conditioned degree KL (lower = better)
tcdkl = type_conditioned_degree_kl(graph_dicts, training_dicts, n_max=8)
print(f"Per-type degree KL: {tcdkl['degree_kl_per_type_mean']:.4f}")

# Mode coverage (higher = better)
mc = mode_coverage(graph_dicts, training_dicts)
print(f"Mode coverage: {mc['mode_coverage']:.1%}")

# Visualize
fig = draw_bubble_diagram(graph_dict, title="Sample 1")
fig.savefig("sample.png")
```

## Sample Variance in Metrics

Evaluation metrics computed on finite sample sets have **inherent statistical variance**. A metric change between two runs does not necessarily indicate a model improvement or regression — it may simply be noise from a different random draw.

### Empirical variance (200 samples, 100 steps, temp=0.0, same checkpoint)

Five runs with different random seeds produced:

| Seed | Validity | Diversity | Node KL | Edge KL | Rooms KL |
|------|----------|-----------|---------|---------|----------|
| 42   | 98.5%    | 1.000     | 0.0498  | 0.2815  | 0.0041   |
| 123  | 99.5%    | 1.000     | 0.0473  | 0.2489  | 0.0117   |
| 456  | 100.0%   | 0.990     | 0.0487  | 0.3211  | 0.0099   |
| 789  | 99.5%    | 0.995     | 0.0508  | 0.3077  | 0.0072   |
| 1337 | 100.0%   | 1.000     | 0.0485  | 0.1931  | 0.0093   |

Edge KL ranges from **0.19 to 0.32** across seeds — a 1.7x difference from the same model. This variance comes from:

1. **`num_rooms` sampling** — different seeds draw different room counts, changing the edge budget
2. **Unmasking randomness** — `rand < p_unmask` decides which positions unmask at each step
3. **Small sample KL instability** — KL divergence between two empirical histograms is noisy when counts are low in some bins

### Guidelines for reliable comparisons

- **Use >= 1000 samples** (the eval config default) for stable KL-based metrics. At 200 samples, Edge KL variance is ~0.13; at 1000 samples it drops to ~0.03.
- **Fix the seed** (`seed: 42`) when comparing two model checkpoints to eliminate sampling randomness.
- **Report mean +/- std** over 3-5 seeds when claiming a metric improvement.
- **Validity, diversity, and novelty** are more stable at small sample sizes because they are based on counting (binary or hash-based), not distributional distances.
- **KL-based metrics** (Edge KL, Node KL, conditional edge KL, type-conditioned degree KL) are the most sensitive to sample size because they estimate probability distributions from finite counts.

## Configuration

`configs/eval/default.yaml`:

```yaml
num_samples: 1000        # Total samples to generate per seed
sampling_steps: 100      # Denoising steps per sample
temperature: 0.0         # 0 = argmax, >0 = stochastic
unmasking_mode: random   # "random" (MDLM) or "llada" (LLaDA-style)
metrics: [validity, novelty, diversity, distribution_match, conditional_edge_kl, graph_structure_mmd, spatial_transitivity, type_conditioned_degree_kl, mode_coverage]
checkpoint_path: null     # Required: set via CLI
batch_size: 64           # Samples per generation batch
save_samples: true       # Save tokens to .pt
visualize: true          # Save visualization grid
num_viz_samples: 16      # How many to visualize
remasking:
  enabled: false
  strategy: "cap"
  eta: 0.1
# Multi-seed
seeds: [42, 123, 456, 789, 1337]
# Conditional top-N pairs (null to disable)
conditional_topN_pairs: 20
# Stratified drill-down metrics
stratified: true
# Denoising eval (model quality)
run_denoising_eval: true
denoising_t_grid: [0.1, 0.3, 0.5, 0.7, 0.9]
denoising_max_batches: 50
```

## Testing

```bash
# All evaluation tests
pytest tests/test_validity.py tests/test_metrics.py -v

# Specific test
pytest tests/test_validity.py::TestConnectivity -v
```
