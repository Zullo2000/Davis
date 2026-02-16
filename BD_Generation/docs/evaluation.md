# Evaluation Module

Phase 5 of the BD Generation pipeline. Provides validity checking, generation quality metrics, visualization, and evaluation scripts.

## Architecture

```
bd_gen/eval/
├── validity.py     # Graph validity checker (connectivity, constraints, sanity)
└── metrics.py      # Quality metrics (validity rate, novelty, diversity, distribution match)

bd_gen/viz/
└── graph_viz.py    # Bubble diagram visualization (networkx + matplotlib)

scripts/
├── sample.py       # Generate and visualize samples from checkpoint
└── evaluate.py     # Full evaluation pipeline with metrics and wandb logging
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
- **`consistent` check simplified** — the upper-triangle token format stores each (i,j) pair once, preventing directional contradictions. Full spatial transitivity checking deferred to v2.

## Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Validity rate | `validity_rate(results)` | Fraction of samples passing all checks |
| Diversity | `diversity(graph_dicts)` | Unique graphs / total (hash-based) |
| Novelty | `novelty(samples, training_set)` | Fraction not in training set (exact match) |
| Distribution match | `distribution_match(samples, training_set)` | KL divergence of node/edge/num_rooms histograms |
| Per-class accuracy | `per_class_accuracy(preds, targets, mask)` | Training-time accuracy per class |
| Graph structure MMD | `graph_structure_mmd(samples, reference)` | MMD on degree, clustering, spectral features |

### Graph Structure MMD (DiGress / MELD / GraphARM Protocol)

`graph_structure_mmd(samples, reference, n_max=8)` computes MMD^2 with Gaussian RBF kernel for three topology-only statistics:

| Sub-metric | Feature | What it catches |
|------------|---------|----------------|
| `mmd_degree` | Normalized degree histogram per graph | Connectivity density |
| `mmd_clustering` | Binned clustering coefficient histogram | Local cliquishness |
| `mmd_spectral` | Normalized Laplacian eigenvalues | Global graph shape |

Lower values = generated graphs are more structurally similar to the reference set. Uses the median heuristic for kernel bandwidth if `sigma` is not specified.

These metrics are **topology-only** (edge types ignored), complementing `distribution_match` which captures node/edge type distributions.

### Novelty Implementation

Uses exact-match novelty (hash-based, O(N+M)) instead of graph edit distance. GED is NP-hard and impractical at scale (1000 samples x 64K training set). Each graph is hashed as `(num_rooms, tuple(node_types), tuple(sorted(edge_triples)))`.

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
    wandb.mode=disabled
```

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
from bd_gen.eval import check_validity, validity_rate, novelty, diversity, graph_structure_mmd
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

# Visualize
fig = draw_bubble_diagram(graph_dict, title="Sample 1")
fig.savefig("sample.png")
```

## Configuration

`configs/eval/default.yaml`:

```yaml
num_samples: 1000        # Total samples to generate
sampling_steps: 100      # Denoising steps per sample
temperature: 0.0         # 0 = argmax, >0 = stochastic
metrics: [validity, novelty, diversity, distribution_match, graph_structure_mmd]
checkpoint_path: null     # Required: set via CLI
batch_size: 64           # Samples per generation batch
save_samples: true       # Save tokens to .pt
visualize: true          # Save visualization grid
num_viz_samples: 16      # How many to visualize
```

## Testing

```bash
# All evaluation tests
pytest tests/test_validity.py tests/test_metrics.py -v

# Specific test
pytest tests/test_validity.py::TestConnectivity -v
```
