# Dataset (`bd_gen.data.dataset`)

## Purpose

`BubbleDiagramDataset` is the PyTorch `Dataset` that the training loop consumes. It loads graphs, tokenizes them all upfront, splits them into train/val/test, and precomputes class weights and room-count statistics needed by the loss function and sampler.

---

## Quick start

```python
from bd_gen.data.dataset import BubbleDiagramDataset
from bd_gen.data.vocab import RPLAN_VOCAB_CONFIG

# Create the training split
train_ds = BubbleDiagramDataset(
    mat_path="data/data.mat",
    cache_path="data_cache/graph2plan_nmax8.pt",
    vocab_config=RPLAN_VOCAB_CONFIG,
    split="train",       # "train", "val", or "test"
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    seed=42,             # deterministic splits
)

print(len(train_ds))  # ~64,630 (80% of 80,788)

# Access a single sample
sample = train_ds[0]
sample["tokens"]     # shape (36,), dtype torch.long
sample["pad_mask"]   # shape (36,), dtype torch.bool (True = real position)
sample["num_rooms"]  # int, e.g. 6
```

---

## What happens during `__init__`

1. **Load graphs** via `load_graph2plan()` (uses cache if available)
2. **Tokenize all graphs** into `(tokens, pad_mask)` tensors — fits in ~26 MB RAM
3. **Split deterministically** using `torch.randperm` with a fixed seed
4. **Compute statistics** from the training split only:
   - `edge_class_weights` — for the loss function
   - `node_class_weights` — for the loss function (optional, not used in v1)
   - `num_rooms_distribution` — for the sampler during generation

Steps 1-4 take ~5-10 seconds on first run, <1 second when the `.pt` cache exists.

---

## Splits

The dataset is shuffled with a seeded RNG and divided into three non-overlapping splits:

| Split | Default fraction | RPLAN size |
|-------|-----------------|------------|
| train | 80% | ~64,630 |
| val   | 10% | ~8,078 |
| test  | 10% | ~8,080 |

**Determinism**: The same `seed` always produces the same split. Different seeds produce different splits. This is verified by tests.

To get all three splits, create three dataset instances with the same arguments but different `split` values. They share the same underlying data load (via the `.pt` cache) and produce disjoint subsets.

---

## Class weights

Class weights compensate for imbalanced token frequencies in the training data, so the loss function doesn't ignore rare room types or edge types.

### Formula

For each class `c` in the vocabulary:

```
weight[c] = total_non_pad_tokens / (vocab_size * count[c])
```

- `count[c]` = how many non-PAD positions have value `c` in the training set
- `total_non_pad_tokens` = sum of all counts
- Classes that never appear (PAD, MASK) get weight 0

This is standard **inverse-frequency weighting**, producing weights where rare classes get higher weight and common classes get lower weight, with mean approximately 1.0.

### Accessing weights

```python
train_ds.edge_class_weights   # shape (13,), dtype float32
train_ds.node_class_weights   # shape (15,), dtype float32
```

These are `None` for val/test splits (weights should only come from training data).

### Example: passing to loss

```python
loss_fn = ELBOLoss(
    edge_class_weights=train_ds.edge_class_weights,
    node_class_weights=None,  # not used in v1
)
```

---

## Room count distribution

A normalised histogram of how many rooms each training graph has:

```python
train_ds.num_rooms_distribution  # shape (8,), dtype float32, sums to 1.0
```

Index `k` = probability of a graph having `k + 1` rooms. For RPLAN:

| Index | Rooms | Approx. probability |
|------:|------:|-------------------:|
| 0 | 1 | 0.0% |
| 1 | 2 | 0.0% |
| 2 | 3 | 0.0% |
| 3 | 4 | 0.3% |
| 4 | 5 | 7.2% |
| 5 | 6 | 31.1% |
| 6 | 7 | 36.2% |
| 7 | 8 | 25.2% |

This is used during generation to sample realistic room counts:

```python
num_rooms = torch.multinomial(train_ds.num_rooms_distribution, 1).item() + 1
```

This is `None` for val/test splits.

---

## DataLoader usage

The dataset works with standard PyTorch DataLoaders:

```python
from torch.utils.data import DataLoader

loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)

for batch in loader:
    tokens   = batch["tokens"]     # (256, 36)
    pad_mask = batch["pad_mask"]   # (256, 36)
    num_rooms = batch["num_rooms"] # (256,)
    # ... pass to model
```

PyTorch's default collation automatically stacks the tensors into batches.

---

## Where this fits in the pipeline

```
data.mat  -->  loader  -->  graph dicts  -->  tokenizer  -->  tensors
                                                                |
                                                          BubbleDiagramDataset
                                                          (stores tensors, splits,
                                                           class weights, distribution)
                                                                |
                                                          DataLoader  -->  training loop
```
