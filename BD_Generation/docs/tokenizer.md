# Tokenizer (`bd_gen.data.tokenizer`)

## Purpose

This module is the bridge between graph-level data and model-level tensors. It converts between the intermediate graph dictionary format (produced by the loader) and flat token sequences that the diffusion model operates on. It provides two functions: `tokenize` and `detokenize`.

---

## How it works

### Token sequence layout

A graph is flattened into a single 1D tensor of length `seq_len` (36 for RPLAN):

```
[ node_0, node_1, ..., node_7, edge_0, edge_1, ..., edge_27 ]
|<---- 8 node positions ---->|<----- 28 edge positions ----->|
```

- **Node positions `[0:8]`**: Each holds a room type index (0-12), or `NODE_PAD_IDX` (14) if that room slot is unused.
- **Edge positions `[8:36]`**: Each corresponds to a room pair `(i, j)` in upper-triangle order (see `vocab.md`). Can hold:
  - A relationship type (0-9) if the rooms are adjacent
  - `EDGE_NO_EDGE_IDX` (10) if both rooms exist but are not adjacent
  - `EDGE_PAD_IDX` (12) if one or both rooms don't exist in this graph

### The PAD vs. no-edge distinction

This is the most important invariant in the entire data pipeline:

| Token | When used | Model treatment |
|-------|-----------|-----------------|
| **Relationship** (0-9) | Both rooms exist AND are adjacent | Real data, included in loss |
| **no-edge** (10) | Both rooms exist but are NOT adjacent | Real data, included in loss |
| **PAD** (12) | One or both rooms don't exist | Ignored entirely (masked out) |

Getting this wrong would corrupt the loss function — PAD positions must never contribute to the loss, while no-edge is a meaningful signal the model needs to learn.

---

## Usage

### Tokenize (graph dict -> tensors)

```python
from bd_gen.data.tokenizer import tokenize
from bd_gen.data.vocab import RPLAN_VOCAB_CONFIG

graph = {
    "num_rooms": 4,
    "node_types": [0, 1, 3, 7],
    "edge_triples": [(0, 1, 2), (0, 3, 5), (1, 2, 9), (2, 3, 0)],
}

tokens, pad_mask = tokenize(graph, RPLAN_VOCAB_CONFIG)

# tokens:   shape (36,), dtype torch.long  — the token sequence
# pad_mask: shape (36,), dtype torch.bool  — True at real positions
```

### Detokenize (tensors -> graph dict)

```python
from bd_gen.data.tokenizer import detokenize

recovered = detokenize(tokens, pad_mask, RPLAN_VOCAB_CONFIG)
# recovered["num_rooms"]    -> 4
# recovered["node_types"]   -> [0, 1, 3, 7]
# recovered["edge_triples"] -> [(0, 1, 2), (0, 3, 5), (1, 2, 9), (2, 3, 0)]
```

### Round-trip guarantee

`detokenize(tokenize(graph))` always recovers the original graph exactly (edge triple order may differ, but the set is identical). This is verified by 8 parametrised tests across all graph sizes 1-8.

---

## Input validation

Both functions validate their inputs and raise `ValueError` for:

- `num_rooms` outside `[1, n_max]`
- `node_types` length not matching `num_rooms`
- Node type values outside `[0, 12]`
- Edge triples with `i >= j` (must be upper-triangle)
- Edge relationship values outside `[0, 9]`
- Wrong tensor shapes in `detokenize`

---

## Where this fits in the pipeline

```
data.mat  -->  graph2plan_loader  -->  graph dicts  -->  tokenizer  -->  tensors
                                                                          |
                                                         stored in BubbleDiagramDataset
                                                                          |
                                                         fed to diffusion model
```

The tokenizer is called once per graph during dataset construction (not per training step). The resulting tensors are stored in RAM and accessed by index.
