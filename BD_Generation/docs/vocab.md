# Vocabulary Reference (`bd_gen.data.vocab`)

## Purpose

`vocab.py` is the **single source of truth** for all vocabulary constants and dataset-dependent sizing in the BD-Gen project. Every other module that needs to know about token types, special indices, or sequence dimensions **must import from this module** rather than defining its own constants. This eliminates the risk of mismatches between model configuration and data processing.

> **PROVISIONAL WARNING**: The `NODE_TYPES` and `EDGE_TYPES` lists are based on the Graph2Plan paper and **MUST be verified against the actual `data.mat` file** during Phase 1 (data verification task). If the real dataset uses different type names, orderings, or counts, this module must be updated first and all downstream code will automatically pick up the changes.

---

## Constant Reference

### Node Vocabulary

| Index | Name             | Description                     |
|------:|------------------|---------------------------------|
|     0 | `MasterRoom`     | Master bedroom                  |
|     1 | `SecondRoom`     | Secondary bedroom               |
|     2 | `LivingRoom`     | Living room                     |
|     3 | `Kitchen`        | Kitchen                         |
|     4 | `Bathroom`       | Bathroom / toilet               |
|     5 | `Balcony`        | Balcony                         |
|     6 | `Entrance`       | Entrance / foyer                |
|     7 | `DiningRoom`     | Dining room                     |
|     8 | `StudyRoom`      | Study / office                  |
|     9 | `StorageRoom`    | Storage / closet                |
|    10 | `WallIn`         | Interior wall segment           |
|    11 | `ExternalArea`   | External area                   |
|    12 | `ExternalWall`   | External wall boundary          |
|    13 | **MASK**         | Diffusion mask token            |
|    14 | **PAD**          | Padding for unused positions    |

- `NODE_VOCAB_SIZE = 15` (13 room types + MASK + PAD)

### Edge Vocabulary

| Index | Name             | Description                        |
|------:|------------------|------------------------------------|
|     0 | `left-of`        | Node i is left of node j           |
|     1 | `right-of`       | Node i is right of node j          |
|     2 | `above`          | Node i is above node j             |
|     3 | `below`          | Node i is below node j             |
|     4 | `left-above`     | Node i is left-above node j        |
|     5 | `left-below`     | Node i is left-below node j        |
|     6 | `right-above`    | Node i is right-above node j       |
|     7 | `right-below`    | Node i is right-below node j       |
|     8 | `inside`         | Node i is inside node j            |
|     9 | `surrounding`    | Node i surrounds node j            |
|    10 | **no-edge**      | No spatial relationship exists      |
|    11 | **MASK**         | Diffusion mask token               |
|    12 | **PAD**          | Padding for unused edge positions  |

- `EDGE_VOCAB_SIZE = 13` (10 relationships + no-edge + MASK + PAD)

---

## VocabConfig Usage

`VocabConfig` is a frozen dataclass that derives all sequence sizing from a single parameter: `n_max` (maximum number of rooms per graph).

```python
from bd_gen.data.vocab import VocabConfig, RPLAN_VOCAB_CONFIG

# Use the RPLAN preset
vc = RPLAN_VOCAB_CONFIG
print(vc.n_max)     # 8
print(vc.n_edges)   # 28  (= C(8,2) = 8*7/2)
print(vc.seq_len)   # 36  (= 8 nodes + 28 edges)

# Or create a custom config
vc14 = VocabConfig(n_max=14)
print(vc14.n_edges)  # 91  (= C(14,2) = 14*13/2)
print(vc14.seq_len)  # 105 (= 14 nodes + 91 edges)
```

### Preset Configurations

| Config               | `n_max` | `n_edges` | `seq_len` | Dataset  |
|----------------------|--------:|----------:|----------:|----------|
| `RPLAN_VOCAB_CONFIG` |       8 |        28 |        36 | RPLAN    |
| `RESPLAN_VOCAB_CONFIG`|     14 |        91 |       105 | ResPlan  |

---

## Edge Position Encoding

Edges are stored in **upper-triangle row-major order**. For a graph with `n_max` nodes, only edges `(i, j)` where `i < j` are stored (since the adjacency relationship is encoded directionally in the edge type itself).

### Visual Diagram for `n_max = 4`

The upper triangle of a 4x4 adjacency matrix:

```
     j=0  j=1  j=2  j=3
i=0   -   pos0 pos1 pos2
i=1   -    -   pos3 pos4
i=2   -    -    -   pos5
i=3   -    -    -    -
```

Mapping:

| Position | Pair (i, j) |
|---------:|-------------|
|        0 | (0, 1)      |
|        1 | (0, 2)      |
|        2 | (0, 3)      |
|        3 | (1, 2)      |
|        4 | (1, 3)      |
|        5 | (2, 3)      |

Total: `C(4, 2) = 6` edge positions.

### Conversion Functions

```python
vc = VocabConfig(n_max=4)

# Position -> pair
vc.edge_position_to_pair(0)   # (0, 1)
vc.edge_position_to_pair(3)   # (1, 2)

# Pair -> position
vc.pair_to_edge_position(0, 3)  # 2
vc.pair_to_edge_position(2, 3)  # 5

# Swapped pairs are handled automatically
vc.pair_to_edge_position(3, 0)  # 2 (same as (0, 3))
```

These two functions are exact inverses: for all valid inputs, `edge_position_to_pair(pair_to_edge_position(i, j)) == (min(i,j), max(i,j))` and `pair_to_edge_position(*edge_position_to_pair(pos)) == pos`.

---

## Sequence Layout

A single flattened token sequence has the following layout:

```
[ node_0, node_1, ..., node_{n_max-1}, edge_pos_0, edge_pos_1, ..., edge_pos_{n_edges-1} ]
|<-------- n_max node tokens -------->|<------------- n_edges edge tokens ------------->|
```

- **Indices `[0, n_max)`**: Node token positions (room types or NODE_PAD)
- **Indices `[n_max, seq_len)`**: Edge token positions (edge types, no-edge, or EDGE_PAD)

---

## PAD Mask Semantics

The `compute_pad_mask(num_rooms)` method returns a boolean tensor of shape `(seq_len,)`:

- **`True`** = real position (contains meaningful data)
- **`False`** = PAD position (should be ignored by the model)

### Rules

1. **Node positions**: The first `num_rooms` positions are `True`; positions `num_rooms` through `n_max - 1` are `False` (PAD).
2. **Edge positions**: An edge position for pair `(i, j)` is `True` only if **both** `i < num_rooms` **and** `j < num_rooms`. If either endpoint is a PAD node, the edge is also PAD.

### Example: `n_max = 4, num_rooms = 2`

```
Nodes:  [True, True, False, False]    -- rooms 0,1 are real; 2,3 are PAD
Edges:  [True, False, False, False, False, False]
         (0,1)  (0,2)  (0,3)  (1,2)  (1,3)  (2,3)
          ^real   ^pad   ^pad   ^pad   ^pad   ^pad
```

Only edge `(0, 1)` is real because both node 0 and node 1 exist. All other edges involve at least one PAD node.

### PAD vs. No-Edge: A Critical Distinction

These are two fundamentally different concepts:

| Concept    | Token Index        | Meaning                                              |
|------------|--------------------|------------------------------------------------------|
| **PAD**    | `EDGE_PAD_IDX (12)` | This edge position does not exist in the graph because one or both endpoint rooms are not present. The model should **ignore** this position entirely (via attention masking). |
| **no-edge** | `EDGE_NO_EDGE_IDX (10)` | Both rooms exist, but they have **no spatial adjacency relationship**. This is a real, meaningful token that the model must learn to predict. |

This distinction is critical for correct loss computation: PAD positions are excluded from the loss, while no-edge positions contribute to the loss like any other real token.
