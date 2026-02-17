# Vocabulary Reference (`bd_gen.data.vocab`)

## Purpose

`vocab.py` is the **single source of truth** for all vocabulary constants and dataset-dependent sizing in the BD-Gen project. Every other module that needs to know about token types, special indices, or sequence dimensions **must import from this module** rather than defining its own constants. This eliminates the risk of mismatches between model configuration and data processing.

> **VERIFIED (Phase 1)**: The `NODE_TYPES` and `EDGE_TYPES` lists have been verified against Graph2Plan's `get_vocab()` in `Network/model/utils.py` and the actual `data.mat` contents. Data uses 0-based indexing.

---

## Constant Reference

### Node Vocabulary

| Index | Name             | Description                     |
|------:|------------------|---------------------------------|
|     0 | `LivingRoom`     | Living room (every plan has one)|
|     1 | `MasterRoom`     | Master bedroom                  |
|     2 | `Kitchen`        | Kitchen                         |
|     3 | `Bathroom`       | Bathroom / toilet               |
|     4 | `DiningRoom`     | Dining room                     |
|     5 | `ChildRoom`      | Child's bedroom                 |
|     6 | `StudyRoom`      | Study / office                  |
|     7 | `SecondRoom`     | Secondary bedroom (most common) |
|     8 | `GuestRoom`      | Guest room                      |
|     9 | `Balcony`        | Balcony                         |
|    10 | `Entrance`       | Entrance / foyer                |
|    11 | `Storage`        | Storage / closet                |
|    12 | `Wall-in`        | Interior wall segment           |
|    13 | **MASK**         | Diffusion mask token            |
|    14 | **PAD**          | Padding for unused positions    |

- `NODE_VOCAB_SIZE = 15` (13 room types + MASK + PAD)
- Graph2Plan defines 15 room types (0-14) but only 0-12 appear in bubble diagram data. Indices 13 (External) and 14 (ExteriorWall) are repurposed as MASK and PAD.

### Edge Vocabulary

| Index | Name             | Inverse (9-r)     | Description                        |
|------:|------------------|--------------------|------------------------------------|
|     0 | `left-above`     | `right-below` (9)  | Node i is left-above node j        |
|     1 | `left-below`     | `right-above` (8)  | Node i is left-below node j        |
|     2 | `left-of`        | `right-of` (7)     | Node i is left of node j           |
|     3 | `above`          | `below` (6)        | Node i is above node j             |
|     4 | `inside`         | `surrounding` (5)  | Node i is inside node j            |
|     5 | `surrounding`    | `inside` (4)       | Node i surrounds node j            |
|     6 | `below`          | `above` (3)        | Node i is below node j             |
|     7 | `right-of`       | `left-of` (2)      | Node i is right of node j          |
|     8 | `right-above`    | `left-below` (1)   | Node i is right-above node j       |
|     9 | `right-below`    | `left-above` (0)   | Node i is right-below node j       |
|    10 | **no-edge**      | —                  | No spatial relationship exists      |
|    11 | **MASK**         | —                  | Diffusion mask token               |
|    12 | **PAD**          | —                  | Padding for unused edge positions  |

- `EDGE_VOCAB_SIZE = 13` (10 relationships + no-edge + MASK + PAD)
- The inverse of relationship `r` is simply `9 - r` (Graph2Plan ordered types symmetrically).

---

## Edge Directionality

All edges are stored as `(i, j, rel)` triples with **`i < j`** enforced (upper-triangle). The label `rel` is always read as:

> **"node i is [rel] relative to node j"**

The lower-indexed node is always the **subject**. The higher-indexed node is always the **object**.

The inverse relationship (from j's perspective) is `9 - rel` but is **not stored** — it is implicit. For example, if the stored triple is `(2, 5, 2)` (`left-of`), then node 5 is to the right of node 2 (`9 - 2 = 7`, `right-of`). Only the canonical `i < j` form is stored.

### Reading Directionality on Plots

Plots produced by `draw_bubble_diagram` label each node as `i:Type` (e.g. `0:Liv`, `3:Bath`). For any edge with label `rel`:

1. Identify the two node indices from the node labels.
2. The **smaller index** is the subject; the **larger index** is the object.
3. Read: "node \[smaller\] is \[rel\] relative to node \[larger\]".

No arrows are drawn — the node-index labels make the subject unambiguous.

> **Note:** The plot layout uses `spring_layout` (force-directed), which does **not** reflect actual spatial positions. A node labelled "above" may appear visually below on the plot. The layout shows graph topology, not spatial arrangement.

### Worked Example

Suppose a graph with 4 rooms:

| Index | Type       | Plot label |
|------:|------------|------------|
|     0 | LivingRoom | `0:Liv`    |
|     1 | Kitchen    | `1:Kit`    |
|     2 | Bathroom   | `2:Bath`   |
|     3 | MasterRoom | `3:Mstr`   |

Edge triples:

| Triple        | rel name  | Reading                              | Inverse (from j)              |
|:--------------|:----------|:-------------------------------------|:------------------------------|
| `(0, 1, 2)`  | `left-of` | 0:Liv is left of 1:Kit               | 1:Kit is right-of 0:Liv (r=7) |
| `(0, 3, 3)`  | `above`   | 0:Liv is above 3:Mstr                | 3:Mstr is below 0:Liv (r=6)   |
| `(1, 2, 6)`  | `below`   | 1:Kit is below 2:Bath                | 2:Bath is above 1:Kit (r=3)   |

On the plot, the edge between `0:Liv` and `1:Kit` is labelled `left-of`. Since 0 < 1, this means node 0 (Liv) is to the left of node 1 (Kit).

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
