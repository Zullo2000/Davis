# Graph2Plan Loader (`bd_gen.data.graph2plan_loader`)

## Purpose

This module reads the Graph2Plan `data.mat` file and converts it into a clean list of Python dictionaries, one per floorplan graph. It handles all the messy parts of the raw data: MATLAB struct parsing, self-loop filtering, edge direction normalisation, and range validation. The parsed result is cached as a `.pt` file so subsequent loads are instant.

---

## What the raw data looks like

The Graph2Plan dataset ships as a single `data.mat` file (~25 MB) containing 80,788 floorplan records. Each record has:

- **`rType`**: An array of integers representing room types (0-12, see `vocab.py`). For example, `[0, 1, 3, 7]` means LivingRoom, MasterRoom, Bathroom, SecondRoom.
- **`rEdge`**: An Nx3 array where each row is `[u, v, rel_type]` — an edge between room `u` and room `v` with spatial relationship `rel_type` (0-9, see `vocab.py`).

### Quirks handled by the loader

1. **`squeeze_me=True` side effects**: When scipy loads the .mat file, single-element arrays can become scalars and single-row 2D arrays can become 1D. The loader reshapes these back.
2. **Self-loops**: 457 edges in the data have `u == v`. These are meaningless and are silently filtered out.
3. **Edge direction**: The data is already upper-triangle (`u < v`), but the loader defensively re-orders any `u > v` edges and inverts the relationship type using `9 - r`.

---

## Output format

Each graph becomes a dictionary with three keys:

```python
{
    "node_types": [0, 1, 3, 7],         # list[int], room type indices (0-12)
    "edge_triples": [(0, 1, 2), ...],   # list[tuple[int, int, int]], (u, v, rel)
    "num_rooms": 4,                      # int, == len(node_types)
}
```

Guarantees:
- All `node_types` values are in `[0, 12]`
- All edges satisfy `u < v` (strict upper triangle)
- All `rel_type` values are in `[0, 9]`
- `num_rooms` matches `len(node_types)`
- No self-loops
- Graphs with more than `n_max` rooms are excluded

---

## Usage

```python
from bd_gen.data.graph2plan_loader import load_graph2plan

graphs = load_graph2plan(
    mat_path="data/data.mat",       # raw Graph2Plan file
    cache_path="data_cache/g2p.pt", # where to save/load cache
    n_max=8,                        # skip graphs with > 8 rooms
)

print(len(graphs))           # 80,788 for RPLAN
print(graphs[0]["num_rooms"])  # e.g. 6
print(graphs[0]["node_types"]) # e.g. [0, 1, 2, 3, 7, 9]
```

The first call parses `data.mat` (~4 seconds) and saves the cache. Subsequent calls load the cache (<1 second).

---

## Caching

- **Format**: `torch.save()` / `torch.load()` (pickle-based `.pt` file)
- **Cache key**: The cache path is explicit — there is no automatic invalidation. If you change `n_max` or the raw data, use a different cache path or delete the old cache.
- **Size**: ~28 MB for the full RPLAN dataset

---

## Relationship inversion

The module exposes `_invert_relationship(r)` which returns `9 - r`. This works because Graph2Plan ordered its 10 spatial relationships as symmetric pairs:

```
0 (left-above)  <-> 9 (right-below)
1 (left-below)  <-> 8 (right-above)
2 (left-of)     <-> 7 (right-of)
3 (above)       <-> 6 (below)
4 (inside)      <-> 5 (surrounding)
```

This is only needed when an edge `(u, v)` has `u > v` and must be flipped to upper-triangle form. In practice, the raw data is already upper-triangle, so this is purely defensive.

---

## Dataset statistics (RPLAN)

| Metric | Value |
|--------|-------|
| Total records | 80,788 |
| Room count range | 4-8 (none with 1-3) |
| Most common size | 7 rooms (36.2%) |
| Edges per graph | 3-18, mean 10.2 |
| Self-loops filtered | 457 |
| Most common room type | SecondRoom (index 7) |
| Rarest room type | Entrance (index 10, only 292 occurrences) |
