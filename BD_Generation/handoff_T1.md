# Handoff: Phase 1 Data Pipeline — COMPLETE

> **Created:** 2026-02-13
> **Session purpose:** Complete Phase 1 (Data Pipeline) — test, fix, implement dataset, document.

---

## What was accomplished

### From previous session (vocab verification + code writing)
- Downloaded and verified `data.mat` against Graph2Plan source code
- Corrected `vocab.py` NODE_TYPES/EDGE_TYPES mappings (were completely wrong)
- Wrote `graph2plan_loader.py`, `tokenizer.py`, test suites, `prepare_data.py`

### This session (testing + dataset + docs)
1. **Ran tests, fixed 2 failures** in `test_loader.py` — mock tests needed `mat_path.touch()` for existence check
2. **Fixed 2 lint issues** in `test_tokenizer.py` — import sorting, line length
3. **Tested `prepare_data.py` end-to-end** — 80,788 graphs parsed, stats verified
4. **Implemented `bd_gen/data/dataset.py`** — BubbleDiagramDataset with splits, class weights, num_rooms_distribution
5. **Implemented `tests/test_dataset.py`** — 34 tests covering all dataset functionality
6. **Created module docs** — `docs/graph2plan_loader.md`, `docs/tokenizer.md`, `docs/dataset.md`, updated `docs/vocab.md`
7. **Two local commits** on `data/graph2plan-loader`:
   - `48fcdf2` feat(data): implement Phase 1 data pipeline
   - `48ceec0` docs: add module documentation for Phase 1 data pipeline

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Indexing | 0-based, no subtraction | Verified: min(rType)=0, max(rType)=12 |
| Edge inverse | `9 - r` | Symmetric pairing in Graph2Plan vocab |
| Room types 13/14 | Repurposed as MASK/PAD | External/ExteriorWall never in bubble data |
| Class weights | Inverse-frequency, PAD excluded | Standard approach; MASK gets weight 0 |
| num_rooms_distribution | `dist[k]` = P(k+1 rooms) | 0-indexed tensor for 1-based counts |
| Weights on val/test | `None` | Must only come from training data |

## Current state of the codebase

**Branch:** `data/graph2plan-loader` (2 commits ahead of `main`)

**All green:**
- 167/167 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean
- `prepare_data.py` runs end-to-end
- DataLoader iteration verified

**Phase 1 files (all committed):**
- `bd_gen/data/vocab.py` — corrected mappings, VERIFIED status
- `bd_gen/data/graph2plan_loader.py` — .mat parser, caching, validation
- `bd_gen/data/tokenizer.py` — tokenize/detokenize, PAD vs no-edge
- `bd_gen/data/dataset.py` — BubbleDiagramDataset, splits, weights
- `tests/test_loader.py` (23 tests), `tests/test_tokenizer.py` (56 tests), `tests/test_dataset.py` (34 tests)
- `scripts/prepare_data.py` — download + parse + cache + stats
- `docs/vocab.md`, `docs/graph2plan_loader.md`, `docs/tokenizer.md`, `docs/dataset.md`

**Data files (gitignored):**
- `BD_Generation/data/data.mat` — 25.3 MB
- `BD_Generation/data_cache/graph2plan_nmax8.pt` — parsed cache

**Uncommitted:** `README.md` (modified, not part of Phase 1)

## What remains to be done

### Phase 1 wrap-up
1. Merge `data/graph2plan-loader` into `main`
2. Update `implementation_state_T1.md` → COMPLETE after merge

### Phase 2 — Model Architecture
3. Read spec: `BD_Generation/planning_T1.md` sections on model
4. Implement `bd_gen/model/` — BDDenoiser transformer with adaLN
5. Update `conftest.py` `dummy_model()` fixture

### Phase 3-5
6. Diffusion core (noise schedule, forward/reverse process)
7. Training loop
8. Evaluation

## Files to reference in next session

**Read first:**
1. `BD_Generation/implementation_state_T1.md` — phase tracker
2. `BD_Generation/planning_T1.md` — full implementation spec
3. This handoff file

**Phase 1 code (complete):**
4. `BD_Generation/bd_gen/data/dataset.py` — main dataset class
5. `BD_Generation/bd_gen/data/vocab.py` — all constants
6. `BD_Generation/tests/conftest.py` — shared fixtures (`vocab_config`, `sample_batch`)

**Docs:** `BD_Generation/docs/` — vocab, loader, tokenizer, dataset

## Context for the next session

### Dataset facts
- **80,788 graphs**, all 4-8 rooms. Distribution: 7 (36.2%), 6 (31.1%), 8 (25.2%), 5 (7.2%), 4 (0.3%)
- **Edges per graph:** 3-18, mean 10.2. **457 self-loops** filtered by loader.
- **LivingRoom (0)** in every record. **SecondRoom (7)** most common type overall.
- **Rarest:** Entrance (10) — 292 occurrences, GuestRoom (8) — 860.

### Architecture notes
- `VocabConfig(n_max=8)` → `seq_len=36` (8 nodes + 28 edges)
- PAD vs no-edge: PAD excluded from loss, no-edge is real signal
- `__getitem__` returns `{"tokens", "pad_mask", "num_rooms"}`
- Class weights and distribution are attributes on train dataset instance
