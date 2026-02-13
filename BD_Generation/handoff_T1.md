# Handoff: Phase 2 Model Architecture — Code Complete, Pending Commits

> **Created:** 2026-02-13
> **Session purpose:** Implement Phase 2 (Model Architecture) — transformer denoiser with adaLN-Zero for MDLM diffusion.

---

## What was accomplished

1. **Created branch** `model/transformer-denoiser` from `main`
2. **Implemented `bd_gen/model/embeddings.py`** (219 lines) — 4 classes:
   - `NodeEmbedding`: `nn.Embedding(15, d_model)` for room type tokens
   - `EdgeEmbedding`: `nn.Embedding(13, d_model)` for edge type tokens
   - `CompositePositionalEncoding`: 3 learned tables (entity_type, node_index, pair_index) + precomputed buffer indices
   - `TimestepEmbedding`: sinusoidal encoding + MLP projection
3. **Implemented `bd_gen/model/transformer.py`** (170 lines) — 2 classes:
   - `MultiHeadSelfAttention`: combined QKV projection, `F.scaled_dot_product_attention`, `(B,1,1,S)` float additive mask
   - `AdaLNBlock`: 6-param adaLN modulation (shift/scale/gate for attn+FFN), zero-init weights AND bias
4. **Implemented `bd_gen/model/denoiser.py`** (214 lines) — `BDDenoiser`:
   - 11-step forward pass: split tokens → embed → concat → pos enc → timestep cond → transformer blocks → final adaLN → split → heads
   - `_process_t()` helper for flexible timestep input (float, int, 0D/1D tensors)
   - Zero-init final adaLN (2 params: shift+scale), zero-init classification heads
   - `condition=None` placeholder for v2 cross-attention
5. **Wrote `tests/test_embeddings.py`** (221 lines) — 27 tests across all 4 embedding classes
6. **Wrote `tests/test_denoiser.py`** (441 lines) — 28 tests: forward shapes, `_process_t`, gradient flow, adaLN zero-init, param count, PAD mask, unconditional, timestep variation
7. **Updated `bd_gen/model/__init__.py`** — exports all 7 public classes
8. **Updated `tests/conftest.py`** — `dummy_model()` fixture returns real `BDDenoiser(d_model=32, n_layers=1, n_heads=2)` instead of `pytest.skip()`
9. **Created `docs/model.md`** (516 lines) — detailed architecture documentation with motivations, ASCII diagrams, forward pass walkthrough, design rationale
10. **All 222 tests pass**, `ruff check` clean

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Attention impl | `F.scaled_dot_product_attention` | Only 36 tokens for RPLAN; flash attention unnecessary |
| adaLN modulation | Zero-init weights AND bias | Identity modulation + zero gate at init → stable DiT training |
| Final layer | adaLN with 2 params (shift+scale), no gate | No residual at final layer; matches DiDAPS DDitFinalLayer |
| Classification heads | Zero-init weights AND bias | Initial output = all zeros = uniform logits = clean starting point |
| GELU variant | `nn.GELU()` (exact) | Standard; DiDAPS uses approximate but difference negligible |
| QKV projection | Single `Linear(d_model, 3*d_model, bias=True)` | Efficient combined projection |
| PAD mask for SDPA | `(B,1,1,S)` float mask with -inf | Broadcasts across heads and queries |
| Outer SiLU | Applied in `BDDenoiser.forward()` before adaLN | Follows DiT/DiDAPS convention: `c = SiLU(timestep_embedding(t))` |

## Current state of the codebase

**Branch:** `model/transformer-denoiser` — NO COMMITS YET (all changes are unstaged)

**Unstaged/untracked changes:**
- Modified: `bd_gen/model/__init__.py`, `tests/conftest.py`, `implementation_state_T1.md`, `README.md`
- New files: `bd_gen/model/embeddings.py`, `bd_gen/model/transformer.py`, `bd_gen/model/denoiser.py`, `docs/model.md`, `tests/test_embeddings.py`, `tests/test_denoiser.py`

**All green:**
- 222/222 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean
- `from bd_gen.model import BDDenoiser` works
- Small config param count: ~1.28M (within 1-5M target)
- Forward shapes verified: `(4, 8, 15)` and `(4, 28, 13)` for RPLAN

**Previous phases (committed on `main`):**
- Phase 0 (Scaffold): 54 tests, tagged `v0.1.0`
- Phase 1 (Data Pipeline): 167 tests, merged to `main`

**Data files (gitignored):**
- `BD_Generation/data/data.mat` — 25.3 MB
- `BD_Generation/data_cache/graph2plan_nmax8.pt` — parsed cache

## What remains to be done

### Phase 2 wrap-up (immediate)
1. **Run tests + ruff** to verify everything still passes (do this FIRST)
2. **Git commit** the Phase 2 files on `model/transformer-denoiser` branch (suggested commits below)
3. **Merge** `model/transformer-denoiser` into `main` with `--no-ff`
4. **Tag** `v0.3.0` on main
5. **Update `implementation_state_T1.md`** — Phase 2 summary (ONLY when user says to mark COMPLETE)

**Suggested commit structure:**
```
feat(model): implement embedding modules (NodeEmbedding, EdgeEmbedding, CompositePositionalEncoding, TimestepEmbedding)
feat(model): implement adaLN-Zero transformer block and MHSA
feat(model): implement BDDenoiser top-level model
test(model): add embedding and denoiser tests (55 tests)
docs(model): add detailed module documentation
```

### Phase 3 — Diffusion Core (next phase)
6. Read spec: `planning_T1.md` sections on noise schedules, forward/reverse process
7. Implement `bd_gen/diffusion/` — noise schedule, forward noising, ELBO loss

### Phase 4-5
8. Training loop
9. Evaluation

## Files to reference in next session

**Read first (in order):**
1. `BD_Generation/implementation_state_T1.md` — phase tracker with rules
2. `BD_Generation/planning_T1.md` — full implementation spec (Sections 3.2, 5.2, 6 for Phase 2)
3. This handoff file

**Phase 2 code (complete, uncommitted):**
4. `BD_Generation/bd_gen/model/denoiser.py` — top-level BDDenoiser class
5. `BD_Generation/bd_gen/model/embeddings.py` — 4 embedding classes
6. `BD_Generation/bd_gen/model/transformer.py` — MHSA + AdaLNBlock
7. `BD_Generation/bd_gen/model/__init__.py` — exports
8. `BD_Generation/docs/model.md` — detailed architecture documentation

**Phase 2 tests:**
9. `BD_Generation/tests/test_denoiser.py` — 28 tests
10. `BD_Generation/tests/test_embeddings.py` — 27 tests
11. `BD_Generation/tests/conftest.py` — updated `dummy_model()` fixture + `sample_batch`

**Key dependencies:**
12. `BD_Generation/bd_gen/data/vocab.py` — VocabConfig, vocab sizes, edge_position_to_pair()
13. `BD_Generation/configs/model/small.yaml` — d_model=128, n_layers=4, n_heads=4
14. `BD_Generation/configs/model/base.yaml` — d_model=256, n_layers=6, n_heads=8

**Reference (read-only, NOT a dependency):**
15. `DiDAPS_COPY/backbones/dit.py` — TimestepEmbedder, DDiTBlock, DDitFinalLayer patterns

## Context for the next session

### Architecture facts
- `VocabConfig(n_max=8)` → `seq_len=36` (8 nodes + 28 edges)
- Forward pass: tokens (B,36) → split → embed separately → concat → pos enc → SiLU(timestep_emb) → N transformer blocks → final adaLN + norm → split → heads → (B,8,15) + (B,28,13)
- Small config: ~1.28M params. Base config: ~5M params.

### Bug fix during implementation
- `test_different_pad_masks_produce_different_outputs` initially failed because zero-init adaLN gates (gate=0) cause transformer blocks to contribute nothing → attention masking has no observable effect. **Fix:** randomize ALL weights to simulate a trained model, then verify mask matters. Comment in test explains this.

### Important rules (from implementation_state_T1.md)
- Parallelize workstreams within a phase, NOT across phases
- Create a docs `.md` for each module
- Only mark phase COMPLETE when user explicitly says so
- Alert at 80% token usage for handoff
