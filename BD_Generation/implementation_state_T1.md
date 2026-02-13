
# Implementation State

> Updated after each phase. Coordinator reads this + the spec before starting work.
> Rule: keep each phase summary under 60 lines. Capture decisions and deviations, not raw logs.
> Rule: For this first implementation, parallelize with sub-agents the workstreams within a single phase as described in planning_T1.md but not the phases.
> Rule: create a documentation .md file for each module.
> Rule: only when I have understood everything about a Phase and I tell you to do so, you can change the status of the Phase into "COMPLETE". Until the previous phase is not in that status you cannot proceed to the next one. 
> Rule: when you arrive to 80% of the tokens tell me because I will use /create-handoff create a new session to optimally continue


## Overall Status
- Current phase: Phase 2 (NOT STARTED)
- Last completed: Phase 1
- Spec corrections: vocab.py NODE_TYPES/EDGE_TYPES name-to-index mappings corrected (Phase 1 Step 0)

## Phase 0 — Scaffold
Status: COMPLETE
Branch: `setup/repo-scaffold` → merged to `main`, tagged `v0.1.0`

### Deliverables (all verified)
- `pip install -e "BD_Generation/[dev]"` succeeds (Python 3.14.2, PyTorch 2.10.0)
- `from bd_gen.data.vocab import NODE_VOCAB_SIZE, RPLAN_VOCAB_CONFIG` works
- 54/54 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean
- All 8 Hydra YAML configs parse with OmegaConf

### Files created (25 new, 1 modified)
- `.gitignore` — rewritten with whitelist pattern (`/*` then `!/BD_Generation/`)
- `pyproject.toml`, `Makefile`, `README.md` — build tooling
- `bd_gen/` — 7 subpackages with `__init__.py` files
- `bd_gen/data/vocab.py` — all vocab constants, VocabConfig, pad mask, edge mapping
- `tests/test_vocab.py`, `tests/conftest.py` — 54 tests + shared fixtures
- `configs/` — 8 YAML files (model, data, noise, training, eval)
- `docs/vocab.md` — module documentation
- `scripts/.gitkeep`, `notebooks/.gitkeep` — placeholder dirs

### Deviations from spec
- None. All implementations match planning_T1.md exactly.

### Issues resolved
- hydra-core 1.3.2 installs fine on Python 3.14.2 (no compatibility issue)
- All dependencies resolved without conflicts

### Key decisions
- `.gitignore` uses `/*` exclusion pattern per spec Section 11.1
- Existing BD_Generation/ markdown files committed alongside scaffold
- `dummy_model()` fixture uses `pytest.skip()` until Phase 2

## Phase 1 — Data Pipeline
Status: COMPLETE
Branch: `data/graph2plan-loader` → merged to `main`

### Deliverables (all verified)
- 167/167 tests pass (`pytest BD_Generation/tests/ -v`)
- `ruff check` clean (all source, tests, scripts)
- `prepare_data.py` runs end-to-end: 80,788 graphs parsed, cached as `.pt`
- DataLoader smoke test passes (batch iteration works)

### Files created/modified (8 new, 1 modified)
- `bd_gen/data/vocab.py` — MODIFIED: corrected NODE_TYPES/EDGE_TYPES names (verified against Graph2Plan source)
- `bd_gen/data/graph2plan_loader.py` — NEW: .mat parser with caching, self-loop filtering, validation
- `bd_gen/data/tokenizer.py` — NEW: tokenize/detokenize with PAD vs no-edge invariant
- `bd_gen/data/dataset.py` — NEW: BubbleDiagramDataset with splits, class weights, num_rooms_distribution
- `tests/test_loader.py` — NEW: 23 tests (synthetic + real data integration)
- `tests/test_tokenizer.py` — NEW: 56 tests (roundtrip, PAD correctness, edge cases)
- `tests/test_dataset.py` — NEW: 34 tests (splits, weights, PAD invariant, DataLoader)
- `scripts/prepare_data.py` — NEW: auto-download + parse + cache + stats

### Key decisions
- Data is 0-based; no index subtraction needed (verified: min(rType)=0, max(rType)=12)
- Edge inverse: `9 - r` (verified against symmetric pairing in Graph2Plan vocab)
- Graph2Plan room types 13/14 repurposed as MASK/PAD (never appear in bubble data)
- n_max filtering not needed for RPLAN (all 80,788 graphs have 4-8 rooms)
- Class weights: inverse-frequency, PAD excluded; MASK class gets weight 0
- num_rooms_distribution: normalised histogram from training split, index k = P(k+1 rooms)

### Deviations from spec
- None. All implementations match planning_T1.md specs.

## Phase 2 — Model Architecture
Status: NOT STARTED

## Phase 3 — Diffusion Core
Status: NOT STARTED

## Phase 4 — Training Loop
Status: NOT STARTED

## Phase 5 — Evaluation
Status: NOT STARTED
