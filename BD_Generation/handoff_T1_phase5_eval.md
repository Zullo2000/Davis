# Handoff: Phase 5 — Evaluation & Metrics

> **Created:** 2026-02-16
> **Session purpose:** Implement Phase 5 (evaluation, metrics, visualization) of the BD Generation pipeline

---

## What was accomplished

All Phase 5 code is written and tests pass. Only minor linting cleanup remains.

### New files created (11 files):
- `bd_gen/eval/validity.py` — Graph validity checker (connectivity BFS, room-type constraints, MASK/range checks)
- `bd_gen/eval/metrics.py` — Evaluation metrics (validity_rate, diversity, novelty, distribution_match, per_class_accuracy)
- `bd_gen/viz/graph_viz.py` — Bubble diagram visualization (networkx + matplotlib, 13-color room-type palette)
- `tests/test_validity.py` — 16 validity tests (valid graphs, disconnected, MASK injection, type constraints, batch)
- `tests/test_metrics.py` — 23 metrics tests (all metric functions + edge cases)
- `scripts/sample.py` — Generate + visualize samples from checkpoint (Hydra Compose API)
- `scripts/evaluate.py` — Full evaluation pipeline (generate → validate → metrics → wandb → JSON)
- `docs/evaluation.md` — Module documentation
- `notebooks/04_sample_analysis.ipynb` — Sample analysis notebook

### Modified files (3 files):
- `bd_gen/eval/__init__.py` — Added exports for all eval functions
- `bd_gen/viz/__init__.py` — Added exports for draw_bubble_diagram, draw_bubble_diagram_grid
- `configs/eval/default.yaml` — Expanded from 4 to 9 lines (added checkpoint_path, batch_size, viz options)

### Test results:
- **348/348 tests pass** (309 existing + 39 new)
- All new validity and metrics tests green

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| `vocab_config` param on `check_validity` | Added (not in spec) | Same precedent as Phase 3: `detokenize` requires it |
| `novelty` metric | Exact-match hash, not GED | GED is NP-hard; 1000 samples × 64K training = impractical |
| `consistent` check | Simplified — format prevents contradictions | Upper-triangle stores each (i,j) once; full transitivity = v2 |
| Visualization backend | `matplotlib.use("Agg")` | Headless server compatibility |
| Room-type constraints | At most 1 LivingRoom (idx 0), at most 1 Entrance (idx 10) | RPLAN domain knowledge |

## Current state

### What works:
- All 11 new files written and functional
- 348/348 tests pass
- Branch `eval/metrics-and-validity` created and active

### What remains (minor cleanup):
1. **3 ruff import-sorting errors** (I001) in:
   - `bd_gen/eval/metrics.py`
   - `tests/test_metrics.py`
   - `tests/test_validity.py`
   - Fix: `python -m ruff check --fix BD_Generation/bd_gen/ BD_Generation/tests/ BD_Generation/scripts/`

2. **Verify all tests still pass** after ruff fix: `pytest BD_Generation/tests/ -v`

3. **Verify imports work**:
   - `from bd_gen.eval import check_validity, validity_rate, novelty, diversity, distribution_match`
   - `from bd_gen.viz import draw_bubble_diagram, draw_bubble_diagram_grid`

4. **Commit all changes** to branch `eval/metrics-and-validity`

5. **Merge to main** with `--no-ff` and tag `v0.6.0` (requires user approval)

6. **Update `implementation_state_T1.md`** — Add Phase 5 summary (mark as COMPLETE after merge)

## What remains to be done (ordered)

1. Run `ruff check --fix` to auto-fix the 3 import sorting issues
2. Run full test suite one more time to confirm
3. Commit with message: `feat(eval): implement Phase 5 evaluation metrics, validity, and visualization`
4. Ask user to approve merge to main + tag v0.6.0
5. Update `implementation_state_T1.md` with Phase 5 summary

## Files to reference in next session

**Read first (rules & state):**
1. `BD_Generation/CLAUDE.md` — Agent rules (always read first)
2. `BD_Generation/implementation_state_T1.md` — Current state (Phase 5 NOT STARTED → update to COMPLETE)
3. This handoff: `BD_Generation/handoff_T1_phase5_eval.md`

**Key implementation files (already written, may need minor lint fixes):**
4. `BD_Generation/bd_gen/eval/validity.py` — Core validity checker
5. `BD_Generation/bd_gen/eval/metrics.py` — All metrics
6. `BD_Generation/bd_gen/viz/graph_viz.py` — Visualization
7. `BD_Generation/scripts/evaluate.py` — Evaluation script
8. `BD_Generation/scripts/sample.py` — Sampling script

**Tests:**
9. `BD_Generation/tests/test_validity.py` — 16 tests
10. `BD_Generation/tests/test_metrics.py` — 23 tests

## Context for the next session

- **Git branch:** `eval/metrics-and-validity` (already checked out, all changes unstaged)
- **No conftest.py changes needed** — existing fixtures (`vocab_config`, `sample_batch`, `dummy_model`) were sufficient; test files create their own graph dicts via `tokenize()`
- **Pad mask reconstruction in scripts:** Both `sample.py` and `evaluate.py` reconstruct pad masks from generated tokens by counting non-PAD nodes. This is necessary because `sample()` returns raw tokens without masks.
- **The `consistent` check is deliberately simple** — the upper-triangle format inherently prevents the "left-of in both directions" contradiction. Document this in implementation_state_T1.md.
- **networkx >= 3.1** is already in `pyproject.toml` dependencies — no changes needed.
- **Training checkpoint** exists locally (from Phase 4 GPU training on university server). The scripts accept `eval.checkpoint_path=<path>` as CLI override.
