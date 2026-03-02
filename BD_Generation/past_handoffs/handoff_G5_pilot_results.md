# Handoff: G5 Pilot Results — Analysis & Next Experiments

> **Created:** 2026-03-02
> **Session purpose:** Synced guidance code to jabiru, fixed critical repeat_interleave bug, ran 6-config pilot, copied results locally, generated distribution/trajectory plots, began results analysis.

---

## What was accomplished

- **Fixed critical K-candidate expansion bug** in `bd_gen/guidance/guided_sampler.py`:
  - `repeat_interleave(K, dim=0)` + `.view(K, B, SEQ)` was mixing batch elements across candidates — `candidates[k, b]` pointed to the wrong batch element
  - Root cause: `repeat_interleave` groups K copies per element `[b0,b0,b0,b1,b1,b1,...]` but `.view(K, B, SEQ)` expects K full-batch copies `[b0,b1,...,bN,b0,b1,...,bN,...]`
  - Fix: changed to `.repeat(K, 1)` (and `.repeat(K, 1, 1)` for 3D logit tensors)
  - This caused PAD tokens (14) at real node positions, crashing `detokenize` with `ValueError: tokens[5] = 14 not in valid node range [0, 13)`
  - Commit: `693a07a`

- **Fixed Python 3.9 compatibility** in `scripts/analyze_guidance_stats.py`:
  - Backslash in f-string expression (`{'\u0394_remask':>9}`) is Python 3.12+ only
  - Jabiru runs Python 3.9
  - Commit: `a8b59e9`

- **Synced code to jabiru**, resolved git conflict in `comparison.md`, moved `_samples.pt` files from old `loglinear/` to `loglinear_noise_sc/` (24 files) and `linear/` to `linear_noise_sc/` (12 files)

- **Ran full G5 pilot on jabiru** — `bash scripts/run_g5_pilot.sh all` completed successfully:
  - Step 1: Calibrated P90 normalizers → `configs/guidance/calibration_v1_no_remask.json`
  - Step 2: Generated 6 guided configs (GPU, ~30 min)
  - Step 3: Evaluated baseline + 6 guided models with constraint metrics
  - Step 4: Built comparison table → `eval_results/loglinear_noise_sc/comparison_guided_pilot.md`

- **Copied results locally**: 6 guided JSONs, comparison table, calibration JSON

- **Ran distribution + trajectory plots** on jabiru (after fixing the Python 3.9 f-string issue). PNGs need to be copied locally.

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Fix repeat_interleave bug | Changed to `.repeat(K, 1)` | Correct layout for `.view(K, B, SEQ)` — K full-batch copies, not K copies per element |
| Pilot reward mode | Soft (from YAML default) | Smoother gradient signal for candidate selection; hard mode comparison deferred |
| Pilot grid | α ∈ {0.1, 1.0, 5.0} × K ∈ {4, 16} | Coarse sweep to find the right α range |

## Pilot results summary

**Constraint Satisfaction (overall — all 4 constraints simultaneously):**

| Config | Satisfaction | Diversity | Cond. Edge TV |
|--------|-------------|-----------|---------------|
| Baseline (no guidance) | 43.3% | 0.945 | 0.472 |
| K=4, α=0.1 | **68.5%** | 0.903 | 0.487 |
| K=16, α=0.1 | **77.0%** | 0.909 | 0.517 |
| K=4, α=1.0 | 47.0% | 0.938 | 0.473 |
| K=16, α=1.0 | 48.5% | 0.932 | 0.477 |
| K=4, α=5.0 | 43.5% | 0.944 | 0.470 |
| K=16, α=5.0 | 43.7% | 0.936 | 0.474 |

**Key findings:**
- **α=0.1 is dramatically better** — 77% overall satisfaction vs 43% baseline (~2x). Small α = sharper softmax(`reward/α`) = stronger guidance.
- **α=1.0 and α=5.0 barely move the needle** — softmax too flat, guidance doesn't steer.
- **K=16 > K=4** within same α (77% vs 68.5% at α=0.1).
- **Hardest constraint**: `no_bath_kitchen` (52% → 77.5% at best). Easiest: `one_living` (100% everywhere).
- **Quality impact at α=0.1**: diversity drops ~4% (0.945→0.909), cond. edge TV slightly worse (0.472→0.517). Mode coverage, spatial transitivity, node TV essentially unchanged.
- **Validity**: 100% across all configs. No degradation.

**Interpretation:**
- `weights = softmax(reward / α)`: small α → concentrated weights → strong selection pressure
- The sweet spot is around α=0.1. Need finer grid around this value.
- The 4 constraints in `configs/guidance/example_basic.yaml` are: one_kitchen, one_living (trivially satisfied), kitchen_near_living, no_bath_kitchen

## Current state of the codebase

### What exists and works
- Full guidance module (G1-G4): `bd_gen/guidance/` — all 124 tests pass
- Guided sampler with the `repeat` fix: generates valid guided samples
- Pilot scripts: `run_g5_pilot.sh`, `run_g5_experiments.sh`
- Analysis: `analyze_guidance_stats.py` with `--plot-distributions` and `--plot-trajectories`
- All results from 6-config pilot copied locally (JSONs + comparison table)
- 721 tests pass (3 pre-existing failures unrelated to guidance)

### What is partially done
- Distribution + trajectory PNGs generated on jabiru but **not yet copied locally**
- Results analysis started but not written up formally

### Known issues
- `analyze_guidance_stats.py` had Python 3.9 compatibility issue (fixed)
- `--alpha 0` gotcha in `generate_guided.py` line 219: `0.0` is falsy, falls through to YAML default. Not relevant for current experiments (α ≥ 0.01).
- 3 pre-existing test failures in `test_metrics.py` (`_aggregate_multi_seed` import renamed)

## What remains to be done

1. **Copy PNGs locally** from jabiru:
   ```bash
   scp "amine.chraibi@jabiru.polytechnique.fr:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear_noise_sc/*.png" BD_Generation/eval_results/loglinear_noise_sc/
   ```

2. **Review distribution + trajectory plots** — the user is most interested in:
   - Whether violations bounce back after being resolved
   - Whether guidance fights the sampling process
   - ESS distribution (degenerate or healthy?)

3. **Discuss and write up results** — user wants to comment on pilot results before next experiments

4. **Decide on finer α sweep** — candidates: α ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.5} with K=16
   - Also consider: hard mode comparison at α=0.1

5. **Expand to other model variants** (if pilot analysis is satisfactory):
   - v1 + confidence remasking (tsw=1.0)
   - v2 + no remasking
   - v2 + confidence remasking

6. **Consider different/additional constraints** — current set has one trivially satisfied constraint (one_living)

## Files to reference in next session

1. `BD_Generation/eval_results/loglinear_noise_sc/comparison_guided_pilot.md` — the pilot comparison table (7 models)
2. `BD_Generation/configs/guidance/example_basic.yaml` — the 4 constraints used
3. `BD_Generation/configs/guidance/calibration_v1_no_remask.json` — P90 normalizers
4. `BD_Generation/bd_gen/guidance/guided_sampler.py` — the fixed guided sampler
5. `BD_Generation/scripts/run_g5_pilot.sh` — pilot experiment script
6. `BD_Generation/scripts/analyze_guidance_stats.py` — plotting script
7. `BD_Generation/implementation_state_T1_guidance.md` — full G5 state
8. `BD_Generation/eval_results/loglinear_noise_sc/*.png` — plots (once copied locally)

## Context for the next session

- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`
- **v1 checkpoint**: `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt`
- **Seeds**: `[42, 123, 456, 789, 1337]` from `configs/eval/default.yaml`
- **Jabiru Python**: 3.9 — avoid backslashes in f-string expressions
- **Calibration P90 values**: `{one_kitchen: 1.0, one_living: 1.0, kitchen_near_living: 1.0, no_bath_kitchen: 2.0}`
- **α semantics**: `weights = softmax(reward / α)`. Small α = aggressive guidance. Large α = mild. The pilot showed α=0.1 is in the sweet spot; α=1.0+ is too weak.
- **Remasking delta always 0** in this pilot (no-remasking variant)
- **The `repeat_interleave` bug was critical** — all prior guided sample runs (if any existed before this fix) would have produced garbage results due to cross-batch mixing
