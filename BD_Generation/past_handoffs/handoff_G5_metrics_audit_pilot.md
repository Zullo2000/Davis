# Handoff: G5 Metrics Audit & Pilot Experiment Prep

> **Created:** 2026-02-27
> **Session purpose:** Audit guidance evaluation metrics for correctness, fix comparison table gap, prepare 6-config pilot experiment

---

## What was accomplished

- **Full audit of guidance evaluation metrics** — all constraint violations, satisfaction metrics, reward→weight flow verified correct
  - `edge_triples` only contains real adjacencies (rel 0-9, not NO_EDGE) — RequireAdj/ForbidAdj iterate correctly
  - Reward flow: `energy = Σ (λ/p90) * φ(violation)`, `reward = -energy`, `weights = softmax(rewards/α)` — correct SVDD
  - Satisfaction metrics in `evaluate.py`: `satisfaction_{name}`, `satisfaction_overall`, `mean_violation*` all correct

- **Fixed comparison table gap in `eval_results/save_utils.py`:**
  - Added `_build_constraint_table()` — dynamically detects all `constraint/*` keys from result JSONs, renders satisfaction rates as %, violations as floats, delta column for 2-model comparisons
  - Added guidance config params (K, alpha, reward_mode, phi, num_constraints) to the config comparison table
  - All 593 tests pass (2 skipped, 1 pre-existing failure in test_metrics.py unrelated)

- **Created `scripts/run_g5_pilot.sh`** — focused 6-config pilot experiment (v1 + llada + top-p=0.9 + no remasking, α∈{0.1,1.0,5.0} × K∈{4,16})

- **Created `fake_examples_guidance_stats.md`** — detailed walkthrough with fabricated but realistic data showing every output the user will see (calibration JSON, generation logs, evaluation logs, full comparison table, GuidanceStats diagnostics). Includes interpretation guide.

- **Updated `implementation_state_T1_guidance.md`** — G5 status reflects pilot approach, revised experiment plan, new files listed

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Pilot before full grid | 6 configs (v1 no-remask only) | Validate metrics and comparison pipeline before committing to 60-run grid |
| α grid for pilot | {0.1, 1.0, 5.0} | Covers gentle, moderate, aggressive — wide enough to see trends |
| K grid for pilot | {4, 16} | Tests low vs high candidate count; skips K=8 for speed |
| Skip soft-vs-hard step | Default to soft (from YAML) | Can revisit after pilot validates the pipeline |
| φ = linear | Keep default | P90 calibration already normalizes scales; no reason for nonlinear shaping before seeing results |
| Dynamic constraint table | Auto-detect `constraint/*` keys | Constraint names are config-dependent; can't hardcode in METRIC_FAMILIES |

## Current state of the codebase

### What exists and works
- Full guidance module (G1-G4 complete): `bd_gen/guidance/` — constraints, soft violations, reward composer, guided sampler, calibration
- All scripts ready: `generate_guided.py`, `calibrate_constraints.py`, `evaluate.py --guidance-config`, `analyze_guidance_stats.py`, `run_g5_pilot.sh`
- Comparison tables now include constraint satisfaction section (dynamic)
- 593 tests pass, no regressions

### What is partially done
- G5 experiments: prep complete, **not yet run on jabiru**
- No calibration JSONs yet (produced by step 1 on jabiru)
- `fake_examples_guidance_stats.md` created for user review — user wanted to comment on it in a new session before running real experiments

### Known issues
- 3 pre-existing test failures in `test_metrics.py` (`_aggregate_multi_seed` import renamed but tests not updated). Unrelated to guidance.

## What remains to be done

1. **User reviews `fake_examples_guidance_stats.md`** — may have comments on what metrics to add/change
2. **Sync code to jabiru** (git push + pull)
3. **Run pilot on jabiru:**
   ```bash
   cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation
   bash scripts/run_g5_pilot.sh all    # or step-by-step: calibrate → generate → evaluate → compare
   ```
4. **Copy results back locally** (scp the JSONs + comparison_guided_pilot.md)
5. **Review pilot results** — check satisfaction vs quality tradeoff, decide on reward mode and whether to expand grid
6. **Full grid** (60 configs across 4 variants) if pilot looks good

## Files to reference in next session

1. `BD_Generation/fake_examples_guidance_stats.md` — the fake examples doc the user wants to comment on
2. `BD_Generation/implementation_state_T1_guidance.md` — current state (G5 section)
3. `BD_Generation/scripts/run_g5_pilot.sh` — the 6-config pilot script
4. `BD_Generation/eval_results/save_utils.py` — updated with `_build_constraint_table()` and guidance config params
5. `BD_Generation/scripts/analyze_guidance_stats.py` — GuidanceStats trajectory analysis
6. `BD_Generation/planning_T1_guidance.md` — spec (Section 13 = G5, Section 11 = diagnostics)

## Context for the next session

- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`
- **v1 checkpoint**: `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt`
- **Remasking delta will always be 0** for the pilot (no-remasking variant). Only meaningful when testing confidence remasking later.
- **`--alpha 0` gotcha**: `generate_guided.py` line 219 uses `guidance_args.alpha or guidance_config.alpha` — 0.0 is falsy so would fall through to YAML default. Not relevant for pilot (α=0.1 is truthy) but worth knowing.
- **φ choice is not critical yet** — linear is fine for initial validation. If one constraint dominates energy in results, consider log1p.
- **The 7 models in pilot comparison**: 1 unguided baseline + 6 guided (3α × 2K). Baseline is re-evaluated WITH `--guidance-config` so constraint metrics appear for it too.
