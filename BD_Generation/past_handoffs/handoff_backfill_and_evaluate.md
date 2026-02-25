# Handoff: Backfill Samples & Re-evaluate All Models

> **Created:** 2026-02-24
> **Session purpose:** Commit pipeline split code, push to remote, launch GPU backfill of all 35 models on jabiru, then re-evaluate with new CPU-only evaluate.py.

---

## What was accomplished

- **Fixed missing `inside_validity` export** from `bd_gen/eval/__init__.py` (was defined in metrics.py but not exported)
- **Fixed 8 ruff lint errors** across 4 files (line length, unused import, f-string)
- **Committed** as `1f6c521`: `feat(eval): split eval pipeline (generate vs evaluate) + inside_validity metric`
- **Pushed** to `origin/main`
- **Pulled** on jabiru (`/Data/amine.chraibi/Davis`)
- **Launched GPU backfill** of all 35 models (23 loglinear + 12 linear) on jabiru — currently running

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Run order | Test 1 model first, then batch 34 | Catch errors early before committing to 50+ min batch |
| `&&` chaining | All 34 remaining runs chained with `&&` | Stop on first failure to avoid wasting GPU time |
| v2 noise override | `noise=loglinear` for v2 checkpoint | v2 uses `noise.type=learned` internally but results belong in `eval_results/loglinear/` |
| No tmux | User preference | User explicitly said they don't use tmux |

## Current state of the codebase

- **Local (Windows):** clean working tree, 1 commit ahead of origin (already pushed)
- **Jabiru:** `git pull` done, backfill running. 35 `_samples.pt` files being generated into:
  - `BD_Generation/eval_results/loglinear/*_samples.pt` (23 files)
  - `BD_Generation/eval_results/linear/*_samples.pt` (12 files)
- **No failing tests** — all lint clean before commit

### Checkpoints used for backfill

| Schedule | Checkpoint path |
|---|---|
| loglinear (v1) | `outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt` |
| linear (v1) | `outputs/2026-02-19_15-24-27/checkpoints/checkpoint_final.pt` |
| v2 (MELD) | `outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt` |

## What remains to be done

1. **Wait for jabiru backfill to finish** (~50-70 min total, ~1-2 min per model)
2. **Copy `.pt` files from jabiru to local** — the `.pt` files are in `eval_results/{schedule}/` on jabiru but NOT tracked by git (too large). Options:
   - `scp jabiru:/Data/amine.chraibi/Davis/BD_Generation/eval_results/loglinear/*_samples.pt BD_Generation/eval_results/loglinear/`
   - `scp jabiru:/Data/amine.chraibi/Davis/BD_Generation/eval_results/linear/*_samples.pt BD_Generation/eval_results/linear/`
   - OR run evaluate.py directly on jabiru (it's CPU-only, works anywhere)
3. **Run CPU-only evaluation on all models:**
   ```bash
   cd BD_Generation
   python scripts/evaluate.py --schedule loglinear --update-comparison
   python scripts/evaluate.py --schedule linear --update-comparison
   ```
   This recomputes ALL metrics (including new `inside_validity`) from saved tokens and updates the JSON files + comparison.md tables.
4. **Verify consistency** — spot-check a few models: metrics from new evaluate.py should match existing JSONs (except for newly added `inside_validity` which won't have existed before)
5. **Commit updated JSONs + comparison.md** after re-evaluation
6. **Delete `scripts/generate_and_evaluate.py`** — transition backup, no longer needed once all models have `_samples.pt`
7. **Add `.pt` to `.gitignore`** if not already there (sample files are ~1.6 MB each, 56 MB total — may not want in git)

## Files to reference in next session

1. `BD_Generation/handoff_eval_pipeline_split.md` — original pipeline split design decisions
2. `BD_Generation/scripts/evaluate.py` — CPU-only metric script (the one to run next)
3. `BD_Generation/scripts/generate_samples.py` — GPU generation script (already ran on jabiru)
4. `BD_Generation/eval_results/save_utils.py` — shared utilities (aggregate_multi_seed, comparison table)
5. `BD_Generation/implementation_state_T1.md` — last section has pipeline split status

## Context for the next session

- **Jabiru access:** `ssh jabiru` then `cd /Data/amine.chraibi/Davis && source .venv/bin/activate && cd BD_Generation`
- **The `.pt` files are NOT in git.** They live only on jabiru after generation. Must be copied locally (scp) or evaluation must run on jabiru.
- **Running evaluate.py on jabiru is fine** — it's CPU-only and jabiru has the venv with all dependencies. This avoids the scp step entirely.
- **`eval.top_p=null` worked fine** with Hydra on jabiru (no need for `~eval.top_p` workaround)
- **The v2 model** auto-detects from checkpoint (presence of `rate_network_state_dict` key) and gets `v2_` prefix automatically
- **After evaluation, the comparison.md files will have the new `inside_validity` metric** in the Validity section of the table
- **`generate_and_evaluate.py`** is the old monolithic script kept as transition backup. Safe to delete after confirming all 35 models have working `_samples.pt` + updated `.json`
