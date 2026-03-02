# Handoff: G5 K* Sweep Results + Next: α Fine-Tuning

> **Created:** 2026-03-02
> **Session purpose:** Ran K* sweep to find minimal K for constraint satisfaction, analyzed results, now ready for α fine-tuning at chosen K*.

---

## What was accomplished

- **Designed and ran K* sweep experiment** — 8 K values × 2 variants = 16 configs, 200 samples each
  - Script: `scripts/run_g5_kstar.sh` (new, supports `noremask` and `confidence` variants)
  - Ran on jabiru via tmux: `bash scripts/run_g5_kstar.sh noremask all && bash scripts/run_g5_kstar.sh confidence all`
  - Total runtime: ~1h 41min (53 min no-remask + 48 min confidence) under GPU contention
- **Analyzed results** — identified K* for both variants
- **Documented findings** in `docs/guidance.md` §Round 2 — K* sweep (setup, timing, results tables, 6 findings, per-constraint breakdowns, recommendations)
- **Updated** `implementation_state_T1_guidance.md` — K* sweep marked DONE, experiment plan updated with Round 3 (α fine-tuning)

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Supersede Round 2 fine α sweep | K* sweep instead | Round 2 was too slow at 5000 samples; K* is more actionable |
| Reduced samples | 2 seeds × 100 = 200/config | CI ≈ ±6%, sufficient for plateau detection |
| K grid | {4,8,10,12,14,16,20,24} | Dense in 8–16 transition zone, anchors at 4 and 24 |
| Run variants sequentially | noremask first, then confidence | GPU contention — avoid doubling time |

## K* sweep results summary

### No-remasking: K* ≈ 12
- Baseline: 13.3% → K=12: 56.5% (plateau through K=24 at ~53–57.5%)
- All within ±6% CI from K=12 onward

### Confidence remasking: K* > 24 (no plateau)
- Baseline: 16.7% → K=24: 61.0% (still climbing)
- Remasking fights guidance → needs more candidates

### Key finding
Remasking shifts K* UP. The "self-correction" hypothesis is not supported.

### Bottleneck
`no_bath_kitchen` (ForbidAdj) is the hardest constraint — other 3 saturate by K=8–10.

### Quality tradeoff
Mode coverage drops ~20pp, cond. edge TV worsens ~+0.10. Validity unaffected.

## Current state

- **G1–G4**: COMPLETE (constraints, soft violations, guided sampler, calibration)
- **G5**: IN PROGRESS
  - Round 1 (coarse α/K): DONE — α=0.1 sweet spot
  - Constraint revision: DONE — `one_living` → `between_2_and_3_bathrooms`
  - K* sweep: DONE — K*≈12 (no-remask), K*>24 (confidence)
  - **Next: Round 3 — α fine-tuning at chosen K***

## What remains to be done

1. **Round 3 — α fine-tuning** (NEXT SESSION):
   - No-remasking: sweep α ∈ {0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3} at K=12
   - Confidence: sweep α at K=16 (or K=20 if budget allows)
   - Same reduced setup (2 seeds × 100 = 200/config)
   - Create `scripts/run_g5_alpha.sh` (similar structure to `run_g5_kstar.sh`)
2. Consider hard reward mode comparison at best α/K
3. Expand to v2 variants if warranted
4. Mark G5 COMPLETE when tuning is done

## Files to reference in next session

1. `implementation_state_T1_guidance.md` — current state (read first per CLAUDE.md rules)
2. `planning_T1_guidance.md` — static spec
3. `docs/guidance.md` — Round 1 + Round 2 (K* sweep) findings, all accumulated knowledge
4. `scripts/run_g5_kstar.sh` — template for the α sweep script
5. `eval_results/loglinear_noise_sc/comparison_guided_kstar_noremask.md` — full K* results (no-remask)
6. `eval_results/loglinear_noise_sc/comparison_guided_kstar_confidence.md` — full K* results (confidence)

## Context for the next session

- **Calibration files exist on jabiru**: `configs/guidance/calibration_v1_no_remask.json` and `calibration_v1_confidence.json` — already calibrated for the revised constraint set. No need to re-calibrate.
- **Hydra seed/sample override syntax works**: `eval.seeds=[42,123] eval.num_samples=100` — verified in K* sweep.
- **Tag convention**: Round 1 used `basic`, K* sweep used `kstar`. Use a new tag (e.g., `alpha`) for α sweep to avoid filename collisions.
- **tmux on jabiru**: `tmux new -s <name>` / `tmux attach -t <name>`. Detach: `Ctrl+B` then `D`.
- **jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, workdir: `/Data/amine.chraibi/Davis/BD_Generation`
- **The baseline satisfaction dropped** from 43% (old constraint set with trivial `one_living`) to 13% (new set with hard `between_2_and_3_bathrooms`). α=0.1 was tuned on the old set — it may not be optimal for the new set.
