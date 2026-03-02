# Handoff: G5 Pilot Results — Commentary & Next Steps

> **Created:** 2026-03-02
> **Session purpose:** Analyze and comment on the G5 pilot results (6-config SVDD guidance sweep), review distribution/trajectory plots, decide on next experiments.

---

## What was accomplished

- **G1–G4 complete** (constraint primitives, soft violations, guided sampler, calibration + eval)
- **Critical `repeat_interleave` bug fixed** (commit `693a07a`) — K-candidate expansion was mixing batch elements
- **6-config pilot completed on jabiru**: α ∈ {0.1, 1.0, 5.0} × K ∈ {4, 16}, all using v1 + llada + top-p=0.9 + no remasking, soft reward mode
- **All results copied locally**: 6 guided JSONs, comparison table, calibration JSON, 12 PNGs (distributions + trajectories)
- **Previous handoff** with full technical details: `past_handoffs/handoff_G5_pilot_results.md`

## Pilot results — key numbers

| Config | Satisfaction (all 4) | Diversity | Cond. Edge TV |
|--------|---------------------|-----------|---------------|
| **Baseline** (no guidance) | 43.3% | 0.945 | 0.472 |
| K=4, α=0.1 | 68.5% | 0.903 | 0.487 |
| **K=16, α=0.1** | **77.0%** | 0.909 | 0.517 |
| K=4, α=1.0 | 47.0% | 0.938 | 0.473 |
| K=16, α=1.0 | 48.5% | 0.932 | 0.477 |
| K=4, α=5.0 | 43.5% | 0.944 | 0.470 |
| K=16, α=5.0 | 43.7% | 0.936 | 0.474 |

**Takeaways from previous session (not yet written up formally):**
- α=0.1 is dramatically better (~2x baseline). α ≥ 1.0 barely moves the needle (softmax too flat).
- K=16 > K=4 within same α (77% vs 68.5% at α=0.1).
- Hardest constraint: `no_bath_kitchen` (52% → 77.5%). Easiest: `one_living` (100% everywhere).
- Quality impact at α=0.1: diversity drops ~4%, cond. edge TV slightly worse (+0.045). Mode coverage, spatial transitivity, node TV essentially unchanged.
- 100% validity across all configs — no degradation.

## What remains to be done

1. **Comment on / write up the pilot results** — this is the user's immediate goal for the next session
   - Interpret the distribution plots (reward distributions across candidates, ESS)
   - Interpret the trajectory plots (per-constraint violation over denoising steps)
   - Discuss whether guidance fights the sampling process or cooperates
   - Formally document findings

2. **Decide on finer α sweep** — candidates: α ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.5} with K=16
   - Also consider: hard reward mode comparison at α=0.1

3. **Expand to other model variants** (if pilot analysis is satisfactory):
   - v1 + confidence remasking (tsw=1.0)
   - v2 + no remasking
   - v2 + confidence remasking

4. **Consider different/additional constraints** — current set has one trivially satisfied constraint (`one_living`)

## Files to reference in next session

1. **This handoff** — `BD_Generation/past_handoffs/handoff_G5_results_commentary.md`
2. **Comparison table** — `BD_Generation/eval_results/loglinear_noise_sc/comparison_guided_pilot.md`
3. **Distribution plots** (6 PNGs):
   - `BD_Generation/eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_basic_K{4,16}_a{0.1,1.0,5.0}_distributions.png`
4. **Trajectory plots** (6 PNGs):
   - `BD_Generation/eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_basic_K{4,16}_a{0.1,1.0,5.0}_trajectories_seed42.png`
5. **Constraint config** — `BD_Generation/configs/guidance/example_basic.yaml`
6. **Calibration** — `BD_Generation/configs/guidance/calibration_v1_no_remask.json`
7. **Implementation state** — `BD_Generation/implementation_state_T1_guidance.md` (G5 section)
8. **Previous handoff** — `BD_Generation/past_handoffs/handoff_G5_pilot_results.md` (full technical details)

## Context for the next session

- **α semantics**: `weights = softmax(reward / α)`. Small α = aggressive guidance (concentrated weights). Large α = mild (flat softmax). The pilot showed α=0.1 is in the sweet spot; α ≥ 1.0 is too weak.
- **Reward mode**: all pilot runs used `soft` mode (smoother gradient signal). Hard mode comparison deferred.
- **Remasking delta always 0** in this pilot (no-remasking variant).
- **`--alpha 0` gotcha**: `0.0` is falsy in `generate_guided.py` line 219, falls through to YAML default. Not relevant for current α range.
- **Calibration P90 values**: `{one_kitchen: 1.0, one_living: 1.0, kitchen_near_living: 1.0, no_bath_kitchen: 2.0}`
- **Jabiru SSH**: `ssh amine.chraibi@jabiru.polytechnique.fr`, working dir: `/Data/amine.chraibi/Davis`
- **Jabiru Python**: 3.9 — avoid backslashes in f-string expressions
- **721 tests pass** (3 pre-existing failures unrelated to guidance)