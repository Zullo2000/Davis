# Handoff: v2 Learned Forward Process Planning

> **Created:** 2026-02-20
> **Session purpose:** Plan v2 implementation (MELD-inspired learned forward process) and produce a full standalone spec

---

## What was accomplished

- **Created full standalone spec:** `BD_Generation/planning_T1_with_learned_forward_process.md` (~1000 lines)
  - 14 sections covering: motivation (state clashing), rate network architecture, STGS gradient flow, per-position ELBO loss, sampling changes, training script, config, evaluation plan, implementation phases, 41 test cases
- **Deep codebase exploration:** Read all core modules (`noise_schedule.py`, `forward_process.py`, `loss.py`, `sampling.py`, `denoiser.py`, `embeddings.py`, `remasking.py`, `train.py`, `evaluate.py`, `checkpoint.py`, `vocab.py`)
- **MELD paper analysis:** Extracted algorithmic details from arXiv:2505.16790 (Section 3.3), including polynomial rate parameterization, STGS, per-element embeddings, training objective
- **Design decisions discussed with user:** 4 key choices resolved via Q&A

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Gradient flow | Full STGS (not weight-only) | Captures inter-position coupling; nodes constrain edges. User agreed after explanation of Path 1 vs Path 2 |
| Rate embeddings | Independent d=32 (not shared with denoiser) | "Clean decoupling": zero shared params, zero gradient cross-talk, safe v1 reuse |
| Training script | Separate `train_v2.py` (not extending `train.py`) | v2 loop differs fundamentally: STGS, per-position weights, Gumbel annealing, joint optimizer |
| Doc format | Full standalone spec (not delta document) | User preference for self-contained reference |
| Eval approach | No remasking initially | Isolates learned-rate effect; remasking adaptation deferred to post-v2 |
| Timestep sampling | Uniform (no importance sampling) | Per-position rates invalidate v1 IS transformation; w_l(t) already provides adaptive emphasis |

## Current state of the codebase

- **v1 is complete and working:** All phases done, 22-run remasking comparison complete, best model selected (`llada_topp0.9_remdm_confidence_tsw0.5`)
- **v2 spec is DRAFT:** Written but user indicated they have "some considerations and potential changes" before approving
- **No v2 code written yet:** Spec only, no implementation started
- **Uncommitted changes:** The new spec file + various doc updates (see `git status`)

## What remains to be done

1. **User review of spec** — User said they have considerations/changes to discuss
2. **Implement Phase 1:** `bd_gen/diffusion/rate_network.py` — RateNetwork with polynomial parameterization
3. **Implement Phase 2:** STGS + `forward_mask_learned()` in `forward_process.py`
4. **Implement Phase 3:** `ELBOLossV2` in `loss.py`
5. **Implement Phase 4:** Add `pre_embedded` param to `BDDenoiser.forward()`
6. **Implement Phase 5:** Per-position alpha in `sample()`
7. **Implement Phase 6:** `scripts/train_v2.py` + configs
8. **Implement Phase 7:** Eval integration, train, compare with v1

## Files to reference in next session

**Read first (planning & state):**
1. `BD_Generation/planning_T1_with_learned_forward_process.md` — **THE v2 SPEC** (start here)
2. `BD_Generation/implementation_state_T1.md` — Current v1 implementation state
3. `BD_Generation/CLAUDE.md` — Agent rules and workflow conventions

**Core code to modify (read before implementing):**
4. `bd_gen/diffusion/forward_process.py` — v1 forward_mask (add learned variant)
5. `bd_gen/diffusion/loss.py` — v1 ELBOLoss (add ELBOLossV2)
6. `bd_gen/diffusion/sampling.py` — v1 sample() (add per-position alpha)
7. `bd_gen/model/denoiser.py` — BDDenoiser (add pre_embedded param)
8. `bd_gen/model/embeddings.py` — CompositePositionalEncoding (pattern reference for rate network buffers)
9. `scripts/train.py` — v1 training loop (pattern reference for train_v2.py)

**Context docs:**
10. `BD_Generation/eval_results/loglinear/best_model_according_to_preferred_metrics_on_llada_unm.md` — v1 best model analysis (comparison target)
11. `BD_Generation/docs/evaluation.md` — Evaluation metrics documentation

## Context for the next session

- **MELD paper (arXiv:2505.16790):** Key formula is the polynomial rate parameterization: `γ̂_l(t) = Σ w_k t^k / Σ w_k`, `α_l(t) = σ(-γ_l(t))`. Edge embeddings = sum of endpoint node embeddings. STGS uses Equations 7-8 from the paper.
- **Backward compatibility is critical:** User explicitly said "I don't want the STGS code to corrupt what we already have because it is more likely that in the future I will use what I already have with the fixed forward process and remasking." All changes must be additive (new functions, optional params with None defaults).
- **Denoiser change is minimal:** ONE optional `pre_embedded` parameter. When None → v1 path unchanged. Only train_v2.py passes it.
- **The spec may need revisions:** User rejected ExitPlanMode and said they have considerations. The next session should start by asking the user what changes they want before implementing.
- **Vocab:** 8 nodes (NODE_VOCAB=15), 28 edges (EDGE_VOCAB=13), SEQ_LEN=36, MASK/PAD indices at end of each vocab.
- **v1 best model config:** llada unmasking, top-p 0.9, confidence remasking, t_switch 0.5, log-linear schedule, 100 steps, 1000 samples × 5 seeds.
