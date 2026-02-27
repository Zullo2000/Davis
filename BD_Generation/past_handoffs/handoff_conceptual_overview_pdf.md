# Handoff: Conceptual Overview PDF — User Edits Pending

> **Created:** 2026-02-25
> **Session purpose:** Created a 14-page PDF document covering masked diffusion concepts, experimental analysis, and v2+remasking results. User wants to make reasoning-heavy edits in a follow-up session.

---

## What was accomplished

- **Created** `BD_Generation/masked_diffusion_conceptual_overview.md` — comprehensive markdown source (459 lines)
- **Generated** `BD_Generation/masked_diffusion_conceptual_overview.pdf` — 14-page PDF via pandoc + pdflatex (419 KB)
- Document covers 13 sections:
  1. Introduction (tokenization: 8 nodes + 28 upper-triangle edges = 36 positions)
  2. Forward process (fixed schedule v1, alpha(t), PAD invariance)
  3. Backward process (reverse denoising loop, carry-over property)
  4. Unmasking modes (random vs LLaDA confidence top-k)
  5. Noise schedules (linear vs log-linear, why "linear" is exponential in alpha)
  6. ELBO weight w(t) and importance sampling
  7. Token prediction (argmax, temperature, top-p)
  8. Remasking / ReMDM (cap vs confidence, t_switch, quality tradeoffs)
  9. Learned forward process v2/MELD (state clashing, rate network architecture, STGS)
  10. Inside validity and why LLaDA is necessary
  11. Priority metrics (definitions, TV/JS over KL, JS redundancy, weighted conditioned)
  12. Remasking strategy comparison (LLaDA only, cap vs confidence synthesis)
  13. v2 + remasking (per-position sigma_max, results table, Pareto front)
- Written in first-person plural ("we") as if the user authored it
- Corrected the edge(A,B)/edge(B,A) error from earlier session — Section 9.1 correctly notes upper-triangle only, explains real dependencies (shared nodes, node-edge, transitive chains)

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| PDF toolchain | pandoc + pdflatex (MiKTeX) | Both already installed on user's Windows system |
| LaTeX escaping | Replaced `$\sim$` with `~`, `\%` with `%` | Pandoc was mis-parsing `$\sim$` as `\$\sim` in certain markdown contexts |
| Document voice | First-person plural ("we") throughout | User requested it should look like they wrote it |
| Scope | All session Q&A + reference doc content | User wanted conceptual answers + metric analysis + v2 remasking in one document |

## Current state

- The markdown source was modified by the user (or linter) after generation — the system-reminder shows the file was changed but the diff only shows the existing content (no visible additions/deletions). The user said they want to "do some changes that require reasoning" in another session.
- The PDF was successfully generated and verified (14 pages, all sections render correctly with proper LaTeX math, tables, and formatting).

## What remains to be done

1. **User edits to the markdown** — user mentioned needing to make "changes that require reasoning." The specific changes are unknown. Read the `.md` file at session start to see what they want.
2. **Regenerate PDF after edits** — command: `cd BD_Generation && pandoc masked_diffusion_conceptual_overview.md -o masked_diffusion_conceptual_overview.pdf --pdf-engine=pdflatex -V colorlinks=true -V linkcolor=blue`
3. Possible additions the user might want (speculative):
   - More detailed metric tables from `comparison.md`
   - Figures or diagrams
   - References/bibliography

## Files to reference in next session

1. `BD_Generation/masked_diffusion_conceptual_overview.md` — **READ FIRST** — the markdown source to edit
2. `BD_Generation/masked_diffusion_conceptual_overview.pdf` — current PDF output
3. `BD_Generation/eval_results/loglinear/comparison.md` — full 24-method comparison table (raw data)
4. `BD_Generation/eval_results/loglinear/best_model_according_to_preferred_metrics_on_llada_unm.md` — priority metric analysis
5. `BD_Generation/remasking_design_with_learned_forward_process.md` — v2 remasking design doc
6. `BD_Generation/planning_T1_with_learned_forward_process.md` — v2 planning doc (rate network, STGS details)

## Context for the next session

- **Pandoc gotcha**: `$\sim$` gets mis-parsed by pandoc when inside certain markdown constructs (bold, inline). Use `~` or "approximately" instead. Same issue may affect other LaTeX-in-markdown constructs.
- **PDF regeneration command**: `pandoc masked_diffusion_conceptual_overview.md -o masked_diffusion_conceptual_overview.pdf --pdf-engine=pdflatex -V colorlinks=true -V linkcolor=blue` (run from `BD_Generation/` directory)
- **The document originated from a Q&A session** where the user asked conceptual questions about masked diffusion, noise schedules, w(t)/IS, rate network/STGS. The answers were compiled into this document with extensions from the reference docs.
- **Key correction**: Section 9.1 (state clashing) was corrected from an earlier error where edge(A,B)=left-of and edge(B,A)=right-of were presented as separate positions. The tokenization uses upper-triangle only (one position per pair).
