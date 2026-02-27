# BD_Generation — Agent Rules

current_static_spec = BD_Generation\planning_T1_guidance.md
current_dynamic_file = BD_Generation\implementation_state_T1_guidance.md

## Session startup
- At the start of each session, read current_dynamic_file to understand current state.
- Refer to current_static_spec as the implementation spec (source of truth for what to build).
- Only read the last `BD_Generation/handoff<last>.md` generated if the user explicitly asks you to resume from a handoff.

## Spec and state files
- current_static_spec is the STATIC spec. Only update it for spec bugs. 
- current_dynamic_file is the DYNAMIC state. Update it after each phase with a compact summary.

## effect of progress on old code
- if the implementation of a new model or a new module (ex: guidance) requires fundamental changes in the code always ask the permission emphasizing where they occur and their motivation and the pros and cons of those changes


## Phase workflow
- Only mark a Phase as "COMPLETE" when the user explicitly tells you to. Until the previous phase is COMPLETE you cannot proceed to the next one.
- Parallelize with sub-agents the workstreams within a single phase as described in current_static_spec
- Create a documentation `.md` file in `docs/` for each new module.

## Git conventions
- Branch naming: `feature-type/description` (e.g., `data/graph2plan-loader`, `model/transformer-denoiser`)
- Commit messages: conventional commits (`feat(scope):`, `test(scope):`, `docs:`)
- Merge to `main` with `--no-ff`, then tag (`v0.1.0`, `v0.3.0`, etc.). Never merge to main, create tags, or push to remote without explicit user approval.

## Context management
- After every 3-4 tool calls, estimate whether you are approaching context (tokens) limits. If the conversation has been going on for a long time (many tool calls, large file reads) and the context is near 80%, proactively warn the user so they can run `/create-handoff` to continue in a fresh session.
