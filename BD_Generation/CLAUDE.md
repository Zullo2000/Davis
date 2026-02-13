# BD_Generation — Agent Rules

## Session startup
- At the start of each session, read `BD_Generation/implementation_state_T1.md` to understand current state.
- Refer to `BD_Generation/planning_T1.md` as the implementation spec (source of truth for what to build).
- Only read `BD_Generation/handoff_T1.md` if the user explicitly asks you to resume from a handoff.

## Spec and state files
- `planning_T1.md` is the STATIC spec. Only update it for spec bugs. 
- `implementation_state_T1.md` is the DYNAMIC state. Update it after each phase with a compact summary.

## Phase workflow
- Only mark a Phase as "COMPLETE" when the user explicitly tells you to. Until the previous phase is COMPLETE you cannot proceed to the next one.
- Parallelize with sub-agents the workstreams within a single phase as described in `planning_T1.md`, but NOT across phases. Only in this implementation because it is the first one and we prioritize comprehension over velocity.
- Create a documentation `.md` file in `docs/` for each new module.

## Git conventions
- Branch naming: `feature-type/description` (e.g., `data/graph2plan-loader`, `model/transformer-denoiser`)
- Commit messages: conventional commits (`feat(scope):`, `test(scope):`, `docs:`)
- Merge to `main` with `--no-ff`, then tag (`v0.1.0`, `v0.3.0`, etc.). Never merge to main, create tags, or push to remote without explicit user approval.

## Context management
- After every 3-4 tool calls, estimate whether you are approaching context (tokens) limits. If the conversation has been going on for a long time (many tool calls, large file reads) and the context is near 80%, proactively warn the user so they can run `/create-handoff` to continue in a fresh session.
