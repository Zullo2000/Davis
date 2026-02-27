# USING LLM TO WRITE THE CONSTRAINTS

## 1) Define a minimal constraint DSL (JSON works well)

Make it explicit and machine-checkable (Pydantic / JSON schema):

Example (hard decoded graph version):

``` json
{
  "name": "kitchen_adjacent_living",
  "type": "RequireAdj",
  "a": "Kitchen",
  "b": "LivingRoom",
  "relation": "ADJ",
  "violation": "shortest_path_minus_1",
  "weight": 1.0
}
```

Example (count constraint):

``` json
{
  "name": "exactly_one_kitchen",
  "type": "ExactCount",
  "room_type": "Kitchen",
  "target": 1,
  "violation": "abs_diff",
  "weight": 1.0
}
```

Example (conditional requirement):

``` json
{
  "name": "bathrooms_rule",
  "type": "ConditionalRequirement",
  "if": {"type": "CountRange", "room_type": "Bedroom", "min": 3, "max": 4},
  "then": {"type": "MinCount", "room_type": "Bathroom", "min": 2},
  "else": {"type": "MinCount", "room_type": "Bathroom", "min": 1},
  "weight": 1.0
}
```

This mirrors the primitives list you wrote (ExactCount, CountRange,
RequireAdj, ForbidAdj, Connectivity, AccessRule,
ConditionalRequirement).

------------------------------------------------------------------------

## 2) Implement primitives once, correctly (your "constraint VM")

This is where PAD handling and node/edge structure are enforced
correctly.

Your representation is a fixed-length token sequence with node slots and
edge slots, and PAD positions are invariant and must be ignored.

So your engine should expose something like:

-   `decode_graph(x0_tokens) -> Graph`
    -   drops PAD rooms and incident edges\
    -   builds adjacency\
    -   provides helpers: counts, distances, components, degrees, etc.

Then each primitive computes a nonnegative violation magnitude v_i(x)
(not just satisfied/unsatisfied), consistent with the dense reward
design you're aiming for.

------------------------------------------------------------------------

## 3) Use an LLM to translate natural language constraints → DSL

You prompt the LLM with: - the DSL schema, - allowed room types and
relations, - a few examples.

And you force JSON output.

Key detail: you always validate the JSON against a schema. If parsing
fails → reject.

This makes the LLM useful without trusting it blindly.

------------------------------------------------------------------------

## 4) Compile DSL → executable v_i

Your compiler maps: - `"type": "RequireAdj"` →
`RequireAdjConstraint(...)` - `"type": "Connectivity"` →
`ConnectivityConstraint(...)`, etc.

So adding a new constraint is usually just "new JSON," not "new code."

------------------------------------------------------------------------

# Bonus: make the LLM generate *tests*, not just the constraint

The most underrated trick: after the LLM outputs a constraint spec, ask
it to output **unit tests** (tiny graphs) that should pass/fail and
their expected violation values.

Example prompt: \> "Given this constraint, produce 5 minimal graphs
(node types + edges) that satisfy it and 5 that violate it. For each,
give expected v(x). Keep graphs ≤6 nodes."

Then you run those tests in CI.

This is how you make LLM-assisted constraint authoring safe.

------------------------------------------------------------------------

# How to handle "soft" violations (important for diffusion)

You already wrote down the key move: at timestep x_t, you have per-slot
distributions q from the denoiser logits, so you can compute expected
counts / expected forbidden edges / approximate probabilities of
existence, etc.

So for each DSL constraint, support:

-   `mode: "hard"` → compute on decoded x̂\_0
-   `mode: "soft"` → compute on posterior means from logits
-   `mode: "hybrid"` → soft early, hard late (or confidence-gated)

Then v_i can be defined systematically without bespoke math per
constraint:

-   **ExactCount**: n̂ = Σ_s q_s(type), v = \|n̂ − target\|
-   **ForbidAdj**: expected forbidden adjacency count via pairwise
    probabilities
-   **RequireAdj**: approximate probability that exists an adjacency and
    set v = 1 − P(exists)

This is a huge win for SVDD-style guidance because it reduces reward
noise when states are still very uncertain.

Where the LLM helps: given a natural-language constraint, it can decide
whether the constraint maps cleanly to a soft version (counts/adjacency)
or needs hybrid/hard-only (connectivity, bedroom access).

------------------------------------------------------------------------

# A Concrete Workflow You Can Implement Quickly

1)  **Write primitives** (the exact list you already have is a great
    start).\
2)  **Write the schema** (JSON + Pydantic) for constraints +
    composition.\
3)  **Write the compiler** (schema → Python objects).\
4)  **LLM prompt 1:** natural-language constraint → JSON spec.\
5)  **LLM prompt 2:** JSON spec → test cases (+ expected v).\
6)  Run:
    -   schema validation,
    -   unit tests,
    -   random fuzz tests,
    -   distribution sanity checks (evaluate v_i on unguided samples for
        scaling), which you already know matters when rewards are
        exponentiated in guidance.


------------------------------------------------------------------------

# Can the LLM create new minimal constraints DSL?

------------------------------------------------------------------------

# What Actually Happens When a New Constraint Appears


Suppose you give the LLM a prompt like:

> "The kitchen must be within distance 2 of the entrance."

And your DSL does not include a `MaxGraphDistance` primitive.

Two possible outcomes:

### Case 1 --- The LLM maps it to existing primitives

It might produce:

``` json
{
  "type": "RequireAdj",
  "a": "Kitchen",
  "b": "Entrance",
  "relation": "PATH_DISTANCE",
  "max_distance": 2
}
```

If your schema does not allow `"relation": "PATH_DISTANCE"`, validation
fails.

That tells you:

> This is a new primitive class, not just a new parameterization.

That failure is valuable information.

------------------------------------------------------------------------

### Case 2 --- The LLM invents a new type

``` json
{
  "type": "MaxDistance",
  "room_a": "Kitchen",
  "room_b": "Entrance",
  "max_distance": 2
}
```

Your schema rejects `"MaxDistance"`.

Again: good.

It means:

> The DSL is missing a primitive.

The failure is not a bug --- it is a signal that the constraint space
has expanded beyond what your engine currently supports.

------------------------------------------------------------------------

# So Should the DSL Ever Grow?

Yes --- but only when **you decide to extend it**, not when the LLM
improvises.

The workflow becomes:

1.  The LLM proposes a constraint.
2.  The JSON fails validation.
3.  You inspect the proposed structure.
4.  You decide:
    -   Can this be expressed using existing primitives?
    -   Or is it a genuinely new structural pattern?
5.  If it is genuinely new → you implement a new primitive once.
6.  You add it to the DSL schema.
7.  From that point on, it is supported safely and systematically.

This keeps the system:

-   Finite\
-   Auditable\
-   Composable\
-   Deterministic\
-   Safe for reward shaping and SVDD integration

The DSL grows slowly and deliberately. Each new primitive becomes a
reusable building block rather than an ad-hoc rule.

------------------------------------------------------------------------

# Important Concept: Closed-World vs Open-World DSL

You are building a **closed-world DSL**.

Closed-world means:

> Only constraint types explicitly defined in your schema are allowed.

If the LLM produces a constraint type not listed in the schema,
validation fails immediately.

This is essential because:

-   You must implement both hard and soft violation versions.
-   You must respect PAD invariants and graph decoding rules.
-   You must support early diffusion soft scoring.
-   You must normalize violation magnitudes consistently.
-   You must integrate constraints into a stable SVDD-style energy.

An open-world DSL --- where the LLM invents types freely and they are
automatically accepted --- would break these guarantees. It would make
scaling, calibration, and reproducibility impossible.

The correct mental model is:

-   The DSL is a typed abstract syntax tree (AST).
-   The LLM generates AST instances.
-   Your constraint engine executes them.
-   If the AST contains a node type you do not support, compilation
    fails.

That failure is correct behavior.

The system remains extensible --- but only under controlled evolution.
