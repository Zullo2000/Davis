# Reward Design for Inference-Time Guidance in Discrete Masked Diffusion Bubble-Diagram Generation

## Problem framing and current approach

You are generating **bubble diagrams** as **discrete graphs** using a **masked discrete diffusion** model, where a sample can be viewed as (i) a multiset/sequence of **room-type categorical tokens** (nodes) plus (ii) categorical tokens describing **pairwise relations** (edges). Your notes describe the representation as a fixed-length categorical structure with explicit node and edge slots, with padding used to represent “unused” rooms/edges. fileciteturn0file0

You want **inference-time guidance** (no fine-tuning) and are taking inspiration from **SVDD**, which frames reward-guided generation as sampling from a distribution that balances (a) high reward and (b) closeness to the pretrained model’s “naturalness” distribution. SVDD operationalizes this with **soft value functions** (look-ahead estimates from intermediate noisy states to terminal rewards) and an **importance-sampling-style selection step at each diffusion timestep**, enabling derivative-free use of (potentially non-differentiable) rewards, including in discrete diffusion. citeturn6view0turn5view1turn10view1

In your WIP section you currently propose a reward that is essentially a **weighted sum of constraint-violation indicators/penalties plus a hard “all constraints satisfied” bonus**. fileciteturn0file0 You were told (credibly) that this can be **too sparse**: if most partial or early-step candidates get nearly identical scores (or all receive “failure”), SVDD’s look-ahead/value-weighted resampling has little signal to exploit—even if the model could reach feasibility with a few discrete edits. This is a known pain point in reward-guided schemes when rewards are sparse or discontinuous. citeturn11view0turn13view0turn28view0

The goal of this report is to translate what the literature says into **concrete reward-function patterns** that are (i) informative for discrete graphs, (ii) compatible with SVDD-style weighting (where rewards/value typically appear inside an exponential weight), and (iii) targeted to your specific constraints.

## What the recent literature suggests about reward-guided discrete diffusion and sparsity

A unifying view across recent guidance papers and tutorials is that many methods are trying to sample from a **reward-tilted** distribution of the general form:

- “naturalness prior” from the pretrained model  
- multiplied by an “exponentiated reward” term  
- often with a temperature / scale hyperparameter controlling how strongly reward influences sampling. citeturn24view0turn26view2turn14view0

This matters for reward design because if the algorithm ultimately uses **weights like** `exp(reward / α)` (directly or via a value function), then:
1) **reward scaling is not cosmetic** (small numeric changes can become large multiplicative weight ratios), and  
2) **binary rewards** (0/1 feasibility bonuses) can create weight collapse or near-total degeneracy unless carefully temperature-tuned. citeturn26view2turn14view4turn10view0

### Sparse reward is sample-inefficient in gradient-free steering

On the discrete side, multiple works explicitly call out that a simple baseline—generate many samples and pick the best—becomes **sample-inefficient when reward signals are sparse**. citeturn11view0turn24view0 This applies directly to SVDD-like selection too: if many candidates tie at 0 (or near 0), selection becomes effectively random.

### Discrete rewards are often “non-smooth,” making intermediate evaluation noisy

A key discrete-specific issue: many rewards in scientific/discrete domains behave like **cliff functions**, where changing a single token can flip reward drastically (valid/invalid, connected/disconnected, etc.). A 2026 discrete diffusion paper proposes guidance that **bypasses intermediate rewards entirely** using a Metropolis–Hastings chain over *clean samples*, arguing that intermediate rewards computed on approximations (e.g., predicted x₀) can be uninformative when the reward is highly sensitive. citeturn13view0turn13view1

In graph generation guidance, related concerns appear as well: graph semantics can be **fragile under small structural perturbations**, so “local” stepwise heuristics can be noisy and short-sighted; this motivates lookahead and verifiers that estimate long-term reward from partially denoised graphs. citeturn22view0

### Projection / “constraints-by-construction” is an alternative to reward shaping

Several lines of work reduce reliance on hand-shaped rewards by **enforcing constraints directly in sampling**:

- **Constrained Discrete Diffusion (CDD)** integrates a projection step (including augmented-Lagrangian style objectives) to satisfy constraints while staying close to the model distribution; the paper emphasizes that augmented-Lagrangian penalties keep optimization signals informative even under constraint violation. citeturn1view1turn9view2  
- **ConStruct** addresses graph structural constraints by designing the diffusion process and a projector so that samples remain within the constrained set across the trajectory, aiming for hard guarantees on properties like planarity/acyclicity. citeturn20view1turn20view2  
- In discrete layout generation, **LayoutDM** supports conditional generation via **hard masking (“strong constraints”)** and **logit adjustment (“weak constraints”)**, and explicitly references the tradition of using **hand-crafted violation-degree cost functions** for layout constraints. citeturn19view0  

Even if you stick with SVDD, these works motivate a hybrid mindset: **use reward shaping only where necessary**, and consider “compile-able” constraints as clamping/projection/repair steps.

## Principles for choosing a dense reward in categorical graph domains

This section distills the above into reward-design principles that are **actionable** for your bubble-diagram constraints.

### Preserve the energy-based viewpoint: define costs/violations, let the sampler exponentiate

Many reward-guided samplers are best understood as sampling from something like a **Boltzmann / product-of-experts** distribution, where an objective/energy defines a density factor that concentrates around optima as temperature decreases. citeturn2view0turn26view2turn24view0

Implication: often the cleanest design is:

- define an **energy** (cost) `E(x)` as a **sum of dense violation magnitudes**, not a single feasibility indicator  
- then use `reward(x) = -E(x)` (possibly scaled / normalized), rather than embedding `exp(-x)` inside each term (because the *sampler itself* already applies an exponential weight). citeturn26view2turn10view0

This directly addresses the “exp(-x)” suggestion you heard—but with a nuance: **you usually want exactly one exponential**, not a double exponential, because SVDD/SMC-style weights already contain `exp(value/α)`. citeturn26view2turn14view4

### Make each constraint “graded” using a distance-to-satisfaction, not a binary check

For discrete constraints, a dense reward is less about continuity (impossible) and more about **having multiple informative levels**, so that:
- small edits tend to cause small score changes  
- partial progress gets credited  
- ties are rare.

This aligns with both (i) the augmented-Lagrangian idea of turning hard goals into smooth(ish) penalty landscapes citeturn9view2 and (ii) the observation that sparse terminal feedback leads to instability and motivates richer stepwise rewards. citeturn28view0

For graphs, “distance-to-satisfaction” often means “how many edges/nodes would need to change” or “how far in shortest-path space” rather than “is satisfied.”

### Control reward scale to avoid weight degeneracy and mode collapse

SMC treatments emphasize “weight degeneracy” (variance of importance weights exploding) as targets become sharper (temperature→0), requiring resampling schedules and careful scaling. citeturn14view3turn14view4 SVDD also warns that scaling can become problematic when the temperature-like hyperparameter is too small. citeturn10view0

Therefore, dense reward design should come with **normalization**:
- normalize each violation term to a comparable numeric range  
- then tune global guidance strength (SVDD α / temperature) rather than letting a single term dominate.

### Reduce intermediate reward noise by using expectation/uncertainty when possible

Two relevant patterns:

- Discrete reward papers note that intermediate rewards can be **noisy/inaccurate** when computed on an approximation to the clean sample. citeturn13view0turn13view1  
- Entropy-aware guidance in discrete diffusion LMs explicitly modulates how “hard” vs “soft” the representation is based on uncertainty, improving stability. citeturn11view2turn11view4  

For your setting, this motivates computing (some) reward components on:
- a decoded x₀ sample (simple), *or*
- an **expected** statistic under the model’s categorical distributions (lower variance), *or*
- a mixed strategy: early steps use expected counts/probabilistic adjacency likelihoods; late steps use hard decoded graph checks.

### Consider curricula / staging across diffusion time

The diffusion-RL literature on sparse reward adopts curricula like “start from later timesteps and progressively extend earlier,” explicitly to address sparse terminal feedback. citeturn27search0turn27search4 While that paper is about training, the conceptual analogue at inference-time is:

- apply **stronger guidance later** (when the model is confident / structure is more formed),  
- or stage constraints: first satisfy program counts, then connectivity, then more topological path constraints.

This also aligns with graph guidance work arguing that myopic local corrections can be insufficient and long-horizon structure needs lookahead. citeturn22view0

## Targeted dense reward candidates for your constraints

Let the decoded discrete sample (your predicted clean bubble-graph) be `x`, from which you derive a **room-type multiset** and an **adjacency graph** `G(x)` over *interior rooms*. Your constraints are:

- Required program: exactly one living room, exactly one kitchen  
- Bedroom count: 1–4 bedrooms  
- Bathroom provision: ≥1 bath; if bedrooms ∈ {3,4} then ≥2 baths  
- Connectivity: all interior rooms in a single connected component  
- Kitchen adjacency: kitchen adjacent to living room  
- Bedroom access: no bedroom is only reachable through bedroom(s)  
- Bathroom–kitchen separation: bathroom not adjacent to kitchen  

(As listed in your prompt.)  

Below are reward candidates that explicitly target these constraints while being **graded**.

### A base pattern: sum of dense violation magnitudes

Define an energy:

\[
E(x) = \sum_i \lambda_i \, \phi_i\big(v_i(x)\big)
\]

and set:

\[
r(x) = -E(x)
\]

This “energy-as-violation” pattern matches the reward-tilting view in diffusion guidance tutorials (reward enters exponentiated weights) and also the broader “Boltzmann density with energy defined by the objective” framing used in optimization-with-diffusion work. citeturn26view2turn2view0turn24view0

**Key design choice:** define each `v_i(x)` as a *nonnegative integer or real* that increases smoothly with violation severity (not just 0/1), then choose a shaping `φ_i` that avoids saturation.

Recommended first-pass shaping functions (in an SVDD-like exponentiated-weight system):
- `φ(v)=v` (linear) for most terms (stable, interpretable)  
- `φ(v)=v^2` for “rare but catastrophic” violations (e.g., disconnectedness), if linear isn’t strong enough  
- `φ(v)=log(1+v)` if a term can become large early and you don’t want it to dominate

Augmented-Lagrangian methods are essentially using increasing penalties to keep the “push toward feasibility” informative even when violated; you can mimic the spirit with piecewise / quadratic escalation on key constraints. citeturn9view2

### Program and count constraints: make them edit-distance-like

Let:
- `n_L(x)` = # living rooms  
- `n_K(x)` = # kitchens  
- `n_Bed(x)` = # bedrooms  
- `n_Bath(x)` = # bathrooms  

Use **L1 distance to the feasible set**:

- Required program violation  
  \[
  v_\text{prog}(x)=|n_L(x)-1| + |n_K(x)-1|
  \]
  This yields values 0,1,2,… rather than a boolean.

- Bedroom count range violation  
  \[
  v_\text{bed}(x)=\max(0,1-n_\text{Bed}(x))+\max(0,n_\text{Bed}(x)-4)
  \]

- Bathroom rule violation (piecewise)  
  Let `reqBath(x)= 2` if `n_Bed(x) ∈ {3,4}`, else `1`. Then  
  \[
  v_\text{bath}(x)=\max(0,\text{reqBath}(x)-n_\text{Bath}(x))
  \]

These are inherently dense (multiple levels), and—crucially—can be computed even if the diagram is otherwise messy.

**Variance reduction option:** if at intermediate steps node types are uncertain, compute expected counts under the model’s categorical probabilities instead of hard argmax decoding (a common trick in discrete guidance to reduce discontinuity). This is conceptually aligned with discrete methods using relaxations or entropy-aware soft/hard mixing to stabilize reward signals. citeturn11view2turn9view0

### Connectivity: prefer metrics that reward partial progress

Connectivity is a classic “cliff” property in graphs: one missing edge can break it. Graph guidance work explicitly notes graph outputs can be fragile and hard to correct locally. citeturn22view0 To reduce sparsity, avoid `v_conn(x) ∈ {0,1}`.

Two strong dense options:

1) **Component count distance**
   \[
   v_\text{conn}(x)=\#\text{components}(G(x)) - 1
   \]
   This equals the minimum number of edges needed to connect components (in an unweighted sense). It is integer-valued and graded.

2) **Largest-component coverage**
   \[
   v_\text{conn-cover}(x)=1 - \frac{|C_\text{max}(x)|}{|V_\text{int}(x)|}
   \]
   where `C_max` is the largest connected component and `V_int` the set of interior rooms. This is in [0,1], naturally normalized.

In practice, (2) is often **more informative early**, because it distinguishes “almost connected” from “fully shattered” even when room count varies.

If rewards are still too brittle because connectivity flips frequently under decoding noise, the “clean-sample-only” guidance line suggests a fallback: avoid intermediate connectivity scores and rely on clean-sample evaluation or local MCMC refinement after denoising. citeturn13view0turn13view1

### Kitchen adjacency: replace a boolean with shortest-path distance

Let `d(u,v)` be shortest-path distance on adjacency edges in `G(x)`.

Then define:

\[
v_{\text{k-adj}}(x)=\max(0, d(\text{kitchen},\text{living}) - 1)
\]

- 0 when adjacent  
- 1 when separated by one intermediate room  
- etc.

This is a standard densification trick: “adjacent” is a distance-1 property, so distance-to-1 is a graded violation.

If kitchen or living is missing, add a missing-node penalty via `v_prog` above rather than forcing special cases everywhere.

### Bathroom–kitchen separation: count forbidden adjacencies (and optionally penalize “near misses”)

Forbidden adjacency can also be densified:

\[
v_{\text{bath-k}}(x)=\#\{(b,k)\in E(G(x)):\ b\in \text{Bath},\ k=\text{Kitchen}\}
\]

This naturally allows multiple bathrooms and gives graded penalty if the generator creates multiple forbidden contacts.

If you later discover the learned adjacency relation is noisy, you can use a “near miss” optional term:

\[
v_{\text{bath-k-near}}(x)=\max(0,2-d(\text{nearest bath},\text{kitchen}))
\]

This encourages a buffer room even if not mandated, but only add this if it empirically improves feasibility without harming realism/diversity.

### Bedroom access: approximate the “no bedroom-through-bedroom” rule with a dense local criterion

Your stated constraint is path-based (“no bedroom may be accessible only through another bedroom”). In graphs, fully path-based constraints can be brittle; you want something graded.

A practical first dense surrogate is:

- For each bedroom node `b`, let `deg_nonbed(b)` be the number of neighbors of `b` that are not bedrooms.
- Define per-bedroom violation:
  \[
  v_\text{acc}(b)=\mathbb{1}[\deg_\text{nonbed}(b)=0]
  \]
- Aggregate:
  \[
  v_{\text{bed-access}}(x)=\sum_{b\in \text{Bedrooms}} v_\text{acc}(b)
  \]

This is still discrete but **less sparse** than an all-or-nothing global check, because it counts how many bedrooms are problematic.

A denser version uses a smooth transform of non-bedroom degree:

\[
v_\text{acc-dense}(b)=\frac{1}{1+\deg_\text{nonbed}(b)}
\quad\Rightarrow\quad
v_{\text{bed-access-dense}}(x)=\sum_b v_\text{acc-dense}(b)
\]

This yields gradations when a bedroom has 0,1,2,… “exits” to non-bedroom space.

This kind of local surrogate mirrors the broader diffusion-alignment idea that richer, step-wise or token-level feedback avoids the instability of a single sparse terminal signal. citeturn28view0turn19view0

If you want a closer match to the original path statement, a more faithful (still graded) graph metric is:

- Remove all bedroom nodes except the target bedroom `b`, and compute whether `b` is still connected to the living room; or  
- Compute the number of vertex-disjoint paths from `b` to any non-bedroom node (a “bedroom cluster exit” measure).  

These are more expensive but can be used selectively as late-stage guidance terms.

### Putting it together: a concrete dense “constraint energy”

A sensible initial full energy could be:

\[
E(x)=
\lambda_\text{prog} v_\text{prog}
+\lambda_\text{bed} v_\text{bed}
+\lambda_\text{bath} v_\text{bath}
+\lambda_\text{conn} v_\text{conn-cover}
+\lambda_\text{kadj} v_\text{k-adj}
+\lambda_\text{bacc} v_\text{bed-access-dense}
+\lambda_\text{bk} v_\text{bath-k}
\]

and `r(x)=-E(x)`.

This removes the “all constraints satisfied” bonus entirely. That bonus is exactly the kind of sparse terminal spike that causes sample inefficiency and unstable weight distributions in guidance. citeturn11view0turn28view0turn14view4

If you still want an explicit “feasible bonus,” prefer a **soft feasibility score** that increases smoothly as energy approaches 0, e.g.:

\[
r(x)= -E(x) + \beta \cdot \exp(-E(x))
\]

But note: because SVDD/SMC weights already exponentiate values, you must tune scales carefully to avoid effectively “double exponentiation” collapse. citeturn26view2turn10view0turn14view4

### When to replace reward terms with hard constraint injection or projection

Some of your constraints are structurally simple and may be cheaper to enforce by **editing the sample** or clamping token slots, rather than shaping reward:

- “exactly one living room and one kitchen” can sometimes be enforced by clamping a designated node slot (if your representation allows it)  
- prohibiting bathroom–kitchen adjacency can sometimes be handled by forbidding specific edge labels at those pairs.

This is analogous to discrete layout generation approaches that implement strong constraints via masking/clamping and weak constraints via logit adjustment. citeturn19view0 It is also philosophically aligned with constrained diffusion approaches that use projection operators instead of pure reward shaping. citeturn1view1turn20view1

A hybrid often works well: **hard-enforce what is easy and unambiguous**, and reserve reward shaping for global or preference-like constraints (connectivity, access patterns).

## Calibration and iteration protocol for reward functions under SVDD-style guidance

Reward choice is rarely “one-shot correct,” and the literature repeatedly emphasizes sensitivity to scaling, diversity collapse, and reward hacking. citeturn28view0turn17view4turn11view4 The most effective way to make reward engineering tractable is to systematize it.

### Calibrate violation magnitudes before tuning guidance temperature

1) Sample a few thousand unguided diagrams from the pretrained diffusion model.  
2) Compute each `v_i(x)` distribution (mean, 90th percentile).  
3) Normalize:
   \[
   \tilde v_i = \frac{v_i}{\text{P90}(v_i)+\epsilon}
   \]
   so each term is roughly comparable in scale.

Then set `E(x)=Σ λ_i \tilde v_i` with λ_i initially all 1. This makes subsequent α/temperature tuning much less chaotic and directly addresses the “weight degeneracy” phenomena emphasized in SMC analyses. citeturn14view4turn14view3

### Tune α / temperature to avoid degeneracy

Because SVDD and SMC-style methods involve exponentiated value weights, too-strong guidance can cause near-deterministic selection and diversity collapse. citeturn26view2turn14view4turn17view4 SVDD explicitly notes scaling issues when the temperature-like parameter is too small. citeturn10view0

A practical heuristic: monitor an “effective sample size” proxy (even without full SMC), e.g. entropy of the normalized candidate weights, to avoid collapse.

### Add complexity in stages (curriculum)

A robust order for your constraints is often:

1) room-type counts (program/bed/bath)  
2) kitchen–living distance term  
3) bathroom–kitchen forbidden adjacency  
4) connectivity coverage  
5) bedroom access

This staged addition mirrors the intuition from sparse-reward diffusion RL that it can help to first solve the “last steps / easiest-to-evaluate” objectives and then expand earlier / harder objectives. citeturn27search0turn27search4 It also matches the “graphs are fragile” point: enforce local adjacency relations before global topological properties. citeturn22view0

### Detect “reward hacks” or unnatural graph artifacts

Even if constraints are satisfied, graphs might become unnatural (degree spikes, weird room-type distributions, etc.). General alignment surveys discuss reward hacking / over-optimization risk and the role of regularization. citeturn28view0turn17view4turn11view2

For bubble diagrams, you can build lightweight realism checks:
- compare degree distribution of generated graphs to training data  
- compare room-count histogram  
- compare adjacency-pair frequencies (e.g., kitchen–bath adjacency should be near zero, but living–bed adjacency may have a typical range)

These can remain “diagnostic metrics” rather than reward terms until you see failure modes.

### If intermediate reward remains too noisy: consider clean-sample refinement

If you observe that small edge flips drastically change reward and intermediate scoring is unreliable (common in discrete domains), the clean-sample Markov-chain approach suggests an escape hatch: do local search over *clean* diagrams using diffusion forward–backward proposals and accept/reject based on clean reward, rather than relying on intermediate reward shaping. citeturn13view0turn13view1

This is not required for your first iteration, but it is valuable to keep in mind if the bedroom-access or connectivity constraints prove especially brittle.

---

**Bottom line:** replace sparse “all constraints satisfied” bonuses with an **energy built from graded violation magnitudes** (counts, component coverage, shortest-path distances, forbidden-edge counts, and bedroom-exit degrees). Keep the reward in a scale where SVDD’s exponentiated weighting remains stable, and consider curriculum / staged constraint activation and selective hard enforcement for “obvious” constraints. citeturn26view2turn14view4turn19view0turn9view2turn28view0