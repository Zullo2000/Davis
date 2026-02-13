# Task 1: General Research Considerations — BD and RFP Generation Pipeline

## 1. Clarifying the Long-Term Vision

Based on your review and prompt, the roadmap is:

1. **Unconstrained BD generation** — discrete masked diffusion on small graphs (≤N_MAX nodes, where N_MAX is configurable per dataset: 8 for RPLAN, up to 14 for ResPlan)
2. **Constrained BD generation** — condition on house boundary (HB), then on partial attributes
3. **BD guidance** — training-free and classifier-free guidance mechanisms to steer BD generation toward architectural plausibility, constraint satisfaction, and user intent, without retraining the base model
4. **RFP layout generation** — vectorized or rasterized, BIM-compatible output
5. **Joint BD + RFP co-generation** — bidirectional dependence in a shared latent space via coupled discrete + continuous diffusion

Step 3 is a critical intermediate: once we can generate BDs conditioned on constraints (step 2), we need principled guidance mechanisms that allow fine-grained control — soft preferences (e.g., "prefer open-plan layouts"), hard constraints (e.g., "exactly 3 bedrooms"), and compositional constraints (e.g., "kitchen adjacent to dining, both near the entrance"). These guidance strategies are developed and evaluated on the BD alone before introducing the additional complexity of geometry generation. The techniques developed here (CFG, training-free reweighting, inpainting) carry directly into the joint model at step 5.

The research question is: **what modeling choices now (step 1) will best serve steps 2–5 later?**

---

## 2. Model Family Decision: Why MDLM-Style Masked Diffusion for the BD

You proposed using masked discrete diffusion à la MDLM for the BD, with a vanilla noise schedule, later adding remasking à la ReMDM. I agree this is the right call, and here is a detailed justification with honest caveats.

### 2.1 Why not DiGress (uniform corruption)?

DiGress is the standard reference for graph generation with discrete diffusion. It uses **uniform transition matrices** (D3PM-style) that can flip any node/edge type to any other type during corruption. Its denoiser is a graph transformer that predicts clean node and edge types from the noisy graph.

**Pros of DiGress for BD:**

- Well-tested on graph benchmarks (planar, SBM, molecular), public codebase
- The graph transformer architecture naturally handles both node and edge features
- Marginal-preserving noise model improves sample quality
- Auxiliary graph-theoretic features (cycle counts, spectral features) can be injected during denoising
- Conditional generation via score-based guidance is demonstrated (on molecular properties)

**Cons of DiGress for BD:**

- Uniform corruption introduces more complex intermediate states (a node can take any type during diffusion, not just "known" or "masked"). For a small, semantically structured domain like floorplan BDs, this is wasteful — the intermediate states have no architectural meaning
- The reverse posterior is factorized over nodes and edges independently (Eq. 19 in your review). This is the same independence assumption that MELD identifies as problematic for multimodal posteriors. For molecules this matters a lot; for floorplans with ≤N_MAX nodes (8 for RPLAN) it is less severe, but it still means the model cannot represent "either layout A or layout B" as a coherent joint prediction
- Training requires full dense adjacency tensors (N×N×b), which scales quadratically. Fine for ≤N_MAX nodes, but the framework is heavier than needed
- Guidance in DiGress requires score-based tricks adapted from continuous diffusion; the training-free guidance framework by Kerby & Moon (arXiv:2409.07359) is promising but still early and only tested on simple molecular properties

### 2.2 Why not GraphARM (autoregressive masked)?

GraphARM generates one node + its incident edges at each step, with a learned ordering network.

**Pros:**

- Avoids state-clashing entirely: node type and its edges are predicted jointly at each step
- The learned ordering can capture structural regularities (e.g., always place living room first)
- No fixed graph size needed — the number of generation steps equals the number of nodes

**Cons:**

- **Sequential generation**: for an N_MAX-node BD, you need N_MAX forward passes (8 for RPLAN). Not catastrophic, but much slower than masked diffusion which can unmask all tokens in parallel
- **Error propagation**: a mistake in an early node contaminates all subsequent predictions (same fundamental issue as AR models). Your review correctly flags this
- **No natural inpainting**: if you want to fix some rooms and regenerate others (your editability requirement), AR ordering makes this awkward. You'd need to define a new ordering that respects the fixed subset
- **Guidance is harder**: there's no diffusion-style denoising trajectory to intervene on at test time. Conditioning typically requires retraining or reward-based fine-tuning
- The learned ordering network (policy gradient on discrete σ) adds training complexity and instability

### 2.3 Why MDLM-style masked diffusion is the best starting point

MDLM uses absorbing-state (masking) corruption: each token is either its true value or [MASK]. The reverse process unmasking tokens one (or several) at a time.

**Pros for BD generation:**

- **Simplicity**: the forward process is trivially understood — mask each element with probability (1−αt). The ELBO simplifies to a weighted average of masked cross-entropy losses. This means simpler implementation, easier debugging, and faster iteration
- **Order-agnostic**: graphs have no natural ordering, and masked diffusion respects this. All masked positions are treated symmetrically
- **Parallel generation**: at inference, you can unmask multiple tokens simultaneously, giving O(T) steps where T is the number of diffusion steps (not graph size). For an N_MAX-node BD with ~SEQ_LEN token entries (36 for RPLAN with N_MAX=8), this is a substantial speedup over GraphARM
- **Natural inpainting**: to condition on partial constraints, simply never mask the constrained positions. This is exactly how MDLM handles conditional generation, and it maps perfectly to your editability requirement (fix some rooms/adjacencies, regenerate the rest)
- **Guidance compatibility**: the denoising trajectory provides a natural hook for test-time guidance. The training-free guidance framework (Kerby & Moon, on DiGress) adapts directly to any discrete diffusion model. For masked diffusion specifically, guidance reduces to reweighting the unmasking probabilities at each step — no retraining needed
- **ReMDM upgrade is free**: as your review notes, ReMDM can be applied on top of a pretrained MDLM model (σt=0 during training). This means you train once and get remasking at inference, which mitigates error propagation without additional training cost
- **ELBO invariance to noise schedule**: MDLM's continuous-time ELBO is invariant to the functional form of αt (as long as it's monotonic). This means you can tune the schedule for inference quality without retraining. BD3-LMs showed that while the ELBO is invariant, gradient variance is not — so schedule choice still matters for training efficiency, but you have flexibility

**The state-clashing concern — and why it's manageable here:**

Your review correctly identifies state-clashing (from MELD) as the main theoretical objection to using masked diffusion on graphs. The problem: if you independently unmask a node type and an edge type, they might be incompatible. MELD's solution is learning per-element corruption rates so that coupled elements are masked/unmasked together.

For floorplan BDs, this concern is significantly mitigated:

- **Small graphs** (≤N_MAX nodes, ≤N_EDGES edges — 8 nodes, 28 edges for RPLAN): the combinatorial space of partially masked states is vastly smaller than in molecular graphs with 50+ atoms
- **Weak type-edge coupling**: in floorplans, most room type pairs are compatible with most spatial relationship types. A kitchen can be north, south, east, or west of a bedroom. This is very different from chemistry where carbon-oxygen bonds have strict valence constraints. The "forbidden combinations" are few (e.g., maybe you wouldn't put an entrance inside a bathroom, but these are statistical tendencies, not hard rules)
- **The denoiser sees global context**: a graph transformer (or GNN) used as the denoiser conditions each prediction on all unmasked elements. So even if a node type and an edge are unmasked independently, the denoiser can learn to predict compatible types given the partial context
- **ReMDM provides a safety net**: if an incompatible pair is unmasked, remasking allows the model to "reconsider" and fix it

**Therefore, I agree that learning the forward process (as in MELD) is unnecessary overhead for this application. A vanilla MDLM with a standard linear or cosine noise schedule is the right starting point.**

### 2.4 Practical considerations on the noise schedule

MDLM's ELBO is schedule-invariant, but BD3-LMs showed that gradient variance is not. For your small graphs:

- Start with a **linear schedule** (βt from 0 to 1, or equivalently αt = 1−t)
- The cosine schedule from improved DDPM is also a reasonable default
- MuLAN (same authors as MDLM) offers a data-dependent multivariate schedule that can significantly reduce training steps. This could be worth investigating later but adds complexity — skip it for v1

---

## 3. Graph Representation: How to Flatten a BD for Masked Diffusion

MDLM operates on sequences of tokens. A BD is a graph. The key design decision is **how to represent the graph as a flat sequence of discrete tokens** that MDLM can process.

### 3.1 The flat adjacency + node vector representation

Represent the graph as:

- **Node features**: a sequence of N tokens, each from a vocabulary of room types (13 types in RPLAN + [MASK] + possibly [PAD])
- **Edge features**: the upper triangle of the adjacency matrix flattened into N(N−1)/2 tokens (28 for RPLAN with N_MAX=8), each from a vocabulary of spatial relationship types (e.g., {no-edge, north, south, east, west, north-east, north-west, south-east, south-west, inside, surrounding, [MASK]})

Total sequence length = N_MAX + N_MAX*(N_MAX−1)/2. For RPLAN (N_MAX=8): 8 + 28 = 36 tokens. For ResPlan (N_MAX=14): 14 + 91 = 105 tokens. This is very manageable.

**Pros:** Complete representation, permutation-equivariant if the denoiser architecture respects it, straightforward to implement. Absent edges are naturally represented as a "no-edge" token, which carries important information ("these rooms are NOT adjacent").

### 3.2 The mixed-entity challenge and how to address it

The main conceptual difficulty is that the SEQ_LEN-token sequence is a heterogeneous mix: the first N_MAX positions are node tokens (room types) and the remaining N_EDGES are edge tokens (spatial relationships). These two entity types have fundamentally different semantics — a "kitchen" token and a "north-of" token live in different conceptual spaces, yet they share the same sequence. Without explicit handling, the transformer denoiser would have to implicitly learn which positions are nodes and which are edges, and how they relate to each other, purely from data. This is wasteful and fragile.

We address this through three complementary mechanisms in the transformer denoiser:

**1. Positional encoding that encodes structural role.** Rather than using a single positional embedding for position index 0 through SEQ_LEN−1, we use a composite encoding: (a) an **entity-type embedding** (a learned vector for "node" vs. "edge"), (b) for node positions, a **node-index embedding** (which room slot: 1st through N_MAX-th), and (c) for edge positions, a **pair-index embedding** encoding both endpoints (i, j). This pair embedding can be implemented as the sum of two learned embeddings, one for i and one for j, or as a single learned embedding for the pair. The pair embedding is critical: it tells the model that edge position 47 refers to the relationship between rooms 3 and 8, allowing it to attend to the relevant node tokens when predicting that edge.

**2. Structured attention biases.** The self-attention mechanism can be augmented with learned bias terms that reflect the graph structure: node-to-node attention, node-to-incident-edge attention (node i attends to all edges involving i), edge-to-endpoint-node attention (edge (i,j) attends to nodes i and j), and edge-to-edge attention (edges sharing a node attend to each other). These biases are not hard masks — they are additive terms in the attention logits, so the model can still attend globally but with an inductive bias toward structurally meaningful interactions. This is analogous to how relative position biases work in text transformers, but adapted to graph topology.

**3. Separate embedding tables for node and edge vocabularies.** Nodes and edges draw from different categorical vocabularies (room types vs. spatial relationships). Using separate embedding lookup tables — and projecting both into a shared hidden dimension via separate linear layers — ensures the model never confuses a "kitchen" embedding with a "north-of" embedding, even if they happen to share a token index. The shared hidden dimension allows the transformer to reason jointly across both entity types.

Together, these three mechanisms mean the transformer always knows (a) what kind of entity each position represents, (b) which graph elements are structurally related, and (c) the semantic space each token belongs to. The underlying architecture remains a standard transformer with full self-attention, preserving all the benefits of MDLM (parallel unmasking, global context, inpainting). The additions are lightweight — a few extra embedding tables and an attention bias matrix that is precomputed once per forward pass.

**Why a transformer rather than a GNN?** Because at early diffusion steps, most of the graph is masked, so there's no meaningful graph structure for message passing. A transformer with full attention can reason about which positions should be unmasked next based on global patterns, regardless of current graph connectivity. This is the same insight behind DiGress's choice of graph transformer over standard MPNNs.

Additionally, graph-theoretic features (once some nodes/edges are unmasked) can be injected as auxiliary conditioning, as in DiGress.

---

## 4. The Path to Conditioning and Guidance

Your roadmap: unconstrained → HB-constrained → partial-attribute-constrained → guided generation. Here's how each stage maps to the MDLM framework.

### 4.1 Unconstrained generation (step 1)

Train MDLM on the full RPLAN BD dataset. No conditioning. The model learns p(BD) — the distribution of valid bubble diagrams.

### 4.2 HB-constrained generation (step 2)

The house boundary constrains which BDs are plausible (a 30m² apartment won't have 8 bedrooms). Two approaches:

**Approach A — Classifier-free guidance (recommended):** During training, encode the HB as a conditioning signal (e.g., a vector of boundary features: area, aspect ratio, number of corners, perimeter). With some probability (e.g., 10%), drop the conditioning (replace with null token). At inference, interpolate between conditional and unconditional predictions:

p\_guided(x|HB) ∝ p(x|HB)^(1+w) / p(x)^w

This is the standard CFG trick, adapted to discrete diffusion. It requires no separate classifier and is well-suited to MDLM.

**Approach B — Inpainting (for hard constraints):** If the constraint is "the BD must contain exactly 1 living room and 2 bedrooms," simply clamp those node tokens and let the model fill in the rest. This is native to masked diffusion.

**Approach C — Training-free guidance (for soft constraints):** Use the Kerby & Moon framework: at each denoising step, evaluate a differentiable constraint function on the current (partially unmasked) state and reweight the unmasking probabilities. E.g., a function that returns higher scores for BDs with total room area compatible with the HB.

### 4.3 Partial-attribute-constrained generation (step 3)

This is exactly the MaskPLAN use case: the user specifies some rooms and adjacencies, and the model completes the rest. In MDLM, this is **trivially implemented as inpainting**: keep user-specified positions unmasked, run the reverse process on everything else.

This is a major advantage of masked diffusion over AR models (where partial constraints require complex masking/reordering) and over GANs (where conditioning is baked into training).

### 4.4 BD guidance (step 3 of the roadmap)

Once constrained generation works, the next milestone is developing and evaluating **guidance mechanisms** that go beyond simple conditioning. The goal is fine-grained, composable control over the generated BDs:

**Classifier-free guidance (CFG) tuning:** The weight w in the CFG formula controls the strength of conditioning. Too low → ignores constraints. Too high → mode collapse to the most stereotypical layout for a given boundary. Systematically evaluating w across constraint types (boundary area, room count, adjacency requirements) is essential. Different constraint types may need different guidance weights, motivating a per-constraint or per-modality weighting scheme.

**Training-free guidance for architectural constraints:** The Kerby & Moon framework allows injecting arbitrary differentiable constraint functions at inference time. For BDs, relevant constraints include: (a) room count constraints (penalize deviations from a target room program), (b) adjacency constraints (reward BDs where specific room pairs are adjacent), (c) spatial coherence (penalize BDs with implausible spatial relationships, e.g., a room that is simultaneously "north-of" and "south-of" another room), and (d) area-compatibility (penalize BDs whose implied room sizes are inconsistent with the boundary). Each of these can be formulated as a differentiable function on the partially unmasked state, and their gradients reweight the unmasking probabilities.

**Compositional guidance:** Real architectural briefs combine multiple constraints. The guidance framework should support composing constraints via product-of-experts (multiply the guidance signals) or additive combination (sum the log-probabilities). Evaluating whether composed guidance degrades sample quality or introduces mode collapse is a key research question.

**Evaluation of guidance quality:** Beyond constraint satisfaction rate, we need to measure whether guided samples are still diverse and architecturally plausible. Metrics: (a) constraint satisfaction rate, (b) diversity under fixed constraints (e.g., number of distinct valid BDs generated for the same brief), (c) validity rate (architectural plausibility), and (d) comparison with retrieval-based baselines (does the guided model produce better BDs than the nearest-neighbor from the training set?).

---

## 5. The Long Game: Joint BD + RFP Co-Generation

Your review correctly identifies that BD and RFP should not be generated independently — topology constrains geometry and vice versa. The idea of a shared latent space with bidirectional communication during diffusion is sound. Here is a critical assessment of the key enabler.

### 5.1 CCDD (arXiv:2510.03206) — The real enabler

**This is the paper that matches your co-generation vision.** CCDD defines a joint diffusion process on the union of a continuous representation space and a discrete token space. A single model simultaneously denoises in both spaces, with factored reverse updates:

- The **discrete branch** (CTMC / masked diffusion) handles tokens — in your case, room types and adjacency types (the BD)
- The **continuous branch** (SDE / Gaussian diffusion) handles continuous variables — in your case, room coordinates, bounding box positions, wall junction locations (the RFP geometry)

The key architectural idea: a single time-conditioned network f\_θ(x\_t, z\_t, t) receives both the partially denoised discrete state x\_t and the partially denoised continuous state z\_t, and outputs modality-specific predictions through separate heads. The reverse updates are factored but **conditioned on each other** — the discrete prediction sees the continuous state and vice versa.

**How this maps to BD + RFP:**

- x\_t = partially masked BD (node types + edge types) — discrete, absorbing-state diffusion
- z\_t = partially noised room geometry (bounding box coordinates, corner positions) — continuous, Gaussian diffusion
- The model architecture: a transformer that processes both x\_t and z\_t, with cross-attention between the two modalities
- Reverse step: predict clean BD tokens given (x\_t, z\_t, t), predict clean geometry given (x\_t, z\_t, t), then update both

**Practical adaptation needed:** CCDD was designed for text (both branches represent the same text, just in different spaces). For BD + RFP, the two branches represent genuinely different objects (topology vs. geometry). You'll need:

1. Separate embedding layers for discrete graph tokens and continuous coordinates
2. A fusion mechanism (cross-attention, MM-DiT-style interleaving, or MoE routing) that lets the two streams communicate
3. Possibly asynchronous noise schedules — the BD might benefit from faster unmasking (it has fewer tokens and less ambiguity), while the geometry needs more denoising steps to converge
4. A careful loss balance (λ weighting discrete CE vs. continuous MSE)

### 5.2 Alternative: HouseDiffusion's mixed discrete+continuous approach

HouseDiffusion already does mixed diffusion: continuous Gaussian noise on corner coordinates + a discrete bit-encoding branch activated in later steps. However, it conditions on a **fixed** BD, rather than co-generating it. The architecture (transformer with corner-level attention, CSA + RCA) is directly relevant to the continuous branch of your future joint model.

GSDiff goes further by decoupling node generation (continuous diffusion on coordinates + type embeddings) from edge prediction (transformer classifier). Its alignment losses and mixed-radix coordinate encoding are practical innovations you should adopt.

**Recommended synthesis for the long-term architecture:**

- BD branch: MDLM-style absorbing-state diffusion on a flat graph sequence
- RFP branch: Gaussian diffusion on room geometry vectors (à la HouseDiffusion/GSDiff)
- Coupling: CCDD-style factored reverse process with a shared transformer backbone
- Conditioning: CFG for soft constraints, inpainting for hard constraints, MaskPLAN-style stochastic masking during training for editability

---

## 6. Datasets: A Comparative Analysis

### 6.1 Graph2Plan's RPLAN Extraction (Primary Dataset)

**Source:** 80K floorplans from RPLAN, preprocessed by Graph2Plan (Data.zip publicly available).

**Graph structure:** Each floorplan is represented as a bubble diagram with:

- **Nodes**: 13 room type categories (MasterRoom, SecondRoom, LivingRoom, Kitchen, Bathroom, Balcony, Entrance, DiningRoom, StudyRoom, StorageRoom, WallIn, ExternalArea, ExternalWall)
- **Edges**: (u, v, r) triples encoding spatial relationships. The relationship r is determined by the relative position of room centroids (computed from bounding boxes). Graph2Plan discretizes centroid-to-centroid vectors into 10 categories: left-of, right-of, above, below, left-above, left-below, right-above, right-below, inside, and surrounding
- **Additional data**: room bounding boxes (gtBox, gtBoxNew), room boundary polygons (rBoundary), building boundary with front door

**How spatial relationships are computed:** Graph2Plan computes the centroid of each room's bounding box, then classifies the relative position of room u with respect to room v by looking at which sector of a compass-like grid the vector (centroid\_u − centroid\_v) falls into. The "inside" and "surrounding" categories handle containment (e.g., a bathroom inside a master bedroom suite). This discretization onto a coarse directional grid is the source of the 5×5 grid description — the centroid-to-centroid direction is quantized into one of 8 compass directions plus 2 containment relations.

**Limitations:** The 10-category spatial vocabulary is coarse. Two rooms whose centroids differ by 1° in angle get the same label. There is no encoding of distance (a room barely adjacent vs. far away get the same directional label). The relationship is also asymmetric in a way that depends on centroid computation quality — rooms with irregular shapes may have centroids that poorly represent their spatial extent. Nevertheless, this representation has been successfully used by Graph2Plan, FP-FGNN, and other downstream models.

### 6.2 ResPlan Dataset

**Source:** 17,000 residential floor plans, publicly available (github.com/m-agour/ResPlan, also on Kaggle). Published August 2025 (arXiv:2508.14006).

**Key advantages over RPLAN:**

- **Vector-graphic format**: rooms are precise polygons (not rasterized), with sub-centimeter fidelity. This enables exact centroid computation, area measurement, and direct 3D extrusion
- **Higher structural diversity**: includes non-Manhattan room shapes, more realistic Western residential layouts (sourced from real-estate listings, primarily European/US)
- **Richer annotations**: walls, doors, windows, and balconies are all represented as clean polygons with consistent thickness
- **Open-source pipeline**: the full processing pipeline (parsing → cleaning → alignment → annotation) is provided as a Python package, making it reproducible and extensible

**Graph structure as provided:** ResPlan's bubble diagrams use a different edge vocabulary than Graph2Plan:

- **Nodes**: rooms with attributes including semantic label (bedroom, bathroom, kitchen, living, corridor, etc.), polygon centroid coordinates, and room area
- **Edges**: typed by **connection type** — three categories: (1) **door** (rooms share a doorway), (2) **arch** (rooms connected by an open passage without a door), (3) **shared-wall** (adjacent rooms with a wall between them but no opening)

**The critical difference:** ResPlan edges encode **how** rooms are connected (door/arch/wall), while Graph2Plan edges encode **where** rooms are relative to each other (north/south/east/etc.). For our MDLM-based BD generation, the spatial relationship encoding is more informative — it captures the topological layout that constrains geometry. Connection-type edges are useful downstream (for RFP generation, where you need to know where to place doors), but they tell the diffusion model little about spatial arrangement.

**Can we convert ResPlan to Graph2Plan-style spatial relationships?**

Yes — and this is highly feasible because ResPlan provides everything needed:

1. **Room centroids are already available** as node attributes in ResPlan's graph. Unlike RPLAN where centroids must be estimated from rasterized bounding boxes, ResPlan provides exact polygon centroids computed from vector geometry.
2. **The conversion algorithm** is straightforward: for each pair of adjacent rooms (u, v) in ResPlan's graph, compute the vector from centroid\_v to centroid\_u, then classify the direction into one of the compass categories. We can reuse Graph2Plan's exact discretization logic or improve upon it (see below).
3. **We can retain both edge types** — create a **dual-edge representation** where each edge carries both a spatial relationship label (north, south-east, etc.) and a connection type label (door, arch, shared-wall). For MDLM training on BDs, we use the spatial relationship as the primary edge token. The connection type becomes a secondary attribute that can be used for conditioning or as an auxiliary prediction target.

**Should we use a 5×5 grid? How to make it more refined?**

Graph2Plan's original 10-category spatial vocabulary (8 compass directions + inside + surrounding) is adequate but coarse. With ResPlan's precise vector geometry, we have the opportunity to do better. Here are the options, from simplest to most expressive:

**Option 1 — Replicate Graph2Plan's 10-category scheme (recommended for v1).** This ensures direct compatibility with Graph2Plan-pretrained models, allows apples-to-apples comparison on evaluation metrics, and keeps the edge vocabulary small (important for MDLM — fewer categories means easier learning). To implement: compute angle = atan2(cy\_u − cy\_v, cx\_u − cx\_v) and classify into 8 sectors of 45° each. Check bounding-box containment for inside/surrounding.

**Option 2 — Finer angular discretization (16 directions).** Subdivide each 45° sector into two 22.5° sectors, giving 16 compass directions + 2 containment = 18 categories. This doubles angular resolution and may help distinguish "slightly-above-and-left" from "directly-left" configurations. The cost is a larger edge vocabulary, but 18 categories is still very manageable for MDLM. However, the benefit may be marginal: floorplan layouts are dominated by axis-aligned arrangements, so most relationships cluster around the 4 cardinal directions anyway.

**Option 3 — Add distance quantization.** The most significant limitation of Graph2Plan's scheme is that it ignores distance. A "north" edge means the same thing whether rooms are 1 meter apart or 10 meters apart. We can add a distance dimension by normalizing the centroid-to-centroid distance by the building's diagonal (or by the average room size) and discretizing into 2–3 bins: "close" (adjacent/touching), "medium" (one room apart), "far" (opposite ends of the plan). Combined with 10 directions, this gives 30 + 2 = 32 categories. This is richer but still compact. The question is whether the training data is large enough (17K plans) to learn reliable statistics for 32 edge types.

**Option 4 — Continuous edge features (defer to joint model).** Instead of discretizing at all, encode the centroid-to-centroid vector as a continuous 2D feature and let the model learn its own representation. This is conceptually elegant but incompatible with MDLM's discrete framework — it would only work in the future CCDD-style joint model where edges can have continuous attributes. Not suitable for v1.

**Recommendation:** Start with Option 1 for direct Graph2Plan compatibility and simpler learning. After the base MDLM model is trained and evaluated, experiment with Option 3 (direction + distance) as a data-driven ablation. The 17K plans in ResPlan may be enough if combined with Graph2Plan's 80K RPLAN samples — a merged dataset of ~97K plans with harmonized edge vocabularies would be powerful.

**Practical implementation:** The conversion script would: (a) load each ResPlan JSON, (b) extract room polygons and their centroids, (c) for each edge in ResPlan's adjacency graph, compute the centroid-to-centroid vector, (d) classify into spatial relationship categories, (e) output in Graph2Plan's (u, v, r) format. Optionally also compute bounding boxes (from polygon extents) to match Graph2Plan's gtBox format. This is a one-time preprocessing step — straightforward Python with shapely/geopandas, which ResPlan already depends on.

### 6.3 Modified Swiss Dwellings (MSD) Dataset

**Source:** 5,372 floor plans derived from the Swiss Dwellings database (v3.0.0). Published ECCV 2024 (arXiv:2407.10121). **Publicly available** on Kaggle (kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings). GitHub: github.com/caspervanengelenburg/msd.

**What makes MSD unique:**

- **Multi-apartment building complexes**: unlike RPLAN and ResPlan which focus on single-apartment layouts, MSD contains floor plans with 2–9 apartments per floor, covering 18.9K distinct apartments across 5.3K floor plans. This is the first large-scale dataset targeting the multi-unit scale
- **More complex graphs**: the average floor plan has 25 areas (rooms), compared to ~7 in RPLAN. Individual apartments within MSD have 3–15 rooms, comparable to RPLAN, but the full-floor graphs are much larger
- **Non-Manhattan room shapes**: rooms in MSD are often irregularly shaped (non-axis-aligned), reflecting realistic Swiss residential architecture. This is in sharp contrast to RPLAN's strictly rectilinear rooms
- **Compass orientation**: MSD encodes the compass direction of the building, which is relevant for environmental design (sunlight, views)
- **Structural elements**: load-bearing walls, columns, and staircases are annotated as structural constraints that must be preserved during generation

**Graph structure:** MSD represents floor plans as networkx graphs where:

- **Nodes** correspond to areas (rooms), with attributes: room polygon (sequence of corners), room type category (Bedroom, Livingroom, Kitchen, Dining, Corridor, Stairs, Storeroom, Bathroom, Balcony, Structure, Background), and zoning type (zone1 = private, zone2 = public)
- **Edges** encode **connectivity type**: "door", "front door", or "passage" — similar to ResPlan's edge scheme, but without spatial relationship information

**Relevance to our project:**

MSD is primarily interesting as a **scaling testbed** and a **future extension**, not as a replacement for Graph2Plan/ResPlan for our initial work.

1. **Scale challenge**: with 25+ rooms per floor, the flat adjacency representation explodes: N=25 gives 25 + 300 = 325 tokens; N=50 gives 50 + 1225 = 1275 tokens. This is where MDLM on flat sequences may need to give way to sparse or hierarchical representations. MSD provides a natural benchmark for testing scalability.
2. **Multi-unit structure**: the inter-apartment connectivity in MSD introduces a hierarchical graph structure (apartments within floors within buildings) that our flat-sequence MDLM cannot naturally represent. Future work could use a hierarchical diffusion approach: first generate apartment-level BDs, then arrange them within a floor boundary.
3. **Edge type mismatch**: like ResPlan, MSD's edges encode connection types (door/passage), not spatial relationships. The same centroid-based conversion we described for ResPlan can be applied to MSD, with the added complexity that MSD rooms are non-Manhattan, so centroids may be less representative of spatial extent. Minimum rotated rectangles (MRR) — which MSD's Modified HouseDiffusion baseline already uses — could serve as a better basis for spatial relationship computation than raw centroids.
4. **Data availability and tooling**: MSD is fully public (Kaggle), with graph extraction notebooks and plotting utilities provided. The codebase includes a graph extraction pipeline (NB - Data Curation 2 - Graph Extraction.ipynb) that converts raw floor plans to networkx graphs, which could be extended to add Graph2Plan-style spatial relationship edges.

**Recommendation:** Do not use MSD for v1 training (the multi-unit complexity is premature). Instead, bookmark it as the scaling benchmark for when the model needs to handle larger, multi-apartment layouts. The spatial relationship conversion pipeline we build for ResPlan can be directly reused for MSD with minor adaptations for non-Manhattan room shapes.

### 6.4 Dataset strategy summary

| Dataset | Size | Edge types | Room shapes | Best use for us |
|---|---|---|---|---|
| Graph2Plan (RPLAN) | 80K | Spatial relationships (10 categories) | Manhattan (rectilinear) | **Primary training data for v1** — ready to use, spatial edges match our needs |
| ResPlan | 17K | Connection types (door/arch/wall) | Non-Manhattan, vector polygons | **Supplementary data after conversion** — higher quality geometry, needs spatial edge derivation |
| MSD | 5.3K floors (18.9K apartments) | Connection types (door/passage) | Non-Manhattan, irregular | **Future scaling benchmark** — multi-unit complexity, too large for v1 flat-sequence approach |

The recommended pipeline: train on Graph2Plan RPLAN data first (80K samples, spatial relationships ready). In parallel, build the centroid-based spatial relationship conversion for ResPlan. Once validated, merge both datasets (harmonizing room type vocabularies and spatial relationship categories) for a ~97K-sample training set with greater architectural diversity.

---

## 7. Risks and Open Questions

**Risk 1: Data quality.** Graph2Plan's RPLAN extraction uses 13 room types and directional spatial relationships on a coarse angular grid. Check whether the spatial relationship vocabulary is rich enough for your needs. If you want finer-grained positioning (e.g., "kitchen shares the north wall segment with living room"), you may need to re-extract from RPLAN or use the ResPlan conversion described above.

**Risk 2: Fixed graph size.** MDLM requires a fixed sequence length. With N_MAX configurable per dataset (8 for RPLAN, up to 14 for ResPlan), the sequence is N_MAX + N_MAX*(N_MAX−1)/2 tokens (36 for RPLAN, 105 for ResPlan). Graphs with fewer rooms need padding with a [PAD] token (distinct from [MASK]). This is standard but requires the model to learn that [PAD] positions should remain [PAD]. Right-sizing N_MAX to the dataset minimizes PAD waste. Alternatively, you can condition on the number of rooms N and only operate on the N + N(N−1)/2 active positions.

**Risk 3: Evaluation metrics.** Your review rightly criticizes FID-only evaluation. For BD generation, consider: (a) validity rate (are the generated BDs architecturally plausible?), (b) graph edit distance to nearest training sample (novelty), (c) constraint satisfaction rate (for conditioned generation), (d) diversity under fixed conditions (multiple samples from same constraint). Define these before training.

**Risk 4: The gap between BD generation and BD+RFP co-generation.** The architecture you choose for the BD denoiser now should be upgradable to a joint model later. If you use a transformer on flattened graph tokens, you can later extend the input sequence to include continuous geometry tokens (with a separate embedding) and add cross-attention. If you use a specialized GNN, the upgrade path to CCDD-style joint denoising is less clear.

**Open question: Should edges be generated jointly or in a second stage?** GSDiff and DiffPlanner both generate nodes first, then predict edges. This avoids the N² scaling of full adjacency representation. For ≤N_MAX nodes the full adjacency is fine (SEQ_LEN tokens — 36 for RPLAN, 105 for ResPlan), but if you want to scale to multi-unit buildings later, the two-stage approach may be necessary. My recommendation: start with joint (all-at-once MDLM on nodes+edges), and factor into stages only if needed.
