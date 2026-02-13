# BD Generation with Discrete Masked Diffusion — Planning Input

> **Purpose of this document:** Provide all context needed to plan the implementation of an MDLM-based bubble diagram (BD) generator for residential floorplans. The planning phase should produce a detailed implementation roadmap — file structure, module responsibilities, data pipeline, training loop, evaluation — ready for coding.

---

## 1. Project Goal and Scope

**Immediate goal (this phase):** Train an unconditional discrete masked diffusion model (MDLM) that generates valid residential bubble diagrams. A BD is a small graph (≤N_MAX nodes, configurable; N_MAX=8 for RPLAN) where nodes are room types and edges are spatial relationships between rooms.

**Future extensions (design for, don't implement yet):**
- Constrained BD generation (condition on house boundary, partial room specifications)
- Guidance mechanisms (classifier-free guidance, training-free guidance, inpainting)
- Joint BD + RFP co-generation via coupled discrete + continuous diffusion (CCDD-style)

**Critical constraint:** All architecture decisions must support gradual expansion toward the future extensions. The codebase should be modular enough to add conditioning, guidance, and a continuous geometry branch without rewriting the core.

---

## 2. The Diffusion Framework: MDLM

We use **MDLM (Masked Discrete Language Model)** with absorbing-state corruption.

### How it works
- **Forward process:** Each token is independently masked with probability (1 − α_t). A token is either its true value or [MASK]. No other corruption.
- **Reverse process:** A transformer predicts the clean token for each masked position, conditioned on all unmasked positions. Multiple tokens can be unmasked per step.
- **Training loss:** Weighted masked cross-entropy — predict the true token for each masked position, weighted by the MDLM continuous-time ELBO.
- **Noise schedule:** The noise schedule α_t controls the masking probability at each timestep. MDLM's continuous-time ELBO is **invariant** to the functional form of α_t (as long as it is monotonically decreasing), but BD3-LMs showed that **gradient variance is not** — different schedules lead to different training dynamics. We start with a **linear schedule** (α_t = 1 − t) as the simplest baseline. The schedule module must be designed for easy swapping: cosine schedule (from improved DDPM), geometric schedule, and custom learned schedules (à la MuLAN) should be drop-in replacements requiring only a config change, not code modification.

### Key properties we rely on
- **Inpainting is native:** Fix some tokens, run reverse process on the rest. This gives us constrained generation for free later.
- **ReMDM upgrade:** Train with σ=0 (pure MDLM). At inference, apply remasking (ReMDM) to reduce error propagation. No retraining needed.
- **Guidance hooks:** The denoising trajectory allows test-time reweighting of unmasking probabilities for guidance. Plan the architecture so guidance functions can be injected at each denoising step.

### Reference implementation
- **MDLM code:** `github.com/kuleshov-group/mdlm` — core framework. Written for text (1D token sequences). We adapt it to graph token sequences.
- **ReMDM code:** same group — inference-time remasking on top of MDLM.
- **Key paper:** MDLM (arXiv:2406.07524)

---

## 3. Graph Representation

A BD is flattened into a **SEQ_LEN-token discrete sequence** for MDLM, where SEQ_LEN = N_MAX + C(N_MAX, 2). N_MAX is the maximum number of rooms (configurable per dataset); N_EDGES = N_MAX*(N_MAX-1)/2 is the number of upper-triangle edge slots.

```
[node_1, node_2, ..., node_{N_MAX}, edge_(1,2), edge_(1,3), ..., edge_(N_MAX-1, N_MAX)]
 \________N_MAX tokens_____________/ \_________________N_EDGES tokens_________________/
          room types                         upper-triangle adjacency
```

| Dataset | N_MAX | N_EDGES | SEQ_LEN |
|---------|-------|---------|---------|
| RPLAN   |   8   |   28    |   36    |
| ResPlan |  14   |   91    |  105    |

### Token vocabularies (separate embedding tables)
- **Node vocabulary:** 13 room types + [MASK] + [PAD] = 15 tokens
  - Room types: MasterRoom, SecondRoom, LivingRoom, Kitchen, Bathroom, Balcony, Entrance, DiningRoom, StudyRoom, StorageRoom, WallIn, ExternalArea, ExternalWall
- **Edge vocabulary:** 10 spatial relationships + no-edge + [MASK] + [PAD] = 13 tokens
  - Spatial relationships: left-of, right-of, above, below, left-above, left-below, right-above, right-below, inside, surrounding

### Handling variable graph sizes
- Fixed sequence length = SEQ_LEN = N_MAX + C(N_MAX, 2) per dataset (36 for RPLAN with N_MAX=8).
- Graphs with fewer rooms: [PAD] tokens fill unused node and edge positions.
- [PAD] is distinct from [MASK]. The model must learn that [PAD] stays [PAD].

### PAD handling — critical implementation detail

[PAD] requires special treatment throughout the pipeline to prevent the model from wasting capacity on trivially predictable positions or learning degenerate shortcuts:

1. **During the forward process (masking):** PAD positions are **never masked**. They remain [PAD] at all noise levels. Only real node and edge positions participate in the masking schedule. This is enforced by a binary `is_pad` mask derived from the data, applied before the stochastic masking step.

2. **During loss computation:** PAD positions are **excluded from the loss**. The model should not be rewarded for predicting [PAD] — it would trivially learn to predict [PAD] everywhere if that reduced the loss. The loss mask is the intersection of "this position was masked" AND "this position is not PAD."

3. **During sampling (reverse process):** PAD positions are **clamped to [PAD]** throughout the entire reverse trajectory, analogous to how inpainting clamps known positions. The model never attempts to unmask a PAD position.

4. **Edge PAD logic:** An edge position (i, j) is PAD if **either** node i or node j is PAD. This is derived from the node PAD pattern, not stored independently. When a graph has N < N_MAX rooms, the first N node positions are real and the remaining N_MAX − N are PAD. All edges involving any PAD node are themselves PAD.

5. **[PAD] vs "no-edge":** These are semantically different. "No-edge" means "these two real rooms are not adjacent" — this is meaningful structural information the model must learn. [PAD] means "this position doesn't exist in this graph" — it carries no information. The distinction is critical for evaluation: when checking validity of a generated BD, PAD positions are ignored entirely.

---

## 4. Transformer Denoiser Architecture

### 4.0 Architecture overview and design rationale

The denoiser is a **standard transformer encoder** operating on the SEQ_LEN-token flat sequence. The core challenge is that this sequence is a **heterogeneous mix**: the first N_MAX positions are node tokens (room types) and the remaining N_EDGES are edge tokens (spatial relationships). These entity types have fundamentally different semantics — a "kitchen" token and a "north-of" token live in different conceptual spaces — yet they share the same sequence and must be processed jointly.

Without explicit handling, the transformer would have to implicitly learn which positions are nodes and which are edges, and how they relate structurally, purely from data. This is wasteful and fragile. We address this through three complementary mechanisms that inform the transformer about the graph structure of its input, while preserving the standard self-attention mechanism that enables MDLM's parallel unmasking, global context, and future inpainting/guidance capabilities.

**Why a transformer rather than a GNN:** At early diffusion steps, most tokens are [MASK] — there is no meaningful graph structure for message passing. A GNN's local message-passing would receive mostly uninformative [MASK] embeddings from neighbors. A transformer with full self-attention can reason about which positions should be unmasked next based on **global** patterns (e.g., "if positions 1–3 are already unmasked as Bedroom, Kitchen, Bathroom, then the edge between positions 1 and 2 is likely a spatial relationship, not no-edge"), regardless of current graph connectivity. This is the same insight behind DiGress's choice of graph transformer over standard MPNNs. Additionally, the transformer architecture is directly extensible: to build the future joint BD+RFP model, we can add continuous geometry tokens to the sequence and introduce cross-attention between discrete and continuous branches without changing the backbone.

**Architecture diagram (v1):**

```
Input: SEQ_LEN discrete tokens (N_MAX nodes + N_EDGES edges), each either true value, [MASK], or [PAD]
  │
  ├── Node tokens ──→ [Node Embedding Table] ──→ Linear(vocab=15, d_model) ──┐
  │                                                                           │
  └── Edge tokens ──→ [Edge Embedding Table] ──→ Linear(vocab=13, d_model) ──┤
                                                                              │
                                                              ┌───────────────┘
                                                              ▼
                                              [Composite Positional Encoding]
                                              = entity_type_emb(node|edge)
                                              + node_index_emb(i)        [nodes only]
                                              + pair_emb(i) + pair_emb(j) [edges only]
                                                              │
                                                              ▼
                                              [Time Embedding: sinusoidal(t)]
                                              added to all positions
                                                              │
                                                              ▼
                                          ┌─────────────────────────────────────┐
                                          │  Transformer Encoder (L layers)     │
                                          │  - Full self-attention (SEQ_LEN × SEQ_LEN)  │
                                          │  - LayerNorm + FFN                  │
                                          │  - No causal mask (bidirectional)   │
                                          └─────────────────────────────────────┘
                                                              │
                                          ┌───────────────────┴───────────────────┐
                                          ▼                                       ▼
                              [Node Classification Head]              [Edge Classification Head]
                              Linear(d_model, 15)                     Linear(d_model, 13)
                              → logits over node vocab                → logits over edge vocab
```

**Size (v1):** d_model = 128–256, L = 4–6 layers, 4–8 attention heads. Total parameters: ~1–5M. Deliberately small — the problem (SEQ_LEN tokens, vocab ≤ 15) does not require a large model. For RPLAN (SEQ_LEN=36), the full 36×36 attention matrix is trivially small (1,296 entries); even at N_MAX=14 (SEQ_LEN=105, 11,025 entries) no memory optimization is needed.

The denoiser has three additions to handle the mixed node/edge nature:

### 4.1 Composite positional encoding
Each position gets a sum of:
- **Entity-type embedding:** learned vector for "node" vs "edge" (2 embeddings)
- **Node-index embedding (nodes only):** which room slot (1st through N_MAX-th)
- **Pair-index embedding (edges only):** encodes both endpoints (i, j). Implement as sum of two learned embeddings (one for i, one for j), or a single embedding per pair

### 4.2 Structured attention biases (deferred to v2+)

> **v1 decision:** We skip structured attention biases in the initial implementation. With only SEQ_LEN tokens (36 for RPLAN), the transformer can learn graph structure from the composite positional encodings alone — the pair-index embeddings already tell the model which nodes each edge connects. Adding explicit attention biases would increase implementation complexity without clear benefit at this scale.

> **Future versions:** If v1 evaluation reveals that the model struggles with structural consistency (e.g., predicting edges that contradict the node types at their endpoints), structured attention biases should be introduced. The design would be additive bias terms in attention logits (precomputed once per forward pass): node i ↔ edges involving node i (incident edges), edge (i,j) ↔ nodes i and j (endpoint nodes), edge (i,j) ↔ edges sharing a node with (i,j). These are additive, not masks — the model can still attend globally but with an inductive bias toward structurally meaningful interactions.

### 4.3 Separate embedding tables
- Node tokens → node embedding table → linear projection to hidden dim
- Edge tokens → edge embedding table → linear projection to same hidden dim
- Output: separate classification heads for node positions (13+2 classes) and edge positions (10+3 classes)

### Reference architectures
- **DiGress** (`github.com/cvignac/DiGress`): graph transformer reference, evaluation metrics. DiGress uses edge-modulated attention on a 2D N×N tensor; we flatten to 1D instead (better for absorbing-state diffusion where most edges are [MASK], and directly compatible with MDLM's sequence-based framework).
- **MDLM transformer:** the text-domain transformer from MDLM codebase, adapted with our composite positional encodings and dual embedding tables. The training loop, noise schedule, and loss computation are reused directly.

---

## 5. Data Pipeline

### 5.1 Primary dataset: Graph2Plan's RPLAN extraction
- **Source:** `github.com/HanHan55/Graph2plan/releases/download/data/Data.zip`
- **Size:** ~80K floorplans
- **Format:** MATLAB `.mat` file (`data.mat`) containing:
  - `rType`: room type indices (13 categories)
  - `rEdge`: (u, v, r) triples — room indices and spatial relationship type
  - `gtBox` / `gtBoxNew`: room bounding boxes
  - `rBoundary`: room boundary polygons
  - `boundary`: building boundary with front door location
- **Spatial relationships in rEdge:** Computed from bounding-box centroids. Direction of vector (centroid_u → centroid_v) classified into 8 compass sectors (45° each) + inside + surrounding = 10 categories.
- **Status:** Ready to use. No preprocessing needed beyond loading and converting to our flat sequence format.

### 5.2 Supplementary dataset: ResPlan (needs conversion)
- **Source:** `github.com/m-agour/ResPlan` (also on Kaggle)
- **Size:** ~17K floorplans
- **Format:** JSON per plan. Keys are semantic categories; values are Shapely Polygon/MultiPolygon objects. Room adjacency graph provided as NetworkX object.
- **Current edge types:** door, arch, shared-wall (connection types, NOT spatial relationships)
- **Node attributes already available:** semantic label, polygon centroid, room area

**Conversion required:** Derive Graph2Plan-style spatial relationship edges from ResPlan data:
1. For each edge (u, v) in ResPlan's adjacency graph, get centroids (already stored as node attributes)
2. Compute angle = atan2(cy_u − cy_v, cx_u − cx_v)
3. Classify into 8 compass sectors (45° each) → one of {left-of, right-of, above, below, left-above, left-below, right-above, right-below}
4. Check bounding-box containment → inside / surrounding
5. Output (u, v, r) in same format as Graph2Plan's rEdge

**Design requirement:** The data loading code must be **unified** — a single `BubbleDiagramDataset` class that can load from either Graph2Plan or converted ResPlan data, producing identical flat token sequences. This means defining a common intermediate format (e.g., a list of dicts with `node_types`, `edge_triples`, `num_rooms`) that both loaders produce.

### 5.3 Room type vocabulary harmonization
ResPlan uses different room labels than RPLAN (e.g., "bedroom" instead of "MasterRoom"/"SecondRoom"). The conversion script must include a mapping table. Define this mapping explicitly and keep it configurable.

---

## 6. Training Plan

### 6.1 Training loop
- Standard MDLM training: sample t ~ Uniform(0,1), mask tokens with probability (1 − α_t), predict masked tokens, compute weighted CE loss
- Optimizer: AdamW
- Batch size, learning rate, warmup: hyperparameters to tune. Start with MDLM paper defaults adapted for our small sequence length (SEQ_LEN=36 for RPLAN vs. thousands in text)

### 6.2 What to log
- Training loss (weighted masked CE)
- Validation loss
- Sample quality metrics at regular intervals (generate N samples, compute validity + diversity)

### 6.3 Evaluation metrics (define before training)
- **Validity rate:** Is the generated BD architecturally plausible? (no duplicate room types that shouldn't be duplicated, connected graph, no contradictory spatial relationships)
- **Novelty:** Graph edit distance to nearest training sample
- **Diversity:** Number of distinct valid BDs in a batch of N samples
- **Distribution match:** Compare node-type histograms and edge-type histograms between generated and training data

---

## 7. Key Reference Papers and Repos

| Resource | URL | Use |
|----------|-----|-----|
| **MDLM** (paper) | arXiv:2406.07524 | Core diffusion framework |
| **MDLM** (code) | github.com/kuleshov-group/mdlm | Adapt from text to graph tokens |
| **ReMDM** (code) | same group | Inference remasking (later) |
| **DiGress** (code) | github.com/cvignac/DiGress | Graph transformer reference, eval metrics |
| **Graph2Plan** (code+data) | github.com/HanHan55/Graph2plan | Primary BD dataset (Data.zip) |
| **ResPlan** (code+data) | github.com/m-agour/ResPlan | Supplementary dataset (needs conversion) |
| **MaskPLAN** (code) | github.com/HangZhangZ/MaskPLAN | Reference for stochastic masking during training |
| **CCDD** (paper) | arXiv:2510.03206 | Future joint model architecture reference |
| **Training-free guidance** (paper) | arXiv:2409.07359 | Future guidance mechanism reference |

---

## 8. Architecture Extensibility Checklist

When planning the implementation, verify these future-proofing requirements:

- [ ] **Conditioning input slot:** The denoiser forward pass should accept an optional `condition` tensor (unused now, will carry HB features later). Pass `None` during unconditional training.
- [ ] **Guidance hook:** The sampling loop should have a pluggable step where a guidance function can modify unmasking probabilities before token selection.
- [ ] **Inpainting mask:** The sampling loop should accept a binary mask specifying which positions are fixed (clamped) vs. free. For unconditional generation, all positions are free.
- [ ] **Modular token embedding:** The embedding layer should be swappable — later we'll add a continuous embedding branch for geometry tokens alongside the discrete node/edge embeddings.
- [ ] **Separate prediction heads:** Node and edge output heads should be separate modules, so we can later add a geometry prediction head without touching the BD heads.
- [ ] **Dataset abstraction:** `BubbleDiagramDataset` returns flat token sequences + metadata. The model never sees raw Graph2Plan or ResPlan data structures.
- [ ] **Noise schedule as swappable module:** The schedule (α_t function) should be a standalone module selected by config, so that linear, cosine, geometric, or learned schedules can be compared without touching training code.

---

## 9. What to Build from Existing Code vs from Scratch

### Reuse from MDLM repo (`github.com/kuleshov-group/mdlm`)
- Noise schedule implementations (linear, cosine)
- Training loop structure (sample t, mask, compute loss, backprop)
- Continuous-time ELBO loss computation
- Sampling loop (ancestral sampling; later ReMDM from `github.com/kuleshov-group/remdm`)
- Hydra config structure for experiment management

### Reuse from Graph2Plan repo (`github.com/HanHan55/Graph2plan`)
- Data loading utilities for the `.mat` file (`scipy.io.loadmat`)
- Room type vocabulary definition (13 categories)
- Spatial relationship computation logic (centroid-to-centroid angle classification)

### Build new
- `BubbleDiagramDataset`: loads Graph2Plan `.mat` data, converts to SEQ_LEN-token flat sequences with PAD handling
- Composite positional encoding module (entity-type + node-index + pair-index embeddings)
- Dual embedding layer (separate node/edge embedding tables → shared hidden dim)
- Dual output heads (separate node/edge classification heads)
- The full denoiser model class that wraps transformer + embeddings + heads
- Graph validity checker (for evaluation: connectivity, spatial consistency, room-type constraints)
- BD visualization utilities (draw generated BDs as labeled graphs with edge types)

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Graph2Plan `.mat` file is hard to parse or has undocumented structure | Medium | Blocks data pipeline | Use `scipy.io.loadmat`; inspect a few samples manually first; Graph2Plan repo has example loading code |
| Model generates trivial outputs (all PAD, or all same room type) | Medium | Wastes training time | Monitor diversity metrics from early epochs; check PAD handling is correct (excluded from loss, never masked) |
| Edge class imbalance causes model to predict "no-edge" everywhere | High | Poor edge generation | See Section 11 (edge sparsity). Use class-weighted loss or focal loss on edge positions |
| Spatial relationship inconsistencies in generated BDs (e.g., A left-of B AND B left-of A) | Medium | Low validity rate | Add explicit validity checks in evaluation; later use ReMDM to fix at inference |
| Model overfits on 80K samples | Low | Poor generalization | Monitor train/val loss gap; 80K samples for a 1–5M param model on 36–105 tokens is likely sufficient |
| MDLM codebase is text-specific and hard to adapt | Low | Slower development | The core (noise schedule, ELBO loss, sampling) is domain-agnostic; only the model architecture needs replacement |

---

## 11. Edge Sparsity — Key Design Consideration

In a typical RPLAN floorplan with N = 4–8 rooms and N_MAX=8:
- **N_EDGES = 28 edge positions** total (upper triangle of N_MAX × N_MAX)
- **PAD depends on actual room count N.** For N=8 (maximum), all 28 edges are real (zero PAD). For N=6: C(6,2)=15 real edges, 13 PAD edges.
- Of the real edges, **most are "no-edge"** (rooms that are not adjacent). A typical 6–8 room floorplan has ~10–15 actual adjacencies, so the remainder of real edge positions are "no-edge."
- Only **~10–15 positions** (out of 28) carry actual spatial relationship tokens.

Right-sizing N_MAX to the dataset (8 for RPLAN instead of, say, 16) significantly improves this ratio: with N_MAX=16 there would be 120 edge slots, of which 92+ would be PAD for a typical RPLAN graph, wasting model capacity on trivially predictable positions.

This class imbalance — where the informative signal (spatial relationships) occupies ~35–55% of real edge positions but a smaller share of total edge positions when PAD is present — is the most likely source of training difficulty.

### Mitigations (implement in order of priority)

1. **PAD exclusion from loss** (already specified in Section 3): removes the dominant PAD class entirely from the learning signal. This is the single most important mitigation.

2. **Class-weighted cross-entropy on edge positions:** Upweight the loss on actual spatial relationship tokens (left-of, right-of, ..., surrounding) relative to "no-edge" tokens. The weight can be set inversely proportional to class frequency in the training set, or tuned as a hyperparameter.

3. **Monitor per-class accuracy during training:** Track accuracy separately for node predictions, no-edge predictions, and spatial-relationship predictions. If the model achieves 95% accuracy on no-edge but 30% on spatial relationships, the weighting needs adjustment.

4. **Focal loss (if needed):** If class weighting alone is insufficient, replace weighted CE with focal loss on edge positions, which downweights easy (no-edge) examples and focuses the gradient on hard (spatial relationship) examples.

5. **Data-level validation:** Before training, compute the actual class distributions in the Graph2Plan dataset. Verify that the 10 spatial relationship types are reasonably balanced among themselves (not all "left-of"), and that the no-edge ratio matches the estimate above. Document these statistics.
