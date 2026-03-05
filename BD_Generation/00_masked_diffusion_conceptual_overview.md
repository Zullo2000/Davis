---
title: "Masked Diffusion for Bubble Diagram Generation: Conceptual Overview and Experimental Analysis"
date: "February 2026"
geometry: "margin=2.5cm"
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{float}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \setlength{\parskip}{0.5em}
  - \setlength{\parindent}{0em}
---

# 1. Introduction

This document provides a conceptual overview of masked discrete diffusion for bubble diagram generation. We cover the forward and backward processes, unmasking strategies, noise schedules, remasking (ReMDM), and the learned forward process (MELD/v2). Each concept is related to our experimental evaluation across 24 configurations tested on the RPLAN dataset (86K samples, 5 seeds each, 1000 samples per seed).

Our bubble diagrams are represented as flat token sequences of length 36: 8 node positions (room types) followed by 28 edge positions (spatial relationships between room pairs, upper-triangle only with $i < j$). Inactive positions (fewer than 8 rooms) are marked as PAD and never participate in the diffusion process.

# 2. The Forward Process (Fixed Schedule, v1)

The forward process progressively corrupts a clean graph $\mathbf{x}_0$ into a fully masked state $\mathbf{x}_1$ over continuous time $t \in [0, 1]$.

At timestep $t$, each non-PAD position is **independently** kept with probability $\alpha(t)$ or replaced with a special MASK token with probability $1 - \alpha(t)$:

$$P(\mathbf{x}_t^l = \text{MASK} \mid \mathbf{x}_0^l) = 1 - \alpha(t)$$

$$P(\mathbf{x}_t^l = \mathbf{x}_0^l \mid \mathbf{x}_0^l) = \alpha(t)$$

The function $\alpha(t)$ is the **noise schedule**. It must satisfy $\alpha(0) = 1$ (fully clean) and $\alpha(1) = 0$ (fully masked). The masking is absorbing: once a token becomes MASK, it cannot spontaneously return to a clean state during the forward process.

Key properties:

- **Position independence**: Each position is masked independently. The only coupling is through the shared timestep $t$ and the global $\alpha(t)$.
- **Node-edge distinction**: Node positions use `NODE_MASK_IDX`, edge positions use `EDGE_MASK_IDX`. These are separate tokens in separate vocabularies.
- **PAD invariance**: PAD positions are never masked. This is the most critical invariant in the entire pipeline.

# 3. The Backward Process (Reverse Denoising)

Generation runs the forward process in reverse: starting from $\mathbf{x}_1$ (fully masked), we progressively unmask tokens over $N = 100$ discrete steps from $t = 1$ to $t = 0$.

At each step $i$ (from $N-1$ down to $0$):

1. **Compute timesteps**: $t_{\text{now}} = (i+1)/N$, $t_{\text{next}} = i/N$.

2. **Model prediction**: The denoiser (a transformer) takes the current partially-masked sequence $\mathbf{x}_t$ and the timestep, and predicts logits for **all** positions simultaneously (including already-unmasked ones, though only masked positions matter).

3. **Token selection**: From the logits, we select a predicted token for each position using one of three strategies (Section 6).

4. **Unmasking probability**: We compute $p_{\text{unmask}} = \frac{\alpha(t_{\text{next}}) - \alpha(t_{\text{now}})}{1 - \alpha(t_{\text{now}})}$, the conditional probability that a currently-masked position should be unmasked at this step.

5. **Unmasking decision**: We decide which masked positions to unmask (Section 4).

6. **Token update**: Unmasked positions get their predicted tokens. Already-decoded positions are **carried over** unchanged (the model cannot override past decisions without remasking).

The carry-over property is fundamental: at each step, the model commits to tokens for the newly unmasked positions, and those tokens persist for all remaining steps. This means early mistakes cannot be corrected — which is precisely the motivation for remasking (Section 7).

# 4. Unmasking Modes: Random vs. LLaDA

The unmasking mode determines **which** masked positions get unmasked at each step.

## 4.1 Random Unmasking (Original MDLM)

Each masked position independently flips a biased coin with probability $p_{\text{unmask}}$. If the coin lands heads, the position is unmasked; otherwise it stays masked. This is a direct implementation of the MDLM reverse process.

Properties:

- No dependency ordering: a semantically "hard" edge might unmask before its endpoint node types are known.
- Maximum stochasticity: different runs with the same seed can unmask very different positions at each step.
- Simple and fast: no sorting or ranking needed.

## 4.2 LLaDA Unmasking (Confidence-Based Top-$k$)

Instead of independent coin-flips, we compute a **budget** $k = \lceil p_{\text{unmask}} \times |\text{masked positions}| \rceil$ and unmask the $k$ positions where the model is **most confident** in its prediction (highest $P(\hat{x}^l \mid \mathbf{x}_t)$ from the softmax).

This was introduced by Nie et al. (LLaDA) and fundamentally changes the unmasking dynamics:

1. **High-confidence positions unmask first.** Node types and "easy" edges (adjacency between common room pairs) tend to have high confidence early in the denoising process, because the model has strong priors about which room types appear together.

2. **Hard decisions are deferred.** Containment edges (inside/surrounding) and rare spatial relationships unmask later, when more context is available from already-decoded positions.

3. **Implicit dependency resolution.** The confidence ranking acts as an automatic dependency resolver — structural "anchor" tokens (node types, high-frequency edges) lock in first, providing context for lower-confidence decisions.

**Based on our experimental analysis, we recommend LLaDA unmasking for all production configurations.** It preserves inside\_validity at $>93\%$ across all settings, while random unmasking degrades to 54–84% (see Section 10 for details). The confidence-based ordering acts as an implicit dependency resolver — structural anchor tokens lock in first, providing context for harder decisions — which is critical for architectural validity.

# 5. Noise Schedules and the Alpha-Sigma Relationship

## 5.1 Linear vs. Log-Linear: Why "Linear" Is Not Really Linear

A common source of confusion: the "linear" schedule refers to **linearity in $\sigma$-space**, not in $\alpha$-space. The relationship between $\sigma$ and $\alpha$ is exponential.

**The chain:**

- The noise schedule is defined in terms of $\sigma(t)$, where $\sigma$ is the log signal-to-noise ratio.
- For the "linear" schedule: $\sigma(t) = t \cdot \sigma_{\max}$ (grows linearly with $t$).
- But the actual masking probability uses $\alpha(t) = e^{-\sigma(t)} = e^{-t \cdot \sigma_{\max}}$.

Since $\alpha = \exp(-\sigma)$, a linear growth in $\sigma$ produces **exponential decay** in $\alpha$:

| $t$ | $\sigma(t) = 5t$ | $\alpha(t) = e^{-5t}$ | % masked |
|:---:|:---------:|:----------:|:----------:|
| 0.0 | 0.0 | 1.000 | 0% |
| 0.1 | 0.5 | 0.607 | 39% |
| 0.3 | 1.5 | 0.223 | 78% |
| 0.5 | 2.5 | 0.082 | 92% |
| 0.7 | 3.5 | 0.030 | 97% |
| 1.0 | 5.0 | 0.007 | 99.3% |

This means with the linear schedule, $\alpha(0.5) \approx 0.007$ — the graph is already 99.3% masked at the halfway point. Almost all the "interesting" masking (where the model needs to learn difficult predictions) happens in the first 30% of the timeline.

**The log-linear schedule** is truly linear in $\alpha$-space: $\alpha(t) = 1 - t$, so $\alpha(0.5) = 0.5$ (50% masked at the halfway point). This distributes the learning signal evenly across the timeline.

**Based on our analysis, we recommend the log-linear schedule.** It provides uniform difficulty progression across the timeline — $\alpha(0.5) = 0.5$ means the model trains on a balanced mix of easy and hard examples — and avoids the training instabilities of the linear schedule (Section 5.2). Empirically, log-linear achieves stable convergence where the linear schedule with importance sampling produces collapsed outputs.

## 5.2 Why Importance Sampling Broke the Linear Schedule

Importance Sampling (IS) of timesteps samples $t$ proportional to the ELBO weight $w(t)$ instead of uniformly. With the linear schedule, $w(t)$ is extremely concentrated near $t \to 0$, so IS samples almost exclusively from the "nearly clean" regime. The model trains almost exclusively on easy examples (1–2 tokens masked), never learning to denoise from heavy corruption. This produced models with near-zero diversity and collapsed outputs.

With the log-linear schedule, $w(t) = 1/t$ is more uniform (see Section 6.2), so IS does not pathologically collapse training onto a single regime. We use IS with the log-linear schedule: it allocates more training signal to timesteps near $t \to 0$ where the model makes its highest-stakes final predictions, improving accuracy where it matters most for generation quality (see Section 6.3).

# 6. The ELBO Weight $w(t)$ and Importance Sampling

## 6.1 Definition

The ELBO loss for masked diffusion is a time-integral:

$$\mathcal{L} = \mathbb{E}_{t \sim U[0,1]} \left[ \sum_{l=1}^{L} w_l(t) \cdot \text{CE}\left(p_\theta(\cdot \mid \mathbf{x}_t), \mathbf{x}_0^l\right) \right]$$

where the per-position ELBO weight is:

$$w_l(t) = \frac{-\alpha'_l(t)}{1 - \alpha_l(t)}$$

For the **fixed** forward process (v1), $\alpha_l = \alpha$ for all positions, so $w(t)$ is a scalar depending only on time.

## 6.2 Concrete Values for Different Schedules

**Log-linear** ($\alpha(t) = 1 - t$): $w(t) = \frac{1}{t}$, which is large near $t = 0$ but integrable.

| $t$ | $w(t)$ |
|:---:|:------:|
| 0.01 | 100 |
| 0.1 | 10 |
| 0.5 | 2 |
| 1.0 | 1 |

**Linear** ($\alpha(t) = e^{-\sigma_{\max} t}$): $w(t) = \frac{\sigma_{\max}}{e^{\sigma_{\max} t} - 1}$.

| $t$ | $w(t)$ |
|:---:|:------:|
| 0.01 | 9,990 |
| 0.1 | 75 |
| 0.5 | 0.6 |
| 1.0 | 0.034 |

The linear schedule's $w(t)$ is **5 orders of magnitude higher** near $t = 0$ compared to $t = 1$. This extreme concentration means that if we sample $t \propto w(t)$ (importance sampling), we would train almost exclusively at $t \approx 0$ where the graph is nearly clean.

## 6.3 Relationship to Importance Sampling

Standard ELBO training samples $t \sim U[0,1]$ and weights the loss by $w(t)$. Importance Sampling (IS) replaces this with $t \sim q(t) \propto w(t)$ and removes the weight (or applies the appropriate correction factor). The goal is to reduce gradient variance by sampling more frequently from timesteps with high ELBO contribution.

**Why IS + linear schedule breaks training**: $w(t)$ for the linear schedule concentrates 99%+ of probability mass in $t < 0.05$. The model trains almost exclusively on nearly-clean inputs (1–2 tokens masked), never learning to handle heavy corruption. During generation, early denoising steps ($t$ near 1, heavy masking) produce garbage predictions because the model was never trained on them. This combination is unusable.

**Why we use IS with the log-linear schedule**: With the log-linear schedule, $w(t) = 1/t$ is much less concentrated than the linear case — the effective range (with clamping at $t_{\min} = 0.001$) is $[1, 1000]$, three orders of magnitude versus five. This means IS does not pathologically collapse training onto a single regime. Instead, it provides a principled benefit: it allocates more training signal to timesteps near $t \to 0$, where the model makes its final, highest-stakes predictions (assigning the last few tokens). These late-stage predictions have the highest ELBO weight because each remaining masked token carries disproportionate information about the graph. By sampling these timesteps more frequently, IS improves the model's accuracy exactly where it matters most for generation quality — the final refinement steps. The clamping at $t_{\min}$ prevents the $1/t$ singularity from dominating, ensuring the model still sees a healthy distribution of intermediate noise levels.

# 7. Token Prediction Strategies

At each denoising step, we must select a token for each masked position from the model's predicted logits. Three strategies are available:

**Argmax (temperature = 0)**: Select the highest-logit token. Purely deterministic — given the same model state, always predicts the same token. This produces near-zero diversity (5 unique archetypes in our experiments) because every generation follows the same greedy path.

**Temperature sampling (temperature $> 0$)**: Add Gumbel noise scaled by temperature, then take argmax. Higher temperature = more randomness. We use float64 for the Gumbel noise computation to prevent underflow in $\log(-\log(u))$.

**Top-$p$ (nucleus) sampling ($p = 0.9$)**: Keep the smallest set of tokens whose cumulative softmax probability $\geq p$, zero out the rest, and sample from the filtered distribution. This preserves the top of the distribution while cutting off the long tail of unlikely tokens. Top-$p$ = 0.9 is our standard choice — it provides enough stochasticity for diversity while staying within the model's confident predictions.

**Based on our analysis, we recommend top-$p$ = 0.9 for all remasking experiments.** Remasking requires stochastic predictions to explore alternatives (Section 8): when a position is remasked, the model must be able to propose a different token than its previous greedy choice. With argmax (temperature = 0), remasked positions simply re-predict the same token deterministically, defeating the purpose of remasking. Top-$p$ = 0.9 provides the necessary stochasticity while staying within the model's confident predictions, avoiding the low-quality tail of the distribution.

# 8. Remasking (ReMDM)

## 8.1 Motivation

Standard masked diffusion commits irrevocably to each unmasked token. If the model makes an early mistake (e.g., predicting the wrong edge type when context is sparse), that error persists and can propagate to subsequent decisions. **Remasking** (from Schiff et al., ReMDM) addresses this by periodically re-masking already-decoded positions, forcing the model to re-predict them with updated context.

## 8.2 The Remasking Loop

At each denoising step, **after** unmasking new positions but **before** the next model call:

1. Identify **decoded positions** (non-PAD, non-MASK) — candidates for remasking.
2. Compute a **remasking budget** $\sigma_{\max}$ from the noise schedule: $\sigma_{\max} = \frac{1 - \alpha(t_{\text{next}})}{\alpha(t_{\text{now}})}$, clamped to $[0, 1]$.
3. Select positions to remask based on the strategy (Section 8.3).
4. Replace selected positions with MASK tokens.

Remasking is **never** applied at the final step ($i = 0$) — all tokens must be finalized.

## 8.3 Remasking Strategies: Cap vs. Confidence

**Cap strategy** ($\eta$ parameter): Each decoded position is remasked with probability $\min(\sigma_{\max}, \eta)$. The $\eta$ parameter caps the maximum per-position remasking rate. This is simple but has a tunable hyperparameter.

**Confidence strategy** (no $\eta$ tuning): Remasking probability is proportional to $\text{softmax}(-\text{confidence})$ over decoded positions, where confidence is the model's softmax probability for the predicted token. Low-confidence predictions are more likely to be remasked. The remasking budget scales with $\sigma_{\max}$, but the allocation across positions is adaptive.

**Key advantage of confidence**: It is self-regulating. At high noise levels (early in denoising), few positions have reliable confidence, so the confidence softmax naturally distributes remasking budget thinly. No $\eta$ tuning is needed. Our experiments confirm confidence matches or beats cap on all priority metrics while requiring zero hyperparameter tuning.

## 8.4 The t\_switch Parameter

$t_{\text{switch}}$ controls **when** remasking activates. Remasking is only applied when $t_{\text{now}} < t_{\text{switch}}$. With $t_{\text{switch}} = 1.0$, remasking is active at all steps (always-on). With $t_{\text{switch}} = 0.5$, remasking only starts in the second half of denoising.

Our experiments show $t_{\text{switch}}$ is non-critical for confidence remasking — the spread across $t_{\text{switch}} \in \{0.3, 0.5, 0.7, 1.0\}$ is only 0.007 on the novelty $\times$ coverage composite. The confidence softmax already provides natural throttling, making the explicit switch largely redundant. **Based on our analysis, we recommend $t_{\text{switch}} = 0.5$** for v1 — it offers a slight inside validity advantage (93.8% vs 93.3%) and follows a conceptually clean "build then refine" pattern: the first half of denoising builds the initial structure unperturbed, and remasking activates in the second half to correct errors with fuller context. See Section 12.2 for a detailed comparison of $t_{\text{switch}}$ values.

## 8.5 Impact of Remasking on Generation Quality

Remasking introduces a tradeoff:

- **Diversity boost**: Unique archetypes increase from 29 to 121 (4.2$\times$) with confidence remasking on v1.
- **Distributional cost**: Conditional edge TV (weighted) degrades from 0.472 to 0.571 (approximately 21% relative increase). Stochastic re-prediction after remasking introduces variance in edge-type assignment.
- **Structural cost**: Spatial transitivity drops from 99.9% to 98.7%. Some re-drawn edge types create horizontal ordering cycles.
- **Validity cost**: Inside validity drops from 99.4% to 93.3% (LLaDA); from 84% to approximately 55% (random). This is why LLaDA unmasking is essential when remasking is enabled.

# 9. The Learned Forward Process (v2 / MELD)

## 9.1 The State Clashing Problem

In v1, the forward process applies **uniform masking**: every non-PAD position is independently masked with the same probability $1 - \alpha(t)$ at timestep $t$. This means structurally distinct bubble diagrams can collapse to **identical masked states** at intermediate timesteps.

Consider two graphs that differ only in their spatial edge assignments: Graph A has rooms 2 and 5 in a left-of relationship, while Graph B has them right-of. At $t = 0.5$, both graphs might produce the same partially-masked sequence — if the edge between rooms 2 and 5 happens to be masked along with the same set of other positions. The denoiser then faces an ambiguous posterior: the observed masked state is consistent with both graphs. Since our denoiser makes **independent per-position predictions** (factorized output), it cannot represent the multimodal posterior and is forced to spread probability mass, producing high-entropy predictions.

**Where state clashing actually matters in our tokenization:**

Our sequence uses upper-triangle edges only — there is exactly one position per room pair $(i, j)$ with $i < j$. So there is no edge$(A,B)$/edge$(B,A)$ redundancy. The real dependencies that cause clashing are:

- **Edges sharing a node**: edges $(1,3)$ and $(2,3)$ both involve room 3. Their correct edge types depend on room 3's type, but room 3 may be masked while these edges are being predicted.
- **Node-edge dependencies**: a node's type constrains which edge types are valid for its connections (e.g., a Bathroom cannot have an "inside" relationship with a LivingRoom).
- **Transitive chains**: if room A is left-of room B and room B is left-of room C, then room A should be left-of room C. Uniform masking can mask the A-C edge while leaving A-B and B-C unmasked, losing the transitivity constraint.

## 9.2 From v1 to v2: What Changes and Why

The MELD solution replaces the fixed forward process with a **learned** one. To see exactly what changes, compare the two pipelines end-to-end:

```
v1 (fixed forward process):
  x_0 → forward_mask(α(t)) → x_t → denoiser(x_t, t) → logits → ELBO(w(t))
         scalar α(t)                discrete tokens        scalar w(t)

v2 (learned forward process):
  x_0 → rate_network(t) → α_l(t) → STGS → soft_emb → denoiser(soft_emb, t) → logits → ELBO(w_l(t))
         per-position α_l(t)         Gumbel-Softmax   pre_embedded path       per-pos w_l(t)
```

Three things change from v1 to v2 — everything else (the denoiser architecture, the vocabulary, the data pipeline, the evaluation metrics) stays the same:

1. **The masking schedule becomes learned and per-position.** In v1, a single scalar $\alpha(t)$ (from the log-linear or linear noise schedule) is broadcast to all 36 positions. In v2, a small **rate network** with learnable parameters $\phi$ produces a separate $\alpha_l(t, \phi)$ for each position $l$. This is the core change that addresses state clashing: the network can learn to mask nodes before edges, or to keep structurally-coupled positions visible together.

2. **Masking goes through Straight-Through Gumbel-Softmax (STGS).** In v1, masking is a hard Bernoulli sample ($u \geq \alpha \Rightarrow \text{mask}$), and the denoiser receives **discrete token indices**. In v2, STGS produces hard keep/mask decisions on the forward pass (identical behavior at inference) but replaces them with **soft mixing weights** on the backward pass. The denoiser receives **soft embeddings** — weighted combinations of the clean and MASK embeddings — so gradients from the loss can flow back through the masking decisions into $\phi$. This is why the denoiser's `forward()` gains a `pre_embedded` flag: when True, it skips its own embedding lookup and directly processes the soft embeddings from STGS.

3. **The ELBO weights become per-position.** In v1, the scalar weight $w(t) = -\alpha'(t) / (1 - \alpha(t))$ is the same for all positions at a given timestep. In v2, each position gets its own weight $w_l(t) = -\alpha'_l(t) / (1 - \alpha_l(t))$, derived from its position-specific schedule. Positions that the rate network masks aggressively (large $|\alpha'_l|$) receive proportionally higher loss weight, ensuring the ELBO remains a valid variational bound.

**What stays the same:** the denoiser transformer (same architecture, same parameters), the vocabulary and tokenization, the data loader, the evaluation pipeline, and the sampling loop structure (which simply uses $\alpha_l(t)$ instead of a scalar $\alpha(t)$ for unmasking probabilities).

## 9.2.1 The Rate Network

The rate network (~5K parameters) is the only new module. It maps a timestep $t$ to 36 position-specific keeping probabilities $\alpha_l(t)$.

**Architecture (step by step):**

1. **Learnable node embeddings**: An `nn.Embedding(8, 32)` table gives each of the 8 node positions its own 32-dimensional identity vector $\mathbf{h}_i^{\text{node}}$.

2. **Edge embeddings via endpoint summation**: For edge position $(i, j)$, the embedding is $\mathbf{h}_{ij}^{\text{edge}} = \mathbf{h}_i^{\text{node}} + \mathbf{h}_j^{\text{node}}$. No additional parameters — edges inherit structural identity from their endpoints. This means edges sharing a node automatically get similar (but not identical) embeddings, which is desirable since they share structural dependencies.

3. **Type-specific projection**: Separate linear layers for nodes and edges: $\mathbf{z}_l = \text{proj}_{\text{type}}(\mathbf{h}_l)$. This allows the MLP to treat nodes and edges differently.

4. **MLP to polynomial coefficients**: A 2-layer MLP ($32 \to 64 \to K$) with SiLU activation maps each projected embedding to $K = 4$ raw values, passed through softplus to ensure strict positivity: $\mathbf{w}_l = \text{softplus}(\text{MLP}(\mathbf{z}_l)) \in \mathbb{R}^{K}_{>0}$.

5. **Monotonic polynomial evaluation**: $\hat{\gamma}_l(t) = \frac{\sum_{k=1}^{K} w_k^l \cdot t^k}{\sum_{k=1}^{K} w_k^l}$. This is guaranteed monotonically increasing since all $w_k > 0$ and $t^k$ is increasing. Properties: $\hat{\gamma}_l(0) = 0$ and $\hat{\gamma}_l(1) = 1$.

6. **Scale to gamma range**: $\gamma_l(t) = \hat{\gamma}_l(t) \cdot (\gamma_{\max} - \gamma_{\min}) + \gamma_{\min}$, where $\gamma_{\min} = -13$ and $\gamma_{\max} = 5$. At $t = 0$: $\gamma = -13$, so $\alpha = \sigma(-(-13)) = \sigma(13) \approx 1.0$ (clean). At $t = 1$: $\gamma = 5$, so $\alpha = \sigma(-5) \approx 0.007$ (masked).

7. **Sigmoid to alpha**: $\alpha_l(t) = \sigma(-\gamma_l(t))$.

8. **Analytical derivative**: $\alpha'_l(t) = -\alpha_l(1 - \alpha_l) \cdot \frac{d\gamma_l}{dt}$, computed in a single forward pass alongside $\alpha_l$ for efficiency.

**PAD handling**: PAD positions are forced to $\alpha = 1.0$ and $\alpha' = 0.0$ after computation, ensuring they are never masked.

The key insight is that the rate network learns position-specific corruption trajectories: it might learn to mask node types early and edges late (or vice versa), or to mask edges between certain room pairs at different rates. This breaks the symmetry that causes state clashing.

## 9.3 Why We Need Gradient Flow Through Discrete Decisions (STGS)

The training loss for v2 is:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{t} \left[ \sum_{l} w_l(t, \phi) \cdot \text{CE}(p_\theta(\cdot \mid \mathbf{x}_t), \mathbf{x}_0^l) \right]$$

The rate network $\phi$ affects this loss through **two pathways**:

**Path 1 — Through the ELBO weights** $w_l(t, \phi) = -\alpha'_l / (1 - \alpha_l)$: These are smooth, differentiable functions of $\phi$. Gradients flow freely. This tells the rate network "this position is hard/easy to reconstruct at this timestep."

**Path 2 — Through the masked input** $\mathbf{x}_t$: Changing $\phi$ changes $\alpha_l(t)$, which changes **which positions are masked**, which changes the denoiser's input, which changes the loss. But the masking step is a **discrete Bernoulli sample** — we draw $u \sim \text{Uniform}(0,1)$ and mask if $u \geq \alpha_l$. This hard threshold has zero gradient almost everywhere (it is a step function of $\alpha_l$).

**Why Path 2 matters**: Path 1 alone tells the rate network which positions are hard, but misses the crucial inter-position coupling. For bubble diagrams, the question is not just "is position $l$ hard?" but "does masking position $A$ instead of position $B$ give the denoiser better context for predicting position $C$?" Nodes constrain their edges, edge transitivity requires multiple edges to be visible simultaneously — these couplings only flow through Path 2.

**How STGS fixes it**: Straight-Through Gumbel-Softmax replaces the hard Bernoulli masking step with a differentiable approximation:

1. **Logits**: For each position, compute $[\log \alpha_l, \log(1 - \alpha_l)]$ — log-probabilities of keep/mask.

2. **Gumbel noise**: Add i.i.d. Gumbel noise (in float64 for numerical stability).

3. **Soft weights**: Apply softmax with temperature $\tau$: $\mathbf{w}^{\text{soft}} = \text{softmax}((\text{logits} + \text{Gumbel}) / \tau)$. This gives a soft 2-vector $[w_{\text{keep}}, w_{\text{mask}}]$ per position.

4. **Hard decisions (forward pass)**: Take argmax to get one-hot $[1, 0]$ (keep) or $[0, 1]$ (mask). These are used to build the actual masked sequence and compute the loss.

5. **Soft gradients (backward pass)**: The straight-through trick: forward uses hard one-hot, backward uses soft Gumbel-Softmax gradients. Implemented as: $\text{output} = \mathbf{w}^{\text{hard}} - \text{sg}(\mathbf{w}^{\text{soft}}) + \mathbf{w}^{\text{soft}}$, where $\text{sg}$ is stop-gradient.

6. **Soft embeddings**: Instead of looking up a discrete clean or MASK embedding, we **mix** them: $\mathbf{e}_l = w_{\text{keep}} \cdot \mathbf{e}_{\text{clean}}^l + w_{\text{mask}} \cdot \mathbf{e}_{\text{MASK}}^l$. These soft embeddings are passed to the denoiser's transformer, which processes them like normal embeddings. Gradients from the denoiser's loss flow back through the mixing weights to $\alpha_l$ and hence to $\phi$.

This gives us the best of both worlds: **forward pass** uses hard discrete decisions (matching actual inference), while the **backward pass** gets smooth gradient signal through the masking decisions, enabling the rate network to learn inter-position coupling.

# 10. Inside Validity and Why LLaDA Is Necessary

## 10.1 Definition

Inside validity checks whether generated bubble diagrams contain architecturally forbidden containment relationships. We define 69 forbidden pairs — combinations where one room type cannot physically be inside another (e.g., LivingRoom inside Bathroom). A graph fails if it contains **even one** violation. The RPLAN training set baseline is 99.78%.

## 10.2 The Decisive Gap

| Method | Inside validity |
|:-------|:--------------:|
| llada\_argmax\_no\_remask | **100.0%** |
| llada\_topp0.9\_no\_remask | **99.4%** |
| llada + conf tsw=0.3 | 95.9% |
| llada + conf tsw=1.0 | 93.3% |
| random\_argmax\_no\_remask | 94.6% |
| random\_topp0.9\_no\_remask | 84.0% |
| random + cap eta=0.2 | 59.3% |
| random + confidence (any tsw) | 54–56% |
| v2\_llada\_no\_remask | **100.0%** |
| v2\_llada + conf tsw=1.0 | **99.5%** |

LLaDA methods stay at 93–100%, while random methods collapse to 54–84%. This is a hard requirement: a model that produces architecturally impossible floorplans in ~45% of samples is unusable regardless of its distributional metrics.

## 10.3 Why Random Unmasking Fails

The root cause is **unmasking order vs. semantic dependencies**. The inside validity check enforces constraints between pairs of room types and their spatial relationship (inside/surrounding edges). To produce a valid containment assignment, the model needs to know the node types of both rooms before assigning an inside/surrounding edge between them.

**LLaDA unmasking** naturally resolves this: high-confidence positions (typically node types and common edges) unmask first, providing context for later, harder decisions like containment edges.

**Random unmasking** breaks this dependency ordering: inside/surrounding edges may unmask before their node types, forcing the model to "guess" containment without knowing which rooms are involved. The model falls back on marginal priors, producing forbidden pairs.

## 10.4 Why Remasking Amplifies the Problem for Random

Random unmasking drops from 84% (no remask) to ~55% (with remasking) — a **29pp collapse**. LLaDA drops from 99.4% to ~93% — only a **6pp drop**.

With random unmasking, remasking is doubly destructive: it re-masks already-decided positions, and the re-prediction may happen in a context where surrounding node types have also been remasked. Each remasking cycle compounds the probability of forbidden-pair violations. With LLaDA, high-confidence node types are unlikely to be remasked (confidence strategy targets low-confidence positions), so edge re-predictions still have node-type context.

# 11. Priority Metrics

## 11.1 Metric Definitions

We evaluate generated bubble diagrams using five priority metrics. Each is defined precisely below.

**1. Novelty.** The fraction of generated graphs that do not appear in the training set (exact hash-based match). Each graph is hashed as `(num_rooms, tuple(node_types), tuple(sorted(edge_triples)))`. Novelty = 1.0 means every generated graph is new; novelty = 0.0 means pure memorization. Computed in $O(N + M)$ via hash lookup.

**2. Mode coverage (weighted).** An "archetype" is the sorted multiset of room types in a graph — it captures the floorplan *composition* regardless of spatial connectivity. For example, `(LivingRoom, Kitchen, Bathroom, MasterRoom, Balcony)` is one archetype. Mode coverage measures what fraction of training archetypes appear at least once in the generated set. The **weighted** variant weights each archetype by its frequency in the training data, so covering the 10 most common archetypes (which may account for 70%+ of training mass) scores higher than covering 10 rare ones. A large gap between weighted and unweighted coverage means the model generates common configurations well but misses rare ones.

**3. Spatial transitivity.** Checks whether the spatial relationships in a generated graph are physically realizable in 2D. Each of the 10 spatial relationship types is decomposed into horizontal and vertical ordering constraints (e.g., "left-of" implies $A.x < B.x$). Two directed graphs are built — one per axis — and a directed cycle in either graph means no valid 2D placement exists. `transitivity_score` is the fraction of graphs with no contradictions on either axis. This catches the most dangerous class of generation failures: graphs that pass all validity checks but can never become a real floorplan.

**4. Conditional edge TV (weighted).** For each canonical room-type pair $(r_i, r_j)$, computes the Total Variation distance between the edge-type distribution in generated data vs. training data. Pairs are canonicalized by room type (if $\text{type}_i > \text{type}_j$, swap and apply the spatial inverse $\text{rel} = 9 - \text{rel}$). The **weighted** variant weights each pair by its training frequency, so common pairs (e.g., LivingRoom–Bathroom) contribute more than rare ones (e.g., Storage–Balcony). This detects models that produce correct marginal edge distributions while systematically misassigning spatial relationships for specific room combinations.

**5. Type-conditioned degree TV (weighted).** For each room type, computes the TV distance between the degree distribution (number of spatial connections) in generated vs. training data. The **weighted** variant weights by room-type frequency. This catches models that match the global degree distribution while giving, e.g., bathrooms too many connections (4–5 neighbors instead of the typical 1–2).

**6. Node TV.** TV distance between the overall room-type distribution in generated vs. training data. Measures whether the model produces the right mix of LivingRooms, Kitchens, Bathrooms, etc.

## 11.2 The Composite Metric: Novelty $\times$ Mode Coverage (Weighted)

We combine novelty and weighted mode coverage into a single composite: their product. This composite is our primary diversity metric because the two components are complementary and neither alone is sufficient.

**Why they must go together.** A model that memorizes the entire training set would achieve perfect weighted mode coverage (every archetype reproduced) but zero novelty — it is not generative. Conversely, a model that generates 1000 unique, novel graphs that all have the same basic room composition (e.g., always LivingRoom + Kitchen + 2 Bedrooms) would achieve perfect novelty but near-zero mode coverage — it has mode-collapsed. Only models that *both* produce new graphs *and* cover the breadth of training archetypes score well on the product.

**Why weighted mode coverage specifically.** The weighted variant matters because unweighted coverage treats all archetypes equally: covering a rare 3-room studio counts the same as covering the most common 5-room apartment. Weighted coverage reflects practical importance — users care most about the most common floorplan configurations being realistic and available. A model covering only the 10 most frequent archetypes may score 70%+ on weighted coverage (because those archetypes dominate training mass), which correctly reflects that it handles the bulk of realistic floorplans. The unweighted variant remains useful as a stricter diagnostic for mode collapse on the tail.

## 11.2 Why TV and JS Over KL Divergence

We prefer Total Variation (TV) distance and Jensen-Shannon (JS) divergence over the standard KL divergence for three reasons:

**Symmetry**: KL divergence is asymmetric — $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$. This means the measured "distance" depends on which distribution we call P and which we call Q. TV and JS are both symmetric, providing an unambiguous distance measure.

**Boundedness**: KL divergence is unbounded — it goes to $+\infty$ when $Q$ assigns zero probability to an event that $P$ considers possible. This makes it hypersensitive to rare events and zero-probability tokens. In our evaluation, generated distributions may assign zero probability to certain rare room-type pairs, causing KL to explode. TV is bounded in $[0, 1]$ and JS is bounded in $[0, \ln 2]$, providing stable, interpretable measurements.

**Interpretability**: TV has a direct operational interpretation — it equals the maximum difference in probability assigned to any event. A TV of 0.15 means the generated distribution differs from the real one by at most 15% on any event. KL values like 0.34 or 1.2 have no such intuitive interpretation.

We computed both TV and JS for all three metric families (conditional edge, type-conditioned degree, node) and found JS produces **identical rank orderings** to TV in all cases (Section 11.3). We therefore report only TV to avoid redundancy.

## 11.3 JS Redundancy

For each metric where both TV and JS were computed, we verified that JS does not change rankings or reveal information that TV misses:

- **Conditional edge**: TV and JS rankings are identical across all 8 LLaDA remasking configurations. Rank correlation = 1.0.
- **Type-conditioned degree**: Minor rank swaps in top 3, but absolute differences are tiny (entire spread is 0.009). No practical conclusion changes.
- **Node**: Rankings essentially identical.

**Conclusion**: We report TV only. JS carries the same information and can be dropped from the preferred metrics set without losing signal.

## 11.4 Why Weighted Conditioned Versions

The "weighted" modifier in our conditional metrics (cond. edge TV weighted, type-cond. degree TV weighted) means we weight each conditioning category by its frequency in the training data.

**Without weighting**: Each room-type pair or room type contributes equally. This gives disproportionate influence to rare pairs (e.g., Storage–Balcony appears in $<$1% of training samples). A model could produce perfect distributions for rare pairs but poor distributions for common ones, and the unweighted metric would not capture this.

**With weighting**: Common pairs (LivingRoom–Kitchen, Bedroom–Bathroom) dominate the metric. This reflects practical importance — users of generated floorplans care most about the most common room configurations being realistic. A model that gets LivingRoom–Kitchen spatial relationships right but occasionally misassigns Storage–Balcony is more useful than the reverse.

# 12. Remasking Strategy Comparison (LLaDA Only)

## 12.1 Cap vs. Confidence

| Criterion | Winner | Margin |
|:----------|:------:|:------:|
| Novelty $\times$ Mode cov (wt.) | **Confidence** | +0.02 (0.73 vs 0.71) |
| Spatial transitivity | Tie | within noise |
| Cond. edge TV (wt.) | **Conf tsw=0.3** | marginal |
| Type-cond. degree TV (wt.) | Tie | within noise |
| Node TV | Tie | within noise |

Confidence remasking is slightly but consistently better than cap. The advantage concentrates on mode coverage (+2–4 pp weighted), the metric that most directly measures diversity. On distributional and structural metrics, the two are equivalent.

## 12.2 Recommendation (v1): Confidence Remasking with $t_{\text{switch}} = 0.5$

Based on our analysis of the full sweep, we recommend **confidence remasking with $t_{\text{switch}} = 0.5$** as the default v1 configuration. The choice between $t_{\text{switch}} = 0.5$ and $t_{\text{switch}} = 1.0$ is a close call — all differences fall within noise — but $t_{\text{switch}} = 0.5$ offers a slight edge on inside validity, which is the hardest constraint to satisfy.

**Head-to-head comparison ($t_{\text{switch}} = 0.5$ vs $1.0$):**

| Metric | tsw=0.5 | tsw=1.0 | Winner |
|:-------|:---:|:---:|:------:|
| Novelty $\times$ Mode cov (wt) | 0.727 | 0.732 | tsw=1.0 (+0.005) |
| Spatial transitivity | 98.5% | 98.7% | tsw=1.0 (+0.2pp) |
| Cond. edge TV (wt) | 0.580 | 0.571 | tsw=1.0 (-0.009) |
| Type-cond degree TV (wt) | 0.166 | 0.169 | tsw=0.5 (-0.003) |
| Node TV | 0.203 | 0.199 | tsw=1.0 (-0.004) |
| Inside validity | 93.8% | 93.3% | tsw=0.5 (+0.5pp) |

All differences are within one standard deviation of the 5-seed evaluation. The rationale for preferring $t_{\text{switch}} = 0.5$:

- **Inside validity is the hardest constraint.** At 93.8% vs 93.3%, tsw=0.5 is marginally safer. Inside validity violations are binary (a single forbidden containment pair fails the sample) and architecturally unrecoverable, making every fraction of a percentage point valuable.
- **Conceptual advantage.** $t_{\text{switch}} = 0.5$ means remasking activates only in the second half of denoising, after the model has built a first draft of the graph from high-confidence positions. This lets the initial structure form without perturbation, then refines it. With $t_{\text{switch}} = 1.0$ (always-on), remasking starts from the first step when almost everything is masked, which is less meaningful — the confidence softmax naturally throttles remasking at high noise levels (few decoded positions to remask), so the practical difference is small, but the conceptual clarity of "build then refine" is an advantage.
- **Both are self-regulating.** Confidence remasking requires no $\eta$ tuning regardless of $t_{\text{switch}}$.

The main cost of any remasking on v1 (vs. no remasking) remains approximately 0.1 on cond. edge TV (weighted) — a 21% degradation in conditional edge fidelity — in exchange for +4pp mode coverage and a 4$\times$ archetype increase (29 $\to$ 121).

# 13. v2 + Remasking: Adapting Remasking to Per-Position Alpha

## 13.1 Technical Adaptation

The v1 remasking schedule computes a scalar $\sigma_{\max}$ from the global noise schedule. For v2, we replace this with a per-position $\sigma_{\max}^l$:

$$\sigma_{\max}^l = \text{clamp}\left(\frac{1 - \alpha_l(t_{\text{next}})}{\alpha_l(t_{\text{now}})}, 0, 1\right) \quad \text{shape: (B, SEQ\_LEN)}$$

where $\alpha_l(t)$ comes from the rate network. The formula is identical to v1 — only the shape changes from scalar to per-position.

**Physical interpretation**: Positions that the rate network keeps clean longer (higher $\alpha_l$ at $t_{\text{next}}$) get lower $\sigma_{\max}$ (less remasking budget). This is desirable — the rate network has learned which positions are "easy" and should be kept stable.

**PAD positions**: The rate network returns $\alpha = 1.0$ for PAD positions, so $\sigma_{\max}^l = 0$ for PAD. PAD positions can never be remasked — correct by construction.

## 13.2 Design Choices Inherited from v1

All 22 configurations tested in v1 settle the design choices for v2:

| Choice | Status | v1 Evidence |
|:-------|:------:|:------------|
| Random unmasking | ELIMINATED | inside validity 54–59% |
| Cap strategy | ELIMINATED | $\leq$ confidence, but needs $\eta$ tuning |
| Argmax sampling | ELIMINATED | 5 archetypes, mode collapse |
| $t_{\text{switch}}$ sweep | ELIMINATED | 0.3–1.0 all equivalent |
| **Confidence + LLaDA + top-$p$** | **SELECTED** | Best composite, self-regulating |

**One experiment, not a sweep**: v2 remasking runs a single configuration (`v2_llada_topp0.9_remdm_confidence_tsw1.0`). All design choices were settled by v1's 22-run experiment suite.

## 13.3 Results: v2 + Remasking

| Metric | v1 no remask | v1 + conf tsw=1.0 | v2 no remask | **v2 + conf tsw=1.0** |
|:-------|:---:|:---:|:---:|:---:|
| Inside validity | 99.4% | 93.3% | 100.0% | **99.5%** |
| Nov. $\times$ Mode cov (wt) | 0.678 | 0.732 | 0.677 | **0.707** |
| Spatial transitivity | 99.9% | 98.7% | 100.0% | **100.0%** |
| Cond. edge TV (wt) | 0.472 | 0.571 | 0.441 | **0.432** |
| Type-cond degree TV (wt) | 0.159 | 0.169 | 0.178 | 0.180 |
| Node TV | 0.119 | 0.199 | 0.054 | **0.059** |

## 13.4 Metric-by-Metric Assessment

1. **Novelty $\times$ Mode coverage (wt)**: v2+remasking scores 0.707 (novelty 0.861 $\times$ mode coverage 82.0%). This falls between v1 no-remask (0.678) and v1+remasking (0.732). The novelty penalty (0.861 vs 0.999 for v1 remasking) outweighs the mode coverage gain (82.0% vs 73.3%). **v1 + conf tsw=1.0 remains the winner** on this metric.

2. **Spatial transitivity**: v2+remasking achieves **100.0%** — perfect, matching v2 no-remask and beating all v1 remasking methods (98.2–98.7%). The learned forward process produces graphs with no spatial ordering cycles even after remasking perturbation. **v2+remasking wins.**

3. **Cond. edge TV (wt)**: v2+remasking achieves **0.432** — the best of any method, including the v1 no-remask baseline (0.472). Remarkably, remasking on v2 actually *improved* conditional edge fidelity (0.441 $\to$ 0.432), the opposite of v1 where remasking degraded it (0.472 $\to$ 0.571). This suggests the v2 denoiser's stronger per-position predictions are robust to remasking perturbation. **v2+remasking wins.**

4. **Type-cond degree TV (wt)**: v2+remasking scores 0.180, slightly worse than v1 no-remask (0.159) and v1+remasking (0.169). The spread is small (0.021 from best to worst) but consistent. **v1 conf tsw=1.0 wins.**

5. **Node TV**: v2+remasking achieves **0.059**, far better than any v1 method (best v1: 0.119 no-remask, 0.199 with remasking). The learned forward process produces near-perfect room type distributions even after remasking. **v2+remasking wins.**

6. **Inside validity (decisive)**: v2+remasking achieves **99.5%**, the highest of any remasked method by a wide margin. For context: v1+remasking achieves 93.3%. v2+remasking is only 0.5pp below the v2 no-remask baseline (100.0%). **v2+remasking wins decisively.**

## 13.5 Summary and Pareto Front

v2+remasking wins on **4 of 5 preferred metrics** (spatial transitivity, cond. edge TV, node TV, inside validity) and loses on 1 (novelty $\times$ mode coverage). The metric it loses on is the diversity-focused one — v2's learned forward process produces more deterministic trajectories that limit diversity (40 unique archetypes vs v1's 121).

**The choice depends on application priority:**

- **If inside validity is a hard constraint ($>$95%)**: Use **v2 + conf tsw=1.0**. It is the only remasked method that stays above 99%, with best-in-class distributional fidelity. The diversity tradeoff (40 vs 121 archetypes) is acceptable if sample quality matters more than variety.

- **If maximum diversity is the goal**: Use **v1 + conf tsw=1.0**. It produces 3$\times$ more unique archetypes (121 vs 40) with near-perfect novelty (0.999), at the cost of 93.3% inside validity and significantly worse distributional fidelity.

- **If balance is needed**: No single method dominates both. v2+remasking occupies a distinct Pareto point: best quality, moderate diversity. v1+remasking occupies the other: best diversity, moderate quality.

# 14. Derivative-Free Guidance with Soft Value-Based Decoding (SVDD)

Guidance is the most important feature of this model: even if a version from those above proved to be the best overall, if it is not guidance-adaptable we would have to disregard it. The purpose of guidance is to steer generated bubble diagrams toward satisfying user-specified architectural constraints — "exactly one kitchen", "kitchen must be adjacent to the living room", "between 2 and 3 bathrooms" — without retraining the model.

SVDD (Soft Value-Based Decoding, Li et al., arXiv:2408.08252) achieves this at inference time. The core idea is simple: instead of sampling one transition per denoising step, sample $K$ candidate transitions, score them against a reward function derived from the constraints, and select one via importance-weighted resampling. This is repeated at every denoising step, so the cumulative effect of consistently picking slightly-better candidates compounds over 100 steps into a large improvement in final constraint satisfaction.

The key property that makes SVDD attractive is that it is **derivative-free**: it requires no gradients through the model, no differentiable reward, and no retraining. It works as a black-box wrapper around any trained denoiser. This is critical for discrete diffusion where the token space is non-differentiable.

## 14.1 The SVDD Mechanism Step by Step

Consider a single denoising step going from time $t_{\text{now}}$ to $t_{\text{next}}$ (recall we denoise backward from $t = 1$ to $t = 0$). The standard (unguided) procedure is:

1. Run the model on $\mathbf{x}_t$ to get logits.
2. Unmask a subset of MASK positions (via LLaDA ordering + top-$p$ sampling).
3. Proceed to $t_{\text{next}}$.

SVDD wraps this in a **propose–score–select** loop:

**Step 1 — Model call (shared, 1$\times$ cost).** Run the denoiser on $\mathbf{x}_t$ to produce node logits and edge logits. This is the expensive step, and it is done **once** regardless of how many candidates $K$ we generate.

**Step 2 — Expand to $K \cdot B$ candidates.** Replicate the current state $\mathbf{x}_t$ and the logits $K$ times. We now have $K$ copies of each of the $B$ samples in the batch.

**Step 3 — Stochastic unmasking (propose).** Run the standard unmasking step (`_single_step_unmask`) on all $K \cdot B$ candidates. Because unmasking is stochastic — top-$p$ sampling selects different tokens for each candidate, and LLaDA ordering assigns different unmask priorities via random confidence-breaking — the $K$ candidates for a given sample will differ at the newly unmasked positions. Positions that were already committed (non-MASK) stay identical across candidates. This means:

- **Early in denoising** (most positions still MASK): candidates differ at many positions, providing rich variation for the reward function to differentiate.
- **Late in denoising** (most positions committed): candidates differ at very few positions, so rewards converge and guidance has less leverage — but by this point most constraints are already resolved.

**Step 4 — Score (reward).** Compute a scalar reward $r_k$ for each candidate $k \in \{1, \ldots, K\}$ using a `RewardComposer`. The reward is the negative energy: $r = -E(\mathbf{x})$ where $E$ aggregates weighted constraint violations (details in §14.5). Two scoring modes exist — soft and hard (§14.4) — which differ in how they estimate constraint violations at partially-masked states.

**Step 5 — Importance weights.** Convert rewards to normalized selection probabilities via a temperature-controlled softmax over the $K$ candidates for each sample:

$$w_k = \frac{\exp(r_k / \alpha)}{\sum_{k'=1}^{K} \exp(r_{k'} / \alpha)}$$

The parameter $\alpha > 0$ controls how aggressively the weights concentrate on high-reward candidates (§14.3).

**Step 6 — Resample (select).** For each sample in the batch, draw one winner from the categorical distribution defined by $\{w_1, \ldots, w_K\}$ via multinomial sampling. This is **stochastic**, not deterministic argmax — even a below-average candidate has some probability of being selected, which preserves diversity.

**Step 7 — Remask (if enabled).** Apply the remasking step (`_single_step_remask`) to the selected winner only. Remasking is orthogonal to guidance — it operates on the single selected output, not on all $K$ candidates.

**Step 8 — Record diagnostics.** Capture ESS, weight entropy, reward of the winner, mean reward across all $K$, per-constraint violations, and remasking statistics (§14.7).

The procedure then advances to the next denoising step with the selected winner as the new $\mathbf{x}_t$.

## 14.2 The Role of $K$ (Number of Candidates)

$K$ is the size of the candidate pool at each denoising step. It directly controls how many options the guidance has to choose from.

**At $K = 1$**, there is no guidance: a single candidate means uniform weight $w_1 = 1$, so the selected output is always the only candidate. The sampling loop reduces exactly to unguided sampling.

**At $K > 1$**, the $K$ candidates differ at the stochastically unmasked positions. A larger pool means:

- More likely to contain a candidate that happens to satisfy (or move toward satisfying) the constraints.
- The reward function can differentiate more finely, since the best-of-$K$ improves with $K$.
- The guidance has more leverage at each step, compounding over 100 steps.

However, scoring cost scales linearly with $K$ (we evaluate constraint violations on each of $K \cdot B$ candidates). The model call is shared, so the overhead is in the scoring loop, not in the neural network forward pass. Empirically:

| $K$ | Scoring cost per step | Constraint satisfaction (pilot) |
|-----|----------------------|-------------------------------|
| 1 | $B$ scores | 43% (baseline, no guidance) |
| 4 | $4B$ scores | 68.5% |
| 16 | $16B$ scores | 77% |

The improvement from $K = 4$ to $K = 16$ is substantial (+8.5 pp), reflecting the fact that with 16 candidates the guidance almost always finds at least one that improves on the average. Diminishing returns set in for larger $K$: going from 16 to 24 candidates adds compute but contributes fewer novel "good" candidates.

Importantly, all $K$ candidates share the same logits (from the single model call) and differ only in the stochastic token selection. The candidates are not independent samples — they are different realizations of the same proposal distribution. This is why $K$ need not be very large: the candidates explore the local neighborhood of the posterior, not the full generative space.

## 14.3 The Role of $\alpha$ (Guidance Temperature)

$\alpha$ controls the sharpness of the softmax that converts rewards into selection probabilities:

$$w_k = \text{softmax}(r_k / \alpha)$$

**Small $\alpha$** (e.g., 0.01–0.1) makes the softmax sharp. Even moderate reward differences among candidates cause one or two to absorb most of the weight. This means guidance is aggressive: the best candidate is selected with high probability at every step.

**Large $\alpha$** (e.g., 1.0–5.0) makes the softmax flat. All candidates receive nearly equal weight regardless of their rewards. Selection becomes near-uniform — effectively unguided.

The following table summarizes the behavior:

| $\alpha$ | Softmax shape | ESS range | Effect |
|----------|--------------|-----------|--------|
| $\to 0$ | Hard argmax | $\approx 1$ | Deterministic selection — diversity collapse |
| 0.01–0.05 | Very sharp | 2–4 | Strong steering, stochastic diversity preserved |
| 0.1 | Sharp | 3–6 | Active steering (pilot sweet spot) |
| 1.0 | Flat | $\approx K$ | Near-random selection — negligible guidance |
| $\geq 5.0$ | Very flat | $\approx K$ | No guidance |

**Why not $\alpha \to 0$?** With $\alpha \to 0$, the softmax becomes a hard argmax: the highest-reward candidate is deterministically selected at every step. Over 100 denoising steps, this greedy strategy funnels all samples down the same narrow path, producing near-identical outputs regardless of the stochastic variation in the candidates. This is **diversity collapse**. The stochasticity at $\alpha = 0.05$–$0.1$ strikes a balance: strong enough to meaningfully steer (ESS drops to 2–4 when candidates differ), soft enough that below-optimal candidates occasionally win, preserving diverse generation trajectories.

**Interaction with $K$.** $\alpha$ and $K$ interact: larger $K$ provides more candidates, which makes the reward differences among them more extreme, which makes a given $\alpha$ feel "sharper." In practice, the optimal $\alpha$ should be tuned for a given $K$.

## 14.4 Soft Mode vs. Hard Mode

At each denoising step, the candidate state $\mathbf{x}_t$ is partially masked: some positions hold committed tokens, others are still MASK. To compute constraint violations on such a partially-resolved state, we need a way to estimate "how close is this candidate to satisfying the constraint?" Two approaches exist.

### 14.4.1 Hard Mode: Argmax Decode Then Count

Hard mode treats the partially-masked candidate as if all MASK positions were resolved by argmax:

1. For every MASK position, take $\arg\max$ of its logits to get a predicted token.
2. Committed (non-MASK) positions keep their current token.
3. Detokenize the resulting fully-resolved sequence into a `graph_dict`.
4. Evaluate each constraint's `hard_violation()` on this decoded graph.

The violation is a discrete count. For example, `ExactCount(Kitchen, target=1)` would count the actual kitchens in the decoded graph and return $|count - 1|$. The value is an integer: either the constraint is satisfied (violation $= 0$) or it's violated by some discrete amount.

**Advantage**: Accurately reflects what the final decoded output would look like if denoising stopped now.

**Disadvantage**: Discontinuous. A small change in logits can flip the argmax at a position (e.g., a position goes from 49% Kitchen / 51% Bedroom to 51% Kitchen / 49% Bedroom), causing the decoded graph to change discretely and the violation to jump. This makes the reward signal noisy across steps and across candidates.

### 14.4.2 Soft Mode: Probability-Weighted Expectations

Soft mode avoids the argmax entirely and works with the full posterior distributions:

1. For each position, construct an **effective probability vector**:
   - **PAD positions**: all-zeros vector (excluded from scoring).
   - **Committed positions** (already unmasked to a specific token): one-hot vector on that token.
   - **MASK positions**: $\text{softmax}(\text{logits})$ — the model's full belief distribution.

2. Evaluate each constraint's `soft_violation()` on these probability vectors.

The key insight is that soft violations replace discrete counts with **expected counts**. For example, if position 3 has $P(\text{Kitchen}) = 0.7$ and position 5 has $P(\text{Kitchen}) = 0.2$, the expected kitchen count contributed by these two positions is $0.7 + 0.2 = 0.9$. The `ExactCount(Kitchen, target=1)` soft violation is $|0.9 - 1| = 0.1$ — a smooth, continuous value that changes gradually as the model's beliefs evolve.

**Where logits come from.** The logits used to build these probability distributions come from the single shared model call at each denoising step (Step 1 in §14.1). The model takes the current partially-masked state $\mathbf{x}_t$ and the timestep $t$ as input and produces logits for every position. At MASK positions, these logits represent the model's posterior belief about what token should be there. At committed positions, the logits are irrelevant (overridden by the one-hot). At PAD positions, the logits are ignored (zeroed out).

**Where soft violations are computed.** After `build_effective_probs()` constructs the per-position probability vectors, each constraint's `soft_violation()` method operates on these vectors to compute a scalar violation score. The computation is entirely in NumPy/PyTorch — no detokenization or graph construction needed. This makes soft scoring significantly faster than hard scoring, which requires argmax decoding + detokenization for each candidate.

**Convergence guarantee.** As denoising progresses and more positions become committed (one-hot distributions), the soft violation converges to the hard violation. At the final step when all positions are committed, soft violation $=$ hard violation exactly. This means soft mode starts as a smooth approximation and ends as the true discrete answer.

**Advantage**: Smooth and continuous — small logit perturbations produce small violation changes, giving a clean gradient of quality across candidates.

**Disadvantage**: Can be optimistic. A position that is 60% Kitchen and 40% Bedroom contributes 0.6 to the expected kitchen count, but in reality the position will become either Kitchen or Bedroom, not 0.6 of a Kitchen. This means soft violations can underestimate the true violation of the final decoded output, especially when many positions are uncertain.

### 14.4.3 Summary Comparison

| Aspect | Soft mode | Hard mode |
|--------|-----------|-----------|
| **Input** | softmax(logits) at MASK positions | argmax(logits) at MASK positions |
| **Output** | Continuous violation $\geq 0$ | Discrete violation $\in \{0, 1, 2, \ldots\}$ |
| **Smoothness** | Yes — varies continuously with logits | No — argmax flips cause jumps |
| **Accuracy** | Approximate (expected values) | Exact (for the argmax-decoded state) |
| **Speed** | Fast (tensor ops on probability vectors) | Slower (requires detokenization per candidate) |
| **Convergence** | $=$ hard violation when all positions committed | — |

All pilot experiments use soft mode.

## 14.5 The Reward Function

The reward function translates constraint violations into a single scalar score for each candidate. It has three components: violations, energy, and reward.

### 14.5.1 Constraint Violations

Each constraint computes a **graded** violation magnitude $v \geq 0$ where $v = 0$ means satisfied. The four constraint primitives are:

**ExactCount** — "exactly $n$ rooms of a given type."

- *Hard*: $v = |\text{count}(\text{type}) - n|$. Count the actual rooms of that type in the decoded graph, take the absolute difference from the target.
- *Soft*: $v = |\hat{n} - n|$ where $\hat{n} = \sum_i q_i(\text{type})$ sums the probability of each active node position being that type.
- *Example*: `ExactCount(Kitchen, target=1)`. If the graph has 2 kitchens, $v_{\text{hard}} = |2 - 1| = 1$. If three node positions have $P(\text{Kitchen}) = 0.8, 0.3, 0.1$, then $\hat{n} = 1.2$ and $v_{\text{soft}} = |1.2 - 1| = 0.2$. The graded violation provides signal even when the model is "close" — it says "you have 0.2 too many expected kitchens", not just "violated."

**CountRange** — "between $lo$ and $hi$ rooms of a given type."

- *Hard*: $v = \max(0, lo - \text{count}) + \max(0, \text{count} - hi)$. Zero inside the range, grows linearly outside.
- *Soft*: same formula with expected count $\hat{n}$ replacing the discrete count.
- *Example*: `CountRange(Bathroom, lo=2, hi=3)`. If the graph has 1 bathroom, $v = \max(0, 2 - 1) + \max(0, 1 - 3) = 1 + 0 = 1$. If it has 2, $v = 0$. If three node positions have $P(\text{Bathroom}) = 0.9, 0.4, 0.2$, then $\hat{n} = 1.5$ and $v_{\text{soft}} = \max(0, 2 - 1.5) + \max(0, 1.5 - 3) = 0.5$. The model is "half a bathroom short" of the minimum — a useful gradient signal.

**RequireAdj** — "at least one adjacency between type A and type B."

- *Hard*: $v = 0$ if any edge exists between the required types in the decoded graph, $v = 1$ if no such edge exists. This is binary.
- *Soft*: $v = 1 - P(\text{exists})$ where $P(\text{exists})$ is computed in log-space for numerical stability:

  $$P(\text{exists}) = 1 - \prod_{(i,j)} (1 - p_{ij})$$
  $$= 1 - \exp\!\left(\sum_{(i,j)} \log(1 - p_{ij})\right)$$

  Here $p_{ij} = p_{\text{types},ij} \cdot P_{\text{adj},ij}$ is the joint probability that edge position $(i,j)$ connects the right room types *and* carries a spatial relationship (not NO_EDGE). The type probability is:

  $$p_{\text{types},ij} = q_i(A) \cdot q_j(B) + q_i(B) \cdot q_j(A)$$

  accounting for both orderings (or just $q_i(A) \cdot q_j(A)$ if $A = B$). The adjacency probability is $P_{\text{adj},ij} = \sum_{s=1}^{10} q_{ij}(s)$, the probability of any of the 10 spatial relation types (excluding NO_EDGE).

- *Example*: `RequireAdj(Kitchen, LivingRoom)`. If one edge position has nodes that are 80% Kitchen and 70% LivingRoom with 90% adjacency probability, that single position contributes $p_{ij} = 0.8 \times 0.7 \times 0.9 = 0.504$. With only this candidate edge pair, $P(\text{exists}) = 1 - (1 - 0.504) = 0.504$ and $v_{\text{soft}} = 0.496$. With multiple candidate edge pairs, the product of complements drives $P(\text{exists})$ higher (closer to 1) and $v$ lower (closer to 0).

**ForbidAdj** — "no adjacency between type A and type B."

- *Hard*: $v = \text{count of forbidden adjacency pairs}$. Count how many edges in the decoded graph connect the forbidden types.
- *Soft*: $v = \sum_{(i,j)} p_{ij}$ — the expected count of forbidden adjacencies, summing the same $p_{ij}$ terms as RequireAdj but without the complement-product construction.
- *Example*: `ForbidAdj(Bathroom, Kitchen)`. If two edge positions have $p_{ij}$ values of 0.3 and 0.1, then $v_{\text{soft}} = 0.4$ — the model expects 0.4 forbidden adjacencies. If the constraint is satisfied in the final graph, $v_{\text{hard}} = 0$.

### 14.5.2 Energy and Reward

The **energy** aggregates all constraint violations into a single non-negative score:

$$E(\mathbf{x}) = \sum_{i} \frac{\lambda_i}{p_{90,i}} \cdot \varphi(v_i(\mathbf{x}))$$

Where:

- $\lambda_i$ is the weight assigned to constraint $i$ (default 1.0). Allows prioritizing some constraints over others.
- $p_{90,i}$ is a **P90 normalizer** — the 90th percentile of non-zero violations for constraint $i$, computed on unguided baseline samples. This calibration step ensures that constraints with different violation scales (e.g., ExactCount ranges 0–7, RequireAdj ranges 0–1) contribute equally to the energy. Without it, high-scale constraints would dominate.
- $\varphi$ is a **shaping function** applied to the raw violation. Options are:
  - *Linear* ($\varphi(v) = v$, default): proportional penalty.
  - *Quadratic* ($\varphi(v) = v^2$): penalizes large violations disproportionately.
  - *Log1p* ($\varphi(v) = \log(1 + v)$): dampens the effect of very large violations.

The **reward** is simply the negated energy:

$$r(\mathbf{x}) = -E(\mathbf{x})$$

The reward is always $\leq 0$. A fully satisfied candidate has $r = 0$ (all violations zero). More violated candidates have more negative rewards. This reward feeds into the softmax importance weighting:

$$w_k = \text{softmax}(r_k / \alpha) = \frac{\exp(-E_k / \alpha)}{\sum_{k'} \exp(-E_{k'} / \alpha)}$$

**Worked example.** Suppose we have two constraints:

- `one_kitchen`: ExactCount(Kitchen, 1), $\lambda = 1.0$, $p_{90} = 2.3$
- `kitchen_near_living`: RequireAdj(Kitchen, LivingRoom), $\lambda = 1.5$, $p_{90} = 0.8$

A candidate has soft violations $v_1 = 0.5$ (expected 1.5 kitchens) and $v_2 = 0.2$ (80% chance the adjacency exists). With linear shaping:

$$E = \frac{1.0}{2.3} \times 0.5 + \frac{1.5}{0.8} \times 0.2 = 0.217 + 0.375 = 0.592$$

$$r = -0.592$$

If $\alpha = 0.1$ and another candidate has $r = -0.4$, the softmax weights are:

$$w_1 = \frac{e^{-0.592/0.1}}{e^{-0.592/0.1} + e^{-0.4/0.1}} = \frac{e^{-5.92}}{e^{-5.92} + e^{-4.0}} \approx \frac{0.00266}{0.00266 + 0.01832} \approx 0.127$$

The second candidate (reward $-0.4$, less violated) gets weight $\approx 0.873$, so it is selected with 87% probability. With $\alpha = 1.0$ instead, the weights would be $\approx 0.45$ vs $\approx 0.55$ — barely differentiated.

## 14.6 SVDD and Our Masked Diffusion

Our masked diffusion follows the MDLM paradigm: a fixed forward process masks tokens independently with probability $1 - \alpha(t)$, and the backward process progressively unmasks them over $N$ discrete steps. There is no remasking of previously committed tokens during standard sampling (no ReMDM), and no learning of the forward process (v1). This simplicity has specific implications for how SVDD operates.

### 14.6.1 No Remasking Means Monotonic Commitment

In our v1 no-remasking setup, once a token is unmasked at step $t$, it stays committed for all subsequent steps. The set of committed positions grows monotonically from empty (fully masked at $t = 1$) to full (fully unmasked at $t = 0$). This has two consequences for SVDD:

**Candidates only differ at newly unmasked positions.** At each step, the $K$ candidates share the same committed tokens from previous steps — they only diverge at the positions being unmasked in this step. Early in denoising (many MASKs), each step unmasks many positions, so candidates differ significantly. Late in denoising (few MASKs), each step unmasks few positions, so candidates are nearly identical. This means guidance has **most leverage in the middle regime** of denoising, where enough structure exists to differentiate candidates but enough positions remain masked to create meaningful variation.

**Guidance effects accumulate irreversibly.** When guidance selects a candidate at step $t$, that candidate's committed tokens are frozen into all future steps. There is no mechanism to "undo" a committed token. This makes the cumulative effect of guidance particularly strong: over 100 steps, consistently selecting the slightly-better candidate locks in constraint-satisfying structure progressively. However, it also means an unlucky early commitment (e.g., placing a second Kitchen before guidance has enough signal to penalize it) cannot be corrected later.

### 14.6.2 Fixed Forward Process Means Global $p_{\text{unmask}}$

Because we do not learn the forward process (v1), the unmasking probability at each step is a scalar derived from the global noise schedule:

$$p_{\text{unmask}}(t) = \frac{\alpha(t_{\text{next}}) - \alpha(t_{\text{now}})}{1 - \alpha(t_{\text{now}})}$$

This scalar applies uniformly to all MASK positions. Every MASK position has the same probability of being unmasked at each step. This means:

- The number of positions unmasked per step is approximately deterministic (controlled by the noise schedule).
- The *which* positions get unmasked is stochastic (LLaDA confidence ordering with random tie-breaking, or purely random ordering).
- Candidates diverge because the stochastic ordering and top-$p$ token sampling produce different tokens at different positions.

With a learned forward process (v2), each position would have its own $\alpha_l(t)$, meaning positions unmask at different rates. This would change the SVDD dynamics: some positions would be "easy" (high $\alpha$, unmasked early) and their tokens would be shared across candidates early, while "hard" positions (low $\alpha$) would remain masked longer and provide more guidance signal.

### 14.6.3 Shared Model Call — Why SVDD Is Cheap

The central efficiency insight of SVDD is that the model call is shared. In our setup:

1. Run the denoiser **once** on $(\mathbf{x}_t, t)$ to get logits for all $B$ samples.
2. Replicate the logits $K$ times.
3. Sample $K$ different transitions from the same logits.

The denoiser forward pass is the bottleneck (~95% of compute per step). The scoring loop over $K \times B$ candidates involves only constraint evaluation on probability vectors (soft) or detokenized graphs (hard), which are lightweight CPU operations. Therefore:

- $K = 1$: 1 model call + $B$ scores $\approx$ unguided cost.
- $K = 16$: 1 model call + $16B$ scores $\approx$ 1.05$\times$ unguided cost (model-dominated).

The actual wall-clock overhead is higher than 5% because the scoring loop is in Python (not fully vectorized), but the model call remains the dominant term.

### 14.6.4 Compatibility with Remasking

Although our primary configuration uses no remasking, SVDD is designed to be compatible with ReMDM remasking. When remasking is enabled:

1. The $K$ candidates are generated **without** remasking (unmasking only).
2. They are scored on their post-unmask state.
3. The winner is selected via importance-weighted resampling.
4. Remasking is applied **only to the winner** — not to all $K$ candidates.

This design choice is deliberate. Applying remasking before scoring would negate some of the guidance signal (committed tokens would be re-masked, making candidates more similar). Applying it only to the winner means guidance operates on the most differentiated state, and remasking then refines the selected output. The `reward_remasking_delta` diagnostic tracks whether remasking cooperates with guidance (positive delta) or fights it (negative delta).

### 14.6.5 Compatibility with Different Unmasking Strategies

SVDD works with any token selection strategy (argmax, top-$p$, temperature sampling) and any unmasking order (random, LLaDA). The choice affects how much the $K$ candidates differ:

- **Argmax sampling** produces identical candidates (no stochasticity) — SVDD has zero leverage. This is why argmax was already eliminated for other reasons (mode collapse).
- **Top-$p$ sampling** produces diverse candidates by sampling from the truncated distribution — ideal for SVDD.
- **LLaDA ordering** (confidence-based) unmasks the most confident positions first, but adds random noise to break ties. This tie-breaking is the source of candidate diversity in the unmasking order.

## 14.7 Why We Track Guidance Dynamics

SVDD guidance is not a binary "works or doesn't" — its effectiveness depends on the interplay of $\alpha$, $K$, the constraint set, the noise schedule, and the base model quality. To understand **how** guidance steers (not just the final satisfaction rate), we record per-step diagnostics throughout the entire denoising trajectory.

### 14.7.1 Effective Sample Size (ESS)

$$\text{ESS} = \frac{1}{\sum_{k=1}^{K} w_k^2}$$

ESS measures how concentrated the importance weights are. It ranges from 1 (all weight on a single candidate) to $K$ (uniform weights).

**Why track it?** ESS is the primary diagnostic for whether guidance is actually steering. If ESS $\approx K$ at every step, the weights are uniform, meaning the reward function cannot differentiate candidates — guidance is inactive regardless of the $\alpha$ setting. If ESS crashes to 1, guidance is maximally aggressive (one candidate dominates) — this risks diversity collapse. Healthy guidance shows ESS in the range 2–$K/2$, with variation across steps:

- **Early steps** ($t \approx 1$): Most positions are MASK, so candidates are similar and ESS $\approx K$. Guidance has little to differentiate.
- **Middle steps** ($0.3 < t < 0.7$): Structure emerges, candidates meaningfully differ, ESS drops to 2–6. This is where guidance exerts its leverage.
- **Late steps** ($t \approx 0$): Most positions are committed, candidates reconverge, ESS rises back toward $K$.

This U-shaped ESS trajectory is a signature of healthy, active guidance. A flat ESS $\approx K$ means $\alpha$ is too high. A flat ESS $\approx 1$ means $\alpha$ is too low.

### 14.7.2 Reward Trajectories

We track both the **reward of the selected candidate** and the **mean reward across all $K$ candidates** at each step. The difference (reward gap) measures the per-step advantage of guidance.

**Why track it?** The reward of the selected candidate increases over denoising steps, but most of this increase comes from the base model's denoising (tokens getting unmasked, structure emerging), not from guidance selection per se. What guidance contributes is the **gap** between the selected candidate and the average: at each step, the winner is slightly better than the mean, and this slight advantage compounds over 100 steps.

The reward gap can be **negative** at individual steps because selection is stochastic multinomial sampling, not argmax. At $\alpha = 0.1$, negative gaps are rare ($\sim$5% of steps); at $\alpha = 1.0$, they occur $\sim$50% of the time (consistent with near-random selection). A persistently positive reward gap confirms that guidance is systematically selecting better candidates.

### 14.7.3 Per-Constraint Violation Trajectories

For each constraint, we record the violation of the selected candidate at every denoising step, producing a trajectory $[v_0, v_1, \ldots, v_{99}]$.

**Why track them?** These trajectories reveal:

1. **When each constraint is resolved.** Some constraints (e.g., ExactCount on a common room type) may be resolved early because the base model naturally gets them right. Others (e.g., ForbidAdj on rare pairs) may only converge in the last 20 steps. Knowing this helps understand which constraints actually benefit from guidance.

2. **Whether constraints compete.** If constraint A's violation decreases while constraint B's increases, they may be in tension — satisfying one forces violations of the other. The energy function aggregates them, so the winner minimizes total energy, but per-constraint trajectories reveal the trade-offs.

3. **Trajectory jumpiness under low $\alpha$.** Under strong guidance ($\alpha = 0.1$), per-constraint trajectories appear jumpier (non-monotone) than under weak guidance. This is because a *different* candidate often wins at adjacent steps. Candidate $k$ that minimizes total energy at step $t$ may have higher violation on constraint A than candidate $k'$ that won at step $t - 1$. With high $\alpha$ (near-random selection), the trajectory follows the model's natural denoising curve — smooth but unsteered. **Trajectory smoothness is not a proxy for guidance quality.**

### 14.7.4 Putting It Together: Diagnosing Guidance Health

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| ESS $\approx K$ at all steps | $\alpha$ too high, guidance inactive | Decrease $\alpha$ |
| ESS $\approx 1$ at most steps | $\alpha$ too low, diversity collapse | Increase $\alpha$ |
| ESS dips in middle, recovers at ends | Healthy guidance | No change needed |
| Reward gap $\approx 0$ | Guidance not differentiating | Decrease $\alpha$ or increase $K$ |
| Reward gap consistently positive | Guidance active and effective | Confirm with satisfaction rate |
| Constraint $i$ violation stays high | This constraint is hard for the base model | Consider increasing $\lambda_i$ or $K$ |
| Two constraints oscillate | Competing constraints | Examine constraint compatibility |
| Remasking delta negative | Remasking undoes guidance progress | Consider lower $t_{\text{switch}}$ or no remasking |

# 15. Tension Between Importance-Sampling Guidance and Confidence Remasking

## 15.1 The Problem

SVDD guidance and confidence remasking operate on the same denoising step but with **misaligned information**. The root cause is architectural: the model is called **once** at the top of each step (on the pre-unmask $\mathbf{x}_t$), and the resulting logits are reused for both unmasking (step 4d in the SVDD loop) and remasking (step 4i). Concretely:

- **Line 341** (`guided_sampler.py`): `node_logits, edge_logits = model(x_t, pad_mask, t)` — computed on pre-unmask $\mathbf{x}_t$.
- **Lines 380–416**: $K$ candidates are generated, scored, and a winner is selected. $\mathbf{x}_t$ is now the post-unmask, post-resample winner.
- **Lines 422–425**: `_single_step_remask(x_t, ..., node_logits, edge_logits)` — remasking uses the **same pre-unmask logits** to compute confidence.

The model has never seen the post-unmask state. The confidence values used for remasking reflect the model's beliefs **before** the tokens were revealed, not after. This creates a systematic tension: SVDD selects tokens based on reward (constraint satisfaction), while confidence remasking removes tokens based on model confidence (how likely the model thought they were *a priori*). These two signals are **anti-correlated precisely when guidance is doing useful work** — guidance is most valuable when it overrides the model's prior to satisfy a constraint, but those overridden tokens are exactly the low-confidence ones that remasking targets.

When SVDD is not doing useful work (the model's preferred tokens already satisfy constraints), there is no tension. The problem only manifests when guidance matters.

## 15.2 Worked Example

### Setup

3 rooms. Positions 0–2 are nodes, positions 3–5 are edges for pairs $(0,1)$, $(0,2)$, $(1,2)$.

**State at the start of step $i$ (step 60 of 100, $t_{\text{now}} = 0.61$):**

| Position | Type | Current value | Status |
|----------|------|---------------|--------|
| 0 | node | Kitchen | committed (step 85) |
| 1 | node | Bedroom | committed (step 80) |
| 2 | node | Bathroom | committed (step 75) |
| 3 | edge(0,1) | adjacent | committed (step 70) |
| 4 | edge(0,2) | **MASK** | — |
| 5 | edge(1,2) | **MASK** | — |

**Active constraint**: "Bathroom must be **inside** Kitchen" (a containment rule on edge position 4).

### Step 1: Model call on $\mathbf{x}_t$ (pre-unmask)

The model sees positions 4 and 5 as MASK. Its softmax predictions for the masked positions:

| Position | $P(\text{adjacent})$ | $P(\text{inside})$ | $P(\text{no\_edge})$ | Model preference |
|----------|---------------------|--------------------|--------------------|-----------------|
| 4 (Kit–Bath) | **0.45** | 0.35 | 0.20 | prefers adjacent |
| 5 (Bed–Bath) | **0.50** | 0.25 | 0.25 | prefers adjacent |

The model has not seen the "bathroom inside kitchen" pattern often enough to be confident about `inside` — $P(\text{inside}) = 0.35$ at position 4.

### Step 2: Generate $K = 3$ candidates via top-$p$ unmasking

Each candidate samples independently from these logits:

| Candidate | pos4 | pos5 | Satisfies constraint? |
|-----------|------|------|-----------------------|
| 1 | adjacent | adjacent | **No** (bathroom not inside kitchen) |
| **2** | **inside** | adjacent | **Yes** |
| 3 | adjacent | inside | **No** (wrong pair has inside) |

### Step 3: Score and compute importance weights ($\alpha = 0.1$)

| Candidate | Reward | $w_k$ |
|-----------|--------|-------|
| 1 | $-1.0$ | 0.004 |
| **2** | **0.0** | **0.98** |
| 3 | $-0.5$ | 0.016 |

### Step 4: Resample — candidate 2 wins

$\mathbf{x}_t$ becomes: [Kitchen, Bedroom, Bathroom, adjacent, **inside**, adjacent].

SVDD has done its job — it found the one candidate where the bathroom–kitchen edge is `inside`.

### Step 5: Confidence remasking on the winner

The remasking function uses the **same logits from Step 1** to compute confidence for each decoded position in the winner:

| Position | Token in winner | $P(\text{token})$ from pre-unmask logits | $-\text{confidence}$ |
|----------|----------------|----------------------------------------|---------------------|
| 0 | Kitchen | 0.88 | $-0.88$ |
| 1 | Bedroom | 0.82 | $-0.82$ |
| 2 | Bathroom | 0.75 | $-0.75$ |
| 3 | adjacent | 0.70 | $-0.70$ |
| **4** | **inside** | **0.35** | **$-0.35$** |
| 5 | adjacent | 0.50 | $-0.50$ |

After $\text{softmax}(-\text{confidence})$ over decoded positions:

| Position | Remasking weight |
|----------|-----------------|
| 0 | 0.08 |
| 1 | 0.09 |
| 2 | 0.10 |
| 3 | 0.11 |
| **4** | **0.35** |
| 5 | 0.27 |

**Position 4 — the `inside` token that SVDD selected specifically because it satisfies the containment constraint — receives the highest remasking probability.** The model's prior confidence in `inside` was only 0.35 (it preferred `adjacent`), so confidence remasking sees it as the least reliable token and targets it.

### Step 6: Consequence

If position 4 gets remasked, the next model call sees position 4 as MASK again. The model, still preferring `adjacent` at that position, will likely predict `adjacent` with top probability. The constraint-satisfying `inside` token is lost, and SVDD must get lucky again at the next step to re-select it — against the same adverse confidence gradient.

## 15.3 When the Tension Is Strongest

The tension is worst when all three conditions hold simultaneously:

1. **Guidance is active and effective** — SVDD is selecting candidates that differ from the model's mode (low $\alpha$, moderate $K$, reward gap consistently positive).
2. **Confidence remasking is active** — $t_{\text{now}} < t_{\text{switch}}$, so remasking is applied after selection.
3. **The constraint-satisfying tokens are low-confidence** — the model's prior (trained distribution) assigns low probability to the tokens that the reward function prefers.

Condition 3 is precisely the case where guidance adds the most value. If the model already assigns high probability to constraint-satisfying tokens, SVDD is not needed — the base model would satisfy constraints on its own.

The `reward_remasking_delta` diagnostic (§14.7) captures this tension: a persistently negative delta means remasking is systematically undoing guidance progress. The current mitigation is to lower $t_{\text{switch}}$ or disable remasking entirely, but this sacrifices the diversity benefits of remasking (§8.5).

## 15.4 Mitigation Options

Three approaches can reduce or eliminate the guidance–remasking tension, with different cost and complexity tradeoffs.

### Option A: Fresh Logits for Remasking (Extra Model Call)

**Idea.** After selecting the SVDD winner, run the model a second time on the post-unmask state. Use the fresh logits — which now reflect the revealed tokens — for the confidence remasking decision.

**Mechanism.** Currently the loop is:

1. `logits = model(x_t_pre_unmask, t)` — shared across $K$ candidates.
2. Unmask, score, resample → winner.
3. `remask(winner, logits)` — stale logits.

With Option A, step 3 becomes:

1. `logits_fresh = model(winner, t)` — second model call.
2. `remask(winner, logits_fresh)` — fresh logits.

The model now sees the full post-unmask context. In the worked example (§15.2), the model would see `[Kitchen, Bedroom, Bathroom, adjacent, inside, adjacent]` as input. In this context, `inside` at position 4 is consistent with the surrounding tokens — the model may assign it confidence 0.65 instead of 0.35, naturally protecting it from remasking.

**Cost.** Doubles the model calls per step (2 instead of 1). For our small transformer (36 tokens, ~2M parameters), this is modest in absolute terms but still a 2$\times$ overhead on the dominant cost term. For larger models, this would be more significant.

**Advantage.** Most principled: the confidence values accurately reflect the model's beliefs given the actual state being remasked. No new hyperparameters.

**Disadvantage.** Pure compute cost. The fresh logits are used only for remasking — the SVDD scoring still uses the original logits (which is correct, since scoring happens before selection).

### Option B: Protect Just-Unmasked Positions for One Step (Zero Cost)

**Idea.** Exclude positions that were MASK before unmasking from the remasking candidate set for this step only. SVDD just evaluated these positions and selected the winner because of what's at them — let the model see them in context for at least one step before considering whether to remask them.

**Mechanism.** Before unmasking, record which positions are currently MASK:

```
was_mask_before = is_mask.clone()
```

After SVDD selection, modify the remasking candidate set:

```
remask_candidates = (~is_mask) & pad_mask & (~was_mask_before)
```

This means only previously-committed positions (those that were already decoded before this step) are eligible for remasking. The just-unmasked positions are protected for exactly one step. At the next denoising step, they become ordinary committed positions and are eligible for remasking based on their confidence under the new model call.

**Applied to the worked example (§15.2):**

- Positions 4 and 5 were MASK before this step → excluded from remasking.
- Only positions 0–3 (the previously committed nodes and edge) are remask candidates.
- The `inside` token at position 4 survives this step. At the next step, the model sees it in context and can form an informed confidence estimate.

**Cost.** Zero — one extra boolean tensor clone and a mask intersection.

**Advantage.** Directly addresses the root cause: SVDD selected these tokens for reward reasons, so don't undo that decision before the model has a chance to see them. Simple to implement, no hyperparameters.

**Disadvantage.** Blunt. Protects all just-unmasked positions equally, regardless of whether guidance actually cared about them. A position that was unmasked with a random token (not reward-relevant) is also protected. In practice this is mild — the protection is only for one step, and at the next step the model evaluates them with full context.

### Option C: Reward-Attributed Confidence Boosting (Zero Cost, Surgical)

**Idea.** Instead of protecting all just-unmasked positions (Option B) or re-running the model (Option A), use the reward signal from the $K$ candidates to identify which positions actually drove the SVDD selection, and selectively boost their confidence to shield them from remasking.

**Why not compare against the mode.** A naive approach would compare the winner's tokens against the mode (most frequent token) across $K$ candidates and call any differing position "guided." This is unreliable because the mode is a poor statistic:

- **At small $K$ (3–4):** The mode of 3 samples from a distribution where $P(\text{adjacent}) = 0.45$, $P(\text{inside}) = 0.35$ is nearly random. Any token can be the mode. Comparing against it measures sampling noise, not guidance effect.
- **At large $K$ (16–64):** The mode converges to the model's most probable token under top-$p$ — essentially the truncated argmax. So "differs from mode" reduces to "low confidence," which is what standard remasking already measures. The mode comparison adds no new information.
- **$K$-sensitivity:** The set of "guided positions" changes qualitatively with $K$, making any downstream boosting unstable across configurations.

**Per-position reward attribution.** Instead, directly measure how much each position's token correlates with the reward across $K$ candidates. For each position $l$ and each sample $b$:

1. Partition the $K$ candidates into two groups: those that **match** the winner's token at position $l$, and those that **don't**.
2. Compute the average reward of the matching group: $\bar{r}_{\text{match}}^l$.
3. Compute the average reward across all $K$ candidates: $\bar{r}_{\text{all}}$.
4. The **attribution** is: $a_l = \bar{r}_{\text{match}}^l - \bar{r}_{\text{all}}$.

If $a_l$ is large and positive, candidates sharing the winner's token at position $l$ tend to have higher rewards — this token is **reward-aligned**, and SVDD likely selected the winner partly because of it.

If $a_l$ is near zero, the token at position $l$ doesn't correlate with the reward. It ended up in the winner incidentally — guidance didn't care about it.

**Applied to the worked example (§15.2):**

Position 4, winner has `inside`:

| Candidate | pos4 token | Reward |
|-----------|-----------|--------|
| 1 | adjacent | $-1.0$ |
| **2** | **inside** | **0.0** |
| 3 | adjacent | $-0.5$ |

- Matching candidates (token = `inside`): $\{2\}$, avg reward = $0.0$
- All candidates avg reward = $-0.5$
- Attribution: $a_4 = 0.0 - (-0.5) = +0.5$ (strongly positive)

Position 5, winner has `adjacent`:

| Candidate | pos5 token | Reward |
|-----------|-----------|--------|
| 1 | adjacent | $-1.0$ |
| **2** | adjacent | **0.0** |
| 3 | inside | $-0.5$ |

- Matching candidates (token = `adjacent`): $\{1, 2\}$, avg reward = $-0.5$
- All candidates avg reward = $-0.5$
- Attribution: $a_5 = -0.5 - (-0.5) = 0.0$ (neutral)

The attribution correctly identifies position 4 as the one SVDD cared about.

**How attribution modulates remasking.** Boost the effective confidence at positions with positive attribution:

$$\text{conf}_{\text{eff}}^l = \text{conf}^l + \beta \cdot \max(a_l, 0)$$

Use $\text{conf}_{\text{eff}}$ in place of $\text{conf}$ in the $\text{softmax}(-\text{confidence})$ computation for remasking. Positions with high reward attribution get a confidence boost, making them less likely to be remasked; positions with zero attribution are unaffected.

In the example, position 4 gets $\text{conf}_{\text{eff}} = 0.35 + \beta \times 0.5$. With $\beta = 1.0$, that becomes $0.85$ — comparable to the committed node tokens. It is no longer the lowest-confidence position.

**Why this is $K$-robust.** Attribution measures a correlation, not a frequency:

- **$K = 3$:** Noisy estimates, but bounded. Worst case: noisy attributions near zero → $\beta \times 0 \approx 0$ → remasking operates as if unmodified. No harm done.
- **$K = 16$:** Statistically meaningful. Positions where the winner's token genuinely correlates with higher reward get a clear positive signal; incidental positions average out to $\sim 0$.
- **$K = 64$:** Even more precise. The method scales gracefully because it measures reward correlation, not token frequency.

**The $\beta$ hyperparameter.** $\beta$ controls the exchange rate between model confidence and reward alignment. At $\beta = 0$, standard confidence remasking. As $\beta$ increases, reward-aligned positions become increasingly protected. A reasonable default: $\beta = 1 / \overline{\Delta r}$ where $\overline{\Delta r}$ is the mean reward gap (normalizes attribution so the boost is scaled relative to the confidence range). This requires empirical validation.

**Cost.** Zero extra model calls. The computation is $O(K \times B \times \text{SEQ\_LEN})$ — a comparison and averaging loop over candidates — negligible compared to the model call and scoring.

### 15.4.4 Summary Comparison

| Aspect | Option A (fresh logits) | Option B (protect 1 step) | Option C (reward attribution) |
|--------|:-:|:-:|:-:|
| Extra model calls | 1 per step ($2\times$ total) | 0 | 0 |
| New hyperparameters | None | None | $\beta$ |
| Precision | High — true post-unmask confidence | Blunt — protects all just-unmasked | Surgical — protects reward-aligned only |
| Implementation complexity | Minimal (one extra model call) | Minimal (one mask intersection) | Moderate (attribution loop) |
| $K$-sensitivity | None | None | Low (graceful degradation) |
| Risk | Compute cost | Over-protection of irrelevant positions | $\beta$ tuning; noisy at small $K$ |

**Recommendation.** Option B is the pragmatic default — zero cost, no hyperparameters, directly addresses the root cause. Option C is the principled upgrade if Option B proves too blunt (i.e., if protecting all just-unmasked positions sacrifices too much remasking diversity). Option A is the gold standard but may not justify the $2\times$ compute unless the tension is empirically severe.

All three options are compatible with each other: one could use Option B as a baseline, add Option C for fine-grained control, and reserve Option A for a validation experiment to establish the upper bound on what fresh logits can achieve.

