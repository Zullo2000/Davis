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

# 14. Inference-Time Guidance via SVDD (Work in Progress)

*This section is work in progress. 

NB: guidance is the most important feature of this model, therefore even if a version from those above proved to be the best but it's not guidance-adaptable, then we'll have to disregard it.

Questions:

- how does remasking affect the reward based guidance a-la-SVDD?
- how should I choose t_switch?
- how do the unmasking and token selection design choices (those that distinguish the many models we have studied so far) affect the reward based guidance a-la-SVDD?
- how does learning the forward process too affect the reward based guidance a-la-SVDD?
