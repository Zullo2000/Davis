## Checkpoint: 
so far we can generate unconstrained bubble diagrams using discrete masked diffusion, now we wanna add guidance to make sure that they respect some topological constraints.

------------------------------------------------------------------------


## General idea: 
using a mixed strategy Reward (incentivizes) + Projection (enforces) / we start with only Reward ! / an idea could be hard-enforce what is easy and unambiguous, and reserve reward shaping for global or preference-like constraints / or consider curricula like “start from later timesteps and progressively extend earlier” explicitly to address sparse terminal feedback.

------------------------------------------------------------------------


In SVDD they predict the soft value functions v_t(x) with posterior mean approximation, then they use them to compute the importance sampling weights as in SMC methods (that's where \alpha arrives). at the beginning the v_t(x) is just the reward calculated from the predicted \hat{x_0} clean from the training. We start with a SVDD-like model.

in SVDD the final guided distribution is proportional to exp(r(x)/\alpha) times the pretrained distribution.

------------------------------------------------------------------------

## reward options:

a) weighted-indicator-reward: r(x)=-E(x)= - ( m - sum_i \lambda_i I_i(x_0) ). Where I_i tells us whether the constraint i was respected or not. 

------------------------------------------------------------------------

b) weighted-violation-reward: r(x)=-E(x)=sum_i \lambda_i \phi_i(v_i(x_0)). x_0 is the clean generated graph 

------------------------------------------------------------------------

The main concern (that leads us towards respected-violated-reward) is that the number of constraints will grow a lot, and we couldn't manually write a different v_i for each of them (case by case). You'd do it anyway! To compute the indicator you typically have to compute the same underlying quantities you’d need for a magnitude violation v_i(x). Moreover, It is not acceptable that 2 kitchens and 5 kitchens get the same I=0. Remember that from v_i you can always get the I_i for free by doing I_i = indicator(v_i = 0)

Implement a tiny “constraint primitives” library, where most new constraints are just new parameterizations, not new bespoke code.

1.  `ExactCount`
2.  `CountRange`
3.  `RequireAdj`
4.  `ForbidAdj`
5.  `Connectivity` (decoded + proxy)
6.  `AccessRule` (decoded + proxy)
7.  `ConditionalRequirement`

This set covers essentially all examples you listed while keeping code
modular.

------------------------------------------------------------------------


## Smart choice: include the alreay computed posteriors! 

At timestep x_t, your denoiser produces logits for every node and edge
slot.

Define distributions q per slot:

For each node slot i: - If x_t\[i\] = PAD → q_i = δ_PAD (ignore) - If
x_t\[i\] ≠ MASK → q_i = δ\_{x_t\[i\]} (committed token) - Else → q_i =
softmax(logits_i)

For each edge slot (i,j): - If either endpoint is PAD → ignore edge - If
committed → delta distribution - Else → softmax(edge logits)

This produces a posterior-mean-like soft object for scoring without
sampling.

Two ways:

------------------------------------------------------------------------

i) use logits to compute soft violations: Instead of hard indicators, compute expected structural quantities.

example -> Exactly one kitchen / living room

Expected count:

n̂\_K = Σ_i q_i(Kitchen)\
n̂\_L = Σ_i q_i(Living)

Soft violation:

v_prog\^soft = \|n̂\_K − 1\| + \|n̂\_L − 1\|

Optional soft-indicator alternative:

Compute Poisson--binomial probability P(n_K = 1)\
Define v_K = 1 − P(n_K = 1)

the other examples are at the end of the document.

------------------------------------------------------------------------


ii) use logits as a confidence gate (not as the weight itself) 

Do not replace λ_i.

Instead define:

λ_i\^eff(x_t) = λ_i · g_i(x_t)

where g_i measures confidence (entropy or max-prob).

Example confidence:

c(s) = 1 − H(q_s) / log\|V\|\
or\
c(s) = max q_s

Combine relevant confidences per constraint.

This creates an automatic curriculum over timesteps.

------------------------------------------------------------------------

ii) -> example of pligging it into SVDD-PM

Define energy:

E = Σ_i λ_i\^eff · φ_i(v_i)

Reward:

r = −E

At each reverse step:

1.  Sample M candidates from base reverse transition.
2.  For each candidate:
    -   Run denoiser once
    -   Compute soft violations
    -   Compute reward r\^(m)
3.  Resample using weights:

w\^(m) ∝ exp(r\^(m) / α)

------------------------------------------------------------------------

ii) problems with this approach:

Suppose we define:

λ_i\^eff(x_t) = λ_i · g_i(x_t)

where g_i is derived from entropy or max-probability.

Then:

-   The violation v_i(x) is still computed on a hard decoded graph.
-   Logits only scale how much the constraint matters.
-   The *shape* of the reward surface does not change --- only its
    amplitude.

So guidance becomes:

r(x) = - Σ_i λ_i g_i(x_t) v_i(x)

This helps avoid penalizing constraints too early, but it does **not**:

-   Reduce reward noise,
-   Smooth discontinuities,
-   Distinguish "almost correct" vs "completely wrong."

Instead if you use them for soft violations:

Let q_i(K) be the probability that slot i is a kitchen.

Then the expected number of kitchens:

n̂\_K = Σ_i q_i(K)

A soft violation:

v_prog\^soft = \|n̂\_K - 1\|

Now:

-   Early in diffusion:\
    if 3 slots each have 0.33 probability of kitchen →\
    n̂\_K ≈ 0.99 → almost correct.

-   Hard decode would likely produce 0 or 3 kitchens → large penalty.


------------------------------------------------------------------------

I prefer i)

------------------------------------------------------------------------


## pre-modelling:

- find all violations and the associated function (trying to avoid sparsity and discontinuity)

- protect from sensitivity to scaling by calibrating violation magnitudes: sample clean BDs -> compute v_i distributions -> normalize v_i
-- monitor the effective sample size (entropy of the normalized candidate weights) to control scale sensitivity

- decide how to evaluate: 
-- control the tradeoff: how well the model respects the constraints (we observe it only from the reward funciton? we should plot its dynamics) VS how well the model preserves the evaluation metrics from uncontrained generation

versions progress: 

- tune alpha and choose the functions \phi_i for r(x)=-E(x)=sum_i \lambda_i \phi_i(v_i(x_0))

- Add complexity in stages (curriculum). example: room-type counts -> kitchen–living distance term -> bathroom–kitchen forbidden adjacency -> connectivity coverage -> bedroom access

- At the very end: If intermediate reward remains too noisy: consider clean-sample refinement

- At the very end: when to replace reward terms with hard constraint injection or projection? “exactly one living room and one kitchen” can sometimes be enforced by clamping a designated node slot

-- consider that when you fix with projection you don't want to kill the structure obtained by the reward part. Therefore you do projection by finding the closest feasible graph to the candidate (argmin of the distance between the generated graph with reward and a feasible graph)


------------------------------------------------------------------------


## EXAMPLES OF SOFT VIOLATIONS:

#### A) Exactly one kitchen / living room

Expected count:

n̂\_K = Σ_i q_i(Kitchen)\
n̂\_L = Σ_i q_i(Living)

Soft violation:

v_prog\^soft = \|n̂\_K − 1\| + \|n̂\_L − 1\|

Optional soft-indicator alternative:

Compute Poisson--binomial probability P(n_K = 1)\
Define v_K = 1 − P(n_K = 1)

------------------------------------------------------------------------

#### B) Bedroom count in \[1,4\]

n̂\_Bed = Σ_i q_i(Bedroom)

v_bed\^soft = max(0, 1 − n̂\_Bed) + max(0, n̂\_Bed − 4)

------------------------------------------------------------------------

#### C) Conditional bathroom rule

Let n̂\_Bath = Σ_i q_i(Bathroom)

Compute P(n_Bed ≥ 3) using Poisson--binomial DP.

Expected required baths:

reqBath = 1 + P(n_Bed ≥ 3)

Violation:

v_bath\^soft = max(0, reqBath − n̂\_Bath)

------------------------------------------------------------------------

#### D) Required kitchen--living adjacency

For each pair i\<j:

p_ij\^{KL} = (q_i(K) q_j(L) + q_i(L) q_j(K)) · q_ij(ADJ)

Approximate:

P(exists KL adjacency) ≈ 1 − ∏\_{i\<j} (1 − p_ij\^{KL})

Violation:

v_k\_adj\^soft = 1 − P(exists KL adjacency)

------------------------------------------------------------------------

#### E) Forbidden bathroom--kitchen adjacency

Expected forbidden count:

v_bath_k\^soft = Σ\_{i\<j} (q_i(B) q_j(K) + q_i(K) q_j(B)) · q_ij(ADJ)

------------------------------------------------------------------------

#### F) Connectivity / Bedroom Access

Recommended hybrid approach:

-   Early steps → small weight or soft proxy
-   Late steps → compute on hard decoded graph
