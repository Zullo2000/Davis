# Diffusion Module Documentation

## Overview

The `bd_gen.diffusion` module implements the MDLM (Masked Diffusion Language Model) framework for discrete token sequences representing bubble diagrams. The module handles:

1. **Noise schedules** — continuous-time schedules mapping `t ∈ [0, 1]` to masking probability
2. **Forward process** — stochastic masking of clean tokens with PAD protection
3. **ELBO loss** — continuous-time variational bound with dual-vocabulary CE and class weighting
4. **Reverse sampling** — iterative denoising from fully masked to clean sequences

Mathematical reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (MDLM, arXiv:2406.07524).

## Architecture

### Training Pipeline
```
x0 (clean tokens)
   │
   ├── forward_mask(x0, pad_mask, t, schedule) → (x_t, mask_indicators)
   │
   ├── BDDenoiser(x_t, pad_mask, t) → (node_logits, edge_logits)
   │
   └── ELBOLoss(node_logits, edge_logits, x0, x_t, pad_mask, mask_indicators, t, schedule) → scalar loss
```

### Inference Pipeline
```
sample(model, schedule, vocab_config, batch_size, num_steps, ...) → generated tokens (B, SEQ_LEN)
```

## Module Reference

### `noise_schedule.py`

**`NoiseSchedule`** (abstract base, inherits `abc.ABC` and `nn.Module`)
- `sigma(t)` — total noise at time t
- `alpha(t)` — keeping probability: `exp(-sigma(t))`
- `alpha_prime(t)` — derivative `d(alpha)/dt`, needed for ELBO weight
- `importance_sampling_transformation(t)` — maps uniform `t` to importance-weighted `t`

**`LinearSchedule(sigma_min=0.0, sigma_max=10.0)`**
```
sigma(t) = sigma_min + t * (sigma_max - sigma_min)
alpha(t) = exp(-sigma(t))
alpha'(t) = -(sigma_max - sigma_min) * alpha(t)
```

**`CosineSchedule(eps=1e-3)`**
```
alpha(t) = eps + (1 - eps) * cos(t * pi/2)
sigma(t) = -log(alpha(t))
alpha'(t) = -(1 - eps) * (pi/2) * sin(t * pi/2)
```

**`get_noise(config)`** — factory function reading `config.type` to instantiate the right schedule.

### `forward_process.py`

**`forward_mask(x0, pad_mask, t, noise_schedule, vocab_config)`**
- Masks each non-PAD position independently with probability `1 - alpha(t)`
- Node positions → `NODE_MASK_IDX` (13), edge positions → `EDGE_MASK_IDX` (11)
- **PAD positions are NEVER masked** — enforced by `should_mask & pad_mask`
- Returns `(x_t, mask_indicators)` where `mask_indicators` is True where masking occurred

### `loss.py`

**`ELBOLoss(edge_class_weights, node_class_weights=None, vocab_config, eps, t_min)`**

Key design:
- Splits `x0`, `pad_mask`, `mask_indicators` into node `[:, :n_max]` and edge `[:, n_max:]`
- Separate CE computations with separate vocabularies and class weights
- `loss_mask = mask_indicators & pad_mask` (masked AND real positions only)
- Per-sample normalization by `N_active` (count of loss-active positions, clamped to min 1)
- ELBO weight `w(t) = -alpha'(t) / (1 - alpha(t) + eps)`, clamped to max 1000
- Timestep `t` clamped to `[t_min, 1.0]` before `w(t)` computation

### `sampling.py`

**`sample(model, noise_schedule, vocab_config, batch_size, num_steps, ...)`**

Algorithm (MDLM ancestral sampling):
1. Determine `num_rooms` per sample (from `fixed_num_rooms`, `num_rooms_distribution`, or uniform)
2. Initialize fully masked sequence (real positions → MASK, PAD positions → PAD tokens)
3. Loop from `t=1` to `t=0` in `num_steps` steps:
   - Get model logits, apply guidance if provided
   - Compute `p_unmask = (alpha_next - alpha_now) / (1 - alpha_now + eps)`
   - Predict tokens for masked positions (argmax or Gumbel based on temperature)
   - Select which MASK positions to unmask (see **Unmasking Modes** below)
   - Clamp PAD positions, apply fixed tokens and remasking if provided
4. Final cleanup: remaining MASK → argmax at `t ≈ 0`

**Unmasking Modes** (`unmasking_mode` parameter):

| Mode | Strategy | Reference |
|------|----------|-----------|
| `"random"` (default) | Each MASK position is unmasked independently with probability `p_unmask` — a coin-flip per position. This is the standard MDLM approach. | Sahoo et al. (MDLM) |
| `"llada"` | Unmask the positions where the model is most confident first. At each step, a budget of `p_unmask × num_remaining_masked` positions is unmasked, selecting those with the highest `P(predicted_token)`. At the final step (`t→0`), all remaining masks are removed. | Nie et al. (LLaDA) |

**Why confidence-based unmasking?** Random unmasking treats all positions equally, which can cause the model to commit early to low-confidence predictions that are hard to correct later. Confidence-based unmasking lets the model resolve easy/structural positions first (e.g., obvious room types, dominant edge relationships) and defer ambiguous positions to later steps when more context is available. This has been shown to improve sample quality in masked language model diffusion (LLaDA, MDLM++ variants).

Extension hooks:
- `guidance_fn((node_logits, edge_logits), x_t, t, pad_mask)` — logit modification per step
- `fixed_tokens` + `fixed_mask` — inpainting (fixed positions clamped after each step)
- `remasking_fn(x_t, t)` — ReMDM-style remasking after each unmasking step

## Numerical Stability

| Hazard | Mitigation |
|--------|------------|
| `w(t) → ∞` as `t → 0` | Clamp `t ≥ 1e-5` and `w ≤ 1000`; float64 internal computation |
| `log(0)` in cosine sigma | Clamp `alpha_t ≥ 1e-8` before log |
| Gumbel `log(-log(u))` underflow | float64 + clamp `u ∈ [1e-10, 1-1e-10]` |
| `p_unmask` catastrophic cancellation | **float64 via `alpha(t.double())`** + eps + clamp ([arXiv:2409.02908](https://arxiv.org/abs/2409.02908)) |
| `N_active = 0` | Clamp `N_active ≥ 1.0` |
| Importance sampling with `sigma_min=0` | Clamp `sigma_min ≥ 1e-4` in transform |

### Float64 Precision for Transition Probabilities

The sampling loop computes `p_unmask = (α(t_next) − α(t_now)) / (1 − α(t_now))` at each
step. When the number of sampling steps N is large, `t_next` and `t_now` differ by only
`1/N`, making `α(t_next)` and `α(t_now)` very close. In float32 (~7 decimal digits),
the numerator `α(t_next) − α(t_now)` loses most significant digits to catastrophic
cancellation. This effectively lowers the unmasking probability, reducing token diversity.

Identified by Zheng et al. ([arXiv:2409.02908](https://arxiv.org/abs/2409.02908)) as a
fundamental issue in MDLM-family models. The fix: compute `alpha(t)` in float64 (~15
decimal digits) by passing `t.double()` to the existing `alpha()` method. PyTorch
auto-promotes `float32 buffer + float64 tensor → float64`.

Affected code:
- **`sampling.py`** (inference): `p_unmask` computed via `alpha(t.double())`
- **`loss.py`** (training): `w(t)` denominator `1 − α(t)` computed via `alpha(t.double())`
- Model forward pass remains entirely in float32

## Configuration

Noise schedules are configured via Hydra YAML files:

```yaml
# configs/noise/linear.yaml
type: linear
sigma_min: 0.0
sigma_max: 10.0

# configs/noise/cosine.yaml
type: cosine
eps: 1e-3
```

Usage: `schedule = get_noise(cfg.noise)` or `schedule = get_noise(OmegaConf.load("configs/noise/linear.yaml"))`.
