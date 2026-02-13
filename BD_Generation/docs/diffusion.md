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
   - Stochastically unmask MASK positions (argmax or Gumbel based on temperature)
   - Clamp PAD positions, apply fixed tokens and remasking if provided
4. Final cleanup: remaining MASK → argmax at `t ≈ 0`

Extension hooks:
- `guidance_fn((node_logits, edge_logits), x_t, t, pad_mask)` — logit modification per step
- `fixed_tokens` + `fixed_mask` — inpainting (fixed positions clamped after each step)
- `remasking_fn(x_t, t)` — ReMDM-style remasking after each unmasking step

## Numerical Stability

| Hazard | Mitigation |
|--------|------------|
| `w(t) → ∞` as `t → 0` | Clamp `t ≥ 1e-5` and `w ≤ 1000` |
| `log(0)` in cosine sigma | Clamp `alpha_t ≥ 1e-8` before log |
| Gumbel `log(-log(u))` underflow | float64 + clamp `u ∈ [1e-10, 1-1e-10]` |
| `p_unmask` division by ~0 | `+1e-8` denominator + clamp to `[0, 1]` |
| `N_active = 0` | Clamp `N_active ≥ 1.0` |
| Importance sampling with `sigma_min=0` | Clamp `sigma_min ≥ 1e-4` in transform |

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
