# Guidance-Aware Remasking Schedule

## 1. Problem Statement

The current EMA lock mechanism in `guided_sampler.py` uses an abrupt binary
cutoff: once the EMA reward derivative plateaus, remasking is fully disabled
for that sample (the pre-remask state is restored at line 619-620). This
creates two problems:

1. **Abrupt transition**: remasking goes from full intensity to zero in one
   step. At the typical lock point (step 50-65, t=0.35-0.5), `sigma_max` is
   still 0.55-1.0, meaning remasking was flipping 50-94 tokens per step
   right before the lock silenced it entirely.

2. **No error correction post-lock**: after the lock, the sample continues
   denoising with guidance (K-candidate reweighting) but without any
   remasking. Errors present at the lock point propagate uncorrected to `t=0`.
   This is effectively the same as running with `no_remask` for the final
   35-50% of steps.

The root cause is that the loglinear noise schedule produces a `sigma_max`
that stays at 1.0 for the first half of generation and decays too slowly in
the second half to be compatible with guidance.

## 2. Empirical Remasking Budget (Loglinear, 100 steps)

```
Step (i) |  t_now  | sigma_max | ~tokens remasked (of 94)
---------|---------|-----------|-------------------------
   95    |  0.960  |   1.000   |  94.0
   80    |  0.810  |   1.000   |  94.0
   65    |  0.660  |   1.000   |  94.0
   50    |  0.510  |   1.000   |  94.0   <-- hard deadline
   45    |  0.460  |   0.832   |  78.2
   40    |  0.410  |   0.677   |  63.6
   35    |  0.360  |   0.546   |  51.3   <-- typical EMA lock
   30    |  0.310  |   0.434   |  40.8
   20    |  0.210  |   0.253   |  23.8
   10    |  0.110  |   0.112   |  10.5
    5    |  0.060  |   0.053   |   5.0
    0    |  0.010  |   0.000   |   0.0
```

Key insight: `sigma_max = 1.0` for all `t >= 0.5` (steps 0-50). Remasking is
still extremely aggressive at the points where the lock typically fires.
There is no natural taper to rely on.

## 3. Proposed Solution: Guidance-Aware Remasking Schedule

Instead of bolting a lock mechanism onto an incompatible remasking schedule,
**design the remasking schedule itself to taper naturally** when used with
guidance.

### Core idea

Apply a multiplicative decay to `sigma_max`:

```
sigma_max_guided(t) = sigma_max(t) * decay(t)
```

where `decay(t)` is a monotonically increasing function of `t` (recall `t`
goes from 1 → 0 during generation, so `decay` should be large at high `t`
and small at low `t`).

### Design goals

| Step range | t range   | Desired behavior                              |
|-----------|-----------|-----------------------------------------------|
| 0-30      | 1.0-0.70  | Full remasking — explore freely, noise is high |
| 30-50     | 0.70-0.50 | Gradual taper — reduce disruption as structure forms |
| 50-80     | 0.50-0.20 | Light remasking — only ~2-5 tokens/step        |
| 80-100    | 0.20-0.00 | Near-zero remasking — finalize without disruption |

### Candidate decay functions

All candidates are parameterized so that tuning is straightforward.

#### Option A: Power schedule

```
decay(t) = t^p
```

- `p=1`: linear taper (same as bare `t`)
- `p=2`: quadratic (faster taper in second half)
- `p=3`: cubic (aggressive taper)

Example values at key points (for `p=2`):

```
t=0.50 → decay=0.25 → sigma_max_guided = 1.0 * 0.25 = 0.25 → ~24 tokens
t=0.35 → decay=0.12 → sigma_max_guided = 0.55 * 0.12 = 0.07 → ~6 tokens
t=0.20 → decay=0.04 → sigma_max_guided = 0.25 * 0.04 = 0.01 → ~1 token
```

For `p=3`:

```
t=0.50 → decay=0.125 → sigma_max_guided = 1.0 * 0.125 = 0.125 → ~12 tokens
t=0.35 → decay=0.043 → sigma_max_guided = 0.55 * 0.043 = 0.024 → ~2 tokens
t=0.20 → decay=0.008 → sigma_max_guided = 0.25 * 0.008 = 0.002 → ~0 tokens
```

#### Option B: Sigmoid cutoff

```
decay(t) = sigmoid((t - t_mid) / tau)
```

- `t_mid`: center of the transition (e.g. 0.5)
- `tau`: sharpness (smaller = sharper transition)

This gives a smooth S-curve: full remasking above `t_mid`, near-zero below.

#### Option C: Linear ramp with floor

```
decay(t) = max(0, (t - t_off) / (t_on - t_off))
```

- `t_on`: timestep above which remasking is at full intensity (e.g. 0.7)
- `t_off`: timestep below which remasking is zero (e.g. 0.2)
- Linear interpolation between them

```
t=0.70 → decay=1.0 → full remasking
t=0.50 → decay=0.6 → sigma_max_guided = 1.0 * 0.6 = 0.6 → ~56 tokens
t=0.35 → decay=0.3 → sigma_max_guided = 0.55 * 0.3 = 0.16 → ~15 tokens
t=0.20 → decay=0.0 → zero remasking
```

### Recommendation

**Option A with `p=2` or `p=3`** is the simplest starting point:
- Single parameter to tune
- Well-behaved (monotonic, smooth, no discontinuities)
- `p=3` gives ~2 tokens at `t=0.35`, which matches the goal of "just a few"
- Easy to implement: one line change in `_compute_sigma_max` or in the caller

Option B (sigmoid) is more flexible but has two parameters. Option C is
intuitive but has a hard cutoff at `t_off` (same abruptness problem, just
pushed earlier).

## 4. Implementation Plan

### 4.1 Where to apply the decay

The decay multiplier should be applied in `RemaskingSchedule._compute_sigma_max()`
(in `bd_gen/diffusion/remasking.py`, line 78) or as a post-multiplier in
`__call__()`. The cleanest approach is a new parameter on `RemaskingSchedule`:

```python
class RemaskingSchedule:
    def __init__(self, ..., guided_decay_power: float = 0.0):
        self.guided_decay_power = guided_decay_power  # 0 = no decay (backward compat)

    def _compute_sigma_max(self, t_now, t_next, ...):
        sigma_max = ...  # existing computation
        if self.guided_decay_power > 0:
            sigma_max = sigma_max * (t_now ** self.guided_decay_power)
        return sigma_max
```

### 4.2 How to pass the parameter

- Add `guided_decay_power` to `RemaskingSchedule.__init__()` and
  `create_remasking_schedule()`.
- Expose as `--remask-decay-power` CLI arg in `generate_guided.py`.
- When `guided_decay_power > 0`, the EMA lock machinery becomes unnecessary
  and can be skipped entirely.

### 4.3 What to do with the EMA lock

- **Keep the EMA lock code** as an alternative/fallback (controlled by
  `--ema-lock` flag).
- The guided decay schedule and the EMA lock are **mutually exclusive**
  strategies for the same problem. Use one or the other, not both.
- If the decay schedule works well, the lock can be deprecated in a future
  cleanup.

### 4.4 Experiment plan

**Round 7: Guided decay schedule**

Grid (using confidence remasking + Option A/B):

| Config | Remasking strategy | Decay power | EMA lock |
|--------|-------------------|-------------|----------|
| 7a     | confidence + Opt A | p=2         | off      |
| 7b     | confidence + Opt A | p=3         | off      |
| 7c     | confidence + Opt B | p=2         | off      |
| 7d     | confidence + Opt B | p=3         | off      |

Compare against:
- Baseline no-remask (~69%)
- Round 6 best (Option B + lock, ~68.2%)
- Round 6b results (once available)

Fixed: K=16, alpha=0.01, soft reward, v1 loglinear, seeds=[42,123,456],
200 samples/seed.

## 5. Files to Modify

| File | Change |
|------|--------|
| `bd_gen/diffusion/remasking.py` | Add `guided_decay_power` param to `RemaskingSchedule` and `create_remasking_schedule()` |
| `scripts/generate_guided.py` | Add `--remask-decay-power` CLI arg, pass to remasking schedule creation |
| `bd_gen/guidance/guided_sampler.py` | No changes needed (decay is in the schedule itself) |
| `scripts/run_g5_round7.sh` | New experiment script for the decay schedule grid |
| `docs/guidance.md` | Update with decay schedule documentation |

## 6. Open Questions

1. **Should the decay also apply to non-guided sampling?** Probably not — the
   standard (non-guided) sampler benefits from the full remasking budget. The
   decay is specifically to make remasking compatible with guidance.

2. **Interaction with `t_switch`**: Currently `t_switch=1.0` activates
   remasking for all steps. With the decay schedule, `t_switch` becomes less
   important (the decay handles the taper). Consider whether `t_switch` should
   be removed or kept as a hard upper bound.

3. **Per-position vs global decay**: The current proposal applies decay
   globally (all positions get the same multiplier). An alternative is
   confidence-weighted decay: high-confidence positions get stronger decay
   (less remasking) while low-confidence positions retain more remasking
   budget. This is already partially handled by the confidence strategy
   itself, so global decay may be sufficient.

4. **Should we wait for Round 6b results before implementing?** Round 6b
   tests the warmup variant of the lock. If warmup significantly improves
   results, the decay schedule may be less urgent. However, the decay approach
   is architecturally cleaner regardless of Round 6b outcomes.
