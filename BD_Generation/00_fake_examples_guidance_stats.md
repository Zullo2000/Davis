# Fake Examples: What You'll See from G5 Pilot

> All numbers below are **fabricated but realistic**. Use this to understand
> the output format and what to look for before running the real experiments.

---

## Step 1: Calibration output

File: `configs/guidance/calibration_v1_no_remask.json`

```json
{
    "one_kitchen": 1.0,
    "one_living": 1.0,
    "kitchen_near_living": 1.0,
    "no_bath_kitchen": 2.0
}
```

**Where do the 4 constraints come from?** They are user-authored in
`configs/guidance/example_basic.yaml`. You write whatever constraints you
want (ExactCount, CountRange, RequireAdj, ForbidAdj) in the YAML file.
The 4 in this example are a starting set; you can add/remove/change any.

**Interpretation:** Among unguided v1 samples, the 90th-percentile violation for
`one_kitchen` is 1.0 (most violations are |count-1| = 1), and for `no_bath_kitchen`
it's 2.0 (some samples have 2 forbidden Bathroom-Kitchen adjacencies). P90=1.0
means "no scaling needed". P90=2.0 means "violations are divided by 2 so a
violation of 2 contributes the same energy as a violation of 1 for the other
constraints".

---

## Step 2: Generation logs (per run)

For each of the 6 runs, you'll see:

```
2026-02-27 14:30:12 [INFO] Device: cuda
2026-02-27 14:30:12 [INFO] Guidance config: configs/guidance/example_basic.yaml
2026-02-27 14:30:12 [INFO] Guidance: K=4, alpha=1.00, phi=linear, reward_mode=soft, 4 constraints
2026-02-27 14:30:12 [INFO] Calibration loaded: configs/guidance/calibration_v1_no_remask.json
2026-02-27 14:30:13 [INFO] Model loaded from checkpoint_final.pt (v2=False)
2026-02-27 14:30:15 [INFO] Generating 1000 guided samples x 5 seeds (42, 123, 456, 789, 1337)
--- Seed 42 ---
seed 42: 100%|████████████████████| 16/16 [02:45<00:00]
  Generated 1000 guided samples, shape [1000, 44]
--- Seed 123 ---
...
2026-02-27 14:44:30 [INFO] Saved: eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_basic_K4_a1.0_samples.pt (156.2 MB)
2026-02-27 14:44:30 [INFO] Method name: llada_topp0.9_no_remask_guided_basic_K4_a1.0
```

---

## Step 3: Evaluation output (per model)

```
2026-02-27 15:00:01 [INFO] Evaluating: llada_topp0.9_no_remask_guided_basic_K4_a1.0
2026-02-27 15:00:01 [INFO] Loaded 4 constraints from example_basic.yaml
  Seed 42: 1000 samples, seq_len=44
  Seed 123: 1000 samples, seq_len=44
  ...
  validity=96.8% +/- 0.4%, diversity=0.912 +/- 0.003
  Saved: eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_basic_K4_a1.0.json
```

---

## Step 4: Comparison table (the main output)

File: `eval_results/loglinear_noise_sc/comparison_guided_pilot.md`

**Baseline** = the same model variant (v1 + llada + top-p=0.9 + no remasking)
run WITHOUT guidance, re-evaluated with `--guidance-config` so constraint
metrics are computed for it too. Satisfaction rate = (# samples where the
constraint is satisfied) / (1000 per seed), then mean +/- std across 5 seeds.
"Satisfaction (all)" = fraction where ALL constraints are simultaneously met.

### Configuration

| Parameter      | baseline  | K4_a0.1 | K4_a1.0 | K4_a5.0 | K16_a0.1 | K16_a1.0 | K16_a5.0 |
|----------------|:---------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------:|
| Guidance K     | --        | 4       | 4       | 4       | 16       | 16       | 16       |
| Guidance alpha | --        | 0.1     | 1.0     | 5.0     | 0.1      | 1.0      | 5.0      |
| Reward mode    | --        | soft    | soft    | soft    | soft     | soft     | soft     |

### Validity

| Metric           | baseline       | K4_a0.1        | K4_a1.0        | K4_a5.0        | K16_a0.1       | K16_a1.0       | K16_a5.0       |
|------------------|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Validity rate    | 97.2 +/- 0.3% | 96.8 +/- 0.4% | 95.1 +/- 0.5% | 82.3 +/- 1.2% | 97.0 +/- 0.3% | 96.2 +/- 0.4% | 78.5 +/- 1.5% |
| Inside validity  | 91.5 +/- 0.4% | 91.2 +/- 0.5% | 89.8 +/- 0.6% | 75.1 +/- 1.8% | 91.3 +/- 0.4% | 90.5 +/- 0.5% | 71.2 +/- 2.0% |

### Coverage

| Metric    | baseline           | K4_a0.1            | K4_a1.0            | K4_a5.0            | K16_a0.1           | K16_a1.0           | K16_a5.0           |
|-----------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Diversity | 0.9230 +/- 0.003   | 0.9180 +/- 0.003   | 0.8950 +/- 0.005   | 0.7120 +/- 0.015   | 0.9200 +/- 0.003   | 0.9050 +/- 0.004   | 0.6800 +/- 0.020   |
| Novelty   | 0.8100 +/- 0.005   | 0.8050 +/- 0.005   | 0.7800 +/- 0.008   | 0.6500 +/- 0.020   | 0.8080 +/- 0.005   | 0.7950 +/- 0.006   | 0.6200 +/- 0.025   |

### Priority Metrics

| Metric                          | baseline           | K4_a0.1            | K4_a1.0            | K4_a5.0            | K16_a0.1           | K16_a1.0           | K16_a5.0           |
|---------------------------------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Mode coverage (weighted)        | 84.5 +/- 0.8%     | 83.9 +/- 0.9%     | 80.2 +/- 1.2%     | 62.0 +/- 2.5%     | 84.1 +/- 0.8%     | 81.5 +/- 1.0%     | 58.3 +/- 3.0%     |
| Spatial transitivity            | 93.1 +/- 0.3%     | 92.8 +/- 0.3%     | 91.0 +/- 0.5%     | 78.5 +/- 1.5%     | 92.9 +/- 0.3%     | 91.8 +/- 0.4%     | 75.2 +/- 1.8%     |
| Cond. edge TV (weighted)        | 0.0410 +/- 0.003  | 0.0425 +/- 0.003  | 0.0520 +/- 0.004  | 0.1150 +/- 0.012  | 0.0418 +/- 0.003  | 0.0480 +/- 0.004  | 0.1280 +/- 0.015  |
| Type-cond. degree TV (weighted) | 0.0350 +/- 0.002  | 0.0365 +/- 0.002  | 0.0450 +/- 0.003  | 0.0980 +/- 0.010  | 0.0358 +/- 0.002  | 0.0415 +/- 0.003  | 0.1100 +/- 0.012  |
| Node TV                         | 0.0240 +/- 0.002  | 0.0255 +/- 0.002  | 0.0360 +/- 0.003  | 0.0950 +/- 0.010  | 0.0248 +/- 0.002  | 0.0320 +/- 0.003  | 0.1050 +/- 0.012  |

### Constraint Satisfaction

| Metric                              | baseline       | K4_a0.1        | K4_a1.0        | K4_a5.0        | K16_a0.1       | K16_a1.0       | K16_a5.0       |
|-------------------------------------|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| **Satisfaction (all)**              | 28.5 +/- 1.2% | 33.0 +/- 1.5% | 52.4 +/- 2.0% | 71.2 +/- 2.5% | 35.8 +/- 1.3% | 62.0 +/- 1.8% | 76.5 +/- 2.2% |
| Satisfaction: one_kitchen           | 62.0 +/- 2.0% | 66.5 +/- 2.1% | 78.3 +/- 1.5% | 88.0 +/- 1.0% | 68.0 +/- 1.8% | 83.5 +/- 1.2% | 91.2 +/- 0.8% |
| Satisfaction: one_living            | 58.0 +/- 2.5% | 62.0 +/- 2.3% | 75.1 +/- 1.8% | 85.5 +/- 1.2% | 64.5 +/- 2.0% | 80.2 +/- 1.5% | 88.0 +/- 1.0% |
| Satisfaction: kitchen_near_living   | 71.0 +/- 1.5% | 74.0 +/- 1.6% | 83.0 +/- 1.2% | 90.5 +/- 0.8% | 76.0 +/- 1.4% | 87.0 +/- 1.0% | 93.0 +/- 0.6% |
| Satisfaction: no_bath_kitchen       | 80.0 +/- 1.0% | 82.5 +/- 1.1% | 89.0 +/- 0.8% | 94.0 +/- 0.5% | 83.0 +/- 1.0% | 91.5 +/- 0.7% | 95.5 +/- 0.4% |
| Mean violation: one_kitchen         | 0.4200 +/- 0.02 | 0.3700 +/- 0.02 | 0.2300 +/- 0.02 | 0.1200 +/- 0.01 | 0.3500 +/- 0.02 | 0.1700 +/- 0.01 | 0.0900 +/- 0.01 |
| Mean violation: one_living          | 0.4600 +/- 0.03 | 0.4100 +/- 0.02 | 0.2600 +/- 0.02 | 0.1500 +/- 0.01 | 0.3800 +/- 0.02 | 0.2000 +/- 0.02 | 0.1200 +/- 0.01 |
| Mean violation: kitchen_near_living | 0.2900 +/- 0.02 | 0.2600 +/- 0.02 | 0.1700 +/- 0.01 | 0.0950 +/- 0.01 | 0.2400 +/- 0.01 | 0.1300 +/- 0.01 | 0.0700 +/- 0.01 |
| Mean violation: no_bath_kitchen     | 0.2200 +/- 0.01 | 0.1900 +/- 0.01 | 0.1200 +/- 0.01 | 0.0600 +/- 0.01 | 0.1800 +/- 0.01 | 0.0900 +/- 0.01 | 0.0450 +/- 0.00 |

---

## How to read the comparison table

### The key tradeoff: satisfaction vs quality degradation

1. **Does guidance improve constraint satisfaction?** Look at `Satisfaction (all)` --
   it should increase monotonically as alpha decreases (stronger guidance) and K
   increases (more candidates). In the fake example: baseline 28.5% -> K16_a5.0
   achieves 76.5%.

2. **At what cost?** Look at:
   - **Validity rate** dropping: from 97.2% to 78.5% at aggressive settings is a red
     flag -- guidance is pushing the model too far from its learned distribution.
   - **Diversity / Mode coverage (weighted)** dropping: diversity 0.923 -> 0.680 or
     mode coverage 84.5% -> 58.3% means guidance is collapsing mode diversity.
     High mode coverage only matters when novelty is also high -- if novelty tanks,
     mode coverage is meaningless.
   - **Spatial transitivity** dropping: from 93.1% to 75.2% means guidance is
     producing physically unrealizable graphs.
   - **Cond. edge TV / Type-cond. degree TV / Node TV** increasing: these measure
     how far generated distributions drift from training. Small increases are fine;
     large jumps (e.g. Cond. edge TV 0.041 -> 0.128) indicate overfitting to
     constraints at the expense of realism.

3. **Sweet spot**: You want the row where satisfaction is meaningfully higher than
   baseline but validity/diversity/transitivity haven't tanked. In the fake example,
   `K16_a1.0` looks promising: satisfaction 28.5% -> 62.0%, validity only drops
   97.2% -> 96.2%, spatial transitivity 93.1% -> 91.8%, diversity 0.923 -> 0.905.

4. **K effect**: Compare same-alpha across K=4 vs K=16. More candidates should
   always help (higher satisfaction, same or better quality). If K=16 isn't better
   than K=4, something is wrong with the importance sampling.

5. **alpha too aggressive**: If alpha=5.0 has very high satisfaction but validity
   drops below ~90% or diversity drops below ~0.80, it's too aggressive.

---

## GuidanceStats diagnostics

Run `python scripts/analyze_guidance_stats.py --schedule loglinear_noise_sc --model <name>`:

### Part A: Text summary (averaged over 5000 samples)

```
======================================================================
  Model: llada_topp0.9_no_remask_guided_basic_K16_a1.0
  Schedule: loglinear_noise_sc
  Config: K=16, alpha=1.0, mode=soft, phi=linear
  Steps: 100
  Batches/seeds analyzed: 80
======================================================================

  ESS trajectory:
    Mean:  8.52
    Min:   3.41 (step 78)
    Max:   15.20 (step 5)

  Max weight (single candidate dominance):
    Mean: 0.215
    Max:  0.580 (step 78)

  Reward gap (guidance steering signal):
    Mean: 0.1823

  Remasking delta (positive = cooperative):
    Mean:     0.0000
    Negative: 0/100 steps
    (Note: remasking disabled for this variant, so delta is always 0)

  Per-constraint violation trajectories:
    one_kitchen:
      Start: 0.4200  End: 0.0850  Min: 0.0850
      Resolved (<0.1) at step 89/100 (89%)
    one_living:
      Start: 0.4600  End: 0.1050  Min: 0.1050
    kitchen_near_living:
      Start: 0.2900  End: 0.0620  Min: 0.0620
      Resolved (<0.1) at step 82/100 (82%)
    no_bath_kitchen:
      Start: 0.2200  End: 0.0380  Min: 0.0380
      Resolved (<0.1) at step 68/100 (68%)

  Trajectory snapshot (every 20 steps):
      Step    ESS   MaxW    R_gap    R_sel  R_delta  Remask
         0  14.80  0.102   0.0120  -0.8500   0.0000     0.0
        20  12.30  0.145   0.0850  -0.6200   0.0000     0.0
        40  10.10  0.180   0.1520  -0.4100   0.0000     0.0
        60   7.50  0.250   0.2100  -0.2300   0.0000     0.0
        80   4.20  0.420   0.2800  -0.1100   0.0000     0.0
        99   5.80  0.310   0.1950  -0.0450   0.0000     0.0
```

The text summary above is useful as a quick health check but averages over
5000 samples hide important distributional information. The plots below are
the primary diagnostic.

### Part B: Distribution plots (histograms over 5000 samples)

Run with `--plot-distributions`:

```bash
python scripts/analyze_guidance_stats.py \
    --schedule loglinear_noise_sc \
    --model llada_topp0.9_no_remask_guided_basic_K16_a1.0 \
    --plot-distributions
```

Saves: `eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_basic_K16_a1.0_distributions.png`

The figure has one subplot per metric, each showing a histogram over
all 5000 samples (1000 samples x 5 seeds) of the **final denoising step**
value. Each histogram shows mean (red dashed) and median (orange dotted).

**Subplots (left to right, top to bottom):**

```
+---------------------------+---------------------------+---------------------------+
| ESS (final step)          | Reward (selected)         | Reward gap                |
|                           |                           |                           |
| [histogram over 5000      | [histogram]               | [histogram]               |
|  samples]                 |  -- mean=..., median=...  |  -- mean=..., median=...  |
|  -- mean=8.52, med=8.10   |                           |                           |
+---------------------------+---------------------------+---------------------------+
| Remasking delta           | viol: one_kitchen         | viol: one_living          |
|                           |                           |                           |
| [all 0 for no-remasking]  | [histogram: many at 0,    | [histogram: many at 0,    |
|                           |  spike at 1]              |  spike at 1]              |
+---------------------------+---------------------------+---------------------------+
| viol: kitchen_near_living | viol: no_bath_kitchen     |                           |
|                           |                           |                           |
| [histogram: mostly 0,     | [histogram: mostly 0,     |                           |
|  some at 1]               |  some at 1 or 2]          |                           |
+---------------------------+---------------------------+---------------------------+
```

**What to look for:**

- **ESS histogram**: Should be roughly unimodal, centered around K/2.
  If bimodal (cluster near 1 AND near K), guidance is working for some
  samples and failing for others -- investigate which constraint
  causes the split.
- **Reward gap histogram**: Should be mostly positive. If a long left
  tail reaches negative values, importance sampling is sometimes
  selecting worse-than-average candidates (likely noise from low ESS).
- **Violation histograms**: Ideally most mass at 0. A heavy tail
  beyond 1 means guidance can't resolve that constraint for many samples.
  Compare across guided vs unguided: the histogram should shift leftward.

### Part C: Time-evolution trajectory plots (2 individual samples)

Run with `--plot-trajectories`:

```bash
python scripts/analyze_guidance_stats.py \
    --schedule loglinear_noise_sc \
    --model llada_topp0.9_no_remask_guided_basic_K16_a1.0 \
    --plot-trajectories --traj-seed 42
```

Saves: `eval_results/loglinear_noise_sc/llada_topp0.9_no_remask_guided_basic_K16_a1.0_trajectories_seed42.png`

The figure shows 2 individual samples from seed 42 overlaid in each
subplot (blue = sample 0, red = sample 64). Every denoising step
(0..99) is plotted, giving the full time evolution of each metric.

**Subplots (left to right, top to bottom):**

```
+---------------------------+---------------------------+---------------------------+
| ESS                       | Reward (selected)         | Reward gap                |
|                           |                           |                           |
| [two curves, one per      | [two curves: both start   | [two curves: starts near  |
|  sample, step 0..99]      |  at ~-0.85, converge      |  0, peaks mid-process,    |
|  -- starts high (~15),    |  toward ~-0.05]           |  falls back as violations |
|  decreases, recovers      |                           |  are resolved]            |
+---------------------------+---------------------------+---------------------------+
| Remasking delta           | viol: one_kitchen         | viol: one_living          |
|                           |                           |                           |
| [flat at 0 for            | [blue: drops from 1.0     | [blue: drops from 1.0     |
|  no-remasking variant]    |  to 0 at step 75;         |  to 0 at step 82;         |
|                           |  red: drops from 0 (was   |  red: stays at 1.0        |
|                           |  already satisfied)]      |  until step 90 then 0]    |
+---------------------------+---------------------------+---------------------------+
| viol: kitchen_near_living | viol: no_bath_kitchen     |                           |
|                           |                           |                           |
| [blue: 1->0 at step 60;  | [blue: 2->0 at step 45;  |                           |
|  red: 1->0 at step 55]   |  red: 0 throughout]       |                           |
+---------------------------+---------------------------+---------------------------+
```

**What to look for (this is the critical diagnostic):**

- **Violation curves for individual samples**: Each sample has a DISCRETE
  trajectory -- violations are integers (0, 1, 2, ...) for hard constraints.
  The interesting question is: at what step does each constraint get resolved
  (violation drops to 0), and does it STAY resolved or does it bounce back?

- **Bouncing violations**: If a violation drops to 0 at step 60 then jumps
  back to 1 at step 70, it means guidance committed to the right token early
  but a later unmasking step broke it. This would be a red flag -- guidance
  selection at step 60 was correct but subsequent steps undid the progress.

- **ESS per sample**: Unlike the averaged ESS which is smooth, per-sample
  ESS will be noisy. Look for steps where ESS drops to 1 (total collapse) --
  at those steps, all K candidates got reward=0 except one.

- **Reward trajectory shape**: Should monotonically increase (become less
  negative) as violations are resolved. Stalls in the reward curve indicate
  steps where guidance gets stuck. Non-monotonic drops (reward decreases
  after increasing) indicate guidance is actively fighting the sampling process.

- **Divergence between samples**: If both samples show qualitatively similar
  trajectories (same shape, similar resolution times), guidance is working
  robustly. If they diverge wildly (one resolves everything by step 50,
  the other never resolves), the process has high variance -- may need
  higher K or different alpha.

---

## How to read the averaged diagnostics (Part A)

**ESS (Effective Sample Size):**
- ESS ~ K/2 (8.5 out of 16) -> guidance working but not degenerate
- ESS min > 2.0 -> no step dominated by a single candidate
- ESS ~ K at all steps -> alpha too high, guidance has no effect
- ESS < 2 at many steps -> alpha too low, too aggressive

**Max weight:**
- Mean < 0.3 -> healthy diversity across candidates
- Max > 0.95 -> one candidate always wins -> diversity destroyed

**Reward gap:**
- Positive -> guidance is selecting better-than-average candidates
- ~ 0 -> importance weighting isn't helping (alpha too high or constraints
  already trivially satisfied)

**Remasking delta:**
- Always 0 for no-remasking variant (expected)
- For remasking variants: positive = cooperative, negative = remasking fights guidance
- If negative at >50% of steps -> remasking undoes guidance progress

**Per-constraint violation trajectories:**
- Should decrease over time as positions get committed
- "Resolved at step X" = when average violation drops below 0.1
- If a constraint never resolves -> guidance isn't strong enough for that constraint
- If all resolve early (< step 50) -> alpha might be stronger than needed

---

## Reward computation details (for reference)

The reward for each candidate at each step is:

```
For each constraint i:
    term_i = (lambda_i / p90_i) * phi(violation_i)

E(x) = sum(term_i)       (energy, always >= 0)
r(x) = -E(x)             (reward, always <= 0)
weights = softmax(r / alpha)  (importance weights over K candidates)
```

Where:
- `lambda_i` = constraint weight (all 1.0 in our config)
- `p90_i` = calibration normalizer (from step 1)
- `phi` = linear (identity) in our config
- `alpha` = guidance temperature (lower = stronger guidance)

**reward_gap** = `mean(reward_of_selected_candidate - mean_reward_of_all_K_candidates)`

**remasking_delta** = `mean(reward_after_remasking - reward_before_remasking)`

Both are averaged over all samples in the batch, then over all batches/seeds
(for the text summary). Per-sample values are preserved in the distribution
and trajectory plots.
