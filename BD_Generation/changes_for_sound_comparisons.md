# Evaluation Upgrade Plan (JS/TV/W1 + Multi-seed + Stratified Drill-down)

**Audience:** VSCode agent modifying this repository.  
**Goal:** Make evaluation metrics more *stable* and *consistently comparable* across future model/sampler variants (vanilla MDLM, confidence-unmasking, ReMDM remasking, later extensions).

This plan references the current evaluation module and metric set (`bd_gen/eval/metrics.py`, `scripts/evaluate.py`, `configs/eval/default.yaml`) as described in `docs/evaluation.md`.  
It also follows the project structure in `planning_T1.md` (Phase 5 evaluation).  

---

## 1) Make evaluation systematic: separate *model quality* vs *sampler quality*

### 1.1 Motivation
A sampler change (e.g., remasking) can improve samples without changing the denoiser’s intrinsic predictive quality much. Therefore we must report **two families** of metrics:

- **Model-quality (sampler-independent):** evaluate the denoiser on *held-out data* by masking real examples and scoring predictions.
- **Sampler-quality (generation-time):** evaluate generated samples (validity, coverage, structural realism, distribution distances).

### 1.2 Code changes

#### A) Add a denoising evaluation module
Create a new file:

- `bd_gen/eval/denoising_eval.py`

Implement functions (names can vary, but keep outputs JSON-serializable):

```python
def denoising_eval(
    model,
    dataloader,
    noise_schedule,
    vocab_config,
    t_grid: list[float],
    device: str,
    max_batches: int | None = None,
) -> dict:
    """
    For each t in t_grid:
      - forward_mask batch x0 -> x_t, mask_indicators
      - model(x_t, pad_mask, t) -> logits
      - compute masked accuracy separately for nodes and edges:
          acc = correct / total over positions where (mask_indicators & pad_mask)
      - optionally compute masked cross-entropy (unweighted) for nodes and edges
    Aggregate across batches and return:
      {
        "denoise/acc_node@t=0.1": ...,
        "denoise/acc_edge@t=0.1": ...,
        ...
      }
    """
```

Notes:
- Use existing `bd_gen.diffusion.forward_process.forward_mask(...)` and the model forward signature.
- **Mask for scoring**: only positions that are (masked AND non-PAD). (Same logic used in ELBO loss.)
- Do not include `[MASK]` or `[PAD]` prediction probabilities here; your denoiser already clamps those logits to `-inf` (SUBS constraint).

#### B) Add an ELBO-on-validation metric (optional but recommended)
If you can do this easily without refactors:

- Compute average `ELBOLoss` over `val_dataloader` with random `t ~ U(0,1)` OR fixed `t_grid`.
- Return as `denoise/val_elbo`.

This uses existing `bd_gen/diffusion/loss.py: ELBOLoss`.

#### C) Wire denoising eval into `scripts/evaluate.py`
Add config toggles:

- `eval.run_denoising_eval: bool = true`
- `eval.denoising_t_grid: [0.1, 0.3, 0.5, 0.7, 0.9]`
- `eval.denoising_max_batches: int | null` (e.g., 50 for quick runs)

Implementation:
- In `evaluate.py`, after loading the checkpoint/model and dataset:
  - build a **val** dataloader
  - run `denoising_eval(...)`
  - include its outputs in the final metrics dict and wandb logs

#### D) Export it
Update:

- `bd_gen/eval/__init__.py` to export `denoising_eval` (or the chosen API).

### 1.3 Tests
Add tests in `tests/test_metrics.py` or a new `tests/test_denoising_eval.py`:

- Ensure `denoising_eval(...)` returns keys for each t.
- For a synthetic “perfect logits” setup, masked accuracy should be 1.0.
- Ensure PAD positions are excluded from totals.

---

## 3) Replace “headline KL” with JS/TV/W1 (keep marginal KL as diagnostic)

### 3.1 What to implement

#### A) Jensen–Shannon divergence (JS)
**Definition (in nats):**

Let `m = 0.5 * (p + q)`. Then:

`JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)`

**Why JS:**
- Symmetric (`JS(p,q)=JS(q,p)`)
- Bounded and far less “explosive” than KL on sparse empirical histograms
- More stable for finite sample comparisons (your current setup is histogram-based)

Implementation note (no smoothing for now):
- Compute KL terms with the convention `0 * log(0 / x) = 0`.
- If `p_k > 0`, then `m_k > 0` automatically, so `log(p_k / m_k)` is safe.

#### B) Total Variation distance (TV)
**Definition:**

`TV(p, q) = 0.5 * sum_k |p_k - q_k|`

**Why TV:**
- Very stable (no logs)
- Interpretable: “how much probability mass differs”
- Great sanity-check complement to JS

#### C) Wasserstein-1 distance (W1) for `num_rooms`
`num_rooms` is **ordinal** (4,5,6,7,8). KL/JS ignore the ordering.

**Definition (1D discrete W1):**
For support values `x_1 < ... < x_K` with unit spacing, W1 can be computed via CDFs:

`W1(p,q) = sum_{k=1..K} |CDF_p(x_k) - CDF_q(x_k)|`

(If spacing is not 1, multiply by step size. Here it’s 1.)

**Why W1:**
- Penalizes being “one room off” less than “four rooms off”
- Much more meaningful than KL/JS for ordered count variables

### 3.2 Where to add this
Modify:

- `bd_gen/eval/metrics.py`

Add helper functions (numpy or torch OK; keep dependencies minimal):

```python
def total_variation(p: np.ndarray, q: np.ndarray) -> float: ...
def js_divergence(p: np.ndarray, q: np.ndarray) -> float: ...
def wasserstein1_1d_discrete(p: np.ndarray, q: np.ndarray) -> float: ...
```

### 3.3 Update `distribution_match(...)` (do NOT remove KL)
Current `distribution_match` returns KL for node/edge/num_rooms.

Change it to return **both**:
- **Primary distances**: JS + TV (nodes, edges), W1 (num_rooms)
- **Diagnostic legacy**: KL (nodes, edges, num_rooms)

Suggested return keys (keep old ones for backward compatibility):
- `node_kl`, `edge_kl`, `rooms_kl` (existing)
- `node_js`, `edge_js`
- `node_tv`, `edge_tv`
- `rooms_w1`

Optional (recommended) additional scalar:
- `edge_present_rate_gen` and `edge_present_rate_ref`
- `edge_present_rate_abs_diff`

(Edge-present rate is also used in stratified metrics in section 7.)

### 3.4 Documentation update
Update `docs/evaluation.md`:
- Describe JS/TV/W1
- State clearly that marginal KL is *still computed* but no longer the headline distance

### 3.5 Tests
Update `tests/test_metrics.py`:
- JS(p,p)=0, TV(p,p)=0, W1(p,p)=0
- Simple sanity cases (e.g., p=[1,0], q=[0,1]) → TV=1.0
- Ensure `distribution_match` returns all new keys.

---

## 4) Update the “core scoreboard” (adapted to 3)

### 4.1 Scoreboard grouping (log prefixes)
In `scripts/evaluate.py`, group logged metrics into:

#### A) `denoise/*` (sampler-independent)
From section (1):
- `denoise/val_elbo` (if implemented)
- `denoise/acc_node@t=...`
- `denoise/acc_edge@t=...`

#### B) `sampler/validity/*`
From existing validity + section (7) drill-down:
- `sampler/validity/overall`
- `sampler/validity/no_mask_tokens`, `.../connected`, `.../valid_types`, etc.
- plus stratified keys (see section 7)

#### C) `sampler/coverage/*`
Existing:
- `sampler/novelty`
- `sampler/diversity`
- `sampler/mode_coverage` (+ weighted + counts)

#### D) `sampler/distribution/*` (replace “headline KL”)
Primary:
- `sampler/distribution/node_js`, `node_tv`
- `sampler/distribution/edge_js`, `edge_tv`
- `sampler/distribution/rooms_w1`
Also log diagnostic:
- `sampler/distribution/node_kl`, `edge_kl`, `rooms_kl` (but don’t headline them)

#### E) `sampler/structure/*`
Existing:
- MMD degree/clustering/spectral
- spatial transitivity (plus stratified)

### 4.2 Code changes
- In `scripts/evaluate.py`, when assembling the final metrics dict, **rename** keys with the above prefixes OR keep old keys but add prefixed aliases.
- Ensure output JSON includes both raw metric dicts and the prefixed scalars (for easy plotting in wandb).

### 4.3 Acceptance criteria
After running `python scripts/evaluate.py ...`:
- output JSON contains all categories
- wandb charts are organized by prefixes (denoise vs sampler)

---

## 5) Conditional metrics: adapt to JS/TV (+ keep KL) + add top‑N pairs

### 5.1 Conditional edge metrics (room-type pair → edge-type distribution)
Current:
- `conditional_edge_kl(...)` returns mean and weighted KL across eligible pairs.

Change it to compute:
- per-pair KL (existing)
- per-pair JS
- per-pair TV

Return keys (suggested):
- `conditional_edge_kl_mean`, `conditional_edge_kl_weighted` (existing)
- `conditional_edge_js_mean`, `conditional_edge_js_weighted`
- `conditional_edge_tv_mean`, `conditional_edge_tv_weighted`
- `num_pairs_evaluated` (existing)

Implementation details:
- Keep your existing canonicalization logic (type-pair + relation inversion).
- For each eligible pair:
  - build `p_pair` (generated) and `q_pair` (reference) over the 10 spatial relationship types (0..9)
  - compute KL/JS/TV

### 5.2 Add “top‑N pairs” version
Add a second function (or a flag):

```python
def conditional_edge_distances_topN(
    samples,
    reference,
    top_n: int = 20,
) -> dict:
    """
    Select the top-N most frequent canonical room-type pairs in *reference* (by edge count).
    Compute KL/JS/TV for those pairs only.
    Return mean + weighted variants + which pairs were used.
    """
```

Suggested return keys:
- `topN`: the N used
- `pairs_used`: list of `(type_i, type_j)` tuples (optional but useful for debugging)
- `conditional_edge_js_topN_mean`, `conditional_edge_js_topN_weighted`
- `conditional_edge_tv_topN_mean`, `conditional_edge_tv_topN_weighted`
- keep KL analogs too (diagnostic)

Why top‑N:
- Much more stable across runs (uses common pairs with good support)
- Easier to interpret and track regressions

### 5.3 Type-conditioned degree metrics
Current:
- `type_conditioned_degree_kl(...)` returns KL mean/weighted.

Add:
- `degree_js_per_type_mean`, `degree_js_per_type_weighted`
- `degree_tv_per_type_mean`, `degree_tv_per_type_weighted`
Keep KL keys too.

Implementation:
- For each room type, histogram node degree over `[0..n_max-1]` (as today).
- Compare generated vs reference with KL/JS/TV.

### 5.4 Config + evaluate.py wiring
Update `configs/eval/default.yaml`:
- `eval.conditional_topN_pairs: 20` (or null to disable)

In `scripts/evaluate.py`:
- compute both “all eligible pairs” and “top-N pairs” conditional metrics
- log them under `sampler/conditional/*`

### 5.5 Tests
- Add tests ensuring top‑N selection returns exactly N pairs when possible.
- Sanity checks: identical inputs → all distances 0.

---

## 6) Quantify uncertainty via multi-seed evaluation (no bootstrapping)

### 6.1 What to implement
Update `scripts/evaluate.py` to support running the full evaluation multiple times with different RNG seeds and then aggregating results.

Add to `configs/eval/default.yaml`:
- `eval.seeds: [42, 123, 456, 789, 1337]`
- (optional) `eval.num_seeds: null` (if you want to slice the list)

Implementation approach:
1. For each seed in `eval.seeds`:
   - call existing seed utility (`bd_gen/utils/seed.py`) to set seeds
   - run sample generation (same checkpoint + same eval config)
   - compute metrics dict for that seed
2. Aggregate scalar metrics across seeds:
   - `mean` and `std` (population or sample std; pick one and be consistent)
3. Output structure:
```python
{
  "meta": { "checkpoint": ..., "num_samples": ..., "sampling_steps": ..., "seeds": [...] },
  "per_seed": { "42": {...metrics...}, "123": {...}, ... },
  "summary": {
      "metric_name": {"mean": ..., "std": ...},
      ...
  }
}
```

### 6.2 What counts as “scalar” for aggregation
- Aggregate simple floats/ints (validity rate, JS, TV, W1, MMDs, etc.)
- For nested dicts (e.g., stratified-by-num_rooms metrics), aggregate each leaf key separately:
  - e.g. `sampler/validity_by_rooms/overall/n=4` → mean/std across seeds

### 6.3 Logging
- Log **summary means** to wandb (and optionally std as separate series).
- Save full JSON to disk (as you already do with `eval_results/save_utils.py`).

### 6.4 No bootstrap
Do not implement bootstrap resampling. Mean±std over seeds is sufficient for now.

### 6.5 Tests
- Add a small test that runs evaluation for 2 seeds with `num_samples=5` and verifies:
  - `per_seed` has both seeds
  - `summary` contains mean/std keys

---

## 7) Stratify ONLY these drill-down metrics by `num_rooms`

### 7.1 Metrics to stratify (and only these)
Compute per-`num_rooms` breakdown for:

1. **Validity** (overall + key subchecks)
2. **Spatial transitivity**
3. **Edge-present rate / adjacency**

Do **not** stratify novelty/diversity/mode_coverage by default.

### 7.2 Implementation details

#### A) Validity-by-num_rooms
In `scripts/evaluate.py` (or in a helper in `bd_gen/eval/metrics.py`):
- You already have `validity_results` per sample (dict with checks) and each sample has `num_rooms`.
- Group by `num_rooms` and compute the mean of each boolean check.

Return structure suggestion:
```python
{
  "validity_by_num_rooms": {
    "4": {"overall": 0.99, "connected": 0.995, ...},
    "5": {...},
    ...
  }
}
```

#### B) Spatial transitivity-by-num_rooms
Your existing `spatial_transitivity(graph_dicts)` returns:
- `transitivity_score`, `h_consistent`, `v_consistent`

Add a stratified wrapper:
- group `graph_dicts` by `num_rooms`
- run `spatial_transitivity` per group

Return:
```python
{
  "transitivity_by_num_rooms": {
    "4": {"transitivity_score": ..., "h_consistent": ..., "v_consistent": ...},
    ...
  }
}
```

#### C) Edge-present rate-by-num_rooms
Define edge-present rate per graph:

- Let `n = num_rooms`
- Let `E_possible = n*(n-1)/2`
- Let `E_present = len(edge_triples)` (assuming `edge_triples` includes only non-`no-edge` relationships)
- `edge_present_rate = E_present / E_possible` (define as 0 if E_possible==0)

Compute this per graph, then average by group:
```python
{
  "edge_present_rate_by_num_rooms": {
    "4": 0.33,
    "5": 0.28,
    ...
  }
}
```

Also compute and log the overall mean edge-present rate.

### 7.3 Wiring + logging
- Add these stratified dicts into the output JSON.
- Additionally log flat keys to wandb for easy plotting, e.g.:
  - `sampler/validity/overall@n=4`
  - `sampler/transitivity/score@n=8`
  - `sampler/edge_present_rate@n=6`

### 7.4 Tests
Add tests that:
- build a tiny set of synthetic graphs with known `num_rooms` and edge counts
- verify stratified aggregation matches expected values

---

## Implementation checklist (quick)

- [ ] Add `bd_gen/eval/denoising_eval.py` + exports  
- [ ] Add JS/TV/W1 helpers in `bd_gen/eval/metrics.py`  
- [ ] Extend `distribution_match` with JS/TV/W1; keep KL  
- [ ] Extend `conditional_edge_kl` → also JS/TV; keep KL  
- [ ] Add `conditional_edge_*_topN` (KL/JS/TV)  
- [ ] Extend `type_conditioned_degree_kl` → also JS/TV; keep KL  
- [ ] Update `scripts/evaluate.py` to:
  - [ ] multi-seed loop (mean±std; no bootstrap)
  - [ ] stratified drill-down metrics
  - [ ] log prefixed scoreboard metrics  
- [ ] Update `configs/eval/default.yaml` with `seeds`, `run_denoising_eval`, `conditional_topN_pairs`, etc.  
- [ ] Update `docs/evaluation.md` accordingly  
- [ ] Add/extend tests  

---

## Notes / constraints

- **No smoothing-alpha in this change.** Do not add Dirichlet smoothing parameters yet.  
- **Do not delete KL.** Keep KL for continuity but do not make it the headline distribution metric.  
- Keep all outputs **JSON-serializable** and stable-keyed (important for long-term comparisons).
