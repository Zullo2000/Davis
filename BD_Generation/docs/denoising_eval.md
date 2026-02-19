# Denoising Evaluation (`bd_gen.eval.denoising_eval`)

Sampler-independent evaluation of the denoiser's predictive quality on held-out data.

## Why We Use It

The denoising eval measures how well the neural network (the denoiser) can predict the original clean tokens from a partially masked input, without ever running the full sampling loop. It is a direct probe of the model's core skill.

When we generate full samples, quality depends on two things: how good the denoiser's predictions are, and the sampling strategy (number of steps, random vs. confidence-based unmasking, remasking, etc.). If generated samples look bad, we cannot tell which component is at fault. Denoising eval removes the sampler from the equation by probing the model directly on corrupted validation data.

This gives us:

- **Apples-to-apples model comparison** -- different architectures, hyperparameters, or training runs are evaluated on the same fixed corruption levels, regardless of what sampler is later paired with them.
- **Diagnostic profile across noise levels** -- a model might perform well at `t=0.1` (few masks, lots of context) but collapse at `t=0.9` (nearly everything masked). This per-timestep profile reveals *where* the model struggles, which is invisible in a single aggregate metric.
- **Fast and cheap** -- only forward passes on validation data, no iterative sampling loop. Can be run every N training steps to track progress.
- **Overfitting detection** -- `denoise/val_elbo` mirrors the training loss on held-out data, so divergence between train and val ELBO signals overfitting without running expensive generation.

In short: generation-based metrics (graph structure similarity, MMD) tell us if the full system works end-to-end, while denoising eval tells us if the model itself has learned the data distribution -- and at which difficulty levels it succeeds or fails.

## Functions

### `denoising_eval(model, dataloader, noise_schedule, vocab_config, t_grid, device, max_batches=None)`

For each timestep `t` in `t_grid`:
1. Masks validation examples via `forward_mask(x0, pad_mask, t, noise_schedule)`
2. Runs model forward: `model(x_t, pad_mask, t) -> (node_logits, edge_logits)`
3. Computes accuracy and cross-entropy at masked non-PAD positions

**Scoring mask:** `mask_indicators & pad_mask` (same logic as `ELBOLoss`).

**Returns** dict with keys:
- `denoise/acc_node@t={t}` -- node prediction accuracy at noise level t
- `denoise/acc_edge@t={t}` -- edge prediction accuracy at noise level t
- `denoise/ce_node@t={t}` -- node cross-entropy at noise level t
- `denoise/ce_edge@t={t}` -- edge cross-entropy at noise level t

### `denoising_val_elbo(model, dataloader, noise_schedule, elbo_loss, vocab_config, device, max_batches=None)`

Computes average ELBO loss over a validation set with random `t ~ U(0,1)`.

**Returns** dict with key `denoise/val_elbo`.

## Config keys

```yaml
eval:
  run_denoising_eval: true
  denoising_t_grid: [0.1, 0.3, 0.5, 0.7, 0.9]
  denoising_max_batches: 50
```
