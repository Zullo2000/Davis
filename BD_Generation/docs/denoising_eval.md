# Denoising Evaluation (`bd_gen.eval.denoising_eval`)

Sampler-independent evaluation of the denoiser's predictive quality on held-out data.

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
