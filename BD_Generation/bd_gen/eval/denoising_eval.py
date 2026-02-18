"""Denoising evaluation: model-quality metrics independent of the sampler.

Evaluates the denoiser's predictive quality on held-out data by masking real
examples at various noise levels and scoring predictions.  These metrics
are sampler-independent and measure the intrinsic quality of the learned
denoiser.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from bd_gen.data.vocab import VocabConfig
from bd_gen.diffusion.forward_process import forward_mask
from bd_gen.diffusion.loss import ELBOLoss
from bd_gen.diffusion.noise_schedule import NoiseSchedule


def denoising_eval(
    model: torch.nn.Module,
    dataloader: DataLoader,
    noise_schedule: NoiseSchedule,
    vocab_config: VocabConfig,
    t_grid: list[float],
    device: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate denoiser accuracy and cross-entropy at multiple noise levels.

    For each *t* in *t_grid*, masks validation examples via ``forward_mask``
    and measures how accurately the model reconstructs the masked positions.

    Args:
        model: Trained denoiser (already in eval mode).
        dataloader: Validation dataloader yielding dicts with ``tokens``
            and ``pad_mask`` keys.
        noise_schedule: The noise schedule used for masking.
        vocab_config: Vocabulary configuration.
        t_grid: List of timestep values in (0, 1) to evaluate.
        device: Device string (``"cpu"`` or ``"cuda"``).
        max_batches: If set, stop after this many batches.

    Returns:
        Dict with keys like ``denoise/acc_node@t=0.1``,
        ``denoise/acc_edge@t=0.1``, ``denoise/ce_node@t=0.1``,
        ``denoise/ce_edge@t=0.1``.
    """
    n_max = vocab_config.n_max
    results: dict[str, float] = {}

    for t_val in t_grid:
        node_correct = 0
        node_total = 0
        edge_correct = 0
        edge_total = 0
        node_ce_sum = 0.0
        edge_ce_sum = 0.0

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            B = tokens.size(0)

            t_tensor = torch.full((B,), t_val, device=device)
            x_t, mask_indicators = forward_mask(
                tokens, pad_mask, t_tensor, noise_schedule, vocab_config,
            )

            with torch.no_grad():
                node_logits, edge_logits = model(x_t, pad_mask, t_tensor)

            # Scoring mask: masked AND non-PAD
            score_mask = mask_indicators & pad_mask

            # --- Node accuracy ---
            node_score_mask = score_mask[:, :n_max]
            if node_score_mask.any():
                node_preds = node_logits.argmax(dim=-1)  # (B, n_max)
                node_targets = tokens[:, :n_max]
                node_correct += int(
                    (node_preds[node_score_mask] == node_targets[node_score_mask])
                    .sum().item()
                )
                node_total += int(node_score_mask.sum().item())

                # Cross-entropy
                node_ce = torch.nn.functional.cross_entropy(
                    node_logits[node_score_mask],
                    node_targets[node_score_mask],
                    reduction="sum",
                )
                node_ce_sum += node_ce.item()

            # --- Edge accuracy ---
            edge_score_mask = score_mask[:, n_max:]
            if edge_score_mask.any():
                edge_preds = edge_logits.argmax(dim=-1)  # (B, n_edges)
                edge_targets = tokens[:, n_max:]
                edge_correct += int(
                    (edge_preds[edge_score_mask] == edge_targets[edge_score_mask])
                    .sum().item()
                )
                edge_total += int(edge_score_mask.sum().item())

                # Cross-entropy
                edge_ce = torch.nn.functional.cross_entropy(
                    edge_logits[edge_score_mask],
                    edge_targets[edge_score_mask],
                    reduction="sum",
                )
                edge_ce_sum += edge_ce.item()

        t_str = f"{t_val:.1f}"
        results[f"denoise/acc_node@t={t_str}"] = (
            node_correct / node_total if node_total > 0 else 0.0
        )
        results[f"denoise/acc_edge@t={t_str}"] = (
            edge_correct / edge_total if edge_total > 0 else 0.0
        )
        results[f"denoise/ce_node@t={t_str}"] = (
            node_ce_sum / node_total if node_total > 0 else 0.0
        )
        results[f"denoise/ce_edge@t={t_str}"] = (
            edge_ce_sum / edge_total if edge_total > 0 else 0.0
        )

    return results


def denoising_val_elbo(
    model: torch.nn.Module,
    dataloader: DataLoader,
    noise_schedule: NoiseSchedule,
    elbo_loss: ELBOLoss,
    vocab_config: VocabConfig,
    device: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute average ELBO loss on a validation set.

    Args:
        model: Trained denoiser (already in eval mode).
        dataloader: Validation dataloader.
        noise_schedule: The noise schedule.
        elbo_loss: Pre-constructed ``ELBOLoss`` instance.
        vocab_config: Vocabulary configuration.
        device: Device string.
        max_batches: If set, stop after this many batches.

    Returns:
        Dict with ``denoise/val_elbo`` (average ELBO loss).
    """
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        tokens = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        B = tokens.size(0)

        t = torch.rand(B, device=device)
        x_t, mask_indicators = forward_mask(
            tokens, pad_mask, t, noise_schedule, vocab_config,
        )

        with torch.no_grad():
            node_logits, edge_logits = model(x_t, pad_mask, t)
            loss = elbo_loss(
                node_logits, edge_logits,
                tokens, x_t, pad_mask, mask_indicators,
                t, noise_schedule,
            )

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return {"denoise/val_elbo": avg_loss}
