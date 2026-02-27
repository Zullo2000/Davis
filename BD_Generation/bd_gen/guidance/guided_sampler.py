"""SVDD-style guided sampling with K-candidate reweighting.

At each denoising step, K candidate transitions are generated from the
base model's proposal distribution, scored by a RewardComposer, and
resampled via importance weights w_k = softmax(reward / alpha).  The
selected winner is then (optionally) remasked.

The model call is shared across K candidates (1x cost); only the
stochastic transition + scoring is K-fold.

Reference: Li et al., "Derivative-Free Guidance in Continuous and
Discrete Diffusion Models with Soft Value-Based Decoding"
(arXiv:2408.08252).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from bd_gen.data.tokenizer import detokenize
from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    VocabConfig,
)
from bd_gen.diffusion.noise_schedule import NoiseSchedule
from bd_gen.diffusion.sampling import _clamp_pad, _single_step_remask, _single_step_unmask
from bd_gen.guidance.reward import RewardComposer
from bd_gen.guidance.soft_violations import (
    build_effective_probs,
    hard_decode_x0,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class GuidanceStatsStep(TypedDict):
    """Diagnostics for a single denoising step (batch-averaged scalars)."""

    ess: float
    max_weight: float
    weight_entropy: float
    reward_selected: float
    reward_all_candidates: float
    reward_gap: float
    reward_pre_remask: float
    reward_post_remask: float
    reward_remasking_delta: float
    positions_remasked: float
    violations: dict[str, float]


class GuidanceStatsStepPerSample(TypedDict):
    """Per-sample diagnostics for a single step."""

    ess: Tensor  # (B,)
    reward_selected: Tensor  # (B,)
    reward_all_candidates: Tensor  # (B,)
    reward_pre_remask: Tensor  # (B,)
    reward_post_remask: Tensor  # (B,)
    violations: dict[str, Tensor]  # name -> (B,)


@dataclass
class GuidanceStats:
    """Complete diagnostics for a guided generation run."""

    steps: list[GuidanceStatsStep] = field(default_factory=list)
    steps_per_sample: list[GuidanceStatsStepPerSample] = field(default_factory=list)
    final_satisfaction: dict[str, float] = field(default_factory=dict)
    final_mean_violation: dict[str, float] = field(default_factory=dict)
    final_mean_violation_when_failed: dict[str, float] = field(default_factory=dict)
    final_violation_histograms: dict[str, dict[str, int]] = field(default_factory=dict)
    satisfaction_overall: float = 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_candidates_soft(
    candidates: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
    composer: RewardComposer,
    K: int,
    B: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score K*B candidates using soft (posterior expectation) violations.

    Args:
        candidates: (K, B, SEQ_LEN) candidate tokens after unmasking.
        node_logits: (B, n_max, NODE_V) shared logits from single model call.
        edge_logits: (B, n_edges, EDGE_V) shared logits.
        pad_mask: (B, SEQ_LEN) bool.
        vocab_config: Vocabulary configuration.
        composer: RewardComposer instance.
        K: Number of candidates.
        B: Batch size.

    Returns:
        (rewards, per_constraint_violations) where rewards is (K, B) float64
        and per_constraint_violations is {name: (K, B) float64}.
    """
    constraint_names = [c.name for c in composer.constraints]
    rewards = torch.zeros(K, B, dtype=torch.float64)
    per_violations: dict[str, Tensor] = {
        name: torch.zeros(K, B, dtype=torch.float64) for name in constraint_names
    }

    for k in range(K):
        for b in range(B):
            node_probs, edge_probs = build_effective_probs(
                candidates[k, b],
                node_logits[b],
                edge_logits[b],
                pad_mask[b],
                vocab_config,
            )
            reward, violations = composer.compute_reward_soft(
                node_probs, edge_probs, pad_mask[b], vocab_config,
            )
            rewards[k, b] = reward
            for name in constraint_names:
                if name in violations:
                    per_violations[name][k, b] = violations[name]

    return rewards, per_violations


def _score_candidates_hard(
    candidates: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
    composer: RewardComposer,
    K: int,
    B: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score K*B candidates using hard (argmax decode) violations.

    Each candidate is hard-decoded to x̂_0, detokenized to graph_dict,
    then scored via hard_violation().
    """
    constraint_names = [c.name for c in composer.constraints]
    rewards = torch.zeros(K, B, dtype=torch.float64)
    per_violations: dict[str, Tensor] = {
        name: torch.zeros(K, B, dtype=torch.float64) for name in constraint_names
    }

    for k in range(K):
        for b in range(B):
            x0 = hard_decode_x0(
                candidates[k, b],
                node_logits[b],
                edge_logits[b],
                pad_mask[b],
                vocab_config,
            )
            graph_dict = detokenize(x0, pad_mask[b], vocab_config)
            reward, results = composer.compute_reward_hard(graph_dict)
            rewards[k, b] = reward
            for name in constraint_names:
                if name in results:
                    per_violations[name][k, b] = results[name].violation

    return rewards, per_violations


def _score_single_soft(
    x_t: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
    composer: RewardComposer,
    B: int,
) -> Tensor:
    """Score B samples in soft mode. Returns (B,) float64 rewards."""
    rewards = torch.zeros(B, dtype=torch.float64)
    for b in range(B):
        node_probs, edge_probs = build_effective_probs(
            x_t[b], node_logits[b], edge_logits[b], pad_mask[b], vocab_config,
        )
        r, _ = composer.compute_reward_soft(
            node_probs, edge_probs, pad_mask[b], vocab_config,
        )
        rewards[b] = r
    return rewards


def _score_single_hard(
    x_t: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
    composer: RewardComposer,
    B: int,
) -> Tensor:
    """Score B samples in hard mode. Returns (B,) float64 rewards."""
    rewards = torch.zeros(B, dtype=torch.float64)
    for b in range(B):
        x0 = hard_decode_x0(
            x_t[b], node_logits[b], edge_logits[b], pad_mask[b], vocab_config,
        )
        graph_dict = detokenize(x0, pad_mask[b], vocab_config)
        r, _ = composer.compute_reward_hard(graph_dict)
        rewards[b] = r
    return rewards


# ---------------------------------------------------------------------------
# Main guided sampling function
# ---------------------------------------------------------------------------


@torch.no_grad()
def guided_sample(
    model: torch.nn.Module,
    noise_schedule: NoiseSchedule,
    vocab_config: VocabConfig,
    reward_composer: RewardComposer,
    batch_size: int,
    num_steps: int,
    num_candidates: int = 8,
    guidance_alpha: float = 1.0,
    temperature: float = 0.0,
    top_p: float | None = None,
    unmasking_mode: str = "random",
    t_switch: float = 1.0,
    fixed_tokens: Tensor | None = None,
    fixed_mask: Tensor | None = None,
    remasking_fn: Callable | None = None,
    rate_network: torch.nn.Module | None = None,
    num_rooms_distribution: Tensor | None = None,
    fixed_num_rooms: int | None = None,
    device: str = "cpu",
) -> tuple[Tensor, GuidanceStats]:
    """SVDD-style guided sampling with K-candidate reweighting.

    At each denoising step:
      1. Single model call (shared across K candidates).
      2. Expand x_t to K*B candidates.
      3. Run _single_step_unmask on K*B (no remasking).
      4. Score each candidate via RewardComposer.
      5. Importance weights: softmax(reward / alpha) per sample over K.
      6. Resample: select winner per sample via multinomial(weights).
      7. Apply _single_step_remask on winner only.
      8. Record diagnostics.

    Args:
        model: BDDenoiser or compatible.
        noise_schedule: NoiseSchedule for alpha(t).
        vocab_config: Vocabulary configuration.
        reward_composer: RewardComposer with constraints and settings.
        batch_size: Number of samples (B).
        num_steps: Number of denoising steps (N).
        num_candidates: K candidates per sample.
        guidance_alpha: Temperature for importance weights (higher = weaker).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        unmasking_mode: "random" or "llada".
        t_switch: Remasking activation threshold.
        fixed_tokens: Inpainting tokens.
        fixed_mask: Inpainting mask.
        remasking_fn: Optional ReMDM remasking function.
        rate_network: Optional v2 rate network.
        num_rooms_distribution: Room count distribution.
        fixed_num_rooms: Fixed room count.
        device: Target device string.

    Returns:
        (final_tokens, GuidanceStats) where final_tokens is (B, SEQ_LEN).
    """
    K = num_candidates
    B = batch_size
    n_max = vocab_config.n_max
    seq_len = vocab_config.seq_len
    reward_mode = reward_composer.reward_mode
    constraint_names = [c.name for c in reward_composer.constraints]

    stats = GuidanceStats()

    # --- Step 1: Determine num_rooms per sample ---
    if fixed_num_rooms is not None:
        num_rooms_list = [fixed_num_rooms] * B
    elif num_rooms_distribution is not None:
        room_indices = torch.multinomial(
            num_rooms_distribution, B, replacement=True,
        )
        num_rooms_list = (room_indices + 1).tolist()
    else:
        num_rooms_list = torch.randint(1, n_max + 1, (B,)).tolist()

    # --- Step 2: Compute pad_mask ---
    pad_mask = torch.stack(
        [vocab_config.compute_pad_mask(nr) for nr in num_rooms_list]
    ).to(device)  # (B, SEQ_LEN)

    # --- Step 3: Initialize x_t (fully masked) ---
    x_t = torch.zeros(B, seq_len, dtype=torch.long, device=device)
    for b in range(B):
        nr = num_rooms_list[b]
        x_t[b, :nr] = NODE_MASK_IDX
        x_t[b, nr:n_max] = NODE_PAD_IDX

    edge_pad_mask = pad_mask[:, n_max:]
    x_t[:, n_max:] = torch.where(
        edge_pad_mask,
        torch.tensor(EDGE_MASK_IDX, dtype=torch.long, device=device),
        torch.tensor(EDGE_PAD_IDX, dtype=torch.long, device=device),
    )

    # --- Step 4: Reverse loop with K-candidate reweighting ---
    for i in range(num_steps - 1, -1, -1):
        t_now = (i + 1) / num_steps
        t_next = i / num_steps

        t_tensor = torch.full((B,), t_now, dtype=torch.float32, device=device)

        # 4a. Model call (shared)
        node_logits, edge_logits = model(x_t, pad_mask, t_tensor)

        # 4b. Compute p_unmask
        if rate_network is not None:
            alpha_now_64 = rate_network(t_tensor, pad_mask).double()
            t_next_tensor = torch.full(
                (B,), t_next, dtype=torch.float32, device=device,
            )
            alpha_next_64 = rate_network(t_next_tensor, pad_mask).double()
            p_unmask = (alpha_next_64 - alpha_now_64) / (1.0 - alpha_now_64 + 1e-8)
            p_unmask = torch.clamp(p_unmask, min=0.0, max=1.0).float()
        else:
            alpha_now_64 = noise_schedule.alpha(t_tensor.double())
            alpha_next_64 = noise_schedule.alpha(
                torch.full((B,), t_next, dtype=torch.float64, device=device),
            )
            p_unmask = (alpha_next_64 - alpha_now_64) / (1.0 - alpha_now_64 + 1e-8)
            p_unmask = torch.clamp(p_unmask, min=0.0, max=1.0).float()
            p_unmask = p_unmask.unsqueeze(1)  # (B, 1)

        # 4c. Expand to K*B candidates
        x_t_expanded = x_t.repeat_interleave(K, dim=0)  # (K*B, SEQ)
        node_logits_expanded = node_logits.repeat_interleave(K, dim=0)
        edge_logits_expanded = edge_logits.repeat_interleave(K, dim=0)
        pad_expanded = pad_mask.repeat_interleave(K, dim=0)
        p_expanded = p_unmask.repeat_interleave(K, dim=0)

        # Expand inpainting tensors if present
        fixed_tokens_exp = (
            fixed_tokens.repeat_interleave(K, dim=0) if fixed_tokens is not None else None
        )
        fixed_mask_exp = (
            fixed_mask.repeat_interleave(K, dim=0) if fixed_mask is not None else None
        )

        # 4d. Unmask K*B candidates (no remasking)
        candidates_flat = _single_step_unmask(
            x_t_expanded, node_logits_expanded, edge_logits_expanded,
            pad_expanded, p_expanded,
            i, num_steps, n_max, top_p, temperature, unmasking_mode,
            torch.device(device), fixed_tokens_exp, fixed_mask_exp,
        )  # (K*B, SEQ)

        # 4e. Reshape to (K, B, SEQ)
        candidates = candidates_flat.view(K, B, seq_len)

        # 4f. Score candidates
        if reward_mode == "soft":
            rewards, per_violations = _score_candidates_soft(
                candidates, node_logits, edge_logits, pad_mask,
                vocab_config, reward_composer, K, B,
            )
        else:
            rewards, per_violations = _score_candidates_hard(
                candidates, node_logits, edge_logits, pad_mask,
                vocab_config, reward_composer, K, B,
            )
        # rewards: (K, B), per_violations: {name: (K, B)}

        # 4g. Importance weights
        if len(reward_composer.constraints) == 0:
            # No constraints: uniform weights (all rewards are 0)
            weights = torch.ones(K, B, dtype=torch.float64) / K
        else:
            log_weights = rewards / guidance_alpha  # (K, B) float64
            weights = F.softmax(log_weights, dim=0)  # (K, B), sums to 1 per sample

        # 4h. Resample: select winner per sample
        # weights.T is (B, K)
        selected_k = torch.multinomial(
            weights.T.float(), num_samples=1,
        ).squeeze(-1)  # (B,)
        x_t = candidates[selected_k, torch.arange(B)]  # (B, SEQ)

        # Record pre-remask reward
        reward_pre = rewards[selected_k, torch.arange(B)]  # (B,)

        # 4i. Remask winner only
        x_t = _single_step_remask(
            x_t, remasking_fn, t_now, t_next, t_switch, i,
            pad_mask, node_logits, edge_logits,
        )

        # 4j. Compute post-remask reward
        if remasking_fn is not None and i > 0 and t_now < t_switch:
            if reward_mode == "soft":
                reward_post = _score_single_soft(
                    x_t, node_logits, edge_logits, pad_mask,
                    vocab_config, reward_composer, B,
                )
            else:
                reward_post = _score_single_hard(
                    x_t, node_logits, edge_logits, pad_mask,
                    vocab_config, reward_composer, B,
                )
        else:
            reward_post = reward_pre.clone()

        # Count positions remasked
        # Compare winner before remask vs after. We need to count changes.
        # Since x_t was modified in-place by _single_step_remask, we track
        # positions_remasked via the MASK token count difference.
        is_node_mask_post = x_t[:, :n_max] == NODE_MASK_IDX
        is_edge_mask_post = x_t[:, n_max:] == EDGE_MASK_IDX
        is_mask_post = torch.cat([is_node_mask_post, is_edge_mask_post], dim=1)

        winner_before = candidates[selected_k, torch.arange(B)]
        is_node_mask_pre = winner_before[:, :n_max] == NODE_MASK_IDX
        is_edge_mask_pre = winner_before[:, n_max:] == EDGE_MASK_IDX
        is_mask_pre = torch.cat([is_node_mask_pre, is_edge_mask_pre], dim=1)
        positions_remasked = (is_mask_post & ~is_mask_pre).float().sum(dim=1)  # (B,)

        # 4k. Diagnostics
        ess_per_sample = 1.0 / (weights ** 2).sum(dim=0)  # (B,)
        max_w_per_sample = weights.max(dim=0).values  # (B,)

        # Normalized weight entropy: H(w)/log(K)
        log_w = weights.log().clamp(min=-30.0)
        w_entropy_raw = -(weights * log_w).sum(dim=0)  # (B,)
        if K > 1:
            w_entropy_normalized = w_entropy_raw / math.log(K)
        else:
            w_entropy_normalized = torch.ones(B, dtype=torch.float64)

        reward_selected_per = rewards[selected_k, torch.arange(B)]  # (B,)
        reward_all_per = rewards.mean(dim=0)  # (B,)
        reward_gap_per = reward_selected_per - reward_all_per  # (B,)

        # Per-constraint violations for selected candidate
        violations_selected: dict[str, float] = {}
        violations_selected_per: dict[str, Tensor] = {}
        for name in constraint_names:
            v = per_violations[name][selected_k, torch.arange(B)]  # (B,)
            violations_selected[name] = v.mean().item()
            violations_selected_per[name] = v

        step_stats = GuidanceStatsStep(
            ess=ess_per_sample.mean().item(),
            max_weight=max_w_per_sample.mean().item(),
            weight_entropy=w_entropy_normalized.mean().item(),
            reward_selected=reward_selected_per.mean().item(),
            reward_all_candidates=reward_all_per.mean().item(),
            reward_gap=reward_gap_per.mean().item(),
            reward_pre_remask=reward_pre.mean().item(),
            reward_post_remask=reward_post.mean().item(),
            reward_remasking_delta=(reward_post - reward_pre).mean().item(),
            positions_remasked=positions_remasked.mean().item(),
            violations=violations_selected,
        )
        stats.steps.append(step_stats)

        step_per_sample = GuidanceStatsStepPerSample(
            ess=ess_per_sample,
            reward_selected=reward_selected_per,
            reward_all_candidates=reward_all_per,
            reward_pre_remask=reward_pre,
            reward_post_remask=reward_post,
            violations=violations_selected_per,
        )
        stats.steps_per_sample.append(step_per_sample)

    # --- Step 5: Final cleanup ---
    remaining_node_mask = x_t[:, :n_max] == NODE_MASK_IDX
    remaining_edge_mask = x_t[:, n_max:] == EDGE_MASK_IDX
    has_remaining = remaining_node_mask.any() or remaining_edge_mask.any()

    if has_remaining:
        t_final = torch.full((B,), 1e-5, dtype=torch.float32, device=device)
        node_logits, edge_logits = model(x_t, pad_mask, t_final)
        node_pred = node_logits.argmax(dim=-1)
        edge_pred = edge_logits.argmax(dim=-1)
        x_t[:, :n_max] = torch.where(
            remaining_node_mask, node_pred, x_t[:, :n_max],
        )
        x_t[:, n_max:] = torch.where(
            remaining_edge_mask, edge_pred, x_t[:, n_max:],
        )

    x_t = _clamp_pad(x_t, pad_mask, n_max)

    # --- Step 6: Final per-constraint hard evaluation ---
    if len(reward_composer.constraints) > 0:
        all_satisfied_count = 0
        for b in range(B):
            graph_dict = detokenize(x_t[b], pad_mask[b], vocab_config)
            all_satisfied = True
            for constraint in reward_composer.constraints:
                result = constraint.hard_violation(graph_dict)
                name = constraint.name

                # Accumulate satisfaction
                if name not in stats.final_satisfaction:
                    stats.final_satisfaction[name] = 0.0
                    stats.final_mean_violation[name] = 0.0
                    stats.final_mean_violation_when_failed[name] = 0.0
                    stats.final_violation_histograms[name] = {
                        "0": 0, "1": 0, "2": 0, "3+": 0,
                    }

                if result.satisfied:
                    stats.final_satisfaction[name] += 1.0
                else:
                    all_satisfied = False

                stats.final_mean_violation[name] += result.violation

                # Histogram bin
                v_int = int(result.violation)
                if v_int == 0:
                    stats.final_violation_histograms[name]["0"] += 1
                elif v_int == 1:
                    stats.final_violation_histograms[name]["1"] += 1
                elif v_int == 2:
                    stats.final_violation_histograms[name]["2"] += 1
                else:
                    stats.final_violation_histograms[name]["3+"] += 1

            if all_satisfied:
                all_satisfied_count += 1

        # Normalize
        for name in stats.final_satisfaction:
            stats.final_satisfaction[name] /= B
            total_violation = stats.final_mean_violation[name]
            stats.final_mean_violation[name] = total_violation / B
            num_failed = B - int(stats.final_satisfaction[name] * B + 0.5)
            if num_failed > 0:
                stats.final_mean_violation_when_failed[name] = (
                    total_violation / num_failed
                )

        stats.satisfaction_overall = all_satisfied_count / B

    return x_t, stats
