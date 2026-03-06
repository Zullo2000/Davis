#!/usr/bin/env python3
"""Plot raw + EMA-smoothed ESS and reward trajectories for Round 5 configs.

Run on jabiru where _samples.pt files live:
    python scripts/plot_ema_trajectories.py

Outputs PNGs to eval_results/loglinear_noise_sc/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Configs to compare ──────────────────────────────────────────────
SCHEDULE = "loglinear_noise_sc"
CONFIGS = {
    "Option A (fresh logits)": "llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optA_K16_a0.01",
    "Option B (protect just-unmasked)": "llada_topp0.9_remdm_confidence_tsw1.0_guided_r5optB_K16_a0.01",
    "No-remask (best)": "llada_topp0.9_no_remask_guided_r4soft_K16_a0.01",
}

# EMA smoothing factors: beta close to 1 = heavy smoothing
BETAS = [0.7, 0.85, 0.95]
BETA_STYLES = {0.7: ("--", 1.5), 0.85: ("-", 2.0), 0.95: ("-.", 2.5)}

# How many samples to plot per config
N_SAMPLES = 4
SEED = 42


def compute_ema(values: list[float], beta: float) -> np.ndarray:
    """Exponential moving average: ema[t] = beta * ema[t-1] + (1-beta) * x[t]."""
    arr = np.array(values, dtype=np.float64)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for t in range(1, len(arr)):
        ema[t] = beta * ema[t - 1] + (1 - beta) * arr[t]
    return ema


def load_per_sample_trajectories(
    model: str, schedule: str = SCHEDULE
) -> list[dict[str, list[float]]]:
    """Load per-sample step-level trajectories from _samples.pt.

    Returns list of dicts, each mapping metric_name -> [value_per_step].
    """
    path = PROJECT_ROOT / "eval_results" / schedule / f"{model}_samples.pt"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping")
        return []

    payload = torch.load(path, map_location="cpu", weights_only=False)
    guidance_stats = payload.get("guidance_stats", {})
    config = payload.get("config", {})
    batch_size = config.get("batch_size", 50)

    trajectories: list[dict[str, list[float]]] = []

    for seed_key, seed_data in sorted(guidance_stats.items()):
        per_sample_batches = seed_data.get("batch_stats_per_sample")
        if per_sample_batches is None:
            # Fall back to batch-averaged stats
            print(f"  No per-sample stats for {seed_key}, using batch-averaged")
            batch_stats = seed_data.get("batch_stats", [])
            for batch_steps in batch_stats:
                traj: dict[str, list[float]] = {
                    "ess": [], "reward_selected": [], "reward_all_candidates": [],
                    "reward_gap": [], "remasking_delta": [],
                }
                for step in batch_steps:
                    traj["ess"].append(step.get("ess", 0.0))
                    traj["reward_selected"].append(step.get("reward_selected", 0.0))
                    traj["reward_all_candidates"].append(step.get("reward_all_candidates", 0.0))
                    traj["reward_gap"].append(step.get("reward_gap", 0.0))
                    delta = step.get("reward_remasking_delta", 0.0)
                    traj["remasking_delta"].append(delta)
                trajectories.append(traj)
            continue

        # Per-sample stats available
        for batch_idx, batch_steps in enumerate(per_sample_batches):
            if not batch_steps:
                continue
            n_samples_in_batch = batch_steps[0]["ess"].shape[0]
            for sample_idx in range(n_samples_in_batch):
                traj = {
                    "ess": [], "reward_selected": [], "reward_all_candidates": [],
                    "reward_gap": [], "remasking_delta": [],
                }
                for step in batch_steps:
                    traj["ess"].append(step["ess"][sample_idx].item())
                    traj["reward_selected"].append(step["reward_selected"][sample_idx].item())
                    traj["reward_all_candidates"].append(step["reward_all_candidates"][sample_idx].item())
                    r_pre = step.get("reward_pre_remask")
                    r_post = step.get("reward_post_remask")
                    if r_pre is not None and r_post is not None:
                        traj["reward_gap"].append(
                            (step["reward_selected"][sample_idx] - step["reward_all_candidates"][sample_idx]).item()
                        )
                        traj["remasking_delta"].append(
                            (r_post[sample_idx] - r_pre[sample_idx]).item()
                        )
                    else:
                        traj["reward_gap"].append(
                            (step["reward_selected"][sample_idx] - step["reward_all_candidates"][sample_idx]).item()
                        )
                        traj["remasking_delta"].append(0.0)
                trajectories.append(traj)

    return trajectories


def plot_ema_comparison(
    configs: dict[str, str],
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> None:
    """Plot raw + EMA trajectories for ESS and reward across configs."""
    metrics = ["ess", "reward_selected"]
    metric_labels = {"ess": "ESS", "reward_selected": "Reward (selected)"}

    n_configs = len(configs)
    fig, axes = plt.subplots(
        len(metrics), n_configs,
        figsize=(6 * n_configs, 4 * len(metrics)),
        squeeze=False,
    )

    rng = np.random.RandomState(seed)

    for col, (label, model) in enumerate(configs.items()):
        print(f"Loading {label}...")
        trajs = load_per_sample_trajectories(model)
        if not trajs:
            for row in range(len(metrics)):
                axes[row, col].text(0.5, 0.5, "No data", transform=axes[row, col].transAxes,
                                     ha="center", va="center", fontsize=14)
                axes[row, col].set_title(f"{label}\n{metric_labels[metrics[row]]}")
            continue

        # Pick random samples
        indices = rng.choice(len(trajs), size=min(n_samples, len(trajs)), replace=False)
        sample_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

        for row, metric in enumerate(metrics):
            ax = axes[row, col]

            for i, idx in enumerate(indices):
                raw = trajs[idx][metric]
                steps = np.arange(len(raw))
                color = sample_colors[i % len(sample_colors)]

                # Raw trajectory (thin, transparent)
                ax.plot(steps, raw, color=color, alpha=0.25, linewidth=0.8,
                        label=f"sample {idx} raw" if i == 0 else None)

                # EMA overlays
                for beta in BETAS:
                    ema = compute_ema(raw, beta)
                    ls, lw = BETA_STYLES[beta]
                    ax.plot(steps, ema, color=color, linestyle=ls, linewidth=lw,
                            alpha=0.85,
                            label=f"EMA β={beta}" if i == 0 else None)

            ax.set_title(f"{label}\n{metric_labels[metric]}", fontsize=11)
            ax.set_xlabel("Denoising step")
            ax.set_ylabel(metric_labels[metric])
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Raw vs EMA-smoothed trajectories (K=16, α=0.01)\n"
        f"EMA: β=0.7 (light), β=0.85 (medium), β=0.95 (heavy)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out_path = PROJECT_ROOT / "eval_results" / SCHEDULE / "ema_trajectory_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_ema_ess_with_thresholds(
    configs: dict[str, str],
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> None:
    """Focused plot: EMA(ESS/K) with candidate threshold lines."""
    beta = 0.85  # Medium smoothing
    K = 16
    thresholds = [0.3, 0.5]

    n_configs = len(configs)
    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 4), squeeze=False)

    rng = np.random.RandomState(seed)
    sample_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for col, (label, model) in enumerate(configs.items()):
        trajs = load_per_sample_trajectories(model)
        ax = axes[0, col]

        if not trajs:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title(label)
            continue

        indices = rng.choice(len(trajs), size=min(n_samples, len(trajs)), replace=False)

        for i, idx in enumerate(indices):
            raw_ess = trajs[idx]["ess"]
            ess_ratio = [e / K for e in raw_ess]
            steps = np.arange(len(raw_ess))
            color = sample_colors[i % len(sample_colors)]

            # Raw ESS/K
            ax.plot(steps, ess_ratio, color=color, alpha=0.2, linewidth=0.8)
            # EMA
            ema = compute_ema(ess_ratio, beta)
            ax.plot(steps, ema, color=color, linewidth=2.0, alpha=0.85,
                    label=f"sample {idx}")

            # Mark where EMA first crosses each threshold
            for th in thresholds:
                crossings = np.where(ema < th)[0]
                if len(crossings) > 0:
                    step_cross = crossings[0]
                    ax.axvline(step_cross, color=color, linestyle=":", alpha=0.4)
                    ax.plot(step_cross, ema[step_cross], "v", color=color,
                            markersize=6, alpha=0.7)

        # Threshold lines
        for th in thresholds:
            ax.axhline(th, color="gray", linestyle="--", alpha=0.5,
                       label=f"threshold={th}" if col == 0 else None)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("ESS / K")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"EMA(ESS/K) with β=0.85 — candidate lock thresholds\n"
        f"▼ = first crossing below threshold",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out_path = PROJECT_ROOT / "eval_results" / SCHEDULE / "ema_ess_thresholds.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=== EMA Trajectory Analysis ===\n")
    plot_ema_comparison(CONFIGS)
    print()
    plot_ema_ess_with_thresholds(CONFIGS)
    print("\nDone!")
