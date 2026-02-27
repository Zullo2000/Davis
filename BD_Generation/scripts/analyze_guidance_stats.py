"""Analyze GuidanceStats from guided sample files.

Reads ``_samples.pt`` files produced by ``generate_guided.py`` and reports
trajectory diagnostics: ESS curves, reward trajectories, violation resolution,
remasking cooperation, and final satisfaction rates.

Usage::

    # Analyze a single guided model (text summary)
    python scripts/analyze_guidance_stats.py \
        --schedule loglinear_noise_sc \
        --model llada_topp0.9_no_remask_guided_basic_K16_a1.0

    # Distribution plots (histograms over all 5000 samples)
    python scripts/analyze_guidance_stats.py \
        --schedule loglinear_noise_sc \
        --model llada_topp0.9_no_remask_guided_basic_K16_a1.0 \
        --plot-distributions

    # Time-evolution trajectories for 2 individual samples
    python scripts/analyze_guidance_stats.py \
        --schedule loglinear_noise_sc \
        --model llada_topp0.9_no_remask_guided_basic_K16_a1.0 \
        --plot-trajectories --traj-seed 42

    # Compare soft vs hard for all variants
    python scripts/analyze_guidance_stats.py \
        --schedule loglinear_noise_sc learned_noise_sc \
        --compare-modes

    # Export ESS and reward curves as TSV
    python scripts/analyze_guidance_stats.py \
        --schedule loglinear_noise_sc \
        --model llada_topp0.9_no_remask_guided_basic_K16_a1.0 \
        --export-tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_guidance_stats(
    schedule: str, model: str
) -> tuple[dict, list[dict]]:
    """Load samples.pt and extract guidance_stats + config."""
    path = _PROJECT_ROOT / "eval_results" / schedule / f"{model}_samples.pt"
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    guidance_stats = payload.get("guidance_stats", {})

    # Flatten across seeds — collect all step lists
    all_steps: list[list[dict]] = []
    for seed_key, seed_data in guidance_stats.items():
        batch_stats = seed_data.get("batch_stats", [])
        for batch_steps in batch_stats:
            all_steps.append(batch_steps)

    return config, all_steps


def _load_per_sample_stats(
    schedule: str, model: str
) -> tuple[dict, dict[str, list[list[dict]]], list[int]]:
    """Load per-sample stats keyed by seed.

    Returns
    -------
    config : dict
    per_seed : dict[seed_str -> list[list[dict]]]
        per_seed[seed][batch_idx][step_idx] has tensors of shape (B,).
    seeds : list[int]
    """
    path = _PROJECT_ROOT / "eval_results" / schedule / f"{model}_samples.pt"
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    guidance_stats = payload.get("guidance_stats", {})
    seeds = payload.get("seeds", [])

    per_seed: dict[str, list[list[dict]]] = {}
    for seed_key, seed_data in guidance_stats.items():
        ps = seed_data.get("batch_stats_per_sample")
        if ps is None:
            print(
                f"Error: per-sample stats not found for {model}. "
                "Re-generate with updated generate_guided.py."
            )
            sys.exit(1)
        per_seed[seed_key] = ps

    return config, per_seed, seeds


def _average_steps(all_steps: list[list[dict]]) -> list[dict]:
    """Average step-level stats across batches/seeds."""
    if not all_steps:
        return []

    n_steps = len(all_steps[0])
    averaged = []

    for t in range(n_steps):
        step_data = [batch[t] for batch in all_steps if t < len(batch)]
        if not step_data:
            break

        avg = {}
        scalar_keys = [
            "ess", "max_weight", "weight_entropy",
            "reward_selected", "reward_all_candidates", "reward_gap",
            "reward_pre_remask", "reward_post_remask",
            "reward_remasking_delta", "positions_remasked",
        ]
        for key in scalar_keys:
            vals = [s[key] for s in step_data if key in s]
            avg[key] = sum(vals) / len(vals) if vals else 0.0

        # Average violations per constraint
        violation_keys = set()
        for s in step_data:
            if "violations" in s:
                violation_keys.update(s["violations"].keys())
        avg["violations"] = {}
        for vk in sorted(violation_keys):
            vals = [s["violations"][vk] for s in step_data if vk in s.get("violations", {})]
            avg["violations"][vk] = sum(vals) / len(vals) if vals else 0.0

        averaged.append(avg)

    return averaged


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_single(schedule: str, model: str, export_tsv: bool = False) -> None:
    """Analyze and print diagnostics for a single guided model."""
    config, all_steps = _load_guidance_stats(schedule, model)
    steps = _average_steps(all_steps)

    if not steps:
        print(f"No guidance stats found for {model}")
        return

    n_steps = len(steps)

    print(f"\n{'=' * 70}")
    print(f"  Model: {model}")
    print(f"  Schedule: {schedule}")
    print(f"  Config: K={config.get('num_candidates')}, "
          f"\u03b1={config.get('guidance_alpha')}, "
          f"mode={config.get('reward_mode')}, "
          f"phi={config.get('phi')}")
    print(f"  Steps: {n_steps}")
    print(f"  Batches/seeds analyzed: {len(all_steps)}")
    print(f"{'=' * 70}")

    # --- ESS summary ---
    ess_vals = [s["ess"] for s in steps]
    print(f"\n  ESS trajectory:")
    print(f"    Mean:  {sum(ess_vals)/len(ess_vals):.2f}")
    print(f"    Min:   {min(ess_vals):.2f} (step {ess_vals.index(min(ess_vals))})")
    print(f"    Max:   {max(ess_vals):.2f} (step {ess_vals.index(max(ess_vals))})")
    ess_below_2 = sum(1 for e in ess_vals if e < 2.0)
    if ess_below_2 > 0:
        print(f"    WARNING: ESS < 2 at {ess_below_2}/{n_steps} steps (\u03b1 may be too aggressive)")

    # --- Max weight summary ---
    max_w = [s["max_weight"] for s in steps]
    print(f"\n  Max weight (single candidate dominance):")
    print(f"    Mean: {sum(max_w)/len(max_w):.3f}")
    print(f"    Max:  {max(max_w):.3f} (step {max_w.index(max(max_w))})")
    if max(max_w) > 0.95:
        print(f"    WARNING: max_weight > 0.95 \u2014 diversity may be destroyed")

    # --- Reward gap ---
    gaps = [s["reward_gap"] for s in steps]
    print(f"\n  Reward gap (guidance steering signal):")
    print(f"    Mean: {sum(gaps)/len(gaps):.4f}")
    zero_gaps = sum(1 for g in gaps if abs(g) < 1e-6)
    if zero_gaps > n_steps // 2:
        print(f"    WARNING: reward_gap \u2248 0 at {zero_gaps}/{n_steps} steps \u2014 guidance may be ineffective")

    # --- Remasking delta ---
    deltas = [s["reward_remasking_delta"] for s in steps]
    neg_deltas = sum(1 for d in deltas if d < 0)
    print(f"\n  Remasking delta (positive = cooperative):")
    print(f"    Mean:     {sum(deltas)/len(deltas):.4f}")
    print(f"    Negative: {neg_deltas}/{n_steps} steps")
    if neg_deltas > n_steps * 0.5:
        print(f"    WARNING: remasking fights guidance at >{neg_deltas}/{n_steps} steps")

    # --- Per-constraint violation trajectories ---
    if steps[0].get("violations"):
        constraint_names = sorted(steps[0]["violations"].keys())
        print(f"\n  Per-constraint violation trajectories:")
        for name in constraint_names:
            vals = [s["violations"].get(name, 0.0) for s in steps]
            print(f"    {name}:")
            print(f"      Start: {vals[0]:.4f}  End: {vals[-1]:.4f}  "
                  f"Min: {min(vals):.4f}")
            # When does it first drop below 0.1?
            resolved = next((t for t, v in enumerate(vals) if v < 0.1), None)
            if resolved is not None:
                print(f"      Resolved (<0.1) at step {resolved}/{n_steps} "
                      f"({100*resolved/n_steps:.0f}%)")

    # --- Key sampled timesteps ---
    print(f"\n  Trajectory snapshot (every {max(1, n_steps//5)} steps):")
    print(f"    {'Step':>6} {'ESS':>6} {'MaxW':>6} {'R_gap':>8} {'R_sel':>8} "
          f"{'R_delta':>8} {'Remask':>7}")
    for t in range(0, n_steps, max(1, n_steps // 5)):
        s = steps[t]
        print(f"    {t:>6} {s['ess']:>6.2f} {s['max_weight']:>6.3f} "
              f"{s['reward_gap']:>8.4f} {s['reward_selected']:>8.4f} "
              f"{s['reward_remasking_delta']:>8.4f} {s['positions_remasked']:>7.1f}")
    # Always print last step
    if (n_steps - 1) % max(1, n_steps // 5) != 0:
        s = steps[-1]
        print(f"    {n_steps-1:>6} {s['ess']:>6.2f} {s['max_weight']:>6.3f} "
              f"{s['reward_gap']:>8.4f} {s['reward_selected']:>8.4f} "
              f"{s['reward_remasking_delta']:>8.4f} {s['positions_remasked']:>7.1f}")

    # --- Export TSV ---
    if export_tsv:
        tsv_path = (_PROJECT_ROOT / "eval_results" / schedule
                    / f"{model}_guidance_trajectory.tsv")
        with open(tsv_path, "w") as f:
            headers = ["step", "ess", "max_weight", "weight_entropy",
                       "reward_selected", "reward_all_candidates", "reward_gap",
                       "reward_remasking_delta", "positions_remasked"]
            if steps[0].get("violations"):
                for name in sorted(steps[0]["violations"].keys()):
                    headers.append(f"violation_{name}")
            f.write("\t".join(headers) + "\n")
            for t, s in enumerate(steps):
                row = [str(t), f"{s['ess']:.4f}", f"{s['max_weight']:.4f}",
                       f"{s['weight_entropy']:.4f}",
                       f"{s['reward_selected']:.4f}",
                       f"{s['reward_all_candidates']:.4f}",
                       f"{s['reward_gap']:.4f}",
                       f"{s['reward_remasking_delta']:.4f}",
                       f"{s['positions_remasked']:.1f}"]
                if s.get("violations"):
                    for name in sorted(s["violations"].keys()):
                        row.append(f"{s['violations'].get(name, 0.0):.4f}")
                f.write("\t".join(row) + "\n")
        print(f"\n  TSV exported: {tsv_path}")


# ---------------------------------------------------------------------------
# Distribution plots (histograms over all samples at final step)
# ---------------------------------------------------------------------------


def _collect_final_step_values(
    per_seed: dict[str, list[list[dict]]],
) -> dict[str, list[float]]:
    """Collect per-sample values at the final denoising step.

    Returns dict mapping metric_name -> list of floats (one per sample,
    concatenated across all seeds and batches).
    """
    result: dict[str, list[float]] = {}

    for _seed_key, batches in per_seed.items():
        for batch_steps in batches:
            if not batch_steps:
                continue
            final = batch_steps[-1]  # last denoising step
            for key in ["ess", "reward_selected", "reward_all_candidates",
                        "reward_pre_remask", "reward_post_remask"]:
                if key not in final:
                    continue
                vals = final[key]
                if isinstance(vals, torch.Tensor):
                    vals = vals.tolist()
                result.setdefault(key, []).extend(
                    vals if isinstance(vals, list) else [vals]
                )
            # Derived: reward_gap per sample
            if "reward_selected" in final and "reward_all_candidates" in final:
                sel = final["reward_selected"]
                allc = final["reward_all_candidates"]
                if isinstance(sel, torch.Tensor) and isinstance(allc, torch.Tensor):
                    gap = (sel - allc).tolist()
                    result.setdefault("reward_gap", []).extend(gap)
            # Derived: remasking_delta per sample
            if "reward_post_remask" in final and "reward_pre_remask" in final:
                post = final["reward_post_remask"]
                pre = final["reward_pre_remask"]
                if isinstance(post, torch.Tensor) and isinstance(pre, torch.Tensor):
                    delta = (post - pre).tolist()
                    result.setdefault("remasking_delta", []).extend(delta)
            # Violations
            if "violations" in final:
                for vname, vvals in final["violations"].items():
                    if isinstance(vvals, torch.Tensor):
                        vvals = vvals.tolist()
                    result.setdefault(f"violation_{vname}", []).extend(
                        vvals if isinstance(vvals, list) else [vvals]
                    )

    return result


def plot_distributions(schedule: str, model: str) -> None:
    """Plot histograms of per-sample metrics at the final denoising step.

    One figure with subplots: ESS, reward_selected, reward_gap,
    remasking_delta, plus one subplot per constraint violation.
    Saved as ``{model}_distributions.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    config, per_seed, seeds = _load_per_sample_stats(schedule, model)
    values = _collect_final_step_values(per_seed)

    if not values:
        print(f"No per-sample stats available for {model}")
        return

    # Determine subplot layout: core metrics + violation metrics
    core_keys = ["ess", "reward_selected", "reward_gap", "remasking_delta"]
    violation_keys = sorted(k for k in values if k.startswith("violation_"))
    plot_keys = [k for k in core_keys if k in values] + violation_keys
    n_plots = len(plot_keys)

    if n_plots == 0:
        print("No metrics to plot.")
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    labels = {
        "ess": "ESS (final step)",
        "reward_selected": "Reward (selected)",
        "reward_gap": "Reward gap",
        "remasking_delta": "Remasking delta",
    }

    K = config.get("num_candidates", "?")
    alpha = config.get("guidance_alpha", "?")
    n_samples = len(next(iter(values.values())))

    for idx, key in enumerate(plot_keys):
        ax = axes[idx]
        data = np.array(values[key])
        label = labels.get(key, key.replace("violation_", "viol: "))
        ax.hist(data, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={data.mean():.3f}")
        ax.axvline(np.median(data), color="orange", linestyle=":",
                   linewidth=1.5, label=f"median={np.median(data):.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"{model}\nK={K}, \u03b1={alpha} | {n_samples} samples (final step)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_dir = _PROJECT_ROOT / "eval_results" / schedule
    out_path = out_dir / f"{model}_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Distribution plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Time-evolution trajectory plots for individual samples
# ---------------------------------------------------------------------------


def _extract_sample_trajectory(
    batches: list[list[dict]], sample_global_idx: int, batch_size_hint: int,
) -> dict[str, list[float]]:
    """Extract per-step trajectory for one sample from per-sample stats.

    Parameters
    ----------
    batches : list of list of step dicts (per_seed[seed])
    sample_global_idx : global index within this seed (0..num_samples-1)
    batch_size_hint : batch_size used during generation

    Returns
    -------
    dict mapping metric_name -> list[float] (length = n_steps)
    """
    # Find which batch and local index
    batch_idx = sample_global_idx // batch_size_hint
    local_idx = sample_global_idx % batch_size_hint

    if batch_idx >= len(batches):
        return {}
    batch_steps = batches[batch_idx]

    result: dict[str, list[float]] = {}
    for step in batch_steps:
        for key in ["ess", "reward_selected", "reward_all_candidates",
                     "reward_pre_remask", "reward_post_remask"]:
            if key not in step:
                continue
            val = step[key]
            if isinstance(val, torch.Tensor):
                if local_idx < val.numel():
                    result.setdefault(key, []).append(val[local_idx].item())
            else:
                result.setdefault(key, []).append(float(val))
        # Derived: reward_gap
        if "reward_selected" in step and "reward_all_candidates" in step:
            sel = step["reward_selected"]
            allc = step["reward_all_candidates"]
            if isinstance(sel, torch.Tensor) and isinstance(allc, torch.Tensor):
                if local_idx < sel.numel():
                    result.setdefault("reward_gap", []).append(
                        (sel[local_idx] - allc[local_idx]).item()
                    )
        # Derived: remasking_delta
        if "reward_post_remask" in step and "reward_pre_remask" in step:
            post = step["reward_post_remask"]
            pre = step["reward_pre_remask"]
            if isinstance(post, torch.Tensor) and isinstance(pre, torch.Tensor):
                if local_idx < post.numel():
                    result.setdefault("remasking_delta", []).append(
                        (post[local_idx] - pre[local_idx]).item()
                    )
        # Violations
        if "violations" in step:
            for vname, vvals in step["violations"].items():
                if isinstance(vvals, torch.Tensor):
                    if local_idx < vvals.numel():
                        result.setdefault(f"violation_{vname}", []).append(
                            vvals[local_idx].item()
                        )

    return result


def plot_trajectories(
    schedule: str, model: str, traj_seed: int | None = None,
) -> None:
    """Plot time evolution of guidance metrics for 2 samples from one seed.

    Each metric gets its own subplot; both samples are overlaid in the same
    subplot with different colors. Saved as ``{model}_trajectories.png``.
    """
    import matplotlib.pyplot as plt

    config, per_seed, seeds = _load_per_sample_stats(schedule, model)
    batch_size = config.get("batch_size", 64)

    # Pick seed
    if traj_seed is not None:
        seed_key = str(traj_seed)
    elif seeds:
        seed_key = str(seeds[0])
    else:
        seed_key = next(iter(per_seed))

    if seed_key not in per_seed:
        print(f"Error: seed {seed_key} not found. Available: {list(per_seed.keys())}")
        sys.exit(1)

    batches = per_seed[seed_key]

    # Pick 2 samples: first sample of first batch, first sample of second batch
    # (to get diversity in sample characteristics).
    sample_indices = [0, batch_size]
    # Fall back to indices 0, 1 if only one batch
    if len(batches) < 2:
        sample_indices = [0, min(1, batch_size - 1)]

    trajectories = {}
    for si in sample_indices:
        traj = _extract_sample_trajectory(batches, si, batch_size)
        if traj:
            trajectories[si] = traj

    if not trajectories:
        print(f"No per-sample trajectory data for seed {seed_key}")
        return

    # Collect all metric keys across both samples
    all_keys: set[str] = set()
    for traj in trajectories.values():
        all_keys.update(traj.keys())

    # Order: core metrics first, then violations
    core_order = ["ess", "reward_selected", "reward_gap",
                  "remasking_delta"]
    violation_keys = sorted(k for k in all_keys if k.startswith("violation_"))
    plot_keys = [k for k in core_order if k in all_keys] + violation_keys
    n_plots = len(plot_keys)

    if n_plots == 0:
        print("No metrics to plot.")
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    labels = {
        "ess": "ESS",
        "reward_selected": "Reward (selected)",
        "reward_gap": "Reward gap",
        "remasking_delta": "Remasking delta",
    }
    colors = ["#1f77b4", "#d62728"]

    K = config.get("num_candidates", "?")
    alpha = config.get("guidance_alpha", "?")

    for idx, key in enumerate(plot_keys):
        ax = axes[idx]
        label = labels.get(key, key.replace("violation_", "viol: "))
        for ci, (si, traj) in enumerate(trajectories.items()):
            if key in traj:
                steps_range = range(len(traj[key]))
                ax.plot(
                    steps_range, traj[key],
                    color=colors[ci % len(colors)],
                    linewidth=1.2, alpha=0.85,
                    label=f"sample {si}",
                )
        ax.set_xlabel("Denoising step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"{model}\nK={K}, \u03b1={alpha} | seed={seed_key}, "
        f"samples {list(trajectories.keys())}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_dir = _PROJECT_ROOT / "eval_results" / schedule
    out_path = out_dir / f"{model}_trajectories_seed{seed_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Trajectory plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Mode comparison
# ---------------------------------------------------------------------------


def compare_modes(schedules: list[str]) -> None:
    """Compare soft vs hard reward modes across all variants."""
    print(f"\n{'=' * 80}")
    print("  Soft vs Hard Reward Mode Comparison")
    print(f"{'=' * 80}")

    for schedule in schedules:
        eval_dir = _PROJECT_ROOT / "eval_results" / schedule
        if not eval_dir.is_dir():
            print(f"\n  Skipping {schedule}: directory not found")
            continue

        # Find soft_test and hard_test models
        soft_models = sorted(
            p.stem for p in eval_dir.glob("*_guided_soft_test_*_samples.pt")
        )
        hard_models = sorted(
            p.stem for p in eval_dir.glob("*_guided_hard_test_*_samples.pt")
        )

        # Strip _samples suffix
        soft_models = [m.replace("_samples", "") for m in soft_models]
        hard_models = [m.replace("_samples", "") for m in hard_models]

        if not soft_models and not hard_models:
            print(f"\n  No soft/hard comparison models found in {schedule}")
            continue

        print(f"\n  Schedule: {schedule}")
        print(f"  Soft models: {len(soft_models)}")
        print(f"  Hard models: {len(hard_models)}")

        # Pair soft/hard by base variant
        pairs = {}
        for m in soft_models:
            base = m.replace("_guided_soft_test_K8_a1.0", "")
            pairs.setdefault(base, {})["soft"] = m
        for m in hard_models:
            base = m.replace("_guided_hard_test_K8_a1.0", "")
            pairs.setdefault(base, {})["hard"] = m

        # Comparison table header
        print(f"\n  {'Variant':<40} {'Mode':<6} {'ESS_mean':>9} {'ESS_min':>9} "
              f"{'MaxW_max':>9} {'R_gap':>9} {'\u0394_remask':>9}")
        print(f"  {'-'*40} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

        for base, mode_dict in sorted(pairs.items()):
            for mode in ["soft", "hard"]:
                if mode not in mode_dict:
                    continue
                model = mode_dict[mode]
                _, all_steps = _load_guidance_stats(schedule, model)
                steps = _average_steps(all_steps)
                if not steps:
                    continue

                ess_vals = [s["ess"] for s in steps]
                max_w = [s["max_weight"] for s in steps]
                gaps = [s["reward_gap"] for s in steps]
                deltas = [s["reward_remasking_delta"] for s in steps]

                short_base = base[:38]
                print(f"  {short_base:<40} {mode:<6} "
                      f"{sum(ess_vals)/len(ess_vals):>9.2f} "
                      f"{min(ess_vals):>9.2f} "
                      f"{max(max_w):>9.3f} "
                      f"{sum(gaps)/len(gaps):>9.4f} "
                      f"{sum(deltas)/len(deltas):>9.4f}")

    print(f"\n{'=' * 80}")
    print("  Decision guide:")
    print("    - Higher ESS_mean \u2192 more effective use of K candidates")
    print("    - Higher ESS_min \u2192 fewer degenerate steps")
    print("    - Lower MaxW_max \u2192 better diversity preservation")
    print("    - Larger |R_gap| \u2192 stronger guidance signal")
    print("    - Positive \u0394_remask \u2192 remasking cooperates with guidance")
    print(f"{'=' * 80}")


def list_guided_models(schedules: list[str]) -> None:
    """List all guided models found in eval_results."""
    for schedule in schedules:
        eval_dir = _PROJECT_ROOT / "eval_results" / schedule
        if not eval_dir.is_dir():
            continue
        models = sorted(
            p.stem.replace("_samples", "")
            for p in eval_dir.glob("*_guided_*_samples.pt")
        )
        if models:
            print(f"\n{schedule}:")
            for m in models:
                print(f"  {m}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GuidanceStats from guided sample files.",
    )
    parser.add_argument(
        "--schedule", nargs="+", type=str,
        default=["loglinear_noise_sc", "learned_noise_sc"],
        help="Noise schedule subdirectory(ies).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model name to analyze.",
    )
    parser.add_argument(
        "--compare-modes", action="store_true",
        help="Compare soft vs hard reward modes across all variants.",
    )
    parser.add_argument(
        "--export-tsv", action="store_true",
        help="Export ESS/reward trajectory as TSV for plotting.",
    )
    parser.add_argument(
        "--plot-distributions", action="store_true",
        help="Plot histograms of per-sample metrics at the final step "
             "(requires per-sample stats in _samples.pt).",
    )
    parser.add_argument(
        "--plot-trajectories", action="store_true",
        help="Plot time-evolution of guidance metrics for 2 individual "
             "samples from one seed.",
    )
    parser.add_argument(
        "--traj-seed", type=int, default=None,
        help="Seed to use for --plot-trajectories (default: first seed).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all guided models and exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_guided_models(args.schedule)
        return

    if args.compare_modes:
        compare_modes(args.schedule)
        return

    if args.model:
        if len(args.schedule) != 1:
            print("Error: --model requires exactly one --schedule")
            sys.exit(1)
        analyze_single(args.schedule[0], args.model, export_tsv=args.export_tsv)
        if args.plot_distributions:
            plot_distributions(args.schedule[0], args.model)
        if args.plot_trajectories:
            plot_trajectories(
                args.schedule[0], args.model, traj_seed=args.traj_seed,
            )
    else:
        # Analyze all guided models
        for schedule in args.schedule:
            eval_dir = _PROJECT_ROOT / "eval_results" / schedule
            if not eval_dir.is_dir():
                continue
            models = sorted(
                p.stem.replace("_samples", "")
                for p in eval_dir.glob("*_guided_*_samples.pt")
            )
            for model in models:
                analyze_single(schedule, model, export_tsv=args.export_tsv)
                if args.plot_distributions:
                    plot_distributions(schedule, model)
                if args.plot_trajectories:
                    plot_trajectories(
                        schedule, model, traj_seed=args.traj_seed,
                    )


if __name__ == "__main__":
    main()
