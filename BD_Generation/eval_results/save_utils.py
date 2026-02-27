"""Utilities for saving and comparing evaluation results across methods.

V2 format stores multi-seed results with per-seed breakdowns and
summary statistics. V1 files (flat single-seed) are auto-upgraded on load.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Metric family registry -- defines comparison table structure
# ---------------------------------------------------------------------------

# Each entry: (eval_key, display_name, is_percentage, is_diagnostic)
# eval_key is the key as it appears in summary dicts (e.g. "eval/node_js").

_VALIDITY_METRICS: list[tuple[str, str, bool, bool]] = [
    ("eval/validity_rate", "Validity rate", True, False),
    ("eval/connected_rate", "Connected rate", True, False),
    ("eval/valid_types_rate", "Valid types rate", True, False),
    ("eval/no_mask_rate", "No MASK rate", True, False),
    ("eval/inside_validity", "Inside validity", True, False),
]

_COVERAGE_METRICS: list[tuple[str, str, bool, bool]] = [
    ("eval/diversity", "Diversity", False, False),
    ("eval/novelty", "Novelty", False, False),
    ("eval/mode_coverage", "Mode coverage (unweighted)", True, False),
    ("eval/mode_coverage_weighted", "Mode coverage (weighted)", True, False),
    ("eval/num_sample_modes", "Unique archetypes", False, False),
]

_DISTRIBUTION_METRICS: list[tuple[str, str, bool, bool]] = [
    ("eval/node_js", "**Node JS**", False, False),
    ("eval/edge_js", "**Edge JS**", False, False),
    ("eval/node_tv", "**Node TV**", False, False),
    ("eval/edge_tv", "**Edge TV**", False, False),
    ("eval/rooms_w1", "**Rooms W1**", False, False),
    ("eval/node_kl", "Node KL (diag.)", False, True),
    ("eval/edge_kl", "Edge KL (diag.)", False, True),
    ("eval/num_rooms_kl", "Rooms KL (diag.)", False, True),
]

_STRUCTURE_METRICS: list[tuple[str, str, bool, bool]] = [
    ("eval/mmd_degree", "MMD-Degree", False, False),
    ("eval/mmd_clustering", "MMD-Clustering", False, False),
    ("eval/mmd_spectral", "MMD-Spectral", False, False),
    ("eval/transitivity_score", "Spatial transitivity", True, False),
    ("eval/h_consistent", "H-consistent", True, False),
    ("eval/v_consistent", "V-consistent", True, False),
]

_CONDITIONAL_METRICS: list[tuple[str, str, bool, bool]] = [
    ("eval/conditional_edge_js_weighted", "Cond. edge JS (weighted)", False, False),
    ("eval/conditional_edge_tv_weighted", "Cond. edge TV (weighted)", False, False),
    ("eval/conditional_edge_js_topN_weighted",
     "Cond. edge JS top-N (wt.)", False, False),
    ("eval/conditional_edge_tv_topN_weighted",
     "Cond. edge TV top-N (wt.)", False, False),
    ("eval/degree_js_per_type_weighted", "Type-cond. degree JS (wt.)", False, False),
    ("eval/degree_tv_per_type_weighted", "Type-cond. degree TV (wt.)", False, False),
    ("eval/conditional_edge_kl_weighted", "Cond. edge KL (wt., diag.)", False, True),
    ("eval/conditional_edge_kl_topN_weighted",
     "Cond. edge KL top-N (diag.)", False, True),
    ("eval/degree_kl_per_type_weighted", "Type-cond. degree KL (diag.)", False, True),
]

METRIC_FAMILIES: list[tuple[str, list[tuple[str, str, bool, bool]]]] = [
    ("Validity", _VALIDITY_METRICS),
    ("Coverage", _COVERAGE_METRICS),
    ("Distribution (Primary: JS / TV / W1)", _DISTRIBUTION_METRICS),
    ("Graph Structure", _STRUCTURE_METRICS),
    ("Conditional", _CONDITIONAL_METRICS),
]

# Focused metric set for guided experiments: Validity + Coverage + priority metrics
_PRIORITY_METRICS: list[tuple[str, str, bool, bool]] = [
    ("eval/mode_coverage_weighted", "Mode coverage (weighted)", True, False),
    ("eval/transitivity_score", "Spatial transitivity", True, False),
    ("eval/conditional_edge_tv_weighted", "Cond. edge TV (weighted)", False, False),
    ("eval/degree_tv_per_type_weighted", "Type-cond. degree TV (weighted)", False, False),
    ("eval/node_tv", "Node TV", False, False),
]

GUIDED_METRIC_FAMILIES: list[tuple[str, list[tuple[str, str, bool, bool]]]] = [
    ("Validity", _VALIDITY_METRICS),
    ("Coverage", _COVERAGE_METRICS),
    ("Priority Metrics", _PRIORITY_METRICS),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_mean_std(
    mean: float,
    std: float,
    *,
    is_percentage: bool = False,
) -> str:
    """Format mean +/- std for display in markdown tables.

    If *std* is 0.0 (e.g. V1-upgraded single-seed data), only the mean
    is shown, with no misleading ``+/- 0.0``.
    """
    if is_percentage:
        if std == 0.0:
            return f"{100 * mean:.1f}%"
        return f"{100 * mean:.1f} +/- {100 * std:.1f}%"
    # Integer-valued metrics (counts, etc.)
    if mean == int(mean) and std == 0.0:
        return str(int(mean))
    if mean == int(mean) and std == int(std) and std != 0.0:
        return f"{int(mean)} +/- {int(std)}"
    # Non-percentage floats
    if std == 0.0:
        if abs(mean) < 0.01 and mean != 0:
            return f"{mean:.6f}"
        return f"{mean:.4f}"
    if abs(mean) < 0.01 and mean != 0:
        return f"{mean:.6f} +/- {std:.6f}"
    return f"{mean:.4f} +/- {std:.4f}"


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and other non-JSON types."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------


def aggregate_multi_seed(
    per_seed_results: dict[int, dict],
) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics across seeds into mean/std.

    Nested dicts (stratified metrics) are flattened with ``/`` separators
    and aggregated leaf-by-leaf.

    Returns:
        Dict mapping metric_name to {"mean": ..., "std": ...}.
    """

    def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
        flat: dict[str, float] = {}
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, (int, float)):
                flat[key] = float(v)
            elif isinstance(v, dict):
                flat.update(_flatten(v, key))
        return flat

    all_flat: dict[int, dict[str, float]] = {}
    for seed, metrics in per_seed_results.items():
        all_flat[seed] = _flatten(metrics)

    all_keys: set[str] = set()
    for flat in all_flat.values():
        all_keys.update(flat.keys())

    summary: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [flat[key] for flat in all_flat.values() if key in flat]
        if values:
            arr = np.array(values, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
            }

    return summary


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_eval_result(
    path: str | Path,
    method: str,
    config_dict: dict,
    per_seed_metrics: dict[int, dict],
    summary_metrics: dict[str, dict[str, float]],
    denoising_metrics: dict[str, float] | None = None,
) -> None:
    """Save evaluation results in structured V2 JSON format.

    Args:
        path: Output file path (.json).
        method: Method identifier (e.g. ``"mdlm_baseline"``).
        config_dict: Sampling/eval configuration.
        per_seed_metrics: Raw metrics dict per seed.
        summary_metrics: ``{key: {"mean": ..., "std": ...}}`` from aggregation.
        denoising_metrics: Optional seed-independent denoising eval results.
    """
    result = {
        "format_version": 2,
        "method": method,
        "timestamp": datetime.now().isoformat(),
        "config": make_json_serializable(config_dict),
        "per_seed": {
            str(k): make_json_serializable(v)
            for k, v in per_seed_metrics.items()
        },
        "summary": make_json_serializable(summary_metrics),
        "denoising": make_json_serializable(denoising_metrics or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


def load_eval_result(path: str | Path) -> dict:
    """Load evaluation result JSON with backward-compatible V1 → V2 upgrade.

    V1 format: ``{method, timestamp, config, metrics}`` (flat, single-seed).
    V2 format: ``{format_version: 2, method, timestamp, config, per_seed,
    summary, denoising}``.
    """
    data = json.loads(Path(path).read_text())

    if data.get("format_version", 1) >= 2:
        return data

    # --- V1 → V2 upgrade ---
    metrics = data.get("metrics", {})
    seed = str(data.get("config", {}).get("seed", "unknown"))

    summary: dict[str, dict[str, float]] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            summary[k] = {"mean": float(v), "std": 0.0}

    return {
        "format_version": 2,
        "method": data.get("method", "unknown"),
        "timestamp": data.get("timestamp", ""),
        "config": data.get("config", {}),
        "per_seed": {seed: metrics},
        "summary": summary,
        "denoising": {},
        "_upgraded_from_v1": True,
    }


# ---------------------------------------------------------------------------
# Comparison table generation
# ---------------------------------------------------------------------------


def _build_metric_table(
    results: list[dict],
    method_names: list[str],
    metrics_spec: list[tuple[str, str, bool, bool]],
    *,
    primary_only: bool = False,
) -> list[str]:
    """Build markdown table rows for one metric family.

    Returns list of markdown lines (header + separator + data rows),
    or empty list if no data is available for any metric in the spec.
    """
    # Filter diagnostics if requested
    if primary_only:
        metrics_spec = [(k, n, p, d) for k, n, p, d in metrics_spec if not d]

    if not metrics_spec:
        return []

    # Check if there's any data at all for this family
    has_any = False
    for key, _, _, _ in metrics_spec:
        for r in results:
            if key in r["summary"]:
                has_any = True
                break
        if has_any:
            break
    if not has_any:
        return []

    # Header
    two_method = len(results) == 2
    if two_method:
        header = f"| Metric | {method_names[0]} | {method_names[1]} | Delta |"
        sep = "|--------|:---:|:---:|:---:|"
    else:
        header = "| Metric | " + " | ".join(method_names) + " |"
        sep = "|--------|" + "|".join([":---:"] * len(results)) + "|"

    rows = [header, sep]

    for key, display_name, is_pct, _is_diag in metrics_spec:
        vals: list[str] = []
        raw_means: list[float | None] = []
        for r in results:
            entry = r["summary"].get(key)
            if entry is None:
                vals.append("--")
                raw_means.append(None)
            else:
                mean = entry["mean"]
                std = entry["std"]
                vals.append(_format_mean_std(mean, std, is_percentage=is_pct))
                raw_means.append(mean)

        # Skip row if no method has data
        if all(v == "--" for v in vals):
            continue

        if two_method:
            m0, m1 = raw_means[0], raw_means[1]
            if m0 is not None and m1 is not None:
                delta = m1 - m0
                sign = "+" if delta >= 0 else ""
                if is_pct:
                    delta_str = f"{sign}{100 * delta:.1f}%"
                elif delta == int(delta):
                    delta_str = f"{sign}{int(delta)}"
                elif abs(delta) < 0.001 and delta != 0:
                    delta_str = f"{sign}{delta:.6f}"
                else:
                    delta_str = f"{sign}{delta:.4f}"
            else:
                delta_str = "--"
            rows.append(f"| {display_name} | {vals[0]} | {vals[1]} | {delta_str} |")
        else:
            rows.append(f"| {display_name} | " + " | ".join(vals) + " |")

    # Return nothing if only header+sep (no data rows)
    if len(rows) <= 2:
        return []

    return rows


def _build_constraint_table(
    results: list[dict],
    method_names: list[str],
) -> list[str]:
    """Build markdown table for constraint satisfaction metrics.

    Dynamically detects ``constraint/*`` keys from summary dicts.
    Skips histogram keys (non-scalar).
    """
    # Collect all constraint keys across results (skip histograms)
    all_keys: set[str] = set()
    for r in results:
        for k in r.get("summary", {}):
            if k.startswith("constraint/") and "histogram" not in k:
                all_keys.add(k)

    if not all_keys:
        return []

    # Sort: overall first, then satisfaction, then mean_violation, then mean_violation_when_failed
    def _sort_key(k: str) -> tuple[int, str]:
        if "overall" in k:
            return (0, k)
        if "satisfaction" in k:
            return (1, k)
        if "mean_violation_when_failed" in k:
            return (3, k)
        if "mean_violation" in k:
            return (2, k)
        return (4, k)

    sorted_keys = sorted(all_keys, key=_sort_key)

    # Build display names
    def _display_name(k: str) -> str:
        # "constraint/satisfaction_one_kitchen" -> "Satisfaction: one_kitchen"
        suffix = k.replace("constraint/", "")
        if suffix == "satisfaction_overall":
            return "**Satisfaction (all)**"
        if suffix.startswith("satisfaction_"):
            name = suffix.replace("satisfaction_", "")
            return f"Satisfaction: {name}"
        if suffix.startswith("mean_violation_when_failed_"):
            name = suffix.replace("mean_violation_when_failed_", "")
            return f"Mean viol. (failed): {name}"
        if suffix.startswith("mean_violation_"):
            name = suffix.replace("mean_violation_", "")
            return f"Mean violation: {name}"
        return suffix

    # Determine which keys are percentages (satisfaction rates)
    is_pct = {k: "satisfaction" in k for k in sorted_keys}

    # Build table using _build_metric_table's logic (inline for dynamic keys)
    two_method = len(results) == 2
    if two_method:
        header = f"| Metric | {method_names[0]} | {method_names[1]} | Delta |"
        sep = "|--------|:---:|:---:|:---:|"
    else:
        header = "| Metric | " + " | ".join(method_names) + " |"
        sep = "|--------|" + "|".join([":---:"] * len(results)) + "|"

    rows = [header, sep]

    for key in sorted_keys:
        display = _display_name(key)
        pct = is_pct[key]
        vals: list[str] = []
        raw_means: list[float | None] = []
        for r in results:
            entry = r["summary"].get(key)
            if entry is None:
                vals.append("--")
                raw_means.append(None)
            else:
                mean = entry["mean"]
                std = entry["std"]
                vals.append(
                    _format_mean_std(mean, std, is_percentage=pct),
                )
                raw_means.append(mean)

        if all(v == "--" for v in vals):
            continue

        if two_method:
            m0, m1 = raw_means[0], raw_means[1]
            if m0 is not None and m1 is not None:
                delta = m1 - m0
                sign = "+" if delta >= 0 else ""
                if pct:
                    delta_str = f"{sign}{100 * delta:.1f}%"
                else:
                    delta_str = f"{sign}{delta:.4f}"
            else:
                delta_str = "--"
            rows.append(
                f"| {display} | {vals[0]} | {vals[1]} | {delta_str} |",
            )
        else:
            rows.append(f"| {display} | " + " | ".join(vals) + " |")

    if len(rows) <= 2:
        return []

    return rows


def _build_denoising_table(
    results: list[dict],
    method_names: list[str],
) -> list[str]:
    """Build markdown table for denoising (model-quality) metrics."""
    # Collect all denoising keys across results
    all_keys: list[str] = []
    for r in results:
        for k in r.get("denoising", {}):
            if k not in all_keys:
                all_keys.append(k)

    if not all_keys:
        return []

    # Sort by t-value for readability
    all_keys.sort()

    header = "| Metric | " + " | ".join(method_names) + " |"
    sep = "|--------|" + "|".join([":---:"] * len(results)) + "|"
    rows = [header, sep]

    for key in all_keys:
        vals = []
        for r in results:
            v = r.get("denoising", {}).get(key)
            if v is None:
                vals.append("--")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        # Clean up key for display
        display = key.replace("denoise/", "")
        rows.append(f"| {display} | " + " | ".join(vals) + " |")

    if len(rows) <= 2:
        return []

    return rows


def _build_config_table(
    results: list[dict],
    method_names: list[str],
) -> list[str]:
    """Build markdown table comparing run configurations."""
    config_keys = [
        ("seeds", "Seeds"),
        ("num_samples", "Num samples"),
        ("sampling_steps", "Sampling steps"),
        ("temperature", "Temperature"),
        ("top_p", "Top-p"),
        ("unmasking_mode", "Unmasking mode"),
        ("remasking_enabled", "Remasking"),
        ("remasking_strategy", "Remasking strategy"),
        ("remasking_eta", "Remasking eta"),
        ("remasking_t_switch", "Remasking t_switch"),
        ("checkpoint", "Checkpoint"),
        # Guidance-specific (present only for guided models)
        ("num_candidates", "Guidance K"),
        ("guidance_alpha", "Guidance alpha"),
        ("reward_mode", "Reward mode"),
        ("phi", "Phi function"),
        ("num_constraints", "Num constraints"),
    ]

    header = "| Parameter | " + " | ".join(method_names) + " |"
    sep = "|-----------|" + "|".join([":---:"] * len(results)) + "|"
    rows = [header, sep]

    for cfg_key, display in config_keys:
        vals = []
        for r in results:
            v = r.get("config", {}).get(cfg_key)
            if v is None:
                # V1 compat: check "seed" (singular)
                if cfg_key == "seeds":
                    v = r.get("config", {}).get("seed")
                    if v is not None:
                        v = [v]
            if v is None:
                vals.append("--")
            else:
                vals.append(str(v))
        rows.append(f"| {display} | " + " | ".join(vals) + " |")

    return rows


def build_comparison_table(
    result_paths: list[str | Path],
    *,
    primary_only: bool = False,
    guided: bool = False,
) -> str:
    """Build structured markdown comparison from evaluation result JSONs.

    Organizes metrics by family (validity, coverage, distribution, structure,
    conditional, denoising) with JS/TV/W1 as primary distribution metrics.
    Multi-seed results show mean +/- std.

    Args:
        result_paths: Paths to JSON result files (V1 or V2).
        primary_only: If True, hide KL diagnostic metrics.
        guided: If True, use focused metric families for guidance experiments
            (Validity + Coverage + Priority Metrics instead of the full set).

    Returns:
        Full markdown document.
    """
    results = [load_eval_result(p) for p in result_paths]
    if not results:
        return "No results to compare."

    method_names = [r["method"] for r in results]
    has_v1 = any(r.get("_upgraded_from_v1") for r in results)

    families = GUIDED_METRIC_FAMILIES if guided else METRIC_FAMILIES

    lines: list[str] = []

    # --- Header ---
    lines.append("# Evaluation Comparison")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Methods**: {', '.join(method_names)}")
    lines.append("")

    # --- Configuration ---
    lines.append("## Configuration")
    lines.append("")
    lines.extend(_build_config_table(results, method_names))
    lines.append("")

    # --- Metric family sections ---
    for family_title, family_spec in families:
        table_lines = _build_metric_table(
            results, method_names, family_spec, primary_only=primary_only,
        )
        if table_lines:
            lines.append(f"## {family_title}")
            lines.append("")
            lines.extend(table_lines)
            lines.append("")

    # --- Constraint satisfaction (dynamic, only if any result has constraint metrics) ---
    constraint_lines = _build_constraint_table(results, method_names)
    if constraint_lines:
        lines.append("## Constraint Satisfaction")
        lines.append("")
        lines.extend(constraint_lines)
        lines.append("")

    # --- Denoising (model quality) ---
    denoise_lines = _build_denoising_table(results, method_names)
    if denoise_lines:
        lines.append("## Denoising (Model Quality, Seed-Independent)")
        lines.append("")
        lines.extend(denoise_lines)
        lines.append("")

    # --- Footnotes ---
    lines.append("---")
    lines.append(
        "*Auto-generated by `scripts/compare.py`. "
        "Values shown as mean +/- std (population) over N seeds.*"
    )
    lines.append(
        "*JS/TV/W1 are the primary distance measures. "
        "KL metrics marked \"(diag.)\" are diagnostic only.*"
    )
    if has_v1:
        lines.append(
            "*Some results were loaded from V1 format (single-seed) "
            "and are shown without std.*"
        )
    lines.append("")

    return "\n".join(lines)
