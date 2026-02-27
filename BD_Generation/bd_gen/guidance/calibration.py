"""Calibration protocol for constraint P90 normalization.

Different constraints produce violations at different scales: ExactCount may
range 0-7, while RequireAdj is always in {0, 1}. Without normalization,
high-scale constraints dominate the energy and SVDD's importance weights are
insensitive to low-scale constraints.

Calibration normalizes all violations to comparable scales by dividing each
by its 90th percentile on unguided samples.

Protocol:
    1. Load unguided samples → detokenize to graph dicts.
    2. For each constraint, compute hard_violation() on all samples.
    3. P90 = 90th percentile of non-zero violations.
       If all violations are 0, P90 = 1.0 (no normalization needed).
    4. Save calibration to JSON: {constraint_name: p90_value}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bd_gen.guidance.constraints import Constraint


def calibrate_from_samples(
    graph_dicts: list[dict],
    constraints: list[Constraint],
) -> dict[str, float]:
    """Compute P90 normalizers from decoded graph dicts.

    For each constraint, computes hard_violation on all samples, then takes
    the 90th percentile of the non-zero violations.

    Args:
        graph_dicts: Decoded graphs (each with num_rooms, node_types, edge_triples).
        constraints: Compiled constraint objects.

    Returns:
        Dict mapping constraint name to P90 value.
    """
    calibration: dict[str, float] = {}

    for constraint in constraints:
        violations = []
        for gd in graph_dicts:
            result = constraint.hard_violation(gd)
            violations.append(result.violation)

        # Filter to non-zero violations
        nonzero = [v for v in violations if v > 0.0]

        if len(nonzero) == 0:
            # Constraint always satisfied — no normalization needed
            p90 = 1.0
        else:
            p90 = float(np.percentile(nonzero, 90))

        calibration[constraint.name] = p90

    return calibration


def save_calibration(path: Path, calibration: dict[str, float]) -> None:
    """Save calibration dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)


def load_calibration(path: Path) -> dict[str, float]:
    """Load calibration dict from a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
