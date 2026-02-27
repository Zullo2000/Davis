"""RewardComposer: combines multiple constraint violations into a single energy/reward score.

Energy formula:  E(x) = Σ_i (λ_i / p90_i) * φ(v_i(x))
Reward formula:  r(x) = -E(x)

Where:
    λ_i   = constraint weight
    p90_i = P90 normalizer for constraint i (calibrated)
    φ     = shaping function (linear, quadratic, or log1p)
    v_i   = violation value from constraint i
"""

import math
import warnings
from typing import Literal

import torch
from torch import Tensor

from bd_gen.data.vocab import VocabConfig
from bd_gen.guidance.constraints import Constraint, ConstraintResult


class RewardComposer:
    """Combines multiple constraint violations into a single energy/reward score."""

    def __init__(
        self,
        constraints: list[Constraint],
        phi: Literal["linear", "quadratic", "log1p"] = "linear",
        reward_mode: Literal["soft", "hard"] = "soft",
    ):
        self.constraints = constraints
        self.phi = phi
        self.reward_mode = reward_mode

        # Create the shaping function based on phi
        if phi == "linear":
            self._phi_fn = lambda v: v
            self._phi_tensor_fn = lambda v: v
        elif phi == "quadratic":
            self._phi_fn = lambda v: v ** 2
            self._phi_tensor_fn = lambda v: v ** 2
        elif phi == "log1p":
            self._phi_fn = lambda v: math.log1p(v)
            self._phi_tensor_fn = lambda v: torch.log1p(v)
        else:
            raise ValueError(f"Unknown phi function: {phi!r}. Must be 'linear', 'quadratic', or 'log1p'.")

    def _apply_phi(self, v: float) -> float:
        """Apply shaping function to a violation value."""
        return self._phi_fn(v)

    def _apply_phi_tensor(self, v: Tensor) -> Tensor:
        """Apply shaping function to a violation tensor (for soft mode)."""
        return self._phi_tensor_fn(v)

    def compute_energy_hard(
        self, graph_dict: dict,
    ) -> tuple[float, dict[str, ConstraintResult]]:
        """E(x) = Σ (λ_i / p90_i) * φ(v_i(x)) on decoded graph.

        Returns (energy, {constraint_name: ConstraintResult}).
        Energy is always >= 0. When all constraints satisfied, E = 0.
        """
        energy = 0.0
        details: dict[str, ConstraintResult] = {}

        for constraint in self.constraints:
            result = constraint.hard_violation(graph_dict)
            details[constraint.name] = result

            weight = float(constraint.weight)
            p90 = float(constraint.p90_normalizer)
            violation = float(result.violation)

            term = (weight / p90) * self._apply_phi(violation)
            energy += term

        return energy, details

    def compute_reward_hard(
        self, graph_dict: dict,
    ) -> tuple[float, dict[str, ConstraintResult]]:
        """r(x) = -E(x) on decoded graph.

        Returns (reward, {constraint_name: ConstraintResult}).
        Reward is always <= 0.
        """
        energy, details = self.compute_energy_hard(graph_dict)
        return -energy, details

    def compute_energy_soft(
        self,
        node_probs: Tensor,
        edge_probs: Tensor,
        pad_mask: Tensor,
        vocab_config: VocabConfig,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """E(x) = Σ (λ_i / p90_i) * φ(v_i(x)) on posterior distributions.

        Returns (energy_tensor, {constraint_name: violation_tensor}).
        """
        violation_details: dict[str, Tensor] = {}
        energy: Tensor | None = None

        for constraint in self.constraints:
            violation = constraint.soft_violation(
                node_probs, edge_probs, pad_mask, vocab_config,
            )
            violation_details[constraint.name] = violation

            weight = float(constraint.weight)
            p90 = float(constraint.p90_normalizer)

            term = (weight / p90) * self._apply_phi_tensor(violation)

            if energy is None:
                energy = term
            else:
                energy = energy + term

        # If no constraints, return a zero scalar tensor (float64)
        if energy is None:
            energy = torch.tensor(0.0, dtype=torch.float64)

        return energy, violation_details

    def compute_reward_soft(
        self,
        node_probs: Tensor,
        edge_probs: Tensor,
        pad_mask: Tensor,
        vocab_config: VocabConfig,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """r(x) = -E(x) on posterior distributions.

        Returns (reward_tensor, {constraint_name: violation_tensor}).
        """
        energy, violation_details = self.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vocab_config,
        )
        return -energy, violation_details

    def load_calibration(self, calibration: dict[str, float]) -> None:
        """Set P90 normalizers from calibration dict.

        Args:
            calibration: {constraint_name: p90_value}

        Updates each constraint's p90_normalizer if its name is in the dict.
        Warns if a constraint name is not found in the calibration dict.
        """
        for constraint in self.constraints:
            if constraint.name in calibration:
                constraint.p90_normalizer = calibration[constraint.name]
            else:
                warnings.warn(
                    f"Constraint '{constraint.name}' not found in calibration dict. "
                    f"Using default p90_normalizer={constraint.p90_normalizer}.",
                    stacklevel=2,
                )
