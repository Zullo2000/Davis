"""Constraint primitives for guided bubble diagram generation.

Defines the Constraint ABC and four concrete constraint types that evaluate
constraint violations on decoded graphs (hard mode) or probability
distributions (soft mode).

Each constraint computes a *graded* violation magnitude (>= 0, 0 = satisfied)
rather than a binary feasibility flag, providing dense signal for SVDD
importance weighting.

All violation values are float (never int) per spec Section 2.8.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from bd_gen.data.vocab import VocabConfig
from bd_gen.guidance.soft_violations import _compute_adj_terms


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintResult:
    """Result of evaluating one constraint on one decoded graph."""

    name: str
    violation: float  # >= 0, 0 = satisfied
    satisfied: bool  # violation == 0
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """Base class for all architectural constraints.

    Subclasses implement ``hard_violation`` (on decoded graphs) and
    ``soft_violation`` (on probability distributions from logits).

    Attributes:
        name: Human-readable constraint identifier (unique per config).
        weight: Relative importance (λ). Default 1.0.
        p90_normalizer: 90th-percentile normalizer from calibration. Default 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight
        self.p90_normalizer: float = 1.0

    @abstractmethod
    def hard_violation(self, graph_dict: dict) -> ConstraintResult:
        """Evaluate the constraint on a decoded graph.

        Args:
            graph_dict: Decoded graph with keys ``num_rooms``, ``node_types``,
                ``edge_triples``.

        Returns:
            ConstraintResult with graded violation magnitude.
        """

    @abstractmethod
    def soft_violation(
        self,
        node_probs: Tensor,
        edge_probs: Tensor,
        pad_mask: Tensor,
        vocab_config: VocabConfig,
    ) -> Tensor:
        """Evaluate the constraint on posterior probability distributions.

        Args:
            node_probs: (n_max, NODE_VOCAB_SIZE) per-position distributions.
            edge_probs: (n_edges, EDGE_VOCAB_SIZE) per-position distributions.
            pad_mask: (seq_len,) bool — True for real positions.
            vocab_config: Vocabulary configuration.

        Returns:
            Scalar tensor >= 0 (float64).
        """


# ---------------------------------------------------------------------------
# ExactCount
# ---------------------------------------------------------------------------


class ExactCount(Constraint):
    """Require exactly ``target`` rooms of a given type.

    Hard: ``v = |count(type) - target|``
    Soft: ``v = |n̂ - target|`` where ``n̂ = Σ q_i(type)`` over active positions.
    """

    def __init__(
        self,
        name: str,
        room_type_idx: int,
        target: int,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.room_type_idx = room_type_idx
        self.target = target

    def hard_violation(self, graph_dict: dict) -> ConstraintResult:
        node_types: list[int] = graph_dict["node_types"]
        count = sum(1 for t in node_types if t == self.room_type_idx)
        violation = float(abs(count - self.target))
        return ConstraintResult(
            name=self.name,
            violation=violation,
            satisfied=violation == 0.0,
            details={"count": count, "target": self.target},
        )

    def soft_violation(
        self,
        node_probs: Tensor,
        edge_probs: Tensor,
        pad_mask: Tensor,
        vocab_config: VocabConfig,
    ) -> Tensor:
        active = pad_mask[: vocab_config.n_max].double()  # (n_max,)
        q_type = node_probs[:, self.room_type_idx]  # (n_max,)
        n_hat = (q_type * active).sum()
        return torch.abs(n_hat - self.target)


# ---------------------------------------------------------------------------
# CountRange
# ---------------------------------------------------------------------------


class CountRange(Constraint):
    """Require room count of a given type within ``[lo, hi]``.

    Hard: ``v = max(0, lo - count) + max(0, count - hi)``
    Soft: Same formula with expected count ``n̂``.
    """

    def __init__(
        self,
        name: str,
        room_type_idx: int,
        lo: int,
        hi: int,
        weight: float = 1.0,
    ) -> None:
        if lo > hi:
            raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
        super().__init__(name=name, weight=weight)
        self.room_type_idx = room_type_idx
        self.lo = lo
        self.hi = hi

    def hard_violation(self, graph_dict: dict) -> ConstraintResult:
        node_types: list[int] = graph_dict["node_types"]
        count = sum(1 for t in node_types if t == self.room_type_idx)
        violation = float(max(0, self.lo - count) + max(0, count - self.hi))
        return ConstraintResult(
            name=self.name,
            violation=violation,
            satisfied=violation == 0.0,
            details={"count": count, "lo": self.lo, "hi": self.hi},
        )

    def soft_violation(
        self,
        node_probs: Tensor,
        edge_probs: Tensor,
        pad_mask: Tensor,
        vocab_config: VocabConfig,
    ) -> Tensor:
        active = pad_mask[: vocab_config.n_max].double()  # (n_max,)
        q_type = node_probs[:, self.room_type_idx]  # (n_max,)
        n_hat = (q_type * active).sum()
        lo_t = torch.tensor(self.lo, dtype=torch.float64)
        hi_t = torch.tensor(self.hi, dtype=torch.float64)
        return torch.clamp(lo_t - n_hat, min=0.0) + torch.clamp(n_hat - hi_t, min=0.0)


# ---------------------------------------------------------------------------
# RequireAdj
# ---------------------------------------------------------------------------


class RequireAdj(Constraint):
    """Require at least one adjacency between rooms of ``type_a`` and ``type_b``.

    Hard: ``v = 0`` if any edge exists between the required types, else ``v = 1``.
    Soft: ``v = 1 - P(exists)`` via log-space accumulation.
    """

    def __init__(
        self,
        name: str,
        type_a_idx: int,
        type_b_idx: int,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.type_a_idx = type_a_idx
        self.type_b_idx = type_b_idx

    def hard_violation(self, graph_dict: dict) -> ConstraintResult:
        node_types: list[int] = graph_dict["node_types"]
        edge_triples: list[tuple[int, int, int]] = graph_dict["edge_triples"]

        found = False
        for i, j, _rel in edge_triples:
            a, b = node_types[i], node_types[j]
            if self.type_a_idx == self.type_b_idx:
                # Same type: both endpoints must match
                if a == self.type_a_idx and b == self.type_a_idx:
                    found = True
                    break
            else:
                # Different types: either direction
                if (a == self.type_a_idx and b == self.type_b_idx) or (
                    a == self.type_b_idx and b == self.type_a_idx
                ):
                    found = True
                    break

        violation = 0.0 if found else 1.0
        return ConstraintResult(
            name=self.name,
            violation=violation,
            satisfied=found,
            details={
                "found": found,
                "type_a_idx": self.type_a_idx,
                "type_b_idx": self.type_b_idx,
            },
        )

    def soft_violation(
        self,
        node_probs: Tensor,
        edge_probs: Tensor,
        pad_mask: Tensor,
        vocab_config: VocabConfig,
    ) -> Tensor:
        p_ij = _compute_adj_terms(
            node_probs, edge_probs, pad_mask, vocab_config,
            self.type_a_idx, self.type_b_idx,
        )
        # P(exists) = 1 - prod(1 - p_ij) via log-space for stability
        eps = 1e-15
        p_ij_clamped = torch.clamp(p_ij, min=0.0, max=1.0 - eps)
        log_complement = torch.log1p(-p_ij_clamped)  # log(1 - p_ij)
        # PAD positions have p_ij=0, so log1p(0)=0, contributing nothing
        p_exists = 1.0 - torch.exp(log_complement.sum())
        return 1.0 - p_exists


# ---------------------------------------------------------------------------
# ForbidAdj
# ---------------------------------------------------------------------------


class ForbidAdj(Constraint):
    """Forbid any adjacency between rooms of ``type_a`` and ``type_b``.

    Hard: ``v = count of forbidden adjacency pairs``.
    Soft: ``v = Σ p_types * P_adj`` — expected count of forbidden adjacencies.
    """

    def __init__(
        self,
        name: str,
        type_a_idx: int,
        type_b_idx: int,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.type_a_idx = type_a_idx
        self.type_b_idx = type_b_idx

    def hard_violation(self, graph_dict: dict) -> ConstraintResult:
        node_types: list[int] = graph_dict["node_types"]
        edge_triples: list[tuple[int, int, int]] = graph_dict["edge_triples"]

        forbidden_count = 0
        for i, j, _rel in edge_triples:
            a, b = node_types[i], node_types[j]
            if self.type_a_idx == self.type_b_idx:
                if a == self.type_a_idx and b == self.type_a_idx:
                    forbidden_count += 1
            else:
                if (a == self.type_a_idx and b == self.type_b_idx) or (
                    a == self.type_b_idx and b == self.type_a_idx
                ):
                    forbidden_count += 1

        violation = float(forbidden_count)
        return ConstraintResult(
            name=self.name,
            violation=violation,
            satisfied=forbidden_count == 0,
            details={
                "forbidden_count": forbidden_count,
                "type_a_idx": self.type_a_idx,
                "type_b_idx": self.type_b_idx,
            },
        )

    def soft_violation(
        self,
        node_probs: Tensor,
        edge_probs: Tensor,
        pad_mask: Tensor,
        vocab_config: VocabConfig,
    ) -> Tensor:
        p_ij = _compute_adj_terms(
            node_probs, edge_probs, pad_mask, vocab_config,
            self.type_a_idx, self.type_b_idx,
        )
        return p_ij.sum()
