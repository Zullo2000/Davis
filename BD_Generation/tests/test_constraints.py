"""Tests for the guidance constraint system.

Covers:
  - Hard violation tests (spec tests 1-11): ExactCount, CountRange, RequireAdj, ForbidAdj
  - RewardComposer hard mode (spec tests 26-32): energy, reward, phi shaping, calibration
  - Edge cases: validation errors, frozen dataclass, details dicts, soft_violation stub

NODE_TYPES indices used in these tests:
    LivingRoom=0, MasterRoom=1, Kitchen=2, Bathroom=3, DiningRoom=4,
    ChildRoom=5, StudyRoom=6, SecondRoom=7, GuestRoom=8, Balcony=9,
    Entrance=10, Storage=11, Wall-in=12
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from bd_gen.guidance.constraints import (
    Constraint,
    ConstraintResult,
    CountRange,
    ExactCount,
    ForbidAdj,
    RequireAdj,
)
from bd_gen.guidance.reward import RewardComposer


# =========================================================================
# Helpers
# =========================================================================


def _make_graph(
    node_types: list[int],
    edge_triples: list[tuple[int, int, int]] | None = None,
) -> dict:
    """Build a minimal graph_dict for constraint evaluation."""
    return {
        "num_rooms": len(node_types),
        "node_types": node_types,
        "edge_triples": edge_triples or [],
    }


# =========================================================================
# Hard Violation Tests -- ExactCount (spec tests 1-3)
# =========================================================================


class TestExactCount:
    """ExactCount: v = |count(type) - target|."""

    def test_exact_count_satisfied(self):
        """(1) 1 Kitchen in [Kitchen, LivingRoom, Bathroom], target=1 -> v=0."""
        graph = _make_graph(node_types=[2, 0, 3])
        c = ExactCount(name="exactly_1_kitchen", room_type_idx=2, target=1)
        result = c.hard_violation(graph)
        assert result.violation == 0.0
        assert result.satisfied is True

    def test_exact_count_over(self):
        """(2) 3 Kitchens in [Kitchen, Kitchen, Kitchen], target=1 -> v=2.0."""
        graph = _make_graph(node_types=[2, 2, 2])
        c = ExactCount(name="exactly_1_kitchen", room_type_idx=2, target=1)
        result = c.hard_violation(graph)
        assert result.violation == 2.0
        assert result.satisfied is False

    def test_exact_count_under(self):
        """(3) 0 Kitchens in [LivingRoom, Bathroom, SecondRoom], target=1 -> v=1.0."""
        graph = _make_graph(node_types=[0, 3, 7])
        c = ExactCount(name="exactly_1_kitchen", room_type_idx=2, target=1)
        result = c.hard_violation(graph)
        assert result.violation == 1.0
        assert result.satisfied is False

    def test_exact_count_details(self):
        """Details dict contains 'count' and 'target' keys."""
        graph = _make_graph(node_types=[2, 0, 3])
        c = ExactCount(name="exactly_1_kitchen", room_type_idx=2, target=1)
        result = c.hard_violation(graph)
        assert "count" in result.details
        assert "target" in result.details
        assert result.details["count"] == 1
        assert result.details["target"] == 1


# =========================================================================
# Hard Violation Tests -- CountRange (spec tests 4-6)
# =========================================================================


class TestCountRange:
    """CountRange: v = max(0, lo - count) + max(0, count - hi)."""

    def test_count_range_in_range(self):
        """(4) 2 SecondRooms in [SR, SR, LR, Kitchen], range [1, 4] -> v=0."""
        graph = _make_graph(node_types=[7, 7, 0, 2])
        c = CountRange(name="sr_range", room_type_idx=7, lo=1, hi=4)
        result = c.hard_violation(graph)
        assert result.violation == 0.0
        assert result.satisfied is True

    def test_count_range_under(self):
        """(5) 0 SecondRooms in [LR, Kitchen, Bathroom], range [1, 4] -> v=1.0."""
        graph = _make_graph(node_types=[0, 2, 3])
        c = CountRange(name="sr_range", room_type_idx=7, lo=1, hi=4)
        result = c.hard_violation(graph)
        assert result.violation == 1.0
        assert result.satisfied is False

    def test_count_range_over(self):
        """(6) 5 SecondRooms in [SR]*5, range [1, 4] -> v=1.0."""
        graph = _make_graph(node_types=[7, 7, 7, 7, 7])
        c = CountRange(name="sr_range", room_type_idx=7, lo=1, hi=4)
        result = c.hard_violation(graph)
        assert result.violation == 1.0
        assert result.satisfied is False

    def test_count_range_lo_gt_hi_raises(self):
        """CountRange(lo=5, hi=3) should raise ValueError."""
        with pytest.raises(ValueError, match="lo.*must be <= hi"):
            CountRange(name="bad", room_type_idx=0, lo=5, hi=3)


# =========================================================================
# Hard Violation Tests -- RequireAdj (spec tests 7-8)
# =========================================================================


class TestRequireAdj:
    """RequireAdj: v=0 if edge exists between required types, else v=1."""

    def test_require_adj_satisfied(self):
        """(7) Kitchen(2) at 0, LivingRoom(0) at 1, edge (0,1,3) -> v=0."""
        graph = _make_graph(
            node_types=[2, 0],
            edge_triples=[(0, 1, 3)],
        )
        c = RequireAdj(name="kitchen_adj_living", type_a_idx=2, type_b_idx=0)
        result = c.hard_violation(graph)
        assert result.violation == 0.0
        assert result.satisfied is True

    def test_require_adj_missing(self):
        """(8) Kitchen(2) at 0, LivingRoom(0) at 1, no edges -> v=1.0."""
        graph = _make_graph(
            node_types=[2, 0],
            edge_triples=[],
        )
        c = RequireAdj(name="kitchen_adj_living", type_a_idx=2, type_b_idx=0)
        result = c.hard_violation(graph)
        assert result.violation == 1.0
        assert result.satisfied is False


# =========================================================================
# Hard Violation Tests -- ForbidAdj (spec tests 9-11)
# =========================================================================


class TestForbidAdj:
    """ForbidAdj: v = count of forbidden adjacency pairs."""

    def test_forbid_adj_satisfied(self):
        """(9) Kitchen-LivingRoom edge but forbid Bathroom-Kitchen -> v=0."""
        graph = _make_graph(
            node_types=[2, 0],  # Kitchen, LivingRoom
            edge_triples=[(0, 1, 3)],
        )
        # Forbid Bathroom(3)-Kitchen(2), but no Bathroom present -> v=0
        c = ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2)
        result = c.hard_violation(graph)
        assert result.violation == 0.0
        assert result.satisfied is True

    def test_forbid_adj_one_pair(self):
        """(10) Bathroom(3) at 0, Kitchen(2) at 1, edge (0,1,2) -> v=1.0."""
        graph = _make_graph(
            node_types=[3, 2],  # Bathroom, Kitchen
            edge_triples=[(0, 1, 2)],
        )
        c = ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2)
        result = c.hard_violation(graph)
        assert result.violation == 1.0
        assert result.satisfied is False

    def test_forbid_adj_multiple(self):
        """(11) Bathroom at 0 & 2, Kitchen at 1, edges (0,1,2) and (1,2,3) -> v=2.0."""
        graph = _make_graph(
            node_types=[3, 2, 3],  # Bathroom, Kitchen, Bathroom
            edge_triples=[(0, 1, 2), (1, 2, 3)],
        )
        c = ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2)
        result = c.hard_violation(graph)
        assert result.violation == 2.0
        assert result.satisfied is False


# =========================================================================
# RewardComposer -- Hard Mode (spec tests 26-32)
# =========================================================================


class TestRewardComposerHardMode:
    """RewardComposer: E(x) = Sum (lambda_i / p90_i) * phi(v_i(x)), r = -E."""

    def test_energy_non_negative(self):
        """(26) Energy should always be >= 0, even with violations."""
        graph = _make_graph(node_types=[2, 2, 2])  # 3 Kitchens, target 1
        constraints = [
            ExactCount(name="kitchen", room_type_idx=2, target=1),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, details = composer.compute_energy_hard(graph)
        assert energy >= 0.0

    def test_reward_non_positive(self):
        """(27) Reward should always be <= 0."""
        graph = _make_graph(node_types=[2, 2, 2])  # 3 Kitchens, target 1
        constraints = [
            ExactCount(name="kitchen", room_type_idx=2, target=1),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")
        reward, details = composer.compute_reward_hard(graph)
        assert reward <= 0.0

    def test_perfect_graph_zero_energy(self):
        """(28) A graph satisfying all constraints should have E=0, r=0."""
        # 1 Kitchen, 1 Bathroom, Kitchen adj LivingRoom, no Bathroom-Kitchen edge
        graph = _make_graph(
            node_types=[2, 0, 3],  # Kitchen, LivingRoom, Bathroom
            edge_triples=[(0, 1, 3)],  # Kitchen-LivingRoom edge
        )
        constraints = [
            ExactCount(name="1_kitchen", room_type_idx=2, target=1),
            ExactCount(name="1_bathroom", room_type_idx=3, target=1),
            RequireAdj(name="kitchen_adj_living", type_a_idx=2, type_b_idx=0),
            ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, details = composer.compute_energy_hard(graph)
        reward, _ = composer.compute_reward_hard(graph)
        assert energy == 0.0
        assert reward == 0.0
        # All results should be satisfied
        for name, result in details.items():
            assert result.satisfied is True, f"{name} not satisfied"

    def test_phi_quadratic(self):
        """(29) phi='quadratic', violation=2 -> contribution uses 2^2=4."""
        graph = _make_graph(node_types=[2, 2, 2])  # 3 Kitchens, target 1 -> v=2
        constraints = [
            ExactCount(name="kitchen", room_type_idx=2, target=1, weight=1.0),
        ]
        composer = RewardComposer(constraints=constraints, phi="quadratic")
        energy, _ = composer.compute_energy_hard(graph)
        # E = (1.0 / 1.0) * (2.0 ** 2) = 4.0
        assert energy == pytest.approx(4.0)

    def test_phi_log1p(self):
        """(30) phi='log1p', violation=2 -> contribution uses log(1+2)=log(3)."""
        graph = _make_graph(node_types=[2, 2, 2])  # 3 Kitchens, target 1 -> v=2
        constraints = [
            ExactCount(name="kitchen", room_type_idx=2, target=1, weight=1.0),
        ]
        composer = RewardComposer(constraints=constraints, phi="log1p")
        energy, _ = composer.compute_energy_hard(graph)
        # E = (1.0 / 1.0) * log(1 + 2) = log(3)
        assert energy == pytest.approx(math.log(3))

    def test_calibration_loading(self):
        """(31) load_calibration updates p90_normalizer on matching constraints."""
        c1 = ExactCount(name="kitchen", room_type_idx=2, target=1)
        c2 = CountRange(name="sr_range", room_type_idx=7, lo=1, hi=4)
        composer = RewardComposer(constraints=[c1, c2], phi="linear")

        # Default p90 is 1.0
        assert c1.p90_normalizer == 1.0
        assert c2.p90_normalizer == 1.0

        composer.load_calibration({"kitchen": 3.5, "sr_range": 2.0})
        assert c1.p90_normalizer == 3.5
        assert c2.p90_normalizer == 2.0

    def test_calibration_effect(self):
        """(32) With P90=2.0, weight=1.0, violation=4, phi='linear' -> term = (1/2)*4 = 2.0."""
        graph = _make_graph(node_types=[2, 2, 2, 2, 2])  # 5 Kitchens, target 1 -> v=4
        c = ExactCount(name="kitchen", room_type_idx=2, target=1, weight=1.0)
        c.p90_normalizer = 2.0
        composer = RewardComposer(constraints=[c], phi="linear")
        energy, _ = composer.compute_energy_hard(graph)
        # E = (1.0 / 2.0) * 4.0 = 2.0
        assert energy == pytest.approx(2.0)


# =========================================================================
# Additional Edge-Case Tests
# =========================================================================


class TestConstraintEdgeCases:
    """Miscellaneous edge cases and structural checks."""

    def test_constraint_result_dataclass_is_frozen(self):
        """ConstraintResult should be a frozen dataclass (immutable)."""
        result = ConstraintResult(
            name="test", violation=0.0, satisfied=True, details={}
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.violation = 5.0  # type: ignore[misc]

    def test_soft_violation_returns_tensor(self):
        """All constraint soft_violation() methods return a scalar float64 tensor."""
        import torch
        import torch.nn.functional as F
        from bd_gen.data.vocab import (
            NODE_VOCAB_SIZE,
            EDGE_VOCAB_SIZE,
            VocabConfig,
        )

        vc = VocabConfig(n_max=8)
        # Build valid probability distributions (softmax of random logits)
        node_probs = F.softmax(torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64), dim=-1)
        edge_probs = F.softmax(torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64), dim=-1)
        pad_mask = vc.compute_pad_mask(4)

        # Zero PAD positions
        node_probs[~pad_mask[:vc.n_max]] = 0.0
        edge_probs[~pad_mask[vc.n_max:]] = 0.0

        constraints: list[Constraint] = [
            ExactCount(name="a", room_type_idx=0, target=1),
            CountRange(name="b", room_type_idx=0, lo=0, hi=2),
            RequireAdj(name="c", type_a_idx=0, type_b_idx=1),
            ForbidAdj(name="d", type_a_idx=0, type_b_idx=1),
        ]
        for c in constraints:
            v = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
            assert isinstance(v, torch.Tensor), f"{c.name}: not a tensor"
            assert v.dim() == 0, f"{c.name}: not scalar"
            assert v.item() >= 0.0, f"{c.name}: negative violation"

    def test_exact_count_violation_is_float(self):
        """Violation values must always be float, never int (spec Section 2.8)."""
        graph = _make_graph(node_types=[2, 2, 2])
        c = ExactCount(name="kitchen", room_type_idx=2, target=1)
        result = c.hard_violation(graph)
        assert isinstance(result.violation, float)

    def test_count_range_violation_is_float(self):
        """CountRange violations must be float."""
        graph = _make_graph(node_types=[7, 7, 7, 7, 7])
        c = CountRange(name="sr_range", room_type_idx=7, lo=1, hi=4)
        result = c.hard_violation(graph)
        assert isinstance(result.violation, float)

    def test_require_adj_violation_is_float(self):
        """RequireAdj violations must be float."""
        graph = _make_graph(node_types=[2, 0], edge_triples=[])
        c = RequireAdj(name="adj", type_a_idx=2, type_b_idx=0)
        result = c.hard_violation(graph)
        assert isinstance(result.violation, float)

    def test_forbid_adj_violation_is_float(self):
        """ForbidAdj violations must be float."""
        graph = _make_graph(node_types=[3, 2], edge_triples=[(0, 1, 2)])
        c = ForbidAdj(name="forbid", type_a_idx=3, type_b_idx=2)
        result = c.hard_violation(graph)
        assert isinstance(result.violation, float)

    def test_require_adj_reversed_direction(self):
        """RequireAdj should match regardless of edge direction vs type_a/type_b order."""
        # type_a=LivingRoom(0), type_b=Kitchen(2), but edge goes Kitchen -> LivingRoom
        graph = _make_graph(
            node_types=[2, 0],  # node 0 = Kitchen, node 1 = LivingRoom
            edge_triples=[(0, 1, 3)],
        )
        c = RequireAdj(name="living_adj_kitchen", type_a_idx=0, type_b_idx=2)
        result = c.hard_violation(graph)
        assert result.violation == 0.0
        assert result.satisfied is True

    def test_calibration_warns_missing_constraint(self):
        """load_calibration warns when a constraint name is missing from the dict."""
        c = ExactCount(name="kitchen", room_type_idx=2, target=1)
        composer = RewardComposer(constraints=[c], phi="linear")
        with pytest.warns(UserWarning, match="not found in calibration"):
            composer.load_calibration({"nonexistent": 2.0})
        # p90 should remain at default
        assert c.p90_normalizer == 1.0

    def test_reward_composer_multiple_constraints_accumulate(self):
        """Energy accumulates contributions from all constraints."""
        # 3 Kitchens (target 1) -> v=2, 0 SecondRooms (range [1,4]) -> v=1
        graph = _make_graph(node_types=[2, 2, 2])
        constraints = [
            ExactCount(name="kitchen", room_type_idx=2, target=1, weight=1.0),
            CountRange(name="sr_range", room_type_idx=7, lo=1, hi=4, weight=1.0),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, details = composer.compute_energy_hard(graph)
        # E = (1/1)*2.0 + (1/1)*1.0 = 3.0
        assert energy == pytest.approx(3.0)
        assert not details["kitchen"].satisfied
        assert not details["sr_range"].satisfied

    def test_reward_is_negative_energy(self):
        """Reward should be exactly -energy."""
        graph = _make_graph(node_types=[2, 2, 2])
        constraints = [
            ExactCount(name="kitchen", room_type_idx=2, target=1, weight=1.0),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, _ = composer.compute_energy_hard(graph)
        reward, _ = composer.compute_reward_hard(graph)
        assert reward == pytest.approx(-energy)

    def test_phi_linear(self):
        """phi='linear', violation=2 -> contribution = 2.0 (identity)."""
        graph = _make_graph(node_types=[2, 2, 2])  # v=2
        constraints = [
            ExactCount(name="kitchen", room_type_idx=2, target=1, weight=1.0),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, _ = composer.compute_energy_hard(graph)
        assert energy == pytest.approx(2.0)

    def test_invalid_phi_raises(self):
        """Unknown phi function should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown phi"):
            RewardComposer(constraints=[], phi="cubic")  # type: ignore[arg-type]
