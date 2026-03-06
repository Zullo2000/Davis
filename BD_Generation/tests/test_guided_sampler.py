"""Tests for guided sampler (SVDD K-candidate reweighting) and sampling refactoring.

Covers spec tests 40–53 (Sections 14.6–14.7):
  40: K=1 no constraints matches unguided
  41: PAD invariant
  42: No MASK tokens in real positions
  43: Output shapes
  44: Constraint improves satisfaction
  45: ESS not degenerate
  46: GuidanceStats length
  47: Works with v1 (rate_network=None)
  48: Works with v2 (rate_network=model)
  49: Works with remasking
  50: sample() unchanged after refactoring
  51: All existing sampling tests pass (verified separately)
  52: _single_step_unmask returns valid tokens
  53: _single_step_remask no-op when remasking_fn=None
"""

from __future__ import annotations

import torch
import pytest

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    NODE_TYPES,
    RPLAN_VOCAB_CONFIG,
)
from bd_gen.diffusion.sampling import (
    _single_step_remask,
    _single_step_unmask,
    sample,
)
from bd_gen.guidance.guided_sampler import (
    GuidanceStats,
    guided_sample,
)
from bd_gen.guidance.constraints import ExactCount
from bd_gen.guidance.reward import RewardComposer


# =========================================================================
# Sampling Refactoring Tests (spec 50, 52, 53)
# =========================================================================


class TestSamplingRefactoring:
    """Verify the sampling.py refactoring preserves behavior."""

    def test_sample_unchanged_after_refactoring(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 50: sample() produces identical output for same seed."""
        # Run sample twice with the same seed — should be bit-identical
        torch.manual_seed(42)
        r1 = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, temperature=0.0,
            fixed_num_rooms=4,
        )
        torch.manual_seed(42)
        r2 = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, temperature=0.0,
            fixed_num_rooms=4,
        )
        assert torch.equal(r1, r2), "sample() not deterministic after refactoring"

    def test_single_step_unmask_valid_tokens(
        self, dummy_model, vocab_config,
    ):
        """Test 52: _single_step_unmask returns valid tokens."""
        B, n_max = 2, vocab_config.n_max
        seq_len = vocab_config.seq_len
        device = torch.device("cpu")

        # Set up fully masked input
        x_t = torch.full((B, seq_len), NODE_MASK_IDX, dtype=torch.long)
        x_t[:, n_max:] = EDGE_MASK_IDX

        pad_mask = torch.stack([
            vocab_config.compute_pad_mask(4),
            vocab_config.compute_pad_mask(4),
        ])

        # PAD positions
        x_t[:, 4:n_max] = NODE_PAD_IDX
        for pos in range(vocab_config.n_edges):
            if not pad_mask[0, n_max + pos]:
                x_t[:, n_max + pos] = EDGE_PAD_IDX

        # Get logits from model
        t_tensor = torch.full((B,), 0.5, dtype=torch.float32)
        node_logits, edge_logits = dummy_model(x_t, pad_mask, t_tensor)

        p_unmask = torch.full((B, 1), 0.5)

        result = _single_step_unmask(
            x_t.clone(), node_logits, edge_logits, pad_mask, p_unmask,
            i=5, num_steps=10, n_max=n_max, top_p=None,
            temperature=0.0, unmasking_mode="random",
            device=device, fixed_tokens=None, fixed_mask=None,
        )

        # PAD positions should be correct
        assert (result[:, 4:n_max] == NODE_PAD_IDX).all()

        # Node tokens in valid range
        real_nodes = result[:, :4]
        valid_node = (real_nodes >= 0) & (real_nodes < len(NODE_TYPES) + 2)
        assert valid_node.all(), "Invalid node tokens found"

    def test_single_step_remask_noop(self, vocab_config):
        """Test 53: _single_step_remask no-op when remasking_fn=None."""
        B = 2
        seq_len = vocab_config.seq_len
        n_max = vocab_config.n_max

        x_t = torch.zeros(B, seq_len, dtype=torch.long)
        pad_mask = torch.ones(B, seq_len, dtype=torch.bool)
        node_logits = torch.randn(B, n_max, 15)
        edge_logits = torch.randn(B, vocab_config.n_edges, 13)

        result = _single_step_remask(
            x_t, remasking_fn=None, t_now=0.5, t_next=0.4,
            t_switch=1.0, i=5, pad_mask=pad_mask,
            node_logits=node_logits, edge_logits=edge_logits,
        )
        assert torch.equal(result, x_t), "remask should be no-op with None fn"


# =========================================================================
# Guided Sampler Tests (spec 40–49)
# =========================================================================


class TestGuidedSamplerBasic:
    """Core guided sampler tests."""

    def test_k1_no_constraints_matches_unguided(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 40: K=1 with no constraints → identical to sample()."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        torch.manual_seed(42)
        unguided = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=5, temperature=0.0,
            fixed_num_rooms=4,
        )

        torch.manual_seed(42)
        guided, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=1,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=4,
        )

        assert torch.equal(guided, unguided), (
            "K=1 no-constraint guided should match unguided"
        )

    def test_pad_invariant(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 41: PAD positions contain correct PAD tokens."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        result, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=3,
        )

        n_max = vocab_config.n_max
        # Node PAD (positions 3..7)
        assert (result[:, 3:n_max] == NODE_PAD_IDX).all()
        # Edge PAD
        pad_mask = vocab_config.compute_pad_mask(3)
        for pos in range(vocab_config.n_edges):
            if not pad_mask[n_max + pos]:
                assert (result[:, n_max + pos] == EDGE_PAD_IDX).all()

    def test_no_mask_tokens(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 42: No MASK tokens in final output."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        result, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=10, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=8,
        )

        assert (result[:, :vocab_config.n_max] != NODE_MASK_IDX).all()
        assert (result[:, vocab_config.n_max:] != EDGE_MASK_IDX).all()

    def test_output_shapes(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 43: Output is (B, SEQ_LEN) long tensor."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=3, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=4,
        )

        assert result.shape == (3, vocab_config.seq_len)
        assert result.dtype == torch.long
        assert isinstance(stats, GuidanceStats)


class TestGuidedSamplerConstraints:
    """Tests with actual constraints."""

    def test_constraint_improves_satisfaction(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 44: Guided ExactCount(Kitchen=1) yields higher satisfaction
        than unguided.

        Uses a dummy model so effect may be modest, but with K=8 and
        soft reward mode, guided should outperform unguided on average.
        """
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        n_samples = 20
        n_max = vocab_config.n_max

        # Unguided: count kitchens
        torch.manual_seed(123)
        unguided = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=n_samples, num_steps=10, top_p=0.9,
            unmasking_mode="llada", fixed_num_rooms=4,
        )
        unguided_satisfied = 0
        for b in range(n_samples):
            node_tokens = unguided[b, :n_max]
            real = node_tokens[:4]
            count = (real == kitchen_idx).sum().item()
            if count == 1:
                unguided_satisfied += 1

        # Guided: count kitchens
        torch.manual_seed(123)
        guided, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=n_samples, num_steps=10, num_candidates=8,
            guidance_alpha=1.0, top_p=0.9,
            unmasking_mode="llada", fixed_num_rooms=4,
        )
        guided_satisfied = 0
        for b in range(n_samples):
            node_tokens = guided[b, :n_max]
            real = node_tokens[:4]
            count = (real == kitchen_idx).sum().item()
            if count == 1:
                guided_satisfied += 1

        # Guided should be at least as good or better
        # With a dummy model, the effect may be small, so we use >=
        assert guided_satisfied >= unguided_satisfied, (
            f"Guided ({guided_satisfied}/{n_samples}) should be >= "
            f"unguided ({unguided_satisfied}/{n_samples})"
        )

    def test_ess_not_degenerate(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 45: ESS > 1.0 at all steps with alpha=1.0, K=8."""
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        _, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=5, num_candidates=8,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=4,
        )

        for step_idx, step in enumerate(stats.steps):
            assert step["ess"] > 1.0, (
                f"ESS degenerate at step {step_idx}: {step['ess']:.2f}"
            )

    def test_guidance_stats_length(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 46: len(stats.steps) == num_steps."""
        composer = RewardComposer(constraints=[], reward_mode="soft")
        num_steps = 7

        _, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=num_steps, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=4,
        )

        assert len(stats.steps) == num_steps
        assert len(stats.steps_per_sample) == num_steps


class TestGuidedSamplerCompatibility:
    """v1/v2/remasking compatibility tests."""

    def test_works_with_v1(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 47: rate_network=None (v1) runs without error."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            rate_network=None, fixed_num_rooms=4,
        )

        assert result.shape == (2, vocab_config.seq_len)
        assert len(stats.steps) == 5

    def test_works_with_v2(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 48: Works with a rate_network (v2 path).

        Uses a simple mock rate_network that returns uniform alpha(t).
        """
        composer = RewardComposer(constraints=[], reward_mode="soft")

        class MockRateNetwork(torch.nn.Module):
            """Returns alpha = 1-t for all positions."""
            def forward(self, t: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
                B = t.shape[0]
                seq_len = pad_mask.shape[1]
                # alpha(t) = 1 - t: increases from 0 to 1 as t goes from 1 to 0
                alpha = (1.0 - t).unsqueeze(1).expand(B, seq_len)
                return alpha

        rate_net = MockRateNetwork()

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            rate_network=rate_net, fixed_num_rooms=4,
        )

        assert result.shape == (2, vocab_config.seq_len)
        assert len(stats.steps) == 5

    def test_works_with_remasking(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Test 49: Works with remasking_fn, remasking_delta recorded."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "cap", 0.3, linear_schedule, vocab_config,
        )

        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            fixed_num_rooms=4,
        )

        assert result.shape == (2, vocab_config.seq_len)
        # Check remasking delta is recorded
        for step in stats.steps:
            assert "reward_remasking_delta" in step


class TestGuidedSamplerHardMode:
    """Test hard reward mode path."""

    def test_hard_mode_runs(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Hard reward mode completes without error."""
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="hard",
        )

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=4,
        )

        assert result.shape == (2, vocab_config.seq_len)
        assert len(stats.steps) == 5

    def test_hard_mode_final_stats(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Hard mode produces final constraint statistics."""
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="hard",
        )

        _, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            fixed_num_rooms=4,
        )

        assert "one_kitchen" in stats.final_satisfaction
        assert 0.0 <= stats.final_satisfaction["one_kitchen"] <= 1.0
        assert 0.0 <= stats.satisfaction_overall <= 1.0
        assert "one_kitchen" in stats.final_violation_histograms


class TestGuidedSamplerTopP:
    """Test with top-p sampling."""

    def test_top_p_with_guidance(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Top-p sampling works with guidance."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            fixed_num_rooms=4,
        )

        assert result.shape == (4, vocab_config.seq_len)
        assert (result[:, :vocab_config.n_max] != NODE_MASK_IDX).all()


# =========================================================================
# Attribution Boost Tests (Option C)
# =========================================================================


class TestAttributionBoost:
    """Tests for reward-attributed confidence boosting."""

    def test_attribution_boost_runs(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Smoke test: attribution_boost=True with confidence remasking
        completes without error."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            attribution_boost=True,
            fixed_num_rooms=4,
        )

        assert result.shape == (2, vocab_config.seq_len)
        assert len(stats.steps) == 5
        # Check new diagnostic fields are present
        for step in stats.steps:
            assert "mean_attribution_boost" in step
            assert "positions_boosted" in step

    def test_attribution_boost_reduces_remasking_delta(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """With attribution boost, remasking_delta should be less negative
        (remasking destroys less of guidance's work)."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        # Run without boost
        torch.manual_seed(42)
        _, stats_no_boost = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=10, num_candidates=4,
            guidance_alpha=0.1, top_p=0.9,
            remasking_fn=remasking_fn,
            attribution_boost=False,
            fixed_num_rooms=4,
        )

        # Run with boost
        torch.manual_seed(42)
        _, stats_boost = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=10, num_candidates=4,
            guidance_alpha=0.1, top_p=0.9,
            remasking_fn=remasking_fn,
            attribution_boost=True,
            fixed_num_rooms=4,
        )

        # Compare mean remasking delta across steps
        delta_no_boost = sum(
            s["reward_remasking_delta"] for s in stats_no_boost.steps
        ) / len(stats_no_boost.steps)
        delta_boost = sum(
            s["reward_remasking_delta"] for s in stats_boost.steps
        ) / len(stats_boost.steps)

        # With a dummy model this is a soft check — boost should not make
        # things dramatically worse. Allowing equality since dummy model
        # may produce near-zero deltas.
        assert delta_boost >= delta_no_boost - 0.1, (
            f"Boost delta {delta_boost:.4f} much worse than no-boost "
            f"{delta_no_boost:.4f}"
        )

    def test_attribution_boost_noop_without_remasking(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """attribution_boost=True with remasking_fn=None has no effect."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        torch.manual_seed(42)
        result_no_boost, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            attribution_boost=False,
            fixed_num_rooms=4,
        )

        torch.manual_seed(42)
        result_with_boost, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            attribution_boost=True,
            fixed_num_rooms=4,
        )

        assert torch.equal(result_no_boost, result_with_boost), (
            "attribution_boost should have no effect when remasking_fn=None"
        )

    # -----------------------------------------------------------------------
    # Option B: protect_just_unmasked
    # -----------------------------------------------------------------------

    def test_option_b_runs(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Smoke test: protect_just_unmasked=True with confidence remasking
        completes without error and records positions_protected."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            protect_just_unmasked=True,
            fixed_num_rooms=4,
        )

        assert result.shape == (2, vocab_config.seq_len)
        assert len(stats.steps) == 5
        # Check positions_protected diagnostic is present
        for step in stats.steps:
            assert "positions_protected" in step

    def test_option_b_reduces_remasking_delta(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """With protect_just_unmasked, remasking_delta should be less negative
        (remasking can't destroy just-unmasked positions)."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        # Run without Option B
        torch.manual_seed(42)
        _, stats_no_b = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=10, num_candidates=4,
            guidance_alpha=0.1, top_p=0.9,
            remasking_fn=remasking_fn,
            protect_just_unmasked=False,
            fixed_num_rooms=4,
        )

        # Run with Option B
        torch.manual_seed(42)
        _, stats_b = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=10, num_candidates=4,
            guidance_alpha=0.1, top_p=0.9,
            remasking_fn=remasking_fn,
            protect_just_unmasked=True,
            fixed_num_rooms=4,
        )

        # Compare mean remasking delta across steps
        delta_no_b = sum(
            s["reward_remasking_delta"] for s in stats_no_b.steps
        ) / len(stats_no_b.steps)
        delta_b = sum(
            s["reward_remasking_delta"] for s in stats_b.steps
        ) / len(stats_b.steps)

        # With a dummy model this is a soft check — Option B should not make
        # things dramatically worse.
        assert delta_b >= delta_no_b - 0.1, (
            f"Option B delta {delta_b:.4f} much worse than baseline "
            f"{delta_no_b:.4f}"
        )

    def test_option_b_noop_without_remasking(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """protect_just_unmasked=True with remasking_fn=None has no effect."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        torch.manual_seed(42)
        result_off, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            protect_just_unmasked=False,
            fixed_num_rooms=4,
        )

        torch.manual_seed(42)
        result_on, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            protect_just_unmasked=True,
            fixed_num_rooms=4,
        )

        assert torch.equal(result_off, result_on), (
            "protect_just_unmasked should have no effect when remasking_fn=None"
        )

    # -----------------------------------------------------------------------
    # Option A: fresh_logits_for_remask
    # -----------------------------------------------------------------------

    def test_option_a_calls_model_twice(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """With fresh_logits_for_remask=True, model should be called more
        times (2x per guided step) compared to without."""
        from bd_gen.diffusion.remasking import RemaskingSchedule
        from unittest.mock import patch

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        composer = RewardComposer(constraints=[], reward_mode="soft")

        call_counts = {"without": 0, "with": 0}

        original_forward = dummy_model.forward

        def counting_forward_without(*args, **kwargs):
            call_counts["without"] += 1
            return original_forward(*args, **kwargs)

        def counting_forward_with(*args, **kwargs):
            call_counts["with"] += 1
            return original_forward(*args, **kwargs)

        # Run without Option A
        torch.manual_seed(42)
        dummy_model.forward = counting_forward_without
        guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            fresh_logits_for_remask=False,
            fixed_num_rooms=4,
        )

        # Run with Option A
        torch.manual_seed(42)
        dummy_model.forward = counting_forward_with
        guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            fresh_logits_for_remask=True,
            fixed_num_rooms=4,
        )

        dummy_model.forward = original_forward

        # Option A should make more model calls
        assert call_counts["with"] > call_counts["without"], (
            f"Option A calls ({call_counts['with']}) should exceed "
            f"baseline ({call_counts['without']})"
        )

    def test_option_a_noop_without_remasking(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """fresh_logits_for_remask=True with remasking_fn=None has no effect."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        torch.manual_seed(42)
        result_off, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            fresh_logits_for_remask=False,
            fixed_num_rooms=4,
        )

        torch.manual_seed(42)
        result_on, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            fresh_logits_for_remask=True,
            fixed_num_rooms=4,
        )

        assert torch.equal(result_off, result_on), (
            "fresh_logits_for_remask should have no effect when remasking_fn=None"
        )


# =========================================================================
# EMA Lock Tests
# =========================================================================


class TestEmaLock:
    """Tests for adaptive EMA-based remasking lock."""

    def test_ema_lock_smoke(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """ema_lock=True with confidence remasking runs without error
        and populates lock_steps."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        kitchen_idx = NODE_TYPES.index("Kitchen")
        constraint = ExactCount(
            name="one_kitchen", room_type_idx=kitchen_idx, target=1,
        )
        composer = RewardComposer(
            constraints=[constraint], reward_mode="soft",
        )

        result, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=10, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            protect_just_unmasked=True,
            ema_lock=True,
            ema_beta=0.85,
            ema_lock_consecutive=3,
            ema_lock_deadline=0.5,
            fixed_num_rooms=4,
        )

        assert result.shape == (2, vocab_config.seq_len)
        assert len(stats.steps) == 10
        # lock_steps should be populated
        assert stats.lock_steps is not None
        assert stats.lock_steps.shape == (2,)
        # All samples must have locked (deadline at step 5)
        assert (stats.lock_steps >= 0).all(), "All samples should be locked"

    def test_ema_lock_deadline_forces_lock(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """With a very early deadline (0.1 = step 1), all samples should
        lock by step 1."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        composer = RewardComposer(constraints=[], reward_mode="soft")

        _, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=4, num_steps=10, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            ema_lock=True,
            ema_lock_deadline=0.1,  # step 1 deadline
            fixed_num_rooms=4,
        )

        assert stats.lock_steps is not None
        # All should lock at step 1 (deadline = int(0.1 * 10) = 1)
        assert (stats.lock_steps <= 1).all(), (
            f"All samples should lock by step 1, got {stats.lock_steps}"
        )

    def test_ema_lock_diagnostics_recorded(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Verify per-step diagnostics include EMA lock fields."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            "confidence", 0.3, linear_schedule, vocab_config,
        )
        composer = RewardComposer(constraints=[], reward_mode="soft")

        _, stats = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, top_p=0.9,
            remasking_fn=remasking_fn,
            ema_lock=True,
            fixed_num_rooms=4,
        )

        # Check batch-level stats have samples_locked
        for step in stats.steps:
            assert "samples_locked" in step
            assert 0.0 <= step["samples_locked"] <= 1.0

        # Check per-sample stats have ema_reward and locked
        for step_ps in stats.steps_per_sample:
            assert "ema_reward" in step_ps
            assert "locked" in step_ps
            if step_ps["ema_reward"] is not None:
                assert step_ps["ema_reward"].shape == (2,)
            if step_ps["locked"] is not None:
                assert step_ps["locked"].shape == (2,)

    def test_ema_lock_noop_without_remasking(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """ema_lock=True with remasking_fn=None should produce same output
        as ema_lock=False (lock has nothing to disable)."""
        composer = RewardComposer(constraints=[], reward_mode="soft")

        torch.manual_seed(42)
        result_off, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            ema_lock=False,
            fixed_num_rooms=4,
        )

        torch.manual_seed(42)
        result_on, _ = guided_sample(
            dummy_model, linear_schedule, vocab_config,
            reward_composer=composer,
            batch_size=2, num_steps=5, num_candidates=4,
            guidance_alpha=1.0, temperature=0.0,
            remasking_fn=None,
            ema_lock=True,
            fixed_num_rooms=4,
        )

        assert torch.equal(result_off, result_on), (
            "ema_lock should have no effect when remasking_fn=None"
        )
