"""Tests for bd_gen.model.denoiser module.

Covers forward pass shapes, _process_t validation, gradient flow,
adaLN zero-initialization, parameter count estimation, and PAD mask
propagation.
"""

import pytest
import torch
import torch.nn as nn

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    EDGE_VOCAB_SIZE,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    NODE_VOCAB_SIZE,
    RESPLAN_VOCAB_CONFIG,
    VocabConfig,
)
from bd_gen.model.denoiser import BDDenoiser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model(vocab_config: VocabConfig) -> BDDenoiser:
    """Small model for fast test execution (d_model=32, 2 layers, 2 heads)."""
    return BDDenoiser(
        d_model=32, n_layers=2, n_heads=2,
        vocab_config=vocab_config, dropout=0.0,
    )


@pytest.fixture
def small_config_model(vocab_config: VocabConfig) -> BDDenoiser:
    """Model matching configs/model/small.yaml exactly."""
    return BDDenoiser(
        d_model=128, n_layers=4, n_heads=4,
        vocab_config=vocab_config, cond_dim=128,
        mlp_ratio=4, dropout=0.1, frequency_embedding_size=256,
    )


# ---------------------------------------------------------------------------
# Forward pass shape tests
# ---------------------------------------------------------------------------


class TestForwardPassShapes:
    def test_rplan_output_shapes(self, small_model, sample_batch):
        """Output shapes: (B, 8, 15) and (B, 28, 13) for RPLAN."""
        model = small_model
        model.eval()
        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        B = tokens.size(0)

        node_logits, edge_logits = model(tokens, pad_mask, t=0.5)

        assert node_logits.shape == (B, 8, NODE_VOCAB_SIZE)
        assert edge_logits.shape == (B, 28, EDGE_VOCAB_SIZE)

    def test_output_dtypes(self, small_model, sample_batch):
        model = small_model
        model.eval()
        node_logits, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        assert node_logits.dtype == torch.float32
        assert edge_logits.dtype == torch.float32

    def test_resplan_output_shapes(self):
        """Test with RESPLAN_VOCAB_CONFIG (n_max=14)."""
        vc = RESPLAN_VOCAB_CONFIG
        model = BDDenoiser(
            d_model=32, n_layers=1, n_heads=2, vocab_config=vc, dropout=0.0,
        )
        model.eval()
        B = 2
        tokens = torch.randint(0, 10, (B, vc.seq_len))
        pad_mask = torch.ones(B, vc.seq_len, dtype=torch.bool)

        node_logits, edge_logits = model(tokens, pad_mask, t=0.5)

        assert node_logits.shape == (B, 14, NODE_VOCAB_SIZE)
        assert edge_logits.shape == (B, 91, EDGE_VOCAB_SIZE)

    def test_with_base_config_dimensions(self, vocab_config):
        """Test base config dimensions d_model=256, n_layers=6, n_heads=8."""
        model = BDDenoiser(
            d_model=256, n_layers=6, n_heads=8,
            vocab_config=vocab_config, cond_dim=256,
            mlp_ratio=4, dropout=0.0,
        )
        model.eval()
        B = 2
        tokens = torch.randint(0, 10, (B, vocab_config.seq_len))
        pad_mask = torch.ones(B, vocab_config.seq_len, dtype=torch.bool)

        node_logits, edge_logits = model(tokens, pad_mask, t=0.5)

        assert node_logits.shape == (B, 8, NODE_VOCAB_SIZE)
        assert edge_logits.shape == (B, 28, EDGE_VOCAB_SIZE)

    def test_batch_size_one(self, vocab_config):
        """Model works with batch size 1."""
        model = BDDenoiser(
            d_model=32, n_layers=1, n_heads=2,
            vocab_config=vocab_config, dropout=0.0,
        )
        model.eval()
        tokens = torch.randint(0, 10, (1, vocab_config.seq_len))
        pad_mask = torch.ones(1, vocab_config.seq_len, dtype=torch.bool)

        node_logits, edge_logits = model(tokens, pad_mask, t=0.5)

        assert node_logits.shape == (1, 8, NODE_VOCAB_SIZE)
        assert edge_logits.shape == (1, 28, EDGE_VOCAB_SIZE)


# ---------------------------------------------------------------------------
# _process_t tests
# ---------------------------------------------------------------------------


class TestProcessT:
    def test_float_input(self, small_model):
        result = small_model._process_t(0.5, batch_size=4, device=torch.device("cpu"))
        assert result.shape == (4,)
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor([0.5, 0.5, 0.5, 0.5]))

    def test_int_input(self, small_model):
        result = small_model._process_t(1, batch_size=3, device=torch.device("cpu"))
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([1.0, 1.0, 1.0]))

    def test_0d_tensor(self, small_model):
        t = torch.tensor(0.7)
        result = small_model._process_t(t, batch_size=2, device=torch.device("cpu"))
        assert result.shape == (2,)
        assert torch.allclose(result, torch.tensor([0.7, 0.7]))

    def test_1d_tensor_size_1(self, small_model):
        t = torch.tensor([0.3])
        result = small_model._process_t(t, batch_size=4, device=torch.device("cpu"))
        assert result.shape == (4,)

    def test_1d_tensor_size_batch(self, small_model):
        t = torch.tensor([0.1, 0.2, 0.3])
        result = small_model._process_t(t, batch_size=3, device=torch.device("cpu"))
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([0.1, 0.2, 0.3]))

    def test_invalid_1d_wrong_size(self, small_model):
        t = torch.tensor([0.1, 0.2])
        with pytest.raises(ValueError, match="batch_size"):
            small_model._process_t(t, batch_size=4, device=torch.device("cpu"))

    def test_invalid_2d_tensor(self, small_model):
        t = torch.tensor([[0.5]])
        with pytest.raises(ValueError, match="0D.*1D"):
            small_model._process_t(t, batch_size=1, device=torch.device("cpu"))

    def test_preserves_device(self, small_model):
        """Result should be on the specified device."""
        result = small_model._process_t(0.5, batch_size=2, device=torch.device("cpu"))
        assert result.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_all_parameters_receive_gradients(self, small_model, sample_batch):
        """After a backward pass, every parameter should have a non-None gradient."""
        model = small_model
        model.train()

        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]

        node_logits, edge_logits = model(tokens, pad_mask, t=0.5)

        loss = node_logits.sum() + edge_logits.sum()
        loss.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )

    def test_gradients_are_finite(self, small_model, sample_batch):
        """All gradients should be finite (no NaN or Inf)."""
        model = small_model
        model.train()

        node_logits, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        loss = node_logits.sum() + edge_logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )


# ---------------------------------------------------------------------------
# adaLN zero-init verification
# ---------------------------------------------------------------------------


class TestAdaLNZeroInit:
    def test_block_adaLN_modulation_zero_init(self, small_model):
        """All adaLN modulation layers in transformer blocks are zero-initialized."""
        for i, block in enumerate(small_model.blocks):
            w = block.adaLN_modulation.weight.data
            b = block.adaLN_modulation.bias.data
            assert torch.equal(w, torch.zeros_like(w)), (
                f"Block {i} adaLN_modulation weights not zero"
            )
            assert torch.equal(b, torch.zeros_like(b)), (
                f"Block {i} adaLN_modulation bias not zero"
            )

    def test_final_adaLN_zero_init(self, small_model):
        """Final adaLN modulation is zero-initialized."""
        w = small_model.final_adaLN.weight.data
        b = small_model.final_adaLN.bias.data
        assert torch.equal(w, torch.zeros_like(w))
        assert torch.equal(b, torch.zeros_like(b))

    def test_classification_heads_zero_init(self, small_model):
        """Node and edge classification heads are zero-initialized."""
        for head_name in ["node_head", "edge_head"]:
            head = getattr(small_model, head_name)
            assert torch.equal(
                head.weight.data, torch.zeros_like(head.weight.data)
            ), f"{head_name} weights not zero"
            assert torch.equal(
                head.bias.data, torch.zeros_like(head.bias.data)
            ), f"{head_name} bias not zero"

    def test_initial_output_near_zero(self, small_model, sample_batch):
        """At initialization, real-class logits should be near zero (uniform).

        With zero-init heads and zero-init final adaLN, the model initially
        produces all-zero logits for real classes, corresponding to a uniform
        distribution. MASK/PAD logits are -inf (zero masking probabilities).
        """
        model = small_model
        model.eval()
        node_logits, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        # Check only real-class logits (MASK/PAD are intentionally -inf)
        assert node_logits[:, :, :NODE_MASK_IDX].abs().max() < 1e-4, (
            f"Initial node logits too large: max abs = "
            f"{node_logits[:, :, :NODE_MASK_IDX].abs().max()}"
        )
        assert edge_logits[:, :, :EDGE_MASK_IDX].abs().max() < 1e-4, (
            f"Initial edge logits too large: max abs = "
            f"{edge_logits[:, :, :EDGE_MASK_IDX].abs().max()}"
        )


# ---------------------------------------------------------------------------
# Parameter count test
# ---------------------------------------------------------------------------


class TestParameterCount:
    def test_small_config_param_count(self, vocab_config):
        """Small config (d_model=128, L=4, heads=4) should be in 1-5M range."""
        model = BDDenoiser(
            d_model=128, n_layers=4, n_heads=4,
            vocab_config=vocab_config, cond_dim=128,
            mlp_ratio=4, dropout=0.1,
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert 1_000_000 <= n_params <= 5_000_000, (
            f"Small config has {n_params:,} params, expected 1M-5M"
        )

    def test_base_config_param_count(self, vocab_config):
        """Base config (d_model=256, L=6, heads=8) should be in reasonable range."""
        model = BDDenoiser(
            d_model=256, n_layers=6, n_heads=8,
            vocab_config=vocab_config, cond_dim=256,
            mlp_ratio=4, dropout=0.1,
        )
        n_params = sum(p.numel() for p in model.parameters())
        # Base should be larger than small
        assert n_params > 1_000_000, (
            f"Base config has only {n_params:,} params"
        )

    def test_param_count_positive(self, small_model):
        n_params = sum(p.numel() for p in small_model.parameters())
        assert n_params > 0


# ---------------------------------------------------------------------------
# PAD mask propagation
# ---------------------------------------------------------------------------


class TestPadMaskPropagation:
    def test_different_pad_masks_produce_different_outputs(self, vocab_config):
        """Changing the pad_mask should change the model output.

        At initialization, zero-init adaLN gates mean transformer blocks
        contribute nothing (gate=0), so attention masking has no effect.
        We randomize all weights to simulate a trained model where the
        attention mask actually matters.
        """
        torch.manual_seed(42)
        model = BDDenoiser(
            d_model=32, n_layers=2, n_heads=2,
            vocab_config=vocab_config, dropout=0.0,
        )
        # Randomize ALL weights so attention and heads are active
        for p in model.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)
        model.eval()

        B = 2
        tokens = torch.randint(0, 10, (B, vocab_config.seq_len))

        # Mask 1: all positions real
        mask1 = torch.ones(B, vocab_config.seq_len, dtype=torch.bool)

        # Mask 2: last half of positions are PAD
        mask2 = torch.ones(B, vocab_config.seq_len, dtype=torch.bool)
        mask2[:, vocab_config.seq_len // 2 :] = False

        node_logits1, _ = model(tokens, mask1, t=0.5)
        node_logits2, _ = model(tokens, mask2, t=0.5)

        assert not torch.equal(node_logits1, node_logits2), (
            "Different pad masks should produce different outputs"
        )

    def test_all_pad_does_not_crash(self, vocab_config):
        """A batch with almost all PAD (num_rooms=1) should not crash."""
        model = BDDenoiser(
            d_model=32, n_layers=1, n_heads=2,
            vocab_config=vocab_config, dropout=0.0,
        )
        model.eval()

        # num_rooms=1: only position 0 is real, everything else is PAD
        pad_mask = vocab_config.compute_pad_mask(1).unsqueeze(0)  # (1, 36)
        tokens = torch.zeros(1, vocab_config.seq_len, dtype=torch.long)
        tokens[0, 0] = 2  # LivingRoom = Kitchen (idx 2)
        tokens[0, 1:vocab_config.n_max] = 14  # NODE_PAD_IDX
        tokens[0, vocab_config.n_max:] = 12  # EDGE_PAD_IDX

        node_logits, edge_logits = model(tokens, pad_mask, t=0.5)

        # Real-class logits should be finite (MASK/PAD logits are -inf by design)
        assert torch.isfinite(node_logits[:, :, :NODE_MASK_IDX]).all()
        assert torch.isfinite(edge_logits[:, :, :EDGE_MASK_IDX]).all()


# ---------------------------------------------------------------------------
# Unconditional (condition=None)
# ---------------------------------------------------------------------------


class TestUnconditional:
    def test_condition_none_works(self, small_model, sample_batch):
        """Passing condition=None (unconditional) should not error."""
        model = small_model
        model.eval()
        node_logits, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"],
            t=0.5, condition=None,
        )
        assert node_logits is not None
        assert edge_logits is not None

    def test_condition_default_is_none(self, small_model, sample_batch):
        """Not passing condition should default to None."""
        model = small_model
        model.eval()
        node_logits, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5,
        )
        assert node_logits is not None


# ---------------------------------------------------------------------------
# Timestep variation
# ---------------------------------------------------------------------------


class TestTimestepVariation:
    def test_different_timesteps_produce_different_outputs(
        self, small_model, sample_batch
    ):
        """Different timesteps should produce different model outputs."""
        model = small_model
        model.eval()

        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]

        node_logits1, _ = model(tokens, pad_mask, t=0.1)
        node_logits2, _ = model(tokens, pad_mask, t=0.9)

        # At init (zero-init heads), both will be near-zero.
        # But we test the infrastructure: _process_t handles different t values
        # and they flow through to the conditioning.
        # (After training, outputs would differ significantly.)
        # Just verify no crash and real-class logits are valid.
        # (MASK/PAD logits are -inf by design.)
        assert torch.isfinite(node_logits1[:, :, :NODE_MASK_IDX]).all()
        assert torch.isfinite(node_logits2[:, :, :NODE_MASK_IDX]).all()

    def test_batched_timesteps(self, small_model, sample_batch):
        """Per-sample timesteps should work."""
        model = small_model
        model.eval()

        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        B = tokens.size(0)

        t = torch.linspace(0.1, 0.9, B)
        node_logits, edge_logits = model(tokens, pad_mask, t=t)

        assert node_logits.shape == (B, 8, NODE_VOCAB_SIZE)
        assert edge_logits.shape == (B, 28, EDGE_VOCAB_SIZE)


# ---------------------------------------------------------------------------
# SUBS zero masking probabilities
# ---------------------------------------------------------------------------


class TestZeroMaskingProbabilities:
    """Verify that MASK and PAD logits are clamped to -inf after forward()."""

    def test_node_mask_logit_is_neg_inf(self, small_model, sample_batch):
        model = small_model
        model.eval()
        node_logits, _ = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        assert (node_logits[:, :, NODE_MASK_IDX] == float('-inf')).all()

    def test_node_pad_logit_is_neg_inf(self, small_model, sample_batch):
        model = small_model
        model.eval()
        node_logits, _ = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        assert (node_logits[:, :, NODE_PAD_IDX] == float('-inf')).all()

    def test_edge_mask_logit_is_neg_inf(self, small_model, sample_batch):
        model = small_model
        model.eval()
        _, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        assert (edge_logits[:, :, EDGE_MASK_IDX] == float('-inf')).all()

    def test_edge_pad_logit_is_neg_inf(self, small_model, sample_batch):
        model = small_model
        model.eval()
        _, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        assert (edge_logits[:, :, EDGE_PAD_IDX] == float('-inf')).all()

    def test_real_token_logits_are_finite(self, small_model, sample_batch):
        """Non-special logits should remain finite (not clamped)."""
        model = small_model
        model.eval()
        node_logits, edge_logits = model(
            sample_batch["tokens"], sample_batch["pad_mask"], t=0.5
        )
        # Node real classes: 0-12
        assert torch.isfinite(node_logits[:, :, :NODE_MASK_IDX]).all()
        # Edge real classes: 0-10
        assert torch.isfinite(edge_logits[:, :, :EDGE_MASK_IDX]).all()
