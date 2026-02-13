"""Tests for bd_gen.model.embeddings module."""

import torch

from bd_gen.data.vocab import (
    EDGE_VOCAB_SIZE,
    NODE_VOCAB_SIZE,
    RESPLAN_VOCAB_CONFIG,
    VocabConfig,
)
from bd_gen.model.embeddings import (
    CompositePositionalEncoding,
    EdgeEmbedding,
    NodeEmbedding,
    TimestepEmbedding,
)

# ---------------------------------------------------------------------------
# NodeEmbedding
# ---------------------------------------------------------------------------


class TestNodeEmbedding:
    def test_output_shape(self):
        emb = NodeEmbedding(d_model=64)
        x = torch.randint(0, NODE_VOCAB_SIZE, (4, 8))
        out = emb(x)
        assert out.shape == (4, 8, 64)

    def test_output_dtype(self):
        emb = NodeEmbedding(d_model=32)
        x = torch.randint(0, NODE_VOCAB_SIZE, (2, 8))
        assert emb(x).dtype == torch.float32

    def test_single_sample(self):
        emb = NodeEmbedding(d_model=16)
        x = torch.randint(0, NODE_VOCAB_SIZE, (1, 3))
        out = emb(x)
        assert out.shape == (1, 3, 16)

    def test_all_vocab_indices(self):
        """Every valid index (0 to NODE_VOCAB_SIZE-1) should embed without error."""
        emb = NodeEmbedding(d_model=32)
        x = torch.arange(NODE_VOCAB_SIZE).unsqueeze(0)  # (1, 15)
        out = emb(x)
        assert out.shape == (1, NODE_VOCAB_SIZE, 32)


# ---------------------------------------------------------------------------
# EdgeEmbedding
# ---------------------------------------------------------------------------


class TestEdgeEmbedding:
    def test_output_shape(self):
        emb = EdgeEmbedding(d_model=64)
        x = torch.randint(0, EDGE_VOCAB_SIZE, (4, 28))
        out = emb(x)
        assert out.shape == (4, 28, 64)

    def test_output_dtype(self):
        emb = EdgeEmbedding(d_model=32)
        x = torch.randint(0, EDGE_VOCAB_SIZE, (2, 28))
        assert emb(x).dtype == torch.float32

    def test_all_vocab_indices(self):
        """Every valid index (0 to EDGE_VOCAB_SIZE-1) should embed without error."""
        emb = EdgeEmbedding(d_model=32)
        x = torch.arange(EDGE_VOCAB_SIZE).unsqueeze(0)  # (1, 13)
        out = emb(x)
        assert out.shape == (1, EDGE_VOCAB_SIZE, 32)


# ---------------------------------------------------------------------------
# CompositePositionalEncoding
# ---------------------------------------------------------------------------


class TestCompositePositionalEncoding:
    def test_output_shape_rplan(self, vocab_config: VocabConfig):
        pe = CompositePositionalEncoding(vocab_config, d_model=64)
        out = pe()
        assert out.shape == (vocab_config.seq_len, 64)

    def test_output_shape_resplan(self):
        pe = CompositePositionalEncoding(RESPLAN_VOCAB_CONFIG, d_model=32)
        out = pe()
        assert out.shape == (RESPLAN_VOCAB_CONFIG.seq_len, 32)

    def test_output_dtype(self, vocab_config: VocabConfig):
        pe = CompositePositionalEncoding(vocab_config, d_model=32)
        assert pe().dtype == torch.float32

    def test_node_vs_edge_different_entity_type(self, vocab_config: VocabConfig):
        """Nodes and edges get different entity type embeddings."""
        pe = CompositePositionalEncoding(vocab_config, d_model=32)
        out = pe()
        node_region = out[: vocab_config.n_max]
        edge_region = out[vocab_config.n_max :]
        # Mean vectors should differ (different entity types + positions)
        assert not torch.allclose(
            node_region.mean(dim=0), edge_region.mean(dim=0), atol=1e-6
        )

    def test_different_node_positions_differ(self, vocab_config: VocabConfig):
        """Different node positions get different embeddings."""
        pe = CompositePositionalEncoding(vocab_config, d_model=64)
        out = pe()
        # Position 0 vs position 1 should differ (different node_index_emb)
        assert not torch.equal(out[0], out[1])

    def test_different_edge_positions_differ(self, vocab_config: VocabConfig):
        """Different edge positions get different embeddings."""
        pe = CompositePositionalEncoding(vocab_config, d_model=64)
        out = pe()
        n_max = vocab_config.n_max
        # Edge at (0,1) vs edge at (0,2) should differ
        assert not torch.equal(out[n_max], out[n_max + 1])

    def test_deterministic(self, vocab_config: VocabConfig):
        """Same input produces same output (no randomness in forward)."""
        pe = CompositePositionalEncoding(vocab_config, d_model=32)
        out1 = pe()
        out2 = pe()
        assert torch.equal(out1, out2)

    def test_broadcastable_to_batch(self, vocab_config: VocabConfig):
        """Output should broadcast when added to (B, seq_len, d_model) tensor."""
        pe = CompositePositionalEncoding(vocab_config, d_model=32)
        pos_enc = pe()  # (seq_len, d_model)
        batch = torch.randn(4, vocab_config.seq_len, 32)
        result = batch + pos_enc  # should broadcast
        assert result.shape == (4, vocab_config.seq_len, 32)

    def test_buffers_registered(self, vocab_config: VocabConfig):
        """Index tensors should be registered as buffers."""
        pe = CompositePositionalEncoding(vocab_config, d_model=16)
        buffer_names = {name for name, _ in pe.named_buffers()}
        assert "node_indices" in buffer_names
        assert "entity_types" in buffer_names
        assert "edge_i" in buffer_names
        assert "edge_j" in buffer_names

    def test_edge_endpoints_correct(self, vocab_config: VocabConfig):
        """Precomputed edge endpoints match vocab_config.edge_position_to_pair."""
        pe = CompositePositionalEncoding(vocab_config, d_model=16)
        for pos in range(vocab_config.n_edges):
            expected_i, expected_j = vocab_config.edge_position_to_pair(pos)
            assert pe.edge_i[pos].item() == expected_i
            assert pe.edge_j[pos].item() == expected_j


# ---------------------------------------------------------------------------
# TimestepEmbedding
# ---------------------------------------------------------------------------


class TestTimestepEmbedding:
    def test_output_shape_batched(self):
        te = TimestepEmbedding(d_model=64)
        t = torch.rand(4)
        out = te(t)
        assert out.shape == (4, 64)

    def test_output_shape_single(self):
        te = TimestepEmbedding(d_model=64)
        t = torch.tensor([0.5])
        out = te(t)
        assert out.shape == (1, 64)

    def test_output_dtype(self):
        te = TimestepEmbedding(d_model=32)
        t = torch.rand(2)
        assert te(t).dtype == torch.float32

    def test_sinusoidal_encoding_shape(self):
        t = torch.rand(4)
        enc = TimestepEmbedding.sinusoidal_encoding(t, dim=256)
        assert enc.shape == (4, 256)

    def test_sinusoidal_encoding_odd_dim(self):
        t = torch.rand(4)
        enc = TimestepEmbedding.sinusoidal_encoding(t, dim=255)
        assert enc.shape == (4, 255)

    def test_sinusoidal_encoding_deterministic(self):
        t = torch.tensor([0.42, 0.99])
        enc1 = TimestepEmbedding.sinusoidal_encoding(t, dim=128)
        enc2 = TimestepEmbedding.sinusoidal_encoding(t, dim=128)
        assert torch.equal(enc1, enc2)

    def test_different_timesteps_produce_different_embeddings(self):
        te = TimestepEmbedding(d_model=64)
        te.eval()
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        out1 = te(t1)
        out2 = te(t2)
        assert not torch.equal(out1, out2)

    def test_deterministic_in_eval(self):
        te = TimestepEmbedding(d_model=32)
        te.eval()
        t = torch.tensor([0.42])
        out1 = te(t)
        out2 = te(t)
        assert torch.equal(out1, out2)

    def test_custom_frequency_size(self):
        te = TimestepEmbedding(d_model=64, frequency_embedding_size=128)
        t = torch.rand(3)
        out = te(t)
        assert out.shape == (3, 64)

    def test_boundary_timesteps(self):
        """t=0 and t=1 should produce valid (finite) embeddings."""
        te = TimestepEmbedding(d_model=32)
        te.eval()
        t = torch.tensor([0.0, 1.0])
        out = te(t)
        assert torch.isfinite(out).all()
