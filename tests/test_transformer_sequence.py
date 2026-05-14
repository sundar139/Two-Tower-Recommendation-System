from __future__ import annotations

import torch

from movie_recsys.modeling.sequence import (
    TransformerEncoderBlock,
    TransformerUserEncoder,
    build_attention_bias,
)


def test_causal_mask_blocks_future_positions() -> None:
    mask = torch.ones((1, 4), dtype=torch.bool)
    bias = build_attention_bias(mask, dtype=torch.float32)
    for i in range(4):
        for j in range(4):
            value = float(bias[0, 0, i, j].item())
            if j > i:
                assert value < -1e20
            else:
                assert value == 0.0


def test_padding_mask_prevents_padded_effect() -> None:
    encoder = TransformerUserEncoder(
        embedding_dim=8,
        history_length=5,
        layers=1,
        heads=2,
        ffn_dim=32,
        dropout=0.0,
        sequence_pooling="last",
    )
    encoder.eval()

    torch.manual_seed(42)
    x1 = torch.randn(2, 5, 8)
    x2 = x1.clone()
    # Change padded tokens only; pooled output should remain unchanged.
    x2[:, 3:, :] = torch.randn(2, 2, 8) * 100.0

    mask = torch.tensor([[True, True, True, False, False], [True, True, True, False, False]])
    out1 = encoder(x1, mask)
    out2 = encoder(x2, mask)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_transformer_block_shape() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=16,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
    )
    x = torch.randn(3, 7, 16)
    mask = torch.ones((3, 7), dtype=torch.bool)
    y = block(x, mask)
    assert y.shape == (3, 7, 16)


def test_seq_len_one_supported() -> None:
    encoder = TransformerUserEncoder(
        embedding_dim=8,
        history_length=1,
        layers=1,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
        sequence_pooling="last",
    )
    x = torch.randn(4, 1, 8)
    mask = torch.ones((4, 1), dtype=torch.bool)
    out = encoder(x, mask)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_all_padding_rows_do_not_create_non_finite_attention() -> None:
    mask = torch.tensor([[False, False, False, False]], dtype=torch.bool)
    bias = build_attention_bias(mask, dtype=torch.float32)
    assert torch.isfinite(bias).all()


def test_debug_attention_path_returns_weights() -> None:
    encoder = TransformerUserEncoder(
        embedding_dim=8,
        history_length=4,
        layers=1,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
        sequence_pooling="mean",
    )
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]], dtype=torch.bool)

    pooled, attn = encoder(x, mask, return_attention=True)
    assert pooled.shape == (2, 8)
    assert len(attn) == 1
    assert attn[0].shape == (2, 2, 4, 4)
    assert torch.isfinite(attn[0]).all()
