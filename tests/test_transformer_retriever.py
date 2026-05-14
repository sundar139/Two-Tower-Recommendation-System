from __future__ import annotations

import torch

from movie_recsys.modeling.losses import InBatchCrossEntropyLoss
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import load_retrieval_config


def _config():
    cfg = load_retrieval_config("configs/transformer_retrieval.yaml", sample=True)
    cfg.model.dropout = 0.0
    cfg.train.history_length = 50
    return cfg


def _batch(batch_size: int, history_length: int) -> dict[str, torch.Tensor]:
    return {
        "user_idx": torch.randint(0, 32, (batch_size,)),
        "item_idx": torch.randint(1, 64, (batch_size,)),
        "history_item_idx": torch.randint(0, 64, (batch_size, history_length)),
        "history_mask": torch.randint(0, 2, (batch_size, history_length)).bool(),
        "user_features": torch.randn(batch_size, 6),
        "item_features": torch.randn(batch_size, 7),
    }


def test_transformer_user_embedding_norm_and_logits_shape() -> None:
    cfg = _config()
    model = TransformerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )
    out = model(_batch(8, cfg.train.history_length))

    assert out["user_emb"].shape == (8, cfg.model.embedding_dim)
    assert out["item_emb"].shape == (8, cfg.model.embedding_dim)
    assert out["logits"].shape == (8, 8)
    norms = torch.linalg.norm(out["user_emb"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_transformer_loss_finite_and_gradients_flow() -> None:
    cfg = _config()
    model = TransformerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )
    criterion = InBatchCrossEntropyLoss()

    out = model(_batch(8, cfg.train.history_length))
    loss = criterion(out["logits"])
    assert torch.isfinite(loss)
    loss.backward()

    pos_grad = model.user_tower.sequence_encoder.position_embeddings.weight.grad
    assert pos_grad is not None
    assert float(pos_grad.abs().sum().item()) > 0.0

    block_grad = model.user_tower.sequence_encoder.blocks[0].q_proj.weight.grad
    assert block_grad is not None
    assert float(block_grad.abs().sum().item()) > 0.0


def test_empty_history_safe_no_nan() -> None:
    cfg = _config()
    model = TransformerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )
    batch = _batch(4, cfg.train.history_length)
    batch["history_item_idx"] = torch.zeros_like(batch["history_item_idx"])
    batch["history_mask"] = torch.zeros_like(batch["history_mask"])

    out = model(batch)
    assert torch.isfinite(out["user_emb"]).all()
    assert torch.isfinite(out["logits"]).all()


def test_eval_mode_deterministic_with_same_input() -> None:
    torch.manual_seed(123)
    cfg = _config()
    model = TransformerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )
    model.eval()
    batch = _batch(6, cfg.train.history_length)

    out1 = model(batch)
    out2 = model(batch)
    assert torch.allclose(out1["user_emb"], out2["user_emb"], atol=1e-6)
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-6)
