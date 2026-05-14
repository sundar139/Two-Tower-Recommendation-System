from __future__ import annotations

import torch

from movie_recsys.modeling.losses import InBatchCrossEntropyLoss
from movie_recsys.modeling.retrieval import TwoTowerRetriever
from movie_recsys.training.config import load_retrieval_config


def _config():
    return load_retrieval_config("configs/retrieval.yaml", sample=True)


def test_retriever_shapes_and_normalization() -> None:
    cfg = _config()
    model = TwoTowerRetriever(
        config=cfg,
        num_users=100,
        num_items_with_padding=200,
        user_feature_dim=8,
        item_feature_dim=10,
    )

    batch = {
        "user_idx": torch.randint(0, 100, (16,)),
        "item_idx": torch.randint(1, 200, (16,)),
        "history_item_idx": torch.randint(0, 200, (16, 50)),
        "history_mask": torch.randint(0, 2, (16, 50)).bool(),
        "user_features": torch.randn(16, 8),
        "item_features": torch.randn(16, 10),
    }

    out = model(batch)
    assert out["logits"].shape == (16, 16)
    assert torch.isfinite(out["logits"]).all()

    user_norm = torch.linalg.norm(out["user_emb"], dim=1)
    item_norm = torch.linalg.norm(out["item_emb"], dim=1)
    assert torch.allclose(user_norm, torch.ones_like(user_norm), atol=1e-4)
    assert torch.allclose(item_norm, torch.ones_like(item_norm), atol=1e-4)


def test_loss_and_gradients_flow() -> None:
    cfg = _config()
    model = TwoTowerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )
    criterion = InBatchCrossEntropyLoss()

    batch = {
        "user_idx": torch.randint(0, 32, (8,)),
        "item_idx": torch.randint(1, 64, (8,)),
        "history_item_idx": torch.randint(0, 64, (8, 50)),
        "history_mask": torch.randint(0, 2, (8, 50)).bool(),
        "user_features": torch.randn(8, 6),
        "item_features": torch.randn(8, 7),
    }
    out = model(batch)
    loss = criterion(out["logits"])
    assert torch.isfinite(loss)
    loss.backward()

    grad_sum = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_sum += float(param.grad.abs().sum().item())
    assert grad_sum > 0.0
