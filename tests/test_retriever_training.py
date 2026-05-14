from __future__ import annotations

import torch

from movie_recsys.modeling.losses import InBatchCrossEntropyLoss
from movie_recsys.modeling.retrieval import TwoTowerRetriever
from movie_recsys.training.config import load_retrieval_config


def test_small_overfit_loss_decreases() -> None:
    cfg = load_retrieval_config("configs/retrieval.yaml", sample=True)
    model = TwoTowerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = InBatchCrossEntropyLoss()

    torch.manual_seed(42)
    batch = {
        "user_idx": torch.randint(0, 32, (32,)),
        "item_idx": torch.randint(1, 64, (32,)),
        "history_item_idx": torch.randint(0, 64, (32, 50)),
        "history_mask": torch.randint(0, 2, (32, 50)).bool(),
        "user_features": torch.randn(32, 6),
        "item_features": torch.randn(32, 7),
    }

    losses = []
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = criterion(out["logits"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0]
