"""Two-tower retriever module."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from movie_recsys.modeling.towers import ItemTower, UserTower
from movie_recsys.training.config import RetrievalConfig


class TwoTowerRetriever(nn.Module):
    def __init__(
        self,
        *,
        config: RetrievalConfig,
        num_users: int,
        num_items_with_padding: int,
        user_feature_dim: int,
        item_feature_dim: int,
    ) -> None:
        super().__init__()
        self.temperature = config.model.temperature
        self.user_tower = UserTower(
            num_users=num_users,
            num_items_with_padding=num_items_with_padding,
            user_feature_dim=user_feature_dim,
            embedding_dim=config.model.embedding_dim,
            user_id_embedding_dim=config.model.user_id_embedding_dim,
            item_id_embedding_dim=config.model.item_id_embedding_dim,
            feature_hidden_dim=config.model.feature_hidden_dim,
            projection_hidden_dim=config.model.projection_hidden_dim,
            dropout=config.model.dropout,
        )
        self.item_tower = ItemTower(
            num_items_with_padding=num_items_with_padding,
            item_feature_dim=item_feature_dim,
            embedding_dim=config.model.embedding_dim,
            item_id_embedding_dim=config.model.item_id_embedding_dim,
            feature_hidden_dim=config.model.feature_hidden_dim,
            projection_hidden_dim=config.model.projection_hidden_dim,
            dropout=config.model.dropout,
        )

    def encode_user(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.user_tower(
                batch["user_idx"],
                batch["history_item_idx"],
                batch["history_mask"],
                batch["user_features"],
            ),
        )

    def encode_item(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return cast(torch.Tensor, self.item_tower(batch["item_idx"], batch["item_features"]))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        user_emb = self.encode_user(batch)
        item_emb = self.encode_item(batch)
        logits = (user_emb @ item_emb.T) / self.temperature
        return {"user_emb": user_emb, "item_emb": item_emb, "logits": logits}
