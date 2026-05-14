"""Tower definitions for plain two-tower retrieval."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


def masked_mean_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (sequence * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return summed / denom


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(values))


class UserTower(nn.Module):
    def __init__(
        self,
        *,
        num_users: int,
        num_items_with_padding: int,
        user_feature_dim: int,
        embedding_dim: int,
        user_id_embedding_dim: int,
        item_id_embedding_dim: int,
        feature_hidden_dim: int,
        projection_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_id_embedding_dim)
        self.history_embedding = nn.Embedding(
            num_items_with_padding, item_id_embedding_dim, padding_idx=0
        )
        self.user_features_mlp = MLP(
            user_feature_dim, feature_hidden_dim, feature_hidden_dim, dropout
        )
        concat_dim = user_id_embedding_dim + item_id_embedding_dim + feature_hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        history_item_idx: torch.Tensor,
        history_mask: torch.Tensor,
        user_features: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_idx)
        history_emb = self.history_embedding(history_item_idx)
        history_vec = masked_mean_pool(history_emb, history_mask)
        user_feat_vec = self.user_features_mlp(user_features)
        merged = torch.cat([user_emb, history_vec, user_feat_vec], dim=1)
        projected = self.proj(merged)
        return nn.functional.normalize(projected, p=2, dim=1)


class ItemTower(nn.Module):
    def __init__(
        self,
        *,
        num_items_with_padding: int,
        item_feature_dim: int,
        embedding_dim: int,
        item_id_embedding_dim: int,
        feature_hidden_dim: int,
        projection_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(
            num_items_with_padding, item_id_embedding_dim, padding_idx=0
        )
        self.item_features_mlp = MLP(
            item_feature_dim, feature_hidden_dim, feature_hidden_dim, dropout
        )
        concat_dim = item_id_embedding_dim + feature_hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )

    def forward(self, item_idx: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_idx)
        item_feat_vec = self.item_features_mlp(item_features)
        merged = torch.cat([item_emb, item_feat_vec], dim=1)
        projected = self.proj(merged)
        return nn.functional.normalize(projected, p=2, dim=1)
