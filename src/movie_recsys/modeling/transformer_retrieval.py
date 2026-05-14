"""Transformer-enhanced two-tower retriever preserving baseline item tower."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from movie_recsys.modeling.sequence import TransformerUserEncoder
from movie_recsys.modeling.towers import MLP, ItemTower
from movie_recsys.training.config import RetrievalConfig


class TransformerUserTower(nn.Module):
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
        history_length: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_ffn_dim: int,
        sequence_pooling: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_id_embedding_dim)
        self.history_embedding = nn.Embedding(
            num_items_with_padding,
            item_id_embedding_dim,
            padding_idx=0,
        )
        self.sequence_encoder = TransformerUserEncoder(
            embedding_dim=item_id_embedding_dim,
            history_length=history_length,
            layers=transformer_layers,
            heads=transformer_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=dropout,
            sequence_pooling=sequence_pooling,
        )
        self.user_features_mlp = MLP(
            user_feature_dim,
            feature_hidden_dim,
            feature_hidden_dim,
            dropout,
        )
        concat_dim = user_id_embedding_dim + item_id_embedding_dim + feature_hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, projection_hidden_dim),
            nn.GELU(),
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
        sequence_vec = self.sequence_encoder(history_emb, history_mask)
        user_feat_vec = self.user_features_mlp(user_features)
        merged = torch.cat([user_emb, sequence_vec, user_feat_vec], dim=1)
        projected = self.proj(merged)
        return nn.functional.normalize(projected, p=2, dim=1)


class TransformerRetriever(nn.Module):
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
        self.user_tower = TransformerUserTower(
            num_users=num_users,
            num_items_with_padding=num_items_with_padding,
            user_feature_dim=user_feature_dim,
            embedding_dim=config.model.embedding_dim,
            user_id_embedding_dim=config.model.user_id_embedding_dim,
            item_id_embedding_dim=config.model.item_id_embedding_dim,
            feature_hidden_dim=config.model.feature_hidden_dim,
            projection_hidden_dim=config.model.projection_hidden_dim,
            history_length=config.train.history_length,
            transformer_layers=config.model.transformer_layers,
            transformer_heads=config.model.transformer_heads,
            transformer_ffn_dim=config.model.transformer_ffn_dim,
            sequence_pooling=config.model.sequence_pooling,
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
