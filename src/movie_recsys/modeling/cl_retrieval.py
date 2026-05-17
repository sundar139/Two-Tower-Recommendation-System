"""CL-EPIDTN-style contrastive enhancement over residual transformer retrieval."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from movie_recsys.modeling.augmentations import apply_sequence_augmentations
from movie_recsys.modeling.contrastive import symmetric_info_nce_loss
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.training.config import RetrievalConfig


class CLResidualTransformerRetriever(ResidualTransformerRetriever):
    """Residual retriever trained with retrieval + auxiliary contrastive losses."""

    def __init__(
        self,
        *,
        config: RetrievalConfig,
        num_users: int,
        num_items_with_padding: int,
        user_feature_dim: int,
        item_feature_dim: int,
    ) -> None:
        super().__init__(
            config=config,
            num_users=num_users,
            num_items_with_padding=num_items_with_padding,
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim,
        )
        self.contrastive_temperature = float(config.model.contrastive_temperature)
        self.lambda_user_cl = float(config.model.lambda_user_cl)
        self.lambda_item_cl = float(config.model.lambda_item_cl)
        self.lambda_alignment_cl = float(config.model.lambda_alignment_cl)

        self.augmentation_mask_prob = float(config.model.augmentation_mask_prob)
        self.augmentation_dropout_prob = float(config.model.augmentation_dropout_prob)
        self.augmentation_crop_min_ratio = float(config.model.augmentation_crop_min_ratio)
        self.augmentation_reorder_prob = float(config.model.augmentation_reorder_prob)
        self.augmentation_reorder_window = int(config.model.augmentation_reorder_window)

        self.item_id_view_proj = nn.Linear(
            config.model.item_id_embedding_dim,
            config.model.embedding_dim,
        )
        self.item_feature_view_proj = nn.Linear(
            config.model.feature_hidden_dim,
            config.model.embedding_dim,
        )

    @staticmethod
    def _zero_like(loss_ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=loss_ref.dtype, device=loss_ref.device)

    def _user_cl_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history_a, mask_a = apply_sequence_augmentations(
            batch["history_item_idx"],
            batch["history_mask"],
            target_item_idx=batch.get("item_idx"),
            mask_prob=self.augmentation_mask_prob,
            dropout_prob=self.augmentation_dropout_prob,
            crop_min_ratio=self.augmentation_crop_min_ratio,
            reorder_prob=self.augmentation_reorder_prob,
            reorder_window=self.augmentation_reorder_window,
        )
        history_b, mask_b = apply_sequence_augmentations(
            batch["history_item_idx"],
            batch["history_mask"],
            target_item_idx=batch.get("item_idx"),
            mask_prob=self.augmentation_mask_prob,
            dropout_prob=self.augmentation_dropout_prob,
            crop_min_ratio=self.augmentation_crop_min_ratio,
            reorder_prob=self.augmentation_reorder_prob,
            reorder_window=self.augmentation_reorder_window,
        )

        user_view_a = self.user_tower(
            batch["user_idx"],
            history_a,
            mask_a,
            batch["user_features"],
        )
        user_view_b = self.user_tower(
            batch["user_idx"],
            history_b,
            mask_b,
            batch["user_features"],
        )
        loss, _logits_ab, _logits_ba = symmetric_info_nce_loss(
            user_view_a,
            user_view_b,
            temperature=self.contrastive_temperature,
        )
        return loss

    def _item_cl_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item_id_view = self.item_tower.item_embedding(batch["item_idx"])
        item_feature_view = self.item_tower.item_features_mlp(batch["item_features"])

        projected_id = functional.normalize(self.item_id_view_proj(item_id_view), p=2, dim=1)
        projected_feat = functional.normalize(
            self.item_feature_view_proj(item_feature_view),
            p=2,
            dim=1,
        )
        loss, _logits_ab, _logits_ba = symmetric_info_nce_loss(
            projected_id,
            projected_feat,
            temperature=self.contrastive_temperature,
        )
        return loss

    def _alignment_cl_loss(
        self,
        *,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        loss, _logits_ab, _logits_ba = symmetric_info_nce_loss(
            user_emb,
            item_emb,
            temperature=self.contrastive_temperature,
        )
        return loss

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        user_emb = self.encode_user(batch)
        item_emb = self.encode_item(batch)
        logits = (user_emb @ item_emb.T) / self.temperature

        labels = torch.arange(logits.shape[0], device=logits.device)
        retrieval_loss = functional.cross_entropy(logits, labels)

        user_contrastive_loss = self._zero_like(retrieval_loss)
        item_contrastive_loss = self._zero_like(retrieval_loss)
        alignment_contrastive_loss = self._zero_like(retrieval_loss)

        if self.lambda_user_cl > 0.0:
            user_contrastive_loss = self._user_cl_loss(batch)
        if self.lambda_item_cl > 0.0:
            item_contrastive_loss = self._item_cl_loss(batch)
        if self.lambda_alignment_cl > 0.0:
            alignment_contrastive_loss = self._alignment_cl_loss(
                user_emb=user_emb,
                item_emb=item_emb,
            )

        total_loss = retrieval_loss
        total_loss = total_loss + (self.lambda_user_cl * user_contrastive_loss)
        total_loss = total_loss + (self.lambda_item_cl * item_contrastive_loss)
        total_loss = total_loss + (self.lambda_alignment_cl * alignment_contrastive_loss)

        return {
            "user_emb": user_emb,
            "item_emb": item_emb,
            "logits": logits,
            "retrieval_loss": retrieval_loss,
            "user_contrastive_loss": user_contrastive_loss,
            "item_contrastive_loss": item_contrastive_loss,
            "alignment_contrastive_loss": alignment_contrastive_loss,
            "total_loss": total_loss,
        }
