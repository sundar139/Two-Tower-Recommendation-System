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
        self.base_lambda_user_cl = float(config.model.lambda_user_cl)
        self.base_lambda_item_cl = float(config.model.lambda_item_cl)
        self.base_lambda_alignment_cl = float(config.model.lambda_alignment_cl)

        self.effective_lambda_user_cl = self.base_lambda_user_cl
        self.effective_lambda_item_cl = self.base_lambda_item_cl
        self.effective_lambda_alignment_cl = self.base_lambda_alignment_cl

        self.use_contrastive_projection_head = bool(config.model.use_contrastive_projection_head)
        self.contrastive_projection_dim = int(config.model.contrastive_projection_dim)

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
        self.user_contrastive_projection: nn.Sequential | None
        self.item_contrastive_projection: nn.Sequential | None

        if self.use_contrastive_projection_head:
            self.user_contrastive_projection = nn.Sequential(
                nn.Linear(config.model.embedding_dim, config.model.embedding_dim),
                nn.GELU(),
                nn.Linear(config.model.embedding_dim, self.contrastive_projection_dim),
            )
            self.item_contrastive_projection = nn.Sequential(
                nn.Linear(config.model.embedding_dim, config.model.embedding_dim),
                nn.GELU(),
                nn.Linear(config.model.embedding_dim, self.contrastive_projection_dim),
            )
        else:
            self.user_contrastive_projection = None
            self.item_contrastive_projection = None

    def set_effective_contrastive_weights(
        self,
        *,
        lambda_user_cl: float,
        lambda_item_cl: float,
        lambda_alignment_cl: float,
    ) -> None:
        self.effective_lambda_user_cl = float(lambda_user_cl)
        self.effective_lambda_item_cl = float(lambda_item_cl)
        self.effective_lambda_alignment_cl = float(lambda_alignment_cl)

    @staticmethod
    def _zero_like(loss_ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=loss_ref.dtype, device=loss_ref.device)

    def _project_user_for_contrastive(self, user_emb: torch.Tensor) -> torch.Tensor:
        if self.user_contrastive_projection is None:
            return functional.normalize(user_emb, p=2, dim=1)
        projected = self.user_contrastive_projection(user_emb)
        return functional.normalize(projected, p=2, dim=1)

    def _project_item_for_contrastive(self, item_emb: torch.Tensor) -> torch.Tensor:
        if self.item_contrastive_projection is None:
            return functional.normalize(item_emb, p=2, dim=1)
        projected = self.item_contrastive_projection(item_emb)
        return functional.normalize(projected, p=2, dim=1)

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

        user_view_a = self._project_user_for_contrastive(user_view_a)
        user_view_b = self._project_user_for_contrastive(user_view_b)

        loss, _logits_ab, _logits_ba = symmetric_info_nce_loss(
            user_view_a,
            user_view_b,
            temperature=self.contrastive_temperature,
        )
        return loss

    def _item_cl_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item_id_view = self.item_tower.item_embedding(batch["item_idx"])
        item_feature_view = self.item_tower.item_features_mlp(batch["item_features"])

        projected_id = self.item_id_view_proj(item_id_view)
        projected_feat = self.item_feature_view_proj(item_feature_view)

        projected_id = self._project_item_for_contrastive(projected_id)
        projected_feat = self._project_item_for_contrastive(projected_feat)

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
        projected_user = self._project_user_for_contrastive(user_emb)
        projected_item = self._project_item_for_contrastive(item_emb)
        loss, _logits_ab, _logits_ba = symmetric_info_nce_loss(
            projected_user,
            projected_item,
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

        if self.effective_lambda_user_cl > 0.0:
            user_contrastive_loss = self._user_cl_loss(batch)
        if self.effective_lambda_item_cl > 0.0:
            item_contrastive_loss = self._item_cl_loss(batch)
        if self.effective_lambda_alignment_cl > 0.0:
            alignment_contrastive_loss = self._alignment_cl_loss(
                user_emb=user_emb,
                item_emb=item_emb,
            )

        total_loss = retrieval_loss
        total_loss = total_loss + (self.effective_lambda_user_cl * user_contrastive_loss)
        total_loss = total_loss + (self.effective_lambda_item_cl * item_contrastive_loss)
        total_loss = total_loss + (
            self.effective_lambda_alignment_cl * alignment_contrastive_loss
        )

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
