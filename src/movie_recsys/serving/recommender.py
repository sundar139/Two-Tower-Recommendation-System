"""High-level recommendation service built on loaded serving artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
from fastapi import status

from movie_recsys.modeling.faiss_index import search_index
from movie_recsys.ranking.features import _build_split_features
from movie_recsys.serving.errors import ServingError
from movie_recsys.serving.registry import ArtifactRegistry, LoadedArtifacts
from movie_recsys.serving.scorer import PolicySpec, rank_with_policy


@dataclass(slots=True)
class RecommendationRow:
    """Single recommendation score row."""

    item_idx: int
    score: float
    residual_score: float
    ranker_score: float
    popularity_score: float


class RecommendationService:
    """Service orchestrating retrieval and reranking."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        self._registry = registry

    def _loaded_artifacts(self) -> LoadedArtifacts:
        try:
            return self._registry.artifacts
        except RuntimeError as exc:
            raise ServingError(
                message="Serving artifacts are not loaded",
                code="artifacts_not_ready",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ) from exc

    def _user_history(self, *, users_frame: pl.DataFrame, user_idx: int) -> tuple[list[int], int]:
        user_row = users_frame.filter(pl.col("user_idx") == user_idx)
        if user_row.height != 1:
            raise ServingError(
                message=f"Unknown user_idx: {user_idx}",
                code="user_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        raw_history = user_row.get_column("train_history_item_idx").to_list()[0] or []
        history = [int(value) for value in raw_history]
        timestamp_context = 0
        if "last_timestamp" in user_row.columns:
            timestamp_context = int(user_row.get_column("last_timestamp").to_list()[0] or 0)
        return history, timestamp_context

    def _encode_user(
        self,
        *,
        user_idx: int,
        history_item_idx: list[int],
        history_length: int,
        device: torch.device,
        retrieval_model: Any,
        user_features: np.ndarray,
    ) -> np.ndarray:
        history_shifted = [int(item_idx) + 1 for item_idx in history_item_idx[-history_length:]]
        history_tensor = torch.zeros((1, history_length), dtype=torch.long, device=device)
        history_mask = torch.zeros((1, history_length), dtype=torch.bool, device=device)
        if history_shifted:
            history_tensor[0, -len(history_shifted) :] = torch.tensor(
                history_shifted,
                dtype=torch.long,
                device=device,
            )
            history_mask[0, -len(history_shifted) :] = True

        user_features_tensor = torch.from_numpy(user_features).to(device=device).unsqueeze(0)
        with torch.no_grad():
            user_embedding = retrieval_model.encode_user(
                {
                    "user_idx": torch.tensor([user_idx], dtype=torch.long, device=device),
                    "history_item_idx": history_tensor,
                    "history_mask": history_mask,
                    "user_features": user_features_tensor,
                }
            )
        return np.asarray(
            user_embedding.detach().cpu().numpy().astype(np.float32, copy=False),
            dtype=np.float32,
        )

    def _embedding_stats(
        self,
        user_embedding: np.ndarray,
        item_embedding: np.ndarray,
    ) -> dict[str, float]:
        product = user_embedding * item_embedding
        abs_diff = np.abs(user_embedding - item_embedding)
        return {
            "frozen_emb_dot": float(np.dot(user_embedding, item_embedding)),
            "frozen_emb_prod_mean": float(product.mean()),
            "frozen_emb_prod_max": float(product.max()),
            "frozen_emb_prod_min": float(product.min()),
            "frozen_emb_prod_std": float(product.std()),
            "frozen_emb_absdiff_mean": float(abs_diff.mean()),
            "frozen_emb_absdiff_max": float(abs_diff.max()),
            "frozen_emb_absdiff_min": float(abs_diff.min()),
            "frozen_emb_absdiff_std": float(abs_diff.std()),
        }

    def _candidate_frame(
        self,
        *,
        user_idx: int,
        timestamp_context: int,
        history_item_idx: list[int],
        candidate_items: np.ndarray,
        candidate_scores: np.ndarray,
        user_embedding: np.ndarray,
        item_embeddings: np.ndarray,
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for rank, (item_idx, residual_score) in enumerate(
            zip(candidate_items.tolist(), candidate_scores.tolist(), strict=False),
            start=1,
        ):
            item_index = int(item_idx)
            if item_index < 0 or item_index >= item_embeddings.shape[0]:
                continue

            row: dict[str, object] = {
                "query_id": f"serve_u{user_idx:08d}",
                "user_idx": int(user_idx),
                "item_idx": item_index,
                "split": "serve",
                "label": 0,
                "target_item_idx": -1,
                "residual_score": float(residual_score),
                "residual_rank": int(rank),
                "target_injected": False,
                "user_history_length": int(len(history_item_idx)),
                "timestamp_context": int(timestamp_context),
                "candidate_source": "retrieved",
            }
            row.update(
                self._embedding_stats(
                    user_embedding=user_embedding,
                    item_embedding=item_embeddings[item_index],
                )
            )
            rows.append(row)

        if not rows:
            raise ServingError(
                message="No valid candidates available after filtering",
                code="no_candidates",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return pl.DataFrame(rows)

    def recommend(self, *, user_idx: int, top_k: int) -> list[RecommendationRow]:
        """Generate top-k recommendations for a user index."""

        artifacts = self._loaded_artifacts()

        min_top_k = int(artifacts.ranker_config.ranker_top_k)
        configured_min = int(self._registry.config.runtime.min_top_k)
        configured_max = int(self._registry.config.runtime.max_top_k)
        min_allowed = max(configured_min, 1)
        max_allowed = max(configured_max, min_allowed)
        effective_top_k = max(min(top_k, max_allowed), min_allowed)
        if top_k < min_allowed or top_k > max_allowed:
            raise ServingError(
                message=(
                    f"top_k must be between {min_allowed} and {max_allowed}, "
                    f"received {top_k}"
                ),
                code="invalid_top_k",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        if user_idx < 0 or user_idx >= artifacts.feature_tables.user_features.shape[0]:
            raise ServingError(
                message=f"Unknown user_idx: {user_idx}",
                code="user_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        history_item_idx, timestamp_context = self._user_history(
            users_frame=artifacts.users_frame,
            user_idx=user_idx,
        )
        seen_items = set(history_item_idx)
        user_embedding_2d = self._encode_user(
            user_idx=user_idx,
            history_item_idx=history_item_idx,
            history_length=int(artifacts.retrieval_config.train.history_length),
            device=artifacts.device,
            retrieval_model=artifacts.retrieval_model,
            user_features=np.asarray(
                artifacts.feature_tables.user_features[user_idx],
                dtype=np.float32,
            ),
        )
        user_embedding = user_embedding_2d[0]

        candidate_pool_size = max(
            int(self._registry.config.runtime.candidate_top_k),
            max(effective_top_k, min_top_k),
        )
        retrieved_items, retrieved_scores, _latency_ms = search_index(
            artifacts.index,
            user_embedding_2d,
            artifacts.item_mapping,
            candidate_pool_size,
        )

        dedup_items: list[int] = []
        dedup_scores: list[float] = []
        seen_candidate_items: set[int] = set()
        for item_idx, score in zip(
            retrieved_items[0].tolist(),
            retrieved_scores[0].tolist(),
            strict=False,
        ):
            item_index = int(item_idx)
            if item_index in seen_candidate_items or item_index in seen_items:
                continue
            seen_candidate_items.add(item_index)
            dedup_items.append(item_index)
            dedup_scores.append(float(score))

        if not dedup_items:
            raise ServingError(
                message="No unseen candidates available for this user",
                code="no_candidates",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        candidate_frame = self._candidate_frame(
            user_idx=user_idx,
            timestamp_context=timestamp_context,
            history_item_idx=history_item_idx,
            candidate_items=np.asarray(dedup_items, dtype=np.int64),
            candidate_scores=np.asarray(dedup_scores, dtype=np.float32),
            user_embedding=user_embedding,
            item_embeddings=artifacts.item_embeddings,
        )

        feature_frame, _resolved_columns = _build_split_features(
            candidates=candidate_frame,
            users_df=artifacts.users_frame,
            items_df=artifacts.items_frame,
            use_frozen_features=artifacts.ranker_config.use_frozen_retrieval_embeddings,
        )
        missing_feature_columns = [
            name for name in artifacts.feature_columns if name not in feature_frame.columns
        ]
        if missing_feature_columns:
            raise ServingError(
                message=(
                    "Serving feature frame is missing ranker columns: "
                    + ", ".join(missing_feature_columns)
                ),
                code="feature_mismatch",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        feature_matrix = feature_frame.select(artifacts.feature_columns).to_numpy().astype(
            np.float32,
            copy=False,
        )
        with torch.no_grad():
            ranker_scores_tensor = artifacts.ranker_model(
                torch.from_numpy(feature_matrix).to(device=artifacts.device)
            )
        ranker_scores = np.asarray(
            ranker_scores_tensor.detach().cpu().numpy().astype(np.float64, copy=False),
            dtype=np.float64,
        )
        residual_scores = np.asarray(
            feature_frame.get_column("residual_score").to_numpy(),
            dtype=np.float64,
        )
        popularity_scores = (
            np.asarray(feature_frame.get_column("popularity_score").to_numpy(), dtype=np.float64)
            if "popularity_score" in feature_frame.columns
            else np.zeros_like(ranker_scores, dtype=np.float64)
        )
        residual_rank = np.asarray(
            feature_frame.get_column("residual_rank").to_numpy(),
            dtype=np.int64,
        )
        item_idx = np.asarray(feature_frame.get_column("item_idx").to_numpy(), dtype=np.int64)

        policy = PolicySpec(
            policy_name=self._registry.config.scoring.policy_name,
            alpha=float(self._registry.config.scoring.alpha),
            beta=float(self._registry.config.scoring.beta),
            gamma=float(self._registry.config.scoring.gamma),
            top_k_focus=int(self._registry.config.scoring.top_k_focus),
        )
        order, display_scores = rank_with_policy(
            ranker_scores=ranker_scores,
            residual_scores=residual_scores,
            popularity_scores=popularity_scores,
            residual_rank=residual_rank,
            item_idx=item_idx,
            policy=policy,
        )

        final_rows: list[RecommendationRow] = []
        for candidate_index in order[:effective_top_k].tolist():
            idx = int(candidate_index)
            final_rows.append(
                RecommendationRow(
                    item_idx=int(item_idx[idx]),
                    score=float(display_scores[idx]),
                    residual_score=float(residual_scores[idx]),
                    ranker_score=float(ranker_scores[idx]),
                    popularity_score=float(popularity_scores[idx]),
                )
            )

        return final_rows
