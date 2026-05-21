"""High-level recommendation service built on loaded serving artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import polars as pl
import torch

from movie_recsys.modeling.faiss_index import search_index
from movie_recsys.ranking.features import _build_split_features
from movie_recsys.serving.errors import (
    artifacts_not_ready,
    feature_mismatch,
    invalid_candidate_top_k,
    invalid_request,
    invalid_top_k,
    no_candidates,
    user_id_not_found,
    user_not_found,
)
from movie_recsys.serving.registry import ArtifactRegistry, LoadedArtifacts
from movie_recsys.serving.scorer import PolicySpec, rank_with_policy


@dataclass(slots=True)
class RecommendationRow:
    """Single recommendation score row."""

    movie_id: int
    item_idx: int
    title: str
    genres: str
    release_year: int | None
    final_score: float
    residual_score: float | None
    ranker_score: float | None
    popularity_score: float
    rank_position: int
    scorer_policy: str


@dataclass(slots=True)
class RecommendationResult:
    """Top-level recommendation response object returned by the service."""

    user_id: int | None
    user_idx: int | None
    k: int
    cold_start: bool
    scorer_policy: str
    recommendations: list[RecommendationRow]
    debug: dict[str, object] | None = None


@dataclass(slots=True)
class UserHistoryRow:
    """History item returned by serving history endpoint."""

    movie_id: int
    item_idx: int
    title: str
    genres: str
    timestamp: int | None


class ItemMetadata(TypedDict):
    """Static metadata fields attached to recommendation items."""

    movie_id: int
    title: str
    genres: str
    release_year: int | None


class RecommendationService:
    """Service orchestrating retrieval and reranking."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        self._registry = registry

    @property
    def policy_name(self) -> str:
        return self._registry.config.scoring.policy_name

    def resolve_user_idx(self, *, user_id: int) -> int | None:
        """Resolve original user id to internal user_idx."""

        artifacts = self._loaded_artifacts()
        user_row = artifacts.users_frame.filter(pl.col("original_userId") == user_id)
        if user_row.height != 1:
            return None
        return int(user_row.get_column("user_idx").to_list()[0])

    def resolve_user_id(self, *, user_idx: int) -> int | None:
        """Resolve internal user_idx back to original user id."""

        artifacts = self._loaded_artifacts()
        user_row = artifacts.users_frame.filter(pl.col("user_idx") == user_idx)
        if user_row.height != 1:
            return None
        return int(user_row.get_column("original_userId").to_list()[0])

    def get_user_history(
        self,
        *,
        user_id: int,
        limit: int = 100,
    ) -> tuple[int, list[UserHistoryRow]]:
        """Return most recent train-history rows for a user_id, capped by limit."""

        artifacts = self._loaded_artifacts()
        user_idx = self.resolve_user_idx(user_id=user_id)
        if user_idx is None:
            raise user_id_not_found(user_id)

        interactions = artifacts.interactions_train_frame.filter(pl.col("userId") == user_id)
        if interactions.is_empty():
            return user_idx, []

        limited = interactions.sort("timestamp", descending=True).head(max(limit, 1))
        items = artifacts.items_frame.select(
            ["item_idx", "title", "genres", "original_movieId"]
        )
        joined = limited.join(items, on="item_idx", how="left")

        history_rows = [
            UserHistoryRow(
                movie_id=int(row["movieId"]),
                item_idx=int(row["item_idx"]),
                title=str(row["title"] or ""),
                genres=str(row["genres"] or ""),
                timestamp=(int(row["timestamp"]) if row["timestamp"] is not None else None),
            )
            for row in joined.to_dicts()
        ]
        return user_idx, history_rows

    def _item_metadata_map(self, *, item_indices: np.ndarray) -> dict[int, ItemMetadata]:
        artifacts = self._loaded_artifacts()
        metadata_frame = artifacts.items_frame.filter(
            pl.col("item_idx").is_in(item_indices.tolist())
        ).select(["item_idx", "original_movieId", "title", "genres", "release_year"])
        return {
            int(row["item_idx"]): {
                "movie_id": int(row["original_movieId"]),
                "title": str(row["title"] or ""),
                "genres": str(row["genres"] or ""),
                "release_year": (
                    int(row["release_year"]) if row["release_year"] is not None else None
                ),
            }
            for row in metadata_frame.to_dicts()
        }

    def _loaded_artifacts(self) -> LoadedArtifacts:
        try:
            return self._registry.artifacts
        except RuntimeError as exc:
            raise artifacts_not_ready() from exc

    def _user_history(self, *, users_frame: pl.DataFrame, user_idx: int) -> tuple[list[int], int]:
        user_row = users_frame.filter(pl.col("user_idx") == user_idx)
        if user_row.height != 1:
            raise user_not_found(user_idx)

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
            raise no_candidates("No valid candidates available after filtering")
        return pl.DataFrame(rows)

    def _cold_start_recommendations(
        self,
        *,
        top_k: int,
    ) -> list[RecommendationRow]:
        artifacts = self._loaded_artifacts()
        popular_items = artifacts.items_frame.sort(
            by=["popularity_score", "item_idx"],
            descending=[True, False],
        ).head(top_k)

        rows: list[RecommendationRow] = []
        for rank_position, row in enumerate(popular_items.to_dicts(), start=1):
            popularity_score = float(row.get("popularity_score") or 0.0)
            release_year_raw = row.get("release_year")
            rows.append(
                RecommendationRow(
                    movie_id=int(row.get("original_movieId") or 0),
                    item_idx=int(row.get("item_idx") or 0),
                    title=str(row.get("title") or ""),
                    genres=str(row.get("genres") or ""),
                    release_year=(
                        int(release_year_raw) if release_year_raw is not None else None
                    ),
                    final_score=popularity_score,
                    residual_score=None,
                    ranker_score=None,
                    popularity_score=popularity_score,
                    rank_position=rank_position,
                    scorer_policy="popularity_fallback",
                )
            )
        return rows

    def _recommend_known_user(
        self,
        *,
        user_idx: int,
        user_id: int,
        top_k: int,
        candidate_top_k: int,
        exclude_seen: bool,
        include_debug: bool,
    ) -> RecommendationResult:
        artifacts = self._loaded_artifacts()

        if user_idx < 0 or user_idx >= artifacts.feature_tables.user_features.shape[0]:
            raise user_not_found(user_idx)

        effective_top_k = top_k

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
            int(candidate_top_k),
            effective_top_k,
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
            if item_index in seen_candidate_items:
                continue
            if exclude_seen and item_index in seen_items:
                continue
            seen_candidate_items.add(item_index)
            dedup_items.append(item_index)
            dedup_scores.append(float(score))

        if not dedup_items:
            raise no_candidates()

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
            raise feature_mismatch(missing_feature_columns)

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

        metadata_map = self._item_metadata_map(item_indices=item_idx)
        final_rows: list[RecommendationRow] = []
        for rank_position, candidate_index in enumerate(
            order[:effective_top_k].tolist(),
            start=1,
        ):
            idx = int(candidate_index)
            resolved_item_idx = int(item_idx[idx])
            metadata = metadata_map.get(
                resolved_item_idx,
                {
                    "movie_id": resolved_item_idx,
                    "title": "",
                    "genres": "",
                    "release_year": None,
                },
            )
            final_rows.append(
                RecommendationRow(
                    movie_id=int(metadata["movie_id"]),
                    item_idx=resolved_item_idx,
                    title=str(metadata["title"]),
                    genres=str(metadata["genres"]),
                    release_year=(
                        int(metadata["release_year"])
                        if metadata["release_year"] is not None
                        else None
                    ),
                    final_score=float(display_scores[idx]),
                    residual_score=float(residual_scores[idx]),
                    ranker_score=float(ranker_scores[idx]),
                    popularity_score=float(popularity_scores[idx]),
                    rank_position=rank_position,
                    scorer_policy=policy.policy_name,
                )
            )

        debug_payload: dict[str, object] | None = None
        if include_debug:
            debug_payload = {
                "requested_k": top_k,
                "returned_k": len(final_rows),
                "candidate_pool_size": candidate_pool_size,
                "deduplicated_candidate_count": len(dedup_items),
                "exclude_seen": exclude_seen,
            }

        return RecommendationResult(
            user_id=user_id,
            user_idx=user_idx,
            k=top_k,
            cold_start=False,
            scorer_policy=policy.policy_name,
            recommendations=final_rows,
            debug=debug_payload,
        )

    def recommend(
        self,
        *,
        user_idx: int | None,
        user_id: int | None,
        top_k: int,
        exclude_seen: bool = True,
        candidate_top_k: int | None = None,
        allow_cold_start: bool = True,
        include_debug: bool = False,
    ) -> RecommendationResult:
        """Generate top-k recommendations with optional cold-start fallback."""

        configured_min = int(self._registry.config.runtime.min_top_k)
        configured_max = int(self._registry.config.runtime.max_top_k)
        min_allowed = max(configured_min, 1)
        max_allowed = max(configured_max, min_allowed)
        if top_k < min_allowed or top_k > max_allowed:
            raise invalid_top_k(
                top_k=top_k,
                min_allowed=min_allowed,
                max_allowed=max_allowed,
            )

        resolved_candidate_top_k = (
            int(candidate_top_k)
            if candidate_top_k is not None
            else int(self._registry.config.runtime.candidate_top_k)
        )
        if resolved_candidate_top_k < top_k or resolved_candidate_top_k > 500:
            raise invalid_candidate_top_k(
                candidate_top_k=resolved_candidate_top_k,
                requested_top_k=top_k,
                max_allowed=500,
            )

        if user_idx is not None and user_id is not None:
            raise invalid_request("Provide only one of user_idx or user_id")

        if user_idx is None and user_id is None:
            if allow_cold_start:
                cold_start_rows = self._cold_start_recommendations(top_k=top_k)
                return RecommendationResult(
                    user_id=None,
                    user_idx=None,
                    k=top_k,
                    cold_start=True,
                    scorer_policy="popularity_fallback",
                    recommendations=cold_start_rows,
                    debug=(
                        {
                            "requested_k": top_k,
                            "returned_k": len(cold_start_rows),
                            "reason": "missing_user_identifier",
                        }
                        if include_debug
                        else None
                    ),
                )
            raise invalid_request(
                "Either user_idx or user_id is required when allow_cold_start is false"
            )

        resolved_user_idx: int | None = user_idx
        resolved_user_id: int | None = user_id
        if user_idx is not None:
            resolved_user_id = self.resolve_user_id(user_idx=user_idx)
            if resolved_user_id is None:
                if allow_cold_start:
                    cold_start_rows = self._cold_start_recommendations(top_k=top_k)
                    return RecommendationResult(
                        user_id=None,
                        user_idx=user_idx,
                        k=top_k,
                        cold_start=True,
                        scorer_policy="popularity_fallback",
                        recommendations=cold_start_rows,
                        debug=(
                            {
                                "requested_k": top_k,
                                "returned_k": len(cold_start_rows),
                                "reason": "unknown_user_idx",
                            }
                            if include_debug
                            else None
                        ),
                    )
                raise user_not_found(user_idx)
        elif user_id is not None:
            resolved_user_idx = self.resolve_user_idx(user_id=user_id)
            if resolved_user_idx is None:
                if allow_cold_start:
                    cold_start_rows = self._cold_start_recommendations(top_k=top_k)
                    return RecommendationResult(
                        user_id=user_id,
                        user_idx=None,
                        k=top_k,
                        cold_start=True,
                        scorer_policy="popularity_fallback",
                        recommendations=cold_start_rows,
                        debug=(
                            {
                                "requested_k": top_k,
                                "returned_k": len(cold_start_rows),
                                "reason": "unknown_user_id",
                            }
                            if include_debug
                            else None
                        ),
                    )
                raise user_id_not_found(user_id)

        assert resolved_user_idx is not None
        assert resolved_user_id is not None
        return self._recommend_known_user(
            user_idx=resolved_user_idx,
            user_id=resolved_user_id,
            top_k=top_k,
            candidate_top_k=resolved_candidate_top_k,
            exclude_seen=exclude_seen,
            include_debug=include_debug,
        )
