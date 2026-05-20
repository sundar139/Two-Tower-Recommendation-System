"""High-level recommendation service built on loaded serving artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from movie_recsys.serving.registry import ArtifactRegistry


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

    def recommend(self, *, user_idx: int, top_k: int) -> list[RecommendationRow]:
        """Generate top-k recommendations for a user index."""

        _ = (user_idx, top_k)
        msg = "Recommendation logic is not implemented yet"
        raise NotImplementedError(msg)
