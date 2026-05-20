"""Tests for recommendation service baseline behavior."""

from __future__ import annotations

import pytest

from movie_recsys.serving.config import ServingConfig
from movie_recsys.serving.errors import ServingError
from movie_recsys.serving.recommender import RecommendationService
from movie_recsys.serving.registry import ArtifactRegistry


def test_recommendation_service_placeholder() -> None:
    service = RecommendationService(ArtifactRegistry(ServingConfig()))
    with pytest.raises(ServingError):
        service.recommend(user_idx=0, top_k=10)
