"""FastAPI dependency providers for serving runtime objects."""

from __future__ import annotations

from fastapi import Request

from movie_recsys.serving.config import ServingConfig, load_serving_config
from movie_recsys.serving.errors import artifacts_not_ready, service_not_ready
from movie_recsys.serving.recommender import RecommendationService
from movie_recsys.serving.registry import ArtifactRegistry


def get_serving_config() -> ServingConfig:
    """Load serving config using default project path."""

    return load_serving_config()


def get_registry(request: Request) -> ArtifactRegistry:
    """Access app-scoped artifact registry."""

    registry = getattr(request.app.state, "artifact_registry", None)
    if not isinstance(registry, ArtifactRegistry):
        raise artifacts_not_ready()
    return registry


def get_recommendation_service(request: Request) -> RecommendationService:
    """Access app-scoped recommendation service."""

    service = getattr(request.app.state, "recommendation_service", None)
    if not isinstance(service, RecommendationService):
        raise service_not_ready()
    return service
