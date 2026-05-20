"""FastAPI application factory for recommendation serving."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from movie_recsys import __version__
from movie_recsys.serving.config import ServingConfig, load_serving_config
from movie_recsys.serving.dependencies import get_recommendation_service, get_registry
from movie_recsys.serving.errors import ServingError, to_http_exception
from movie_recsys.serving.recommender import RecommendationService
from movie_recsys.serving.registry import ArtifactRegistry
from movie_recsys.serving.schemas import (
    ReadinessResponse,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config: ServingConfig = load_serving_config()
    registry = ArtifactRegistry(config)
    app.state.serving_config = config
    app.state.artifact_registry = registry
    app.state.recommendation_service = RecommendationService(registry)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="Movie Recommender API", version=__version__, lifespan=_lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ReadinessResponse)
    def readyz(registry: ArtifactRegistry = Depends(get_registry)) -> ReadinessResponse:
        return ReadinessResponse(
            status="ready" if registry.is_ready() else "starting",
            ready=registry.is_ready(),
        )

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommend(
        request: RecommendRequest,
        service: RecommendationService = Depends(get_recommendation_service),
    ) -> RecommendResponse:
        try:
            rows = service.recommend(user_idx=request.user_idx, top_k=request.top_k)
        except ServingError as exc:
            raise to_http_exception(exc) from exc

        items = [
            RecommendationItem(
                item_idx=row.item_idx,
                score=row.score,
                residual_score=row.residual_score,
                ranker_score=row.ranker_score,
                popularity_score=row.popularity_score,
            )
            for row in rows
        ]
        return RecommendResponse(
            user_idx=request.user_idx,
            top_k=request.top_k,
            total_candidates=len(items),
            recommendations=items,
        )

    return app
