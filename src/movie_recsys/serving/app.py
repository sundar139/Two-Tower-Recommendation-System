"""FastAPI application factory for recommendation serving."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from movie_recsys import __version__
from movie_recsys.serving.config import ServingConfig, load_serving_config
from movie_recsys.serving.dependencies import get_recommendation_service, get_registry
from movie_recsys.serving.errors import ServingError, to_http_exception
from movie_recsys.serving.recommender import RecommendationService
from movie_recsys.serving.registry import ArtifactRegistry
from movie_recsys.serving.schemas import (
    ErrorResponse,
    ReadinessResponse,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from movie_recsys.serving.telemetry import get_logger, log_event


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = os.environ.get("MOVIE_RECSYS_SERVING_CONFIG", "configs/serving.yaml")
    config: ServingConfig = load_serving_config(config_path)
    registry = ArtifactRegistry(config)
    logger = get_logger()

    app.state.serving_config = config
    app.state.artifact_registry = registry
    app.state.recommendation_service = RecommendationService(registry)
    app.state.startup_error = None

    try:
        registry.load()
        log_event(logger, "serving_startup", ready=True)
    except Exception as exc:  # noqa: BLE001
        app.state.startup_error = str(exc)
        log_event(logger, "serving_startup", ready=False, error=str(exc))

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="Movie Recommender API", version=__version__, lifespan=_lifespan)

    @app.exception_handler(ServingError)
    async def serving_error_handler(
        _request: Request,
        exc: ServingError,
    ) -> JSONResponse:
        http_exc = to_http_exception(exc)
        return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ReadinessResponse)
    def readyz(
        request: Request,
        registry: ArtifactRegistry = Depends(get_registry),
    ) -> ReadinessResponse:
        startup_error = getattr(request.app.state, "startup_error", None)
        if startup_error:
            return ReadinessResponse(
                status="not_ready",
                ready=False,
                model_loaded=False,
                startup_error=str(startup_error),
            )
        return ReadinessResponse(
            status="ready" if registry.is_ready() else "starting",
            ready=registry.is_ready(),
            model_loaded=registry.is_ready(),
            startup_error=None,
        )

    @app.post(
        "/v1/recommend",
        response_model=RecommendResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def recommend(
        request: RecommendRequest,
        service: RecommendationService = Depends(get_recommendation_service),
    ) -> RecommendResponse:
        rows = service.recommend(user_idx=request.user_idx, top_k=request.top_k)

        items = [
            RecommendationItem(
                rank=index,
                item_idx=row.item_idx,
                score=row.score,
                residual_score=row.residual_score,
                ranker_score=row.ranker_score,
                popularity_score=row.popularity_score,
            )
            for index, row in enumerate(rows, start=1)
        ]
        return RecommendResponse(
            user_idx=request.user_idx,
            requested_top_k=request.top_k,
            returned_top_k=len(items),
            policy_name=service.policy_name,
            total_candidates=len(rows),
            recommendations=items,
        )

    return app
