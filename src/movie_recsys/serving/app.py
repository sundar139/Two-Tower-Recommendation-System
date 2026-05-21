"""FastAPI application factory for recommendation serving."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from movie_recsys import __version__
from movie_recsys.serving.config import ServingConfig, load_serving_config
from movie_recsys.serving.dependencies import get_recommendation_service, get_registry
from movie_recsys.serving.errors import ServingError, to_http_exception
from movie_recsys.serving.recommender import RecommendationService
from movie_recsys.serving.registry import ArtifactRegistry
from movie_recsys.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    ReadinessResponse,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
    RootResponse,
    Step5DMetricsSummary,
    UserHistoryItem,
    UserHistoryResponse,
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


def _step5d_metrics_summary() -> Step5DMetricsSummary:
    return Step5DMetricsSummary(
        popularity_val_ndcg10=0.2663275394,
        selected_val_ndcg10=0.3115227229,
        popularity_val_recall50=0.7114404187,
        selected_val_recall50=0.7296580790,
        popularity_test_ndcg10=0.2625267030,
        selected_test_ndcg10=0.3175914351,
        popularity_test_recall50=0.6795100759,
        selected_test_recall50=0.7123735486,
    )


def _build_recommend_response(
    *,
    request: RecommendRequest,
    service: RecommendationService,
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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    config_path = os.environ.get("MOVIE_RECSYS_SERVING_CONFIG", "configs/serving.yaml")
    startup_config = load_serving_config(config_path)
    app = FastAPI(title=startup_config.app_name, version=__version__, lifespan=_lifespan)

    @app.exception_handler(ServingError)
    async def serving_error_handler(
        _request: Request,
        exc: ServingError,
    ) -> JSONResponse:
        http_exc = to_http_exception(exc)
        return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})

    @app.get("/", response_model=RootResponse)
    def root(request: Request) -> RootResponse:
        config: ServingConfig = request.app.state.serving_config
        return RootResponse(
            app_name=config.app_name,
            version=__version__,
            environment=config.environment,
            health_path="/health",
            ready_path="/ready",
            recommend_path="/recommendations",
        )

    @app.get("/health", response_model=HealthResponse)
    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
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

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(
        request: Request,
        registry: ArtifactRegistry = Depends(get_registry),
    ) -> MetadataResponse:
        config: ServingConfig = request.app.state.serving_config
        artifact_paths = registry.validate_paths()
        return MetadataResponse(
            app_name=config.app_name,
            version=__version__,
            environment=config.environment,
            production_policy=config.scoring.policy_name,
            candidate_top_k=config.runtime.candidate_top_k,
            default_k=config.runtime.default_top_k,
            max_k=config.runtime.max_top_k,
            model_artifacts={
                "retriever_config": artifact_paths["retrieval_config"].name,
                "ranker_config": artifact_paths["ranker_config"].name,
                "residual_checkpoint": artifact_paths["residual_checkpoint"].name,
                "ranker_checkpoint": artifact_paths["ranker_checkpoint"].name,
                "faiss_index": artifact_paths["faiss_index"].name,
                "faiss_mapping": artifact_paths["faiss_mapping"].name,
                "faiss_metadata": artifact_paths["faiss_metadata"].name,
            },
            selected_scorer_weights={
                "alpha": config.scoring.alpha,
                "beta": config.scoring.beta,
                "gamma": config.scoring.gamma,
                "top_k_focus": config.scoring.top_k_focus,
            },
            approved_step5d_metrics=_step5d_metrics_summary(),
        )

    @app.post(
        "/recommendations",
        response_model=RecommendResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
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
        return _build_recommend_response(request=request, service=service)

    @app.get(
        "/recommendations/{user_id}",
        response_model=RecommendResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def recommend_by_user_id(
        user_id: int,
        k: int = Query(default=10, ge=1, le=200),
        service: RecommendationService = Depends(get_recommendation_service),
    ) -> RecommendResponse:
        user_idx = service.resolve_user_idx(user_id=user_id)
        if user_idx is None:
            raise to_http_exception(
                ServingError(
                    code="user_not_found",
                    message=f"Unknown user_id: {user_id}",
                    status_code=404,
                )
            )

        request = RecommendRequest(user_idx=user_idx, top_k=k)
        return _build_recommend_response(request=request, service=service)

    @app.get(
        "/users/{user_id}/history",
        response_model=UserHistoryResponse,
        responses={404: {"model": ErrorResponse}},
    )
    def user_history(
        user_id: int,
        service: RecommendationService = Depends(get_recommendation_service),
    ) -> UserHistoryResponse:
        user_idx, rows = service.get_user_history(user_id=user_id, limit=100)
        return UserHistoryResponse(
            user_id=user_id,
            user_idx=user_idx,
            history_count=len(rows),
            history=[
                UserHistoryItem(
                    movie_id=row.movie_id,
                    item_idx=row.item_idx,
                    title=row.title,
                    genres=row.genres,
                    timestamp=row.timestamp,
                )
                for row in rows
            ],
        )

    return app
