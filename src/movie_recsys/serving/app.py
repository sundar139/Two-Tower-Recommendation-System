"""FastAPI application factory for recommendation serving."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from movie_recsys import __version__
from movie_recsys.serving.config import ServingConfig, load_serving_config
from movie_recsys.serving.dependencies import get_recommendation_service, get_registry
from movie_recsys.serving.errors import ServingError, explanation_unavailable, to_http_exception
from movie_recsys.serving.explanations import (
    RecommendationEvidenceItem,
    RecommendationExplanationContext,
    explain_recommendations,
)
from movie_recsys.serving.ollama_client import (
    OllamaClient,
    OllamaClientConfig,
    OllamaUnavailableError,
)
from movie_recsys.serving.recommender import RecommendationService
from movie_recsys.serving.registry import ArtifactRegistry
from movie_recsys.serving.schemas import (
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
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
    app.state.ollama_client = None
    app.state.startup_error = None

    if config.explanations.enabled and config.explanations.provider.lower() == "ollama":
        app.state.ollama_client = OllamaClient(
            OllamaClientConfig(
                base_url=config.explanations.base_url,
                chat_model=config.explanations.chat_model,
                embedding_model=config.explanations.embedding_model,
                timeout_seconds=float(config.explanations.timeout_seconds),
                temperature=float(config.explanations.temperature),
            )
        )

    try:
        registry.load()
        log_event(logger, "serving_startup", ready=True)
    except Exception as exc:  # noqa: BLE001
        app.state.startup_error = str(exc)
        log_event(logger, "serving_startup", ready=False, error=str(exc))

    try:
        yield
    finally:
        ollama_client = getattr(app.state, "ollama_client", None)
        if isinstance(ollama_client, OllamaClient):
            ollama_client.close()


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
    result = service.recommend(
        user_idx=request.user_idx,
        user_id=request.user_id,
        top_k=request.top_k,
        exclude_seen=request.exclude_seen,
        candidate_top_k=request.candidate_top_k,
        allow_cold_start=request.allow_cold_start,
        include_debug=request.include_debug,
    )
    items = [
        RecommendationItem(
            movie_id=row.movie_id,
            item_idx=row.item_idx,
            title=row.title,
            genres=row.genres,
            release_year=row.release_year,
            final_score=row.final_score,
            residual_score=row.residual_score,
            ranker_score=row.ranker_score,
            popularity_score=row.popularity_score,
            rank_position=row.rank_position,
            scorer_policy=row.scorer_policy,
        )
        for row in result.recommendations
    ]
    return RecommendResponse(
        user_id=result.user_id,
        user_idx=result.user_idx,
        k=result.k,
        cold_start=result.cold_start,
        scorer_policy=result.scorer_policy,
        explanation_status="disabled",
        overall_explanation=None,
        recommendations=items,
        debug=result.debug,
    )


def _resolved_user_id(
    *,
    response: RecommendResponse,
    service: RecommendationService,
) -> int | None:
    if response.user_id is not None:
        return response.user_id
    if response.user_idx is not None:
        return service.resolve_user_id(user_idx=response.user_idx)
    return None


def _resolved_style(value: str) -> Literal["concise", "detailed"]:
    return "detailed" if value == "detailed" else "concise"


def _apply_explanations(
    *,
    api_request: Request,
    response: RecommendResponse,
    service: RecommendationService,
    include_explanations: bool,
    explanation_style: str,
    max_explanation_items: int | None,
    include_debug: bool,
) -> RecommendResponse:
    config: ServingConfig = api_request.app.state.serving_config
    if not include_explanations or not config.explanations.enabled:
        response.explanation_status = "disabled"
        response.overall_explanation = None
        return response

    ollama_client = getattr(api_request.app.state, "ollama_client", None)
    if not isinstance(ollama_client, OllamaClient):
        if config.explanations.fail_open:
            response.explanation_status = "unavailable"
            response.overall_explanation = None
            return response
        raise explanation_unavailable("Ollama client is not configured")

    max_items = int(config.explanations.max_items)
    requested_max = max_explanation_items if max_explanation_items is not None else max_items
    resolved_max_items = max(0, min(int(requested_max), max_items, len(response.recommendations)))
    if resolved_max_items == 0:
        response.explanation_status = "generated"
        response.overall_explanation = None
        return response

    resolved_user_id = _resolved_user_id(response=response, service=service)
    recent_titles: list[str] = []
    if resolved_user_id is not None:
        try:
            _resolved_idx, history_rows = service.get_user_history(
                user_id=resolved_user_id,
                limit=8,
            )
            recent_titles = [row.title for row in history_rows if row.title][:8]
        except ServingError:
            recent_titles = []

    top_genres = service.get_user_genre_affinity(
        user_id=resolved_user_id,
        user_idx=response.user_idx,
        top_n=5,
    )

    context = RecommendationExplanationContext(
        user_id=resolved_user_id,
        user_idx=response.user_idx,
        style=_resolved_style(explanation_style),
        include_debug=include_debug,
        scorer_policy=response.scorer_policy,
        items=[
            RecommendationEvidenceItem(
                movie_id=item.movie_id,
                rank_position=item.rank_position,
                title=item.title,
                genres=item.genres,
                release_year=item.release_year,
                final_score=item.final_score,
                ranker_score=item.ranker_score,
                popularity_score=item.popularity_score,
                residual_score=item.residual_score,
                scorer_policy=item.scorer_policy,
            )
            for item in response.recommendations
        ],
        recent_titles=recent_titles,
        top_genres=top_genres,
    )

    try:
        overall, per_item = explain_recommendations(
            context=context,
            client=ollama_client,
            max_items=resolved_max_items,
        )
    except OllamaUnavailableError as exc:
        if config.explanations.fail_open:
            response.explanation_status = "unavailable"
            response.overall_explanation = None
            return response
        raise explanation_unavailable(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        if config.explanations.fail_open:
            response.explanation_status = "failed"
            response.overall_explanation = None
            return response
        raise explanation_unavailable(str(exc)) from exc

    for index, explanation in enumerate(per_item):
        if index >= len(response.recommendations):
            break
        response.recommendations[index].explanation = explanation

    response.explanation_status = "generated"
    response.overall_explanation = overall
    return response


def _explain_from_payload_items(
    *,
    payload: ExplainRequest,
    service: RecommendationService,
) -> ExplainResponse:
    recommendation_items = payload.recommendation_items or []
    scorer_policy = (
        recommendation_items[0].scorer_policy if recommendation_items else service.policy_name
    )
    return ExplainResponse(
        user_id=payload.user_id,
        user_idx=payload.user_idx,
        k=len(recommendation_items),
        cold_start=False,
        scorer_policy=scorer_policy,
        explanation_status="disabled",
        overall_explanation=None,
        recommendations=[item.model_copy(deep=True) for item in recommendation_items],
        debug=None,
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
            explanations_enabled=config.explanations.enabled,
            explanation_provider=config.explanations.provider,
            chat_model=config.explanations.chat_model,
            fail_open=config.explanations.fail_open,
        )

    @app.post(
        "/recommendations",
        response_model=RecommendResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    @app.post(
        "/v1/recommend",
        response_model=RecommendResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    def recommend(
        payload: RecommendRequest,
        api_request: Request,
        service: RecommendationService = Depends(get_recommendation_service),
    ) -> RecommendResponse:
        response = _build_recommend_response(request=payload, service=service)
        return _apply_explanations(
            api_request=api_request,
            response=response,
            service=service,
            include_explanations=payload.include_explanations,
            explanation_style=payload.explanation_style,
            max_explanation_items=payload.max_explanation_items,
            include_debug=payload.include_debug,
        )

    @app.get(
        "/recommendations/{user_id}",
        response_model=RecommendResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def recommend_by_user_id(
        api_request: Request,
        user_id: int,
        k: int = Query(default=10, ge=1, le=200),
        allow_cold_start: bool = Query(default=True),
        exclude_seen: bool = Query(default=True),
        include_debug: bool = Query(default=False),
        include_explanations: bool = Query(default=False),
        explanation_style: Literal["concise", "detailed"] = Query(default="concise"),
        max_explanation_items: int | None = Query(default=None, ge=1),
        service: RecommendationService = Depends(get_recommendation_service),
    ) -> RecommendResponse:
        payload = RecommendRequest(
            user_id=user_id,
            top_k=k,
            allow_cold_start=allow_cold_start,
            exclude_seen=exclude_seen,
            include_debug=include_debug,
            include_explanations=include_explanations,
            explanation_style=explanation_style,
            max_explanation_items=max_explanation_items,
        )
        response = _build_recommend_response(request=payload, service=service)
        return _apply_explanations(
            api_request=api_request,
            response=response,
            service=service,
            include_explanations=payload.include_explanations,
            explanation_style=payload.explanation_style,
            max_explanation_items=payload.max_explanation_items,
            include_debug=payload.include_debug,
        )

    @app.post(
        "/v1/explain",
        response_model=ExplainResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    @app.post(
        "/explanations/recommendations",
        response_model=ExplainResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def explain_recommendations_endpoint(
        payload: ExplainRequest,
        api_request: Request,
        service: RecommendationService = Depends(get_recommendation_service),
    ) -> ExplainResponse:
        if payload.recommendation_items:
            response = _explain_from_payload_items(payload=payload, service=service)
        else:
            recommend_payload = RecommendRequest(
                user_id=payload.user_id,
                user_idx=payload.user_idx,
                top_k=payload.top_k,
                exclude_seen=payload.exclude_seen,
                include_debug=payload.include_debug,
                allow_cold_start=payload.allow_cold_start,
                candidate_top_k=payload.candidate_top_k,
                include_explanations=False,
                explanation_style=payload.style,
                max_explanation_items=payload.max_explanation_items,
            )
            base_response = _build_recommend_response(request=recommend_payload, service=service)
            response = ExplainResponse.model_validate(base_response.model_dump())

        explained = _apply_explanations(
            api_request=api_request,
            response=response,
            service=service,
            include_explanations=True,
            explanation_style=payload.style,
            max_explanation_items=payload.max_explanation_items,
            include_debug=payload.include_debug,
        )
        return ExplainResponse.model_validate(explained.model_dump())

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
