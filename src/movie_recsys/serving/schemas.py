"""Pydantic schemas for serving API requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RecommendationItem(BaseModel):
    """Single recommendation output row."""

    movie_id: int = Field(
        ge=1,
        serialization_alias="movieId",
        validation_alias=AliasChoices("movie_id", "movieId"),
    )
    item_idx: int = Field(ge=0)
    title: str
    genres: str
    release_year: int | None = None
    final_score: float
    residual_score: float | None = None
    ranker_score: float | None = None
    popularity_score: float
    rank_position: int = Field(ge=1)
    scorer_policy: str
    explanation: str | None = None


class RecommendRequest(BaseModel):
    """Recommendation request payload."""

    model_config = ConfigDict(extra="forbid")

    user_id: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("user_id", "userId"),
    )
    user_idx: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("user_idx", "userIdx"),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices("top_k", "k"),
    )
    exclude_seen: bool = Field(
        default=True,
        validation_alias=AliasChoices("exclude_seen", "excludeSeen"),
    )
    include_debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_debug", "includeDebug"),
    )
    allow_cold_start: bool = Field(
        default=True,
        validation_alias=AliasChoices("allow_cold_start", "allowColdStart"),
    )
    candidate_top_k: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("candidate_top_k", "candidateTopK"),
    )
    include_explanations: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_explanations", "includeExplanations"),
    )
    explanation_style: Literal["concise", "detailed"] = Field(
        default="concise",
        validation_alias=AliasChoices("explanation_style", "explanationStyle"),
    )
    max_explanation_items: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max_explanation_items", "maxExplanationItems"),
    )


class RecommendResponse(BaseModel):
    """Recommendation response payload."""

    model_config = ConfigDict(extra="forbid")

    user_id: int | None = None
    user_idx: int | None = None
    k: int
    cold_start: bool
    scorer_policy: str
    explanation_status: Literal["disabled", "generated", "unavailable", "failed"] = "disabled"
    overall_explanation: str | None = None
    recommendations: list[RecommendationItem]
    debug: dict[str, object] | None = None


class ReadinessResponse(BaseModel):
    """Readiness probe response payload."""

    model_config = ConfigDict(extra="forbid")

    status: str
    ready: bool
    model_loaded: bool
    startup_error: str | None = None


class HealthResponse(BaseModel):
    """Liveness payload used by /health and /healthz."""

    model_config = ConfigDict(extra="forbid")

    status: str


class Step5DMetricsSummary(BaseModel):
    """Approved Step 5D metrics included in serving metadata."""

    model_config = ConfigDict(extra="forbid")

    popularity_val_ndcg10: float
    selected_val_ndcg10: float
    popularity_val_recall50: float
    selected_val_recall50: float
    popularity_test_ndcg10: float
    selected_test_ndcg10: float
    popularity_test_recall50: float
    selected_test_recall50: float


class MetadataResponse(BaseModel):
    """Serving metadata and runtime contract payload."""

    model_config = ConfigDict(extra="forbid")

    app_name: str
    version: str
    environment: str
    production_policy: str
    candidate_top_k: int
    default_k: int
    max_k: int
    model_artifacts: dict[str, str]
    selected_scorer_weights: dict[str, float | int]
    approved_step5d_metrics: Step5DMetricsSummary
    explanations_enabled: bool
    explanation_provider: str
    chat_model: str
    fail_open: bool


class RootResponse(BaseModel):
    """Root endpoint payload with service identity and links."""

    model_config = ConfigDict(extra="forbid")

    app_name: str
    version: str
    environment: str
    health_path: str
    ready_path: str
    recommend_path: str


class ExplainRequest(BaseModel):
    """Request payload for dedicated recommendation explanation endpoint."""

    model_config = ConfigDict(extra="forbid")

    user_id: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("user_id", "userId"),
    )
    user_idx: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("user_idx", "userIdx"),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices("top_k", "k"),
    )
    exclude_seen: bool = Field(
        default=True,
        validation_alias=AliasChoices("exclude_seen", "excludeSeen"),
    )
    allow_cold_start: bool = Field(
        default=True,
        validation_alias=AliasChoices("allow_cold_start", "allowColdStart"),
    )
    candidate_top_k: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("candidate_top_k", "candidateTopK"),
    )
    include_debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_debug", "includeDebug"),
    )
    style: Literal["concise", "detailed"] = Field(
        default="concise",
        validation_alias=AliasChoices("style", "explanation_style", "explanationStyle"),
    )
    max_explanation_items: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max_explanation_items", "maxExplanationItems"),
    )
    recommendation_items: list[RecommendationItem] | None = Field(
        default=None,
        validation_alias=AliasChoices("recommendation_items", "recommendationItems"),
    )


class ExplainResponse(RecommendResponse):
    """Response payload for dedicated recommendation explanation endpoint."""


class UserHistoryItem(BaseModel):
    """Single history item returned by /users/{user_id}/history."""

    model_config = ConfigDict(extra="forbid")

    movie_id: int = Field(
        serialization_alias="movieId",
        validation_alias=AliasChoices("movie_id", "movieId"),
    )
    item_idx: int
    title: str
    genres: str
    timestamp: int | None = None


class UserHistoryResponse(BaseModel):
    """User history payload for serving introspection endpoints."""

    model_config = ConfigDict(extra="forbid")

    user_id: int
    user_idx: int
    history_count: int
    history: list[UserHistoryItem]


class ErrorDetail(BaseModel):
    """Structured API error payload details."""

    model_config = ConfigDict(extra="forbid")

    error: str
    message: str


class ErrorResponse(BaseModel):
    """Envelope for API error responses."""

    model_config = ConfigDict(extra="forbid")

    detail: ErrorDetail
