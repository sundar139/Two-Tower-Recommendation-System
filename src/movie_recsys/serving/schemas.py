"""Pydantic schemas for serving API requests and responses."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RecommendationItem(BaseModel):
    """Single recommendation output row."""

    rank: int = Field(ge=1)
    item_idx: int = Field(ge=0)
    score: float
    residual_score: float
    ranker_score: float
    popularity_score: float


class RecommendRequest(BaseModel):
    """Recommendation request payload."""

    user_idx: int = Field(ge=0)
    top_k: int = Field(default=20, ge=1, le=200)


class RecommendResponse(BaseModel):
    """Recommendation response payload."""

    model_config = ConfigDict(extra="forbid")

    user_idx: int
    requested_top_k: int
    returned_top_k: int
    policy_name: str
    total_candidates: int
    recommendations: list[RecommendationItem]


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


class RootResponse(BaseModel):
    """Root endpoint payload with service identity and links."""

    model_config = ConfigDict(extra="forbid")

    app_name: str
    version: str
    environment: str
    health_path: str
    ready_path: str
    recommend_path: str


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
