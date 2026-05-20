"""Pydantic schemas for serving API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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


class ErrorDetail(BaseModel):
    """Structured API error payload details."""

    model_config = ConfigDict(extra="forbid")

    error: str
    message: str


class ErrorResponse(BaseModel):
    """Envelope for API error responses."""

    model_config = ConfigDict(extra="forbid")

    detail: ErrorDetail
