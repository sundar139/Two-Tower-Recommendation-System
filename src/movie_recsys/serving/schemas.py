"""Pydantic schemas for serving API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RecommendationItem(BaseModel):
    """Single recommendation output row."""

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
    top_k: int
    total_candidates: int
    recommendations: list[RecommendationItem]


class ReadinessResponse(BaseModel):
    """Readiness probe response payload."""

    model_config = ConfigDict(extra="forbid")

    status: str
    ready: bool
