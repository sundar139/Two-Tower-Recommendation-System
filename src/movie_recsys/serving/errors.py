"""Typed serving errors and HTTP conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, status


@dataclass(slots=True)
class ServingError(Exception):
    """Domain error raised by serving components."""

    message: str
    code: str = "serving_error"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR


def artifacts_not_ready() -> ServingError:
    return ServingError(
        message="Serving artifacts are not loaded",
        code="artifacts_not_ready",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def service_not_ready() -> ServingError:
    return ServingError(
        message="Recommendation service is not initialized",
        code="service_not_ready",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def user_not_found(user_idx: int) -> ServingError:
    return ServingError(
        message=f"Unknown user_idx: {user_idx}",
        code="user_not_found",
        status_code=status.HTTP_404_NOT_FOUND,
    )


def user_id_not_found(user_id: int) -> ServingError:
    return ServingError(
        message=f"Unknown user_id: {user_id}",
        code="user_not_found",
        status_code=status.HTTP_404_NOT_FOUND,
    )


def invalid_request(message: str) -> ServingError:
    return ServingError(
        message=message,
        code="invalid_request",
        status_code=status.HTTP_400_BAD_REQUEST,
    )


def invalid_candidate_top_k(
    *,
    candidate_top_k: int,
    requested_top_k: int,
    max_allowed: int,
) -> ServingError:
    return ServingError(
        message=(
            "candidate_top_k must be >= requested k and <= "
            f"{max_allowed}, received candidate_top_k={candidate_top_k}, "
            f"requested_k={requested_top_k}"
        ),
        code="invalid_candidate_top_k",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


def invalid_top_k(*, top_k: int, min_allowed: int, max_allowed: int) -> ServingError:
    return ServingError(
        message=(
            f"top_k must be between {min_allowed} and {max_allowed}, "
            f"received {top_k}"
        ),
        code="invalid_top_k",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


def no_candidates(message: str = "No unseen candidates available for this user") -> ServingError:
    return ServingError(
        message=message,
        code="no_candidates",
        status_code=status.HTTP_404_NOT_FOUND,
    )


def feature_mismatch(missing_columns: list[str]) -> ServingError:
    return ServingError(
        message="Serving feature frame is missing ranker columns: " + ", ".join(missing_columns),
        code="feature_mismatch",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def to_http_exception(error: ServingError) -> HTTPException:
    """Convert a domain error into a structured HTTP exception."""

    return HTTPException(
        status_code=error.status_code,
        detail={"error": error.code, "message": error.message},
    )
