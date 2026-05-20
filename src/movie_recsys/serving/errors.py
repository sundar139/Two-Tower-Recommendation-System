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


def to_http_exception(error: ServingError) -> HTTPException:
    """Convert a domain error into a structured HTTP exception."""

    return HTTPException(
        status_code=error.status_code,
        detail={"error": error.code, "message": error.message},
    )
