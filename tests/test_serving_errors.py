"""Tests for serving error translation helpers."""

from __future__ import annotations

from movie_recsys.serving.errors import (
    ServingError,
    artifacts_not_ready,
    invalid_top_k,
    no_candidates,
    to_http_exception,
    user_not_found,
)


def test_to_http_exception_payload() -> None:
    exc = to_http_exception(ServingError(message="boom", code="boom_error", status_code=503))
    assert exc.status_code == 503
    assert exc.detail == {"error": "boom_error", "message": "boom"}


def test_error_factories() -> None:
    assert artifacts_not_ready().code == "artifacts_not_ready"
    assert user_not_found(7).status_code == 404
    assert invalid_top_k(top_k=2000, min_allowed=1, max_allowed=200).status_code == 422
    assert no_candidates().code == "no_candidates"
