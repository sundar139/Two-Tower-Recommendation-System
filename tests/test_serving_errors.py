"""Tests for serving error translation helpers."""

from __future__ import annotations

from movie_recsys.serving.errors import ServingError, to_http_exception


def test_to_http_exception_payload() -> None:
    exc = to_http_exception(ServingError(message="boom", code="boom_error", status_code=503))
    assert exc.status_code == 503
    assert exc.detail == {"error": "boom_error", "message": "boom"}
