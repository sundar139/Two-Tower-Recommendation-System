"""Tests for FastAPI serving app scaffolding."""

from __future__ import annotations

from fastapi.testclient import TestClient

from movie_recsys.serving.app import create_app


def test_healthz_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
