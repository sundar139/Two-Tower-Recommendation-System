"""Tests for FastAPI serving app scaffolding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from movie_recsys.serving.app import create_app
from movie_recsys.serving.dependencies import get_recommendation_service
from movie_recsys.serving.errors import user_not_found
from movie_recsys.serving.recommender import RecommendationRow
from movie_recsys.serving.registry import ArtifactRegistry


def test_healthz_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_startup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken_load(self: ArtifactRegistry) -> None:  # noqa: ARG001
        raise RuntimeError("artifact boot failure")

    monkeypatch.setattr(ArtifactRegistry, "load", _broken_load)
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        assert body["status"] == "not_ready"
        assert "artifact boot failure" in body["startup_error"]


def test_recommend_endpoint_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = SimpleNamespace(
        policy_name="ranker_topk_popularity_backfill",
        recommend=lambda user_idx, top_k: [  # noqa: ARG005
            RecommendationRow(
                item_idx=42,
                score=0.7,
                residual_score=0.2,
                ranker_score=0.6,
                popularity_score=0.4,
            )
        ],
    )
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.post("/v1/recommend", json={"user_idx": 0, "top_k": 5})
        assert response.status_code == 200
        payload = response.json()
        assert payload["policy_name"] == "ranker_topk_popularity_backfill"
        assert payload["returned_top_k"] == 1
        assert payload["recommendations"][0]["item_idx"] == 42
        assert payload["recommendations"][0]["rank"] == 1


def test_recommend_endpoint_serving_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    def _raise_error(user_idx: int, top_k: int) -> list[RecommendationRow]:  # noqa: ARG001
        raise user_not_found(user_idx)

    stub_service = SimpleNamespace(
        policy_name="ranker_topk_popularity_backfill",
        recommend=_raise_error,
    )
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.post("/v1/recommend", json={"user_idx": 999999, "top_k": 5})
        assert response.status_code == 404
        payload = response.json()
        assert payload["detail"]["error"] == "user_not_found"

