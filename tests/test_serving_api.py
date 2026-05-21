"""Tests for Step 6B serving API contract behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from movie_recsys.serving.app import create_app
from movie_recsys.serving.dependencies import get_recommendation_service
from movie_recsys.serving.errors import user_id_not_found
from movie_recsys.serving.ollama_client import OllamaClient, OllamaClientConfig
from movie_recsys.serving.recommender import RecommendationResult, RecommendationRow, UserHistoryRow
from movie_recsys.serving.registry import ArtifactRegistry


@dataclass
class StubService:
    response: RecommendationResult
    history: list[UserHistoryRow]
    raise_unknown_when_cold_start_disabled: bool = False
    last_recommend_kwargs: dict[str, Any] | None = None
    last_history_limit: int | None = None

    @property
    def policy_name(self) -> str:
        return self.response.scorer_policy

    def recommend(
        self,
        *,
        user_idx: int | None,
        user_id: int | None,
        top_k: int,
        exclude_seen: bool,
        candidate_top_k: int | None,
        allow_cold_start: bool,
        include_debug: bool,
    ) -> RecommendationResult:
        self.last_recommend_kwargs = {
            "user_idx": user_idx,
            "user_id": user_id,
            "top_k": top_k,
            "exclude_seen": exclude_seen,
            "candidate_top_k": candidate_top_k,
            "allow_cold_start": allow_cold_start,
            "include_debug": include_debug,
        }
        if self.raise_unknown_when_cold_start_disabled and not allow_cold_start:
            raise user_id_not_found(user_id if user_id is not None else -1)
        if allow_cold_start and user_id == 999999:
            return RecommendationResult(
                user_id=user_id,
                user_idx=None,
                k=top_k,
                cold_start=True,
                scorer_policy="popularity_fallback",
                recommendations=self.response.recommendations,
                debug=None,
            )
        return self.response

    def get_user_history(
        self,
        *,
        user_id: int,
        limit: int = 100,
    ) -> tuple[int, list[UserHistoryRow]]:
        self.last_history_limit = limit
        return (self.response.user_idx or 0), self.history[:limit]

    def resolve_user_id(self, *, user_idx: int) -> int | None:
        if self.response.user_idx == user_idx:
            return self.response.user_id
        return None

    def get_user_genre_affinity(
        self,
        *,
        user_id: int | None,
        user_idx: int | None,
        top_n: int = 5,
    ) -> list[str]:
        _ = user_id, user_idx
        return ["Drama", "Crime"][:top_n]


def _sample_result(*, user_id: int = 709, user_idx: int = 0, k: int = 2) -> RecommendationResult:
    rows = [
        RecommendationRow(
            movie_id=318,
            item_idx=263,
            title="Shawshank Redemption, The (1994)",
            genres="Crime|Drama",
            release_year=1994,
            final_score=0.9,
            residual_score=0.3,
            ranker_score=0.8,
            popularity_score=0.7,
            rank_position=1,
            scorer_policy="ranker_topk_popularity_backfill",
        ),
        RecommendationRow(
            movie_id=296,
            item_idx=248,
            title="Pulp Fiction (1994)",
            genres="Comedy|Crime|Drama|Thriller",
            release_year=1994,
            final_score=0.8,
            residual_score=0.2,
            ranker_score=0.7,
            popularity_score=0.6,
            rank_position=2,
            scorer_policy="ranker_topk_popularity_backfill",
        ),
    ]
    return RecommendationResult(
        user_id=user_id,
        user_idx=user_idx,
        k=k,
        cold_start=False,
        scorer_policy="ranker_topk_popularity_backfill",
        recommendations=rows[:k],
        debug=None,
    )


def _make_mock_ollama_client(
    handler: httpx.BaseTransport,
) -> OllamaClient:
    config = OllamaClientConfig(
        base_url="http://127.0.0.1:11434",
        chat_model="qwen3:4b",
        embedding_model="qwen3-embedding:0.6b",
        timeout_seconds=2.0,
        temperature=0.2,
    )
    http_client = httpx.Client(
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        transport=handler,
    )
    return OllamaClient(config=config, http_client=http_client)


def test_health_aliases_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health")
        healthz = client.get("/healthz")
        assert health.status_code == 200
        assert healthz.status_code == 200
        assert health.json() == healthz.json()


def test_ready_aliases_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()
    with TestClient(app) as client:
        ready = client.get("/ready")
        readyz = client.get("/readyz")
        assert ready.status_code == 200
        assert readyz.status_code == 200
        assert ready.json() == readyz.json()


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


def test_metadata_includes_selected_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/metadata")
        assert response.status_code == 200
        payload = response.json()
        assert payload["production_policy"] == "ranker_topk_popularity_backfill"
        assert payload["explanations_enabled"] is True
        assert payload["explanation_provider"] == "ollama"
        assert payload["chat_model"]
        assert payload["fail_open"] is True


def test_recommendations_and_v1_have_same_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        payload = {"user_idx": 0, "k": 2}
        rec = client.post("/recommendations", json=payload)
        v1 = client.post("/v1/recommend", json=payload)
        assert rec.status_code == 200
        assert v1.status_code == 200
        rec_json = rec.json()
        v1_json = v1.json()
        assert set(rec_json.keys()) == set(v1_json.keys())
        assert set(rec_json["recommendations"][0].keys()) == set(
            v1_json["recommendations"][0].keys()
        )


def test_unknown_user_with_allow_cold_start_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.post(
            "/recommendations",
            json={"user_id": 999999, "k": 2, "allow_cold_start": True},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["cold_start"] is True


def test_unknown_user_with_allow_cold_start_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = StubService(
        response=_sample_result(),
        history=[],
        raise_unknown_when_cold_start_disabled=True,
    )
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.post(
            "/recommendations",
            json={"user_id": 999999, "k": 2, "allow_cold_start": False},
        )
        assert response.status_code == 404


def test_invalid_k_rejected_consistently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.post("/recommendations", json={"user_idx": 0, "k": 0})
        assert response.status_code in {400, 422}


def test_schema_aliases_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.post(
            "/recommendations",
            json={
                "userId": 709,
                "k": 2,
                "excludeSeen": False,
                "includeDebug": True,
                "allowColdStart": True,
            },
        )
        assert response.status_code == 200
        kwargs = stub_service.last_recommend_kwargs
        assert kwargs is not None
        assert kwargs["user_id"] == 709
        assert kwargs["exclude_seen"] is False
        assert kwargs["include_debug"] is True


def test_user_history_caps_at_100(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    history_rows = [
        UserHistoryRow(
            movie_id=idx + 1,
            item_idx=idx,
            title=f"Movie {idx}",
            genres="Drama",
            timestamp=1_500_000_000_000 + idx,
        )
        for idx in range(200)
    ]
    stub_service = StubService(response=_sample_result(), history=history_rows)
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.get("/users/709/history")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["history"]) == 100
        assert stub_service.last_history_limit == 100


def test_exclude_seen_forwarded_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        response = client.post(
            "/recommendations",
            json={"user_idx": 0, "k": 2, "exclude_seen": True},
        )
        assert response.status_code == 200
        kwargs = stub_service.last_recommend_kwargs
        assert kwargs is not None
        assert kwargs["exclude_seen"] is True


def test_deterministic_repeated_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service
    with TestClient(app) as client:
        first = client.post("/recommendations", json={"user_idx": 0, "k": 2})
        second = client.post("/recommendations", json={"user_idx": 0, "k": 2})
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()


def test_include_explanations_false_does_not_call_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    calls: dict[str, int] = {"generate": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            calls["generate"] += 1
            return httpx.Response(status_code=200, json={"response": "unused"})
        return httpx.Response(status_code=404, json={"error": "unexpected path"})

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service

    with TestClient(app) as client:
        client.app.state.ollama_client = _make_mock_ollama_client(httpx.MockTransport(handler))
        response = client.post(
            "/recommendations",
            json={"user_id": 709, "k": 2, "include_explanations": False},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["explanation_status"] == "disabled"
        assert all(row["explanation"] is None for row in payload["recommendations"])
        assert calls["generate"] == 0


def test_recommendations_include_explanations_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    calls: dict[str, int] = {"generate": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            calls["generate"] += 1
            return httpx.Response(
                status_code=200,
                json={
                    "response": (
                        '{"overall_summary": "Top picks align with your recent genre mix.",' 
                        '"item_explanations": ['
                        '{"rank_position": 1, "movie_id": 318, '
                        '"explanation": "Strong genre alignment with your recent history."},'
                        '{"rank_position": 2, "movie_id": 296, '
                        '"explanation": "High combined ranker and popularity support."}'
                        "]}"
                    )
                },
            )
        return httpx.Response(status_code=404, json={"error": "unexpected path"})

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service

    with TestClient(app) as client:
        client.app.state.ollama_client = _make_mock_ollama_client(httpx.MockTransport(handler))
        response = client.post(
            "/recommendations",
            json={
                "user_id": 709,
                "k": 2,
                "include_explanations": True,
                "explanation_style": "concise",
                "max_explanation_items": 1,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["explanation_status"] == "generated"
        assert isinstance(payload["overall_explanation"], str)
        assert payload["recommendations"][0]["explanation"]
        assert payload["recommendations"][1]["explanation"] is None
        assert calls["generate"] == 1


def test_recommendations_fail_open_when_ollama_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    def unavailable_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service

    with TestClient(app) as client:
        client.app.state.ollama_client = _make_mock_ollama_client(
            httpx.MockTransport(unavailable_handler)
        )
        baseline = client.post(
            "/recommendations",
            json={"user_id": 709, "k": 2, "include_explanations": False},
        )
        explained = client.post(
            "/recommendations",
            json={"user_id": 709, "k": 2, "include_explanations": True},
        )
        assert baseline.status_code == 200
        assert explained.status_code == 200

        baseline_ids = [row["movieId"] for row in baseline.json()["recommendations"]]
        explained_ids = [row["movieId"] for row in explained.json()["recommendations"]]
        assert baseline_ids == explained_ids
        assert explained.json()["explanation_status"] == "unavailable"


def test_explanations_do_not_reorder_recommendations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(
                status_code=200,
                json={
                    "response": (
                        '{"overall_summary": "Stable ranking retained.",' 
                        '"item_explanations": ['
                        '{"rank_position": 1, "movie_id": 318, "explanation": "Rank one."},'
                        '{"rank_position": 2, "movie_id": 296, "explanation": "Rank two."}'
                        "]}"
                    )
                },
            )
        return httpx.Response(status_code=404, json={"error": "unexpected path"})

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service

    with TestClient(app) as client:
        client.app.state.ollama_client = _make_mock_ollama_client(httpx.MockTransport(handler))
        baseline = client.post(
            "/recommendations",
            json={"user_id": 709, "k": 2, "include_explanations": False},
        )
        explained = client.post(
            "/recommendations",
            json={"user_id": 709, "k": 2, "include_explanations": True},
        )
        assert baseline.status_code == 200
        assert explained.status_code == 200

        baseline_rows = baseline.json()["recommendations"]
        explained_rows = explained.json()["recommendations"]
        assert [row["movieId"] for row in baseline_rows] == [
            row["movieId"] for row in explained_rows
        ]
        assert [row["rank_position"] for row in baseline_rows] == [
            row["rank_position"] for row in explained_rows
        ]


def test_v1_explain_endpoint_generates_explanations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(
                status_code=200,
                json={
                    "response": (
                        '{"overall_summary": "Explanation endpoint response.",' 
                        '"item_explanations": ['
                        '{"rank_position": 1, "movie_id": 318, "explanation": "Item one."},'
                        '{"rank_position": 2, "movie_id": 296, "explanation": "Item two."}'
                        "]}"
                    )
                },
            )
        return httpx.Response(status_code=404, json={"error": "unexpected path"})

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service

    with TestClient(app) as client:
        client.app.state.ollama_client = _make_mock_ollama_client(httpx.MockTransport(handler))
        response = client.post(
            "/v1/explain",
            json={
                "user_id": 709,
                "top_k": 2,
                "style": "concise",
                "max_explanation_items": 2,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["explanation_status"] == "generated"
        assert payload["overall_explanation"]
        assert len(payload["recommendations"]) == 2


def test_explanations_recommendations_alias_matches_v1_explain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ArtifactRegistry, "load", lambda self: None)
    app = create_app()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(
                status_code=200,
                json={
                    "response": (
                        '{"overall_summary": "Alias endpoint parity.",' 
                        '"item_explanations": ['
                        '{"rank_position": 1, "movie_id": 318, "explanation": "Item one."},'
                        '{"rank_position": 2, "movie_id": 296, "explanation": "Item two."}'
                        "]}"
                    )
                },
            )
        return httpx.Response(status_code=404, json={"error": "unexpected path"})

    stub_service = StubService(response=_sample_result(), history=[])
    app.dependency_overrides[get_recommendation_service] = lambda: stub_service

    payload = {
        "user_id": 709,
        "top_k": 2,
        "style": "concise",
        "max_explanation_items": 2,
    }
    with TestClient(app) as client:
        client.app.state.ollama_client = _make_mock_ollama_client(httpx.MockTransport(handler))
        v1 = client.post("/v1/explain", json=payload)
        alias = client.post("/explanations/recommendations", json=payload)
        assert v1.status_code == 200
        assert alias.status_code == 200
        assert set(v1.json().keys()) == set(alias.json().keys())
