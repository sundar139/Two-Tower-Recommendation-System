from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import httpx
from typer.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def _load_script_module(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    module_name = f"test_{script_name.replace('.', '_')}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load module spec for {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_response(method: str, path: str, status_code: int, payload: dict):
    request = httpx.Request(method, f"http://testserver{path}")
    return httpx.Response(status_code=status_code, json=payload, request=request)


def _make_rows(
    count: int,
    *,
    include_explanations: bool,
    explanation_limit: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(count):
        row: dict[str, object] = {
            "movieId": 1000 + idx,
            "item_idx": idx,
            "title": f"Movie {idx}",
            "genres": "Drama",
            "release_year": 2000 + (idx % 20),
            "final_score": 0.9 - (idx * 0.01),
            "residual_score": 0.3,
            "ranker_score": 0.8,
            "popularity_score": 0.5,
            "rank_position": idx + 1,
            "scorer_policy": "ranker_topk_popularity_backfill",
        }
        if include_explanations:
            row["explanation"] = f"Because signal {idx}" if idx < explanation_limit else ""
        rows.append(row)
    return rows


def _serving_request_stub(
    *,
    captured_limits: list[int],
    timeout_on_explanations: bool = False,
):
    def fake_request(client, *, method, path, params=None, payload=None):  # noqa: ANN001, ARG001
        request_payload = payload or {}

        if method == "GET" and path == "/":
            return _json_response("GET", path, 200, {"status": "ok"}), 1.0
        if method == "GET" and path in {"/health", "/healthz"}:
            return _json_response("GET", path, 200, {"status": "ok"}), 1.0
        if method == "GET" and path in {"/ready", "/readyz"}:
            return _json_response("GET", path, 200, {"ready": True}), 1.0
        if method == "GET" and path == "/metadata":
            metadata = {
                "app_name": "movie-recsys",
                "version": "0.1.0",
                "environment": "dev",
                "production_policy": "strict",
                "candidate_top_k": 100,
                "default_k": 10,
                "max_k": 10,
                "model_artifacts": {},
                "selected_scorer_weights": {},
                "approved_step5d_metrics": {},
                "explanations_enabled": True,
                "explanation_provider": "ollama",
                "chat_model": "qwen3:4b",
                "fail_open": True,
            }
            return _json_response("GET", path, 200, metadata), 1.0

        if method == "GET" and path.startswith("/recommendations/"):
            body = {
                "recommendations": _make_rows(
                    10,
                    include_explanations=False,
                    explanation_limit=0,
                )
            }
            return _json_response("GET", path, 200, body), 1.0
        if method == "GET" and path.startswith("/users/"):
            return _json_response("GET", path, 200, {"history": [{"movieId": 99999}]}), 1.0

        if method == "POST" and path == "/recommendations":
            if int(request_payload.get("k", 10)) > 10:
                return _json_response("POST", path, 422, {"detail": "invalid k"}), 1.0

            allow_cold_start = bool(request_payload.get("allow_cold_start", False))
            include_explanations = bool(request_payload.get("include_explanations", False))
            k_value = int(request_payload.get("k", 10))

            if timeout_on_explanations and include_explanations and not allow_cold_start:
                request = httpx.Request("POST", "http://testserver/recommendations")
                raise httpx.ReadTimeout("timed out", request=request)

            if request_payload.get("user_id") == 2_147_483_000 and not allow_cold_start:
                return _json_response("POST", path, 404, {"detail": "user not found"}), 1.0

            if request_payload.get("user_id") == 2_147_483_000 and allow_cold_start:
                if include_explanations:
                    captured_limits.append(int(request_payload.get("max_explanation_items", -1)))
                    body = {
                        "cold_start": True,
                        "explanation_status": "generated",
                        "recommendations": [
                            {
                                "movieId": 777,
                                "item_idx": 7,
                                "title": "Cold Start",
                                "genres": "Drama",
                                "release_year": 2001,
                                "final_score": 0.75,
                                "popularity_score": 0.75,
                                "rank_position": 1,
                            }
                        ],
                    }
                    return _json_response("POST", path, 200, body), 1.0
                body = {
                    "cold_start": True,
                    "recommendations": [
                        {
                            "movieId": 777,
                            "item_idx": 7,
                            "title": "Cold Start",
                            "genres": "Drama",
                            "release_year": 2001,
                            "final_score": 0.75,
                            "popularity_score": 0.75,
                            "rank_position": 1,
                        }
                    ],
                }
                return _json_response("POST", path, 200, body), 1.0

            if include_explanations:
                explanation_limit = int(request_payload.get("max_explanation_items", 3))
                captured_limits.append(explanation_limit)
                body = {
                    "user_id": 42,
                    "explanation_status": "generated",
                    "recommendations": _make_rows(
                        k_value,
                        include_explanations=True,
                        explanation_limit=explanation_limit,
                    ),
                }
                return _json_response("POST", path, 200, body), 1.0

            body = {
                "user_id": 42,
                "explanation_status": "disabled",
                "recommendations": _make_rows(
                    k_value,
                    include_explanations=False,
                    explanation_limit=0,
                ),
            }
            return _json_response("POST", path, 200, body), 1.0

        if method == "POST" and path == "/v1/recommend":
            k_value = int(request_payload.get("k", 10))
            body = {
                "user_id": 42,
                "explanation_status": "disabled",
                "recommendations": _make_rows(
                    k_value,
                    include_explanations=False,
                    explanation_limit=0,
                ),
            }
            return _json_response("POST", path, 200, body), 1.0

        if method == "POST" and path in {"/v1/explain", "/explanations/recommendations"}:
            explanation_limit = int(request_payload.get("max_explanation_items", 3))
            captured_limits.append(explanation_limit)
            top_k = int(request_payload.get("top_k", 10))
            body = {
                "explanation_status": "generated",
                "recommendations": _make_rows(
                    top_k,
                    include_explanations=True,
                    explanation_limit=explanation_limit,
                ),
            }
            return _json_response("POST", path, 200, body), 1.0

        msg = f"Unhandled request: {method} {path} payload={request_payload} params={params}"
        raise AssertionError(msg)

    return fake_request


def _ollama_request_stub(
    *,
    captured_limits: list[int],
    timeout_on_explanations: bool = False,
):
    def fake_request(client, *, method, path, params=None, payload=None):  # noqa: ANN001, ARG001
        request_payload = payload or {}

        if method == "GET" and path == "/metadata":
            metadata = {
                "explanations_enabled": True,
                "max_k": 10,
                "chat_model": "qwen3:4b",
            }
            return _json_response("GET", path, 200, metadata), 1.0

        if method == "POST" and path == "/recommendations":
            include_explanations = bool(request_payload.get("include_explanations", False))
            k_value = int(request_payload.get("k", 10))

            if timeout_on_explanations and include_explanations:
                request = httpx.Request("POST", "http://testserver/recommendations")
                raise httpx.ReadTimeout("timed out", request=request)

            if include_explanations:
                explanation_limit = int(request_payload.get("max_explanation_items", 3))
                captured_limits.append(explanation_limit)
                body = {
                    "user_id": 42,
                    "explanation_status": "generated",
                    "recommendations": _make_rows(
                        k_value,
                        include_explanations=True,
                        explanation_limit=explanation_limit,
                    ),
                }
                return _json_response("POST", path, 200, body), 1.0

            body = {
                "user_id": 42,
                "explanation_status": "disabled",
                "recommendations": _make_rows(
                    k_value,
                    include_explanations=False,
                    explanation_limit=0,
                ),
            }
            return _json_response("POST", path, 200, body), 1.0

        if method == "POST" and path == "/v1/explain":
            explanation_limit = int(request_payload.get("max_explanation_items", 3))
            captured_limits.append(explanation_limit)
            top_k = int(request_payload.get("top_k", 10))
            body = {
                "explanation_status": "generated",
                "recommendations": _make_rows(
                    top_k,
                    include_explanations=True,
                    explanation_limit=explanation_limit,
                ),
            }
            return _json_response("POST", path, 200, body), 1.0

        msg = f"Unhandled request: {method} {path} payload={request_payload} params={params}"
        raise AssertionError(msg)

    return fake_request


def test_validate_serving_api_custom_timeout_and_max_items(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("validate_serving_api.py")

    captured_timeouts: list[float] = []
    captured_limits: list[int] = []

    class DummyClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            timeout = kwargs.get("timeout")
            captured_timeouts.append(float(timeout))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    monkeypatch.setattr(module.httpx, "Client", DummyClient)
    monkeypatch.setattr(
        module,
        "_request",
        _serving_request_stub(captured_limits=captured_limits),
    )

    report_path = tmp_path / "serving_validation.json"
    result = RUNNER.invoke(
        module.app,
        [
            "--timeout-seconds",
            "77",
            "--max-explanation-items",
            "2",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_timeouts == [77.0]
    assert captured_limits
    assert all(limit == 2 for limit in captured_limits)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["timeout_seconds"] == 77.0
    assert report["max_explanation_items"] == 2


def test_validate_serving_api_timeout_writes_structured_report(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("validate_serving_api.py")

    class DummyClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    monkeypatch.setattr(module.httpx, "Client", DummyClient)
    monkeypatch.setattr(
        module,
        "_request",
        _serving_request_stub(captured_limits=[], timeout_on_explanations=True),
    )

    report_path = tmp_path / "serving_validation_timeout.json"
    result = RUNNER.invoke(module.app, ["--report-path", str(report_path)])

    assert result.exit_code == 1
    assert "Traceback (most recent call last)" not in result.output

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["timeout_error"]["type"] == "timeout"
    assert "explanation validation timeout" in report["timeout_error"]["message"]
    assert "traceback" not in report["timeout_error"]


def test_validate_ollama_explanations_custom_timeout_and_max_items(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module("validate_ollama_explanations.py")

    captured_timeouts: list[float] = []
    captured_limits: list[int] = []

    class DummyClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            timeout = kwargs.get("timeout")
            captured_timeouts.append(float(timeout))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def get(self, url: str):
            return _json_response(
                "GET",
                "/api/tags",
                200,
                {"models": [{"name": "qwen3:4b"}, {"name": "qwen3-embedding:0.6b"}]},
            )

    monkeypatch.setattr(module.httpx, "Client", DummyClient)
    monkeypatch.setattr(
        module,
        "_request",
        _ollama_request_stub(captured_limits=captured_limits),
    )

    report_path = tmp_path / "ollama_validation.json"
    result = RUNNER.invoke(
        module.app,
        [
            "--timeout-seconds",
            "45",
            "--max-explanation-items",
            "4",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_timeouts == [45.0, 45.0]
    assert captured_limits
    assert all(limit == 4 for limit in captured_limits)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["timeout_seconds"] == 45.0
    assert report["max_explanation_items"] == 4


def test_validate_ollama_explanations_debug_timeout_includes_traceback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module("validate_ollama_explanations.py")

    class DummyClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def get(self, url: str):  # noqa: ARG002
            return _json_response(
                "GET",
                "/api/tags",
                200,
                {"models": [{"name": "qwen3:4b"}]},
            )

    monkeypatch.setattr(module.httpx, "Client", DummyClient)
    monkeypatch.setattr(
        module,
        "_request",
        _ollama_request_stub(captured_limits=[], timeout_on_explanations=True),
    )

    report_path = tmp_path / "ollama_validation_timeout.json"
    result = RUNNER.invoke(module.app, ["--debug", "--report-path", str(report_path)])

    assert result.exit_code == 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["timeout_error"]["type"] == "timeout"
    assert "traceback" in report["timeout_error"]