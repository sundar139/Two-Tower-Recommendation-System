from __future__ import annotations

from inspect import signature
from pathlib import Path

import httpx
import yaml

from scripts import run_api
from scripts.check_docker_artifacts import build_preflight_report, print_preflight_report
from scripts.docker_smoke_test import _result, parse_response_payload


def test_dockerfile_exists() -> None:
    assert Path("Dockerfile").exists()


def test_docker_compose_defines_recommender_api() -> None:
    compose_path = Path("docker-compose.yml")
    payload = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    services = payload.get("services")
    assert isinstance(services, dict)

    service = services.get("recommender-api")
    assert isinstance(service, dict)
    assert service.get("container_name") == "movielens-recommender-api"

    ports = service.get("ports")
    assert isinstance(ports, list)
    assert "8000:8000" in ports


def test_dockerignore_excludes_required_paths() -> None:
    ignore_lines = {
        line.strip()
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected = {
        ".git",
        ".venv",
        ".env",
        "mlflow.db",
        "mlruns",
        "artifacts",
        "data/raw",
        "data/interim",
        "data/processed",
        "models",
    }
    assert expected.issubset(ignore_lines)


def test_run_api_accepts_host_port_and_config() -> None:
    params = signature(run_api.main).parameters
    assert "config" in params
    assert "host" in params
    assert "port" in params


def test_preflight_reports_missing_paths_with_temp_fixtures(tmp_path: Path, capsys: object) -> None:
    root = tmp_path
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "faiss").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    (root / "configs" / "serving.yaml").write_text(
        "\n".join(
            [
                "paths:",
                "  retrieval_config: configs/transformer_retrieval_residual.yaml",
                "  ranker_config: configs/ranker.yaml",
                "  faiss_dir: artifacts/faiss",
                "  residual_checkpoint: artifacts/models/best_residual_transformer_retriever.pt",
                "  ranker_checkpoint: artifacts/models/best_neural_ranker.pt",
            ]
        ),
        encoding="utf-8",
    )
    (root / "configs" / "ranker.yaml").write_text("{}\n", encoding="utf-8")
    (root / "configs" / "transformer_retrieval_residual.yaml").write_text(
        "\n".join(
            [
                "paths:",
                "  processed_data_dir: data/processed",
                "files:",
                "  interactions_train: interactions_train.parquet",
                "  users: users.parquet",
                "  items: items.parquet",
                "  user_id_map: user_id_map.parquet",
                "  item_id_map: item_id_map.parquet",
            ]
        ),
        encoding="utf-8",
    )

    (root / "artifacts" / "models" / "best_residual_transformer_retriever.pt").write_bytes(b"x")
    (root / "artifacts" / "models" / "best_neural_ranker.pt").write_bytes(b"x")
    (root / "artifacts" / "faiss" / "index.faiss").write_bytes(b"x")
    (root / "artifacts" / "faiss" / "item_idx_mapping.parquet").write_bytes(b"x")
    (root / "data" / "processed" / "items.parquet").write_bytes(b"x")
    (root / "data" / "processed" / "interactions_train.parquet").write_bytes(b"x")
    (root / "data" / "processed" / "user_id_map.parquet").write_bytes(b"x")
    (root / "data" / "processed" / "item_id_map.parquet").write_bytes(b"x")

    report = build_preflight_report(Path("configs/serving.yaml"), root=root)
    assert report["ok"] is False
    assert "artifacts/faiss/index_metadata.json" in report["missing_paths"]
    assert "data/processed/users.parquet" in report["missing_paths"]

    print_preflight_report(report)
    captured = capsys.readouterr()
    assert "FOUND/MISSING" in captured.out
    assert "missing paths:" in captured.out
    assert "artifacts/faiss/index_metadata.json" in captured.out


def test_smoke_response_parser_handles_success_and_failure() -> None:
    request = httpx.Request("GET", "http://127.0.0.1:8000/health")

    ok_response = httpx.Response(200, json={"status": "ok"}, request=request)
    ok_payload, ok_detail = parse_response_payload(ok_response)
    assert ok_payload == {"status": "ok"}
    assert ok_detail == "ok"

    json_error_response = httpx.Response(400, json={"detail": "invalid k"}, request=request)
    json_error_row = _result(
        name="POST invalid k",
        response=json_error_response,
        latency_ms=1.0,
        passed=False,
    )
    assert json_error_row["details"] == "invalid k"

    text_error_response = httpx.Response(500, text="server exploded", request=request)
    text_payload, text_detail = parse_response_payload(text_error_response)
    assert text_payload is None
    assert text_detail == "server exploded"
