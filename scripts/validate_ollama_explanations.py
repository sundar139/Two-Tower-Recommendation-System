"""Validate Step 7B Ollama explanation flow against a live local server."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import typer

app = typer.Typer(add_completion=False)


def _request(
    client: httpx.Client,
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[httpx.Response, float]:
    start = time.perf_counter()
    response = client.request(method, path, params=params, json=payload)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return response, latency_ms


@app.command()
def main(
    base_url: str = typer.Option("http://127.0.0.1:8000", "--base-url"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", "--ollama-url"),
    known_user_idx: int = typer.Option(0, "--known-user-idx"),
    known_user_id: int | None = typer.Option(None, "--known-user-id"),
    k: int = typer.Option(10, "--k"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
    require_generated: bool = typer.Option(True, "--require-generated/--allow-fail-open"),
    report_path: Path = typer.Option(
        Path("artifacts/reports/ollama_explanation_validation.json"),
        "--report-path",
    ),
) -> None:
    """Run end-to-end explanation checks and write a JSON report."""

    checks: list[dict[str, Any]] = []
    explanation_status: str | None = None
    explanation_count = 0
    explain_latency_ms: float | None = None
    first_recommendation_title: str | None = None
    first_explanation: str | None = None

    def record(*, name: str, passed: bool, latency_ms: float | None, details: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "latency_ms": latency_ms,
                "details": details,
            }
        )

    metadata_body: dict[str, Any] = {}
    allowed_statuses = (
        {"generated"}
        if require_generated
        else {"generated", "unavailable", "failed"}
    )

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds) as api_client:
        metadata_resp, metadata_latency = _request(api_client, method="GET", path="/metadata")
        metadata_ok = metadata_resp.status_code == 200
        if metadata_ok:
            metadata_body = metadata_resp.json()
            metadata_ok = bool(metadata_body.get("explanations_enabled", False))
        record(
            name="GET /metadata explanations enabled",
            passed=metadata_ok,
            latency_ms=metadata_latency,
            details=("ok" if metadata_ok else metadata_resp.text),
        )

        runtime_max_k = int(metadata_body.get("max_k", 200))
        requested_k = max(1, min(k, runtime_max_k))
        max_explanation_items = max(1, min(3, requested_k))

        baseline_payload = {
            "user_idx": known_user_idx,
            "k": requested_k,
            "exclude_seen": True,
            "allow_cold_start": False,
            "include_explanations": False,
        }
        baseline_resp, baseline_latency = _request(
            api_client,
            method="POST",
            path="/recommendations",
            payload=baseline_payload,
        )
        baseline_ok = baseline_resp.status_code == 200
        baseline_body: dict[str, Any] = baseline_resp.json() if baseline_ok else {}
        if baseline_ok:
            baseline_ok = baseline_body.get("explanation_status") == "disabled"
        record(
            name="POST /recommendations include_explanations=false",
            passed=baseline_ok,
            latency_ms=baseline_latency,
            details=("ok" if baseline_ok else baseline_resp.text),
        )

        resolved_user_id = known_user_id
        if resolved_user_id is None and baseline_ok:
            raw_user_id = baseline_body.get("user_id")
            if isinstance(raw_user_id, int):
                resolved_user_id = raw_user_id

        explain_payload = {
            "user_idx": known_user_idx,
            "k": requested_k,
            "exclude_seen": True,
            "allow_cold_start": False,
            "include_explanations": True,
            "explanation_style": "concise",
            "max_explanation_items": max_explanation_items,
        }
        explain_resp, explain_latency = _request(
            api_client,
            method="POST",
            path="/recommendations",
            payload=explain_payload,
        )
        explain_latency_ms = explain_latency
        explain_ok = explain_resp.status_code == 200
        explain_body: dict[str, Any] = explain_resp.json() if explain_ok else {}
        raw_status = explain_body.get("explanation_status")
        explanation_status = raw_status if isinstance(raw_status, str) else None
        if explain_ok:
            explain_ok = explanation_status in allowed_statuses
        record(
            name="POST /recommendations include_explanations=true",
            passed=explain_ok,
            latency_ms=explain_latency,
            details=("ok" if explain_ok else explain_resp.text),
        )

        if explain_ok:
            rows = explain_body.get("recommendations", [])
            if isinstance(rows, list):
                explanation_count = sum(
                    1
                    for row in rows
                    if isinstance(row, dict)
                    and isinstance(row.get("explanation"), str)
                    and bool(row["explanation"])
                )
                if rows and isinstance(rows[0], dict):
                    title = rows[0].get("title")
                    explanation = rows[0].get("explanation")
                    first_recommendation_title = title if isinstance(title, str) else None
                    first_explanation = explanation if isinstance(explanation, str) else None

        explanation_count_ok = True
        if explain_ok and explanation_status == "generated":
            rows = explain_body.get("recommendations", [])
            explained_count = sum(
                1 for row in rows if isinstance(row.get("explanation"), str) and row["explanation"]
            )
            explanation_count_ok = explained_count <= max_explanation_items
        record(
            name="max_explanation_items enforced",
            passed=explanation_count_ok,
            latency_ms=None,
            details=(
                "ok"
                if explanation_count_ok
                else "generated explanation count exceeded max_explanation_items"
            ),
        )

        ordering_ok = False
        if baseline_ok and explain_ok:
            baseline_ids = [
                row["movieId"]
                for row in baseline_body.get("recommendations", [])
                if isinstance(row, dict) and isinstance(row.get("movieId"), int)
            ]
            explain_ids = [
                row["movieId"]
                for row in explain_body.get("recommendations", [])
                if isinstance(row, dict) and isinstance(row.get("movieId"), int)
            ]
            ordering_ok = baseline_ids == explain_ids
        record(
            name="Explanation flow preserves ranking order",
            passed=ordering_ok,
            latency_ms=None,
            details=("ok" if ordering_ok else "movieId ordering changed"),
        )

        v1_explain_payload: dict[str, Any] = {
            "top_k": requested_k,
            "style": "concise",
            "max_explanation_items": max_explanation_items,
        }
        if resolved_user_id is not None:
            v1_explain_payload["user_id"] = resolved_user_id
        else:
            v1_explain_payload["user_idx"] = known_user_idx

        v1_explain_resp, v1_explain_latency = _request(
            api_client,
            method="POST",
            path="/v1/explain",
            payload=v1_explain_payload,
        )
        v1_explain_ok = v1_explain_resp.status_code == 200
        v1_explain_body: dict[str, Any] = v1_explain_resp.json() if v1_explain_ok else {}
        if v1_explain_ok:
            v1_explain_ok = v1_explain_body.get("explanation_status") in allowed_statuses
        record(
            name="POST /v1/explain",
            passed=v1_explain_ok,
            latency_ms=v1_explain_latency,
            details=("ok" if v1_explain_ok else v1_explain_resp.text),
        )

    with httpx.Client(timeout=timeout_seconds) as ollama_client:
        tags_start = time.perf_counter()
        tags_resp = ollama_client.get(f"{ollama_url.rstrip('/')}/api/tags")
        tags_latency = (time.perf_counter() - tags_start) * 1000.0
        tags_ok = tags_resp.status_code == 200
        tags_body: dict[str, Any] = tags_resp.json() if tags_ok else {}
        record(
            name="GET Ollama /api/tags",
            passed=tags_ok,
            latency_ms=tags_latency,
            details=("ok" if tags_ok else tags_resp.text),
        )

        expected_chat_model = str(metadata_body.get("chat_model", "qwen3:4b")).strip()
        chat_model_ok = True
        model_names: list[str] = []
        if tags_ok and expected_chat_model:
            model_names = [
                model.get("name")
                for model in tags_body.get("models", [])
                if isinstance(model, dict) and isinstance(model.get("name"), str)
            ]
            chat_model_ok = expected_chat_model in model_names
        record(
            name="Configured chat model present in Ollama",
            passed=chat_model_ok,
            latency_ms=None,
            details=(
                "ok"
                if chat_model_ok
                else f"chat_model={expected_chat_model!r} not listed in Ollama tags"
            ),
        )

        qwen_chat_model_ok = (not tags_ok) or ("qwen3:4b" in model_names)
        record(
            name="qwen3:4b model present in Ollama",
            passed=qwen_chat_model_ok,
            latency_ms=None,
            details=(
                "ok"
                if qwen_chat_model_ok
                else "qwen3:4b missing from Ollama model list"
            ),
        )

    passed_checks = sum(1 for check in checks if check["passed"])
    failed_checks = len(checks) - passed_checks

    report = {
        "base_url": base_url,
        "ollama_url": ollama_url,
        "requested_k": k,
        "known_user_idx": known_user_idx,
        "known_user_id": known_user_id,
        "require_generated": require_generated,
        "explanation_status": explanation_status,
        "explanation_count": explanation_count,
        "explain_latency_ms": explain_latency_ms,
        "first_recommendation_title": first_recommendation_title,
        "first_explanation": first_explanation,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    typer.echo("Step 7B Ollama Explanation Validation")
    typer.echo("check | result | latency_ms | details")
    for check in checks:
        latency_text = "-"
        if check["latency_ms"] is not None:
            latency_text = f"{check['latency_ms']:.2f}"
        result = "PASS" if check["passed"] else "FAIL"
        typer.echo(f"{check['name']} | {result} | {latency_text} | {check['details']}")

    typer.echo("Explanation summary")
    typer.echo(f"explanation_status={explanation_status or 'unknown'}")
    typer.echo(f"explanations_generated={explanation_count}")
    typer.echo(
        "recommendation_latency_ms="
        + (f"{explain_latency_ms:.2f}" if explain_latency_ms is not None else "-")
    )
    typer.echo(f"first_recommendation_title={first_recommendation_title or ''}")
    typer.echo(f"first_explanation={first_explanation or ''}")

    typer.echo(
        json.dumps(
            {
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if failed_checks > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
