"""Run a concise API smoke test for the Dockerized local serving app."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import typer

app = typer.Typer(add_completion=False)


def parse_response_payload(response: httpx.Response) -> tuple[dict[str, Any] | None, str]:
    """Decode JSON payload when available, otherwise return a text detail."""

    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return None, (text or f"HTTP {response.status_code}")

    if isinstance(payload, dict):
        return payload, "ok"

    return None, "Response JSON must be an object"


def _request(
    client: httpx.Client,
    *,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[httpx.Response | None, float | None, str | None]:
    start = time.perf_counter()
    try:
        response = client.request(method, path, json=payload)
    except httpx.HTTPError as exc:
        return None, None, str(exc)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return response, latency_ms, None


def _result(
    *,
    name: str,
    response: httpx.Response | None,
    latency_ms: float | None,
    passed: bool,
    request_error: str | None = None,
) -> dict[str, Any]:
    if request_error is not None:
        detail = request_error
        status_code: int | None = None
    elif response is None:
        detail = "No response"
        status_code = None
    else:
        payload, payload_detail = parse_response_payload(response)
        status_code = response.status_code
        if passed:
            detail = "ok"
        elif payload is not None and "detail" in payload:
            detail = str(payload["detail"])
        else:
            detail = payload_detail

    return {
        "name": name,
        "passed": bool(passed),
        "status_code": status_code,
        "latency_ms": latency_ms,
        "details": detail,
    }


def run_smoke_test(
    *,
    base_url: str,
    known_user_idx: int,
    k: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute endpoint checks and return a machine-friendly report."""

    checks: list[dict[str, Any]] = []

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds) as client:
        health_resp, health_latency, health_error = _request(client, method="GET", path="/health")
        health_ok = (
            health_resp is not None
            and health_resp.status_code == 200
            and isinstance(health_resp.json(), dict)
            and health_resp.json().get("status") == "ok"
        )
        checks.append(
            _result(
                name="GET /health",
                response=health_resp,
                latency_ms=health_latency,
                passed=health_ok,
                request_error=health_error,
            )
        )

        ready_resp, ready_latency, ready_error = _request(client, method="GET", path="/ready")
        ready_ok = (
            ready_resp is not None
            and ready_resp.status_code == 200
            and isinstance(ready_resp.json(), dict)
            and bool(ready_resp.json().get("ready", False))
        )
        checks.append(
            _result(
                name="GET /ready",
                response=ready_resp,
                latency_ms=ready_latency,
                passed=ready_ok,
                request_error=ready_error,
            )
        )

        metadata_resp, metadata_latency, metadata_error = _request(
            client,
            method="GET",
            path="/metadata",
        )
        metadata_ok = (
            metadata_resp is not None
            and metadata_resp.status_code == 200
            and isinstance(metadata_resp.json(), dict)
            and {"production_policy", "selected_scorer_weights", "model_artifacts"}.issubset(
                set(metadata_resp.json().keys())
            )
        )
        checks.append(
            _result(
                name="GET /metadata",
                response=metadata_resp,
                latency_ms=metadata_latency,
                passed=metadata_ok,
                request_error=metadata_error,
            )
        )

        known_resp, known_latency, known_error = _request(
            client,
            method="POST",
            path="/recommendations",
            payload={
                "user_idx": known_user_idx,
                "k": k,
                "exclude_seen": True,
                "allow_cold_start": False,
            },
        )
        known_ok = (
            known_resp is not None
            and known_resp.status_code == 200
            and isinstance(known_resp.json(), dict)
            and isinstance(known_resp.json().get("recommendations"), list)
            and len(known_resp.json().get("recommendations", [])) > 0
        )
        checks.append(
            _result(
                name="POST /recommendations known user",
                response=known_resp,
                latency_ms=known_latency,
                passed=known_ok,
                request_error=known_error,
            )
        )

        unknown_resp, unknown_latency, unknown_error = _request(
            client,
            method="POST",
            path="/recommendations",
            payload={
                "user_id": 2_147_483_000,
                "k": k,
                "allow_cold_start": True,
            },
        )
        unknown_ok = (
            unknown_resp is not None
            and unknown_resp.status_code == 200
            and isinstance(unknown_resp.json(), dict)
            and bool(unknown_resp.json().get("cold_start", False))
        )
        checks.append(
            _result(
                name="POST unknown user allow_cold_start=true",
                response=unknown_resp,
                latency_ms=unknown_latency,
                passed=unknown_ok,
                request_error=unknown_error,
            )
        )

        max_k = 200
        if metadata_resp is not None and metadata_resp.status_code == 200:
            payload = metadata_resp.json()
            if isinstance(payload, dict):
                raw_max_k = payload.get("max_k")
                if isinstance(raw_max_k, int):
                    max_k = raw_max_k

        invalid_resp, invalid_latency, invalid_error = _request(
            client,
            method="POST",
            path="/recommendations",
            payload={"user_idx": known_user_idx, "k": max_k + 1},
        )
        invalid_ok = invalid_resp is not None and invalid_resp.status_code in {400, 422}
        checks.append(
            _result(
                name="POST invalid k",
                response=invalid_resp,
                latency_ms=invalid_latency,
                passed=invalid_ok,
                request_error=invalid_error,
            )
        )

    passed_checks = sum(1 for check in checks if check["passed"])
    report: dict[str, Any] = {
        "base_url": base_url,
        "known_user_idx": known_user_idx,
        "k": k,
        "checks": checks,
        "passed_checks": passed_checks,
        "failed_checks": len(checks) - passed_checks,
        "ok": passed_checks == len(checks),
    }
    return report


def write_report(report: dict[str, Any], report_path: Path) -> None:
    """Persist smoke test report as formatted JSON."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


@app.command()
def main(
    base_url: str = typer.Option("http://127.0.0.1:8000", "--base-url"),
    known_user_idx: int = typer.Option(0, "--known-user-idx"),
    k: int = typer.Option(10, "--k"),
    timeout_seconds: float = typer.Option(20.0, "--timeout-seconds"),
    report_path: Path = typer.Option(
        Path("artifacts/reports/docker_smoke_test.json"),
        "--report-path",
    ),
) -> None:
    """Run local Docker API smoke checks and save a report."""

    report = run_smoke_test(
        base_url=base_url,
        known_user_idx=known_user_idx,
        k=k,
        timeout_seconds=timeout_seconds,
    )
    write_report(report, report_path)

    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        latency_ms = row["latency_ms"]
        latency_text = f"{latency_ms:.2f}ms" if isinstance(latency_ms, float) else "n/a"
        typer.echo(f"{status} {row['name']} ({latency_text})")

    typer.echo(
        f"summary: {report['passed_checks']}/{len(report['checks'])} passed, "
        f"{report['failed_checks']} failed"
    )
    typer.echo(f"ok: {str(bool(report['ok'])).lower()}")
    if not report["ok"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
