"""Benchmark local serving API latency with sequential recommendation requests."""

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


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(q * (len(sorted_values) - 1))
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


@app.command()
def main(
    base_url: str = typer.Option("http://127.0.0.1:8000", "--base-url"),
    num_users: int = typer.Option(50, "--num-users", min=1),
    k: int = typer.Option(10, "--k", min=1),
    start_user_idx: int = typer.Option(0, "--start-user-idx", min=0),
    timeout_seconds: float = typer.Option(20.0, "--timeout-seconds"),
    report_path: Path = typer.Option(
        Path("artifacts/reports/serving_api_latency.json"),
        "--report-path",
    ),
) -> None:
    """Benchmark sequential recommendation requests for local API latency."""

    preflight: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds) as client:
        for endpoint in ["/health", "/ready", "/metadata"]:
            response, latency_ms = _request(client, method="GET", path=endpoint)
            preflight[endpoint] = {
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "ok": response.status_code == 200,
            }

        for offset in range(num_users):
            user_idx = start_user_idx + offset
            payload = {
                "user_idx": user_idx,
                "k": k,
                "allow_cold_start": False,
                "exclude_seen": True,
                "include_debug": False,
            }
            response, latency_ms = _request(
                client,
                method="POST",
                path="/recommendations",
                payload=payload,
            )
            if response.status_code == 200:
                latencies.append(latency_ms)
            else:
                failures.append(
                    {
                        "user_idx": user_idx,
                        "status_code": response.status_code,
                        "latency_ms": latency_ms,
                        "body": response.text,
                    }
                )

    sorted_latencies = sorted(latencies)
    latency_summary = {
        "p50_ms": _percentile(sorted_latencies, 0.50),
        "p95_ms": _percentile(sorted_latencies, 0.95),
        "max_ms": float(max(sorted_latencies)) if sorted_latencies else 0.0,
        "success_count": len(sorted_latencies),
        "failure_count": len(failures),
    }

    report = {
        "base_url": base_url,
        "num_users": num_users,
        "k": k,
        "start_user_idx": start_user_idx,
        "preflight": preflight,
        "latency": latency_summary,
        "failures": failures,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    typer.echo("Step 6B Serving API Latency Benchmark")
    typer.echo(json.dumps(latency_summary, indent=2, sort_keys=True))
    typer.echo(json.dumps({"report": str(report_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
