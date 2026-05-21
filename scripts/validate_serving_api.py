"""Validate Step 7B serving API contract against a live local server."""

from __future__ import annotations

import json
import math
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


def _is_finite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


@app.command()
def main(
    base_url: str = typer.Option("http://127.0.0.1:8000", "--base-url"),
    known_user_idx: int = typer.Option(0, "--known-user-idx"),
    known_user_id: int | None = typer.Option(None, "--known-user-id"),
    k: int = typer.Option(10, "--k"),
    timeout_seconds: float = typer.Option(20.0, "--timeout-seconds"),
    report_path: Path = typer.Option(
        Path("artifacts/reports/serving_api_validation.json"),
        "--report-path",
    ),
) -> None:
    """Run end-to-end serving API checks and write a JSON report."""

    checks: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {}

    def record(
        *,
        name: str,
        passed: bool,
        latency_ms: float | None,
        details: str,
    ) -> None:
        if latency_ms is not None:
            latencies.setdefault(name, []).append(latency_ms)
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "latency_ms": latency_ms,
                "details": details,
            }
        )

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds) as client:
        root_resp, root_latency = _request(client, method="GET", path="/")
        root_ok = root_resp.status_code == 200
        record(
            name="GET /",
            passed=root_ok,
            latency_ms=root_latency,
            details=("ok" if root_ok else root_resp.text),
        )

        health_resp, health_latency = _request(client, method="GET", path="/health")
        health_ok = health_resp.status_code == 200 and health_resp.json().get("status") == "ok"
        record(
            name="GET /health",
            passed=health_ok,
            latency_ms=health_latency,
            details=("ok" if health_ok else health_resp.text),
        )

        healthz_resp, healthz_latency = _request(client, method="GET", path="/healthz")
        healthz_ok = healthz_resp.status_code == 200 and healthz_resp.json().get("status") == "ok"
        record(
            name="GET /healthz",
            passed=healthz_ok,
            latency_ms=healthz_latency,
            details=("ok" if healthz_ok else healthz_resp.text),
        )

        ready_resp, ready_latency = _request(client, method="GET", path="/ready")
        ready_ok = ready_resp.status_code == 200 and bool(ready_resp.json().get("ready", False))
        record(
            name="GET /ready",
            passed=ready_ok,
            latency_ms=ready_latency,
            details=("ok" if ready_ok else ready_resp.text),
        )

        readyz_resp, readyz_latency = _request(client, method="GET", path="/readyz")
        readyz_ok = readyz_resp.status_code == 200 and bool(readyz_resp.json().get("ready", False))
        record(
            name="GET /readyz",
            passed=readyz_ok,
            latency_ms=readyz_latency,
            details=("ok" if readyz_ok else readyz_resp.text),
        )

        metadata_resp, metadata_latency = _request(client, method="GET", path="/metadata")
        metadata_body: dict[str, Any] = {}
        metadata_ok = metadata_resp.status_code == 200
        if metadata_ok:
            metadata_body = metadata_resp.json()
            required_metadata = {
                "app_name",
                "version",
                "environment",
                "production_policy",
                "candidate_top_k",
                "default_k",
                "max_k",
                "model_artifacts",
                "selected_scorer_weights",
                "approved_step5d_metrics",
                "explanations_enabled",
                "explanation_provider",
                "chat_model",
                "fail_open",
            }
            metadata_ok = required_metadata.issubset(set(metadata_body.keys()))
        record(
            name="GET /metadata",
            passed=metadata_ok,
            latency_ms=metadata_latency,
            details=("ok" if metadata_ok else metadata_resp.text),
        )

        runtime_max_k = int(metadata_body.get("max_k", 200))
        requested_k = max(1, min(k, runtime_max_k))

        known_payload = {
            "user_idx": known_user_idx,
            "k": requested_k,
            "exclude_seen": True,
            "allow_cold_start": False,
            "include_explanations": False,
        }
        rec_resp, rec_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload=known_payload,
        )
        rec_body: dict[str, Any] = rec_resp.json() if rec_resp.status_code == 200 else {}
        rec_ok = rec_resp.status_code == 200 and isinstance(rec_body.get("recommendations"), list)
        record(
            name="POST /recommendations known user",
            passed=rec_ok,
            latency_ms=rec_latency,
            details=("ok" if rec_ok else rec_resp.text),
        )

        include_false_ok = rec_ok and rec_body.get("explanation_status") == "disabled"
        record(
            name="include_explanations=false remains disabled",
            passed=include_false_ok,
            latency_ms=None,
            details=(
                "ok"
                if include_false_ok
                else "response missing explanation_status=disabled for include_explanations=false"
            ),
        )

        v1_resp, v1_latency = _request(
            client,
            method="POST",
            path="/v1/recommend",
            payload=known_payload,
        )
        v1_body: dict[str, Any] = v1_resp.json() if v1_resp.status_code == 200 else {}
        v1_ok = v1_resp.status_code == 200 and isinstance(v1_body.get("recommendations"), list)
        record(
            name="POST /v1/recommend known user",
            passed=v1_ok,
            latency_ms=v1_latency,
            details=("ok" if v1_ok else v1_resp.text),
        )

        schema_match = False
        if rec_ok and v1_ok:
            schema_match = set(rec_body.keys()) == set(v1_body.keys())
            rec_rows = rec_body.get("recommendations", [])
            v1_rows = v1_body.get("recommendations", [])
            if isinstance(rec_rows, list) and isinstance(v1_rows, list) and rec_rows and v1_rows:
                schema_match = schema_match and set(rec_rows[0].keys()) == set(v1_rows[0].keys())
        record(
            name="Schema parity /recommendations vs /v1/recommend",
            passed=schema_match,
            latency_ms=None,
            details=("ok" if schema_match else "response schema mismatch"),
        )

        explanation_limit = max(1, min(3, requested_k))
        explain_payload = {
            "user_idx": known_user_idx,
            "k": requested_k,
            "exclude_seen": True,
            "allow_cold_start": False,
            "include_explanations": True,
            "explanation_style": "concise",
            "max_explanation_items": explanation_limit,
        }
        explain_resp, explain_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload=explain_payload,
        )
        explain_body: dict[str, Any] = (
            explain_resp.json() if explain_resp.status_code == 200 else {}
        )

        explanations_enabled = bool(metadata_body.get("explanations_enabled", False))
        allowed_statuses = {"generated", "unavailable", "failed"}
        if not explanations_enabled:
            allowed_statuses = {"disabled"}

        explain_status = explain_body.get("explanation_status")
        explain_ok = (
            explain_resp.status_code == 200
            and isinstance(explain_body.get("recommendations"), list)
            and explain_status in allowed_statuses
        )
        record(
            name="POST /recommendations include_explanations=true",
            passed=explain_ok,
            latency_ms=explain_latency,
            details=("ok" if explain_ok else explain_resp.text),
        )

        explain_count_ok = True
        if explain_ok and explain_status == "generated":
            explained_rows = explain_body.get("recommendations", [])
            explained_count = sum(
                1
                for row in explained_rows
                if isinstance(row.get("explanation"), str) and row["explanation"]
            )
            explain_count_ok = explained_count <= explanation_limit
        record(
            name="Explanation count respects max_explanation_items",
            passed=explain_count_ok,
            latency_ms=None,
            details=(
                "ok"
                if explain_count_ok
                else "generated explanation count exceeded max_explanation_items"
            ),
        )

        explanation_order_ok = False
        if rec_ok and explain_ok:
            baseline_ids = [
                row["movieId"]
                for row in rec_body.get("recommendations", [])
                if isinstance(row, dict) and isinstance(row.get("movieId"), int)
            ]
            explained_ids = [
                row["movieId"]
                for row in explain_body.get("recommendations", [])
                if isinstance(row, dict) and isinstance(row.get("movieId"), int)
            ]
            explanation_order_ok = baseline_ids == explained_ids
        record(
            name="Explanations do not reorder recommendations",
            passed=explanation_order_ok,
            latency_ms=None,
            details=("ok" if explanation_order_ok else "ranking order changed with explanations"),
        )

        resolved_user_id = known_user_id
        if resolved_user_id is None and rec_ok:
            raw_user_id = rec_body.get("user_id")
            if isinstance(raw_user_id, int):
                resolved_user_id = raw_user_id

        get_rec_ok = False
        history_ok = False
        history_body: dict[str, Any] = {}
        if resolved_user_id is not None:
            get_rec_resp, get_rec_latency = _request(
                client,
                method="GET",
                path=f"/recommendations/{resolved_user_id}",
                params={"k": requested_k, "allow_cold_start": False},
            )
            get_rec_ok = get_rec_resp.status_code == 200
            record(
                name="GET /recommendations/{user_id}",
                passed=get_rec_ok,
                latency_ms=get_rec_latency,
                details=("ok" if get_rec_ok else get_rec_resp.text),
            )

            history_resp, history_latency = _request(
                client,
                method="GET",
                path=f"/users/{resolved_user_id}/history",
            )
            if history_resp.status_code == 200:
                history_body = history_resp.json()
                history_rows = history_body.get("history", [])
                history_ok = isinstance(history_rows, list) and len(history_rows) <= 100
            record(
                name="GET /users/{user_id}/history",
                passed=history_ok,
                latency_ms=history_latency,
                details=("ok" if history_ok else history_resp.text),
            )
        else:
            record(
                name="GET /recommendations/{user_id}",
                passed=False,
                latency_ms=None,
                details="known user_id unavailable from response; pass --known-user-id",
            )
            record(
                name="GET /users/{user_id}/history",
                passed=False,
                latency_ms=None,
                details="known user_id unavailable from response; pass --known-user-id",
            )

        unknown_user_id = 2_147_483_000
        unknown_true_resp, unknown_true_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload={
                "user_id": unknown_user_id,
                "k": requested_k,
                "allow_cold_start": True,
            },
        )
        unknown_true_ok = False
        if unknown_true_resp.status_code == 200:
            unknown_true_body = unknown_true_resp.json()
            unknown_rows = unknown_true_body.get("recommendations", [])
            required_item_keys = {
                "movieId",
                "item_idx",
                "title",
                "genres",
                "release_year",
                "final_score",
                "popularity_score",
                "rank_position",
            }
            unknown_true_ok = bool(unknown_true_body.get("cold_start", False)) and isinstance(
                unknown_rows,
                list,
            )
            if unknown_rows:
                unknown_true_ok = unknown_true_ok and required_item_keys.issubset(
                    set(unknown_rows[0].keys())
                )
        record(
            name="Unknown user allow_cold_start=true",
            passed=unknown_true_ok,
            latency_ms=unknown_true_latency,
            details=("ok" if unknown_true_ok else unknown_true_resp.text),
        )

        unknown_explain_resp, unknown_explain_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload={
                "user_id": unknown_user_id,
                "k": requested_k,
                "allow_cold_start": True,
                "include_explanations": True,
                "explanation_style": "concise",
                "max_explanation_items": explanation_limit,
            },
        )
        unknown_explain_ok = False
        if unknown_explain_resp.status_code == 200:
            unknown_explain_body = unknown_explain_resp.json()
            unknown_explain_ok = bool(unknown_explain_body.get("cold_start", False))
            unknown_explain_ok = unknown_explain_ok and (
                unknown_explain_body.get("explanation_status") in allowed_statuses
            )
        record(
            name="Unknown user with explanations fail-open",
            passed=unknown_explain_ok,
            latency_ms=unknown_explain_latency,
            details=("ok" if unknown_explain_ok else unknown_explain_resp.text),
        )

        unknown_false_resp, unknown_false_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload={
                "user_id": unknown_user_id,
                "k": requested_k,
                "allow_cold_start": False,
            },
        )
        unknown_false_ok = unknown_false_resp.status_code == 404
        record(
            name="Unknown user allow_cold_start=false",
            passed=unknown_false_ok,
            latency_ms=unknown_false_latency,
            details=("ok" if unknown_false_ok else unknown_false_resp.text),
        )

        invalid_k_resp, invalid_k_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload={"user_idx": known_user_idx, "k": runtime_max_k + 1},
        )
        invalid_k_ok = invalid_k_resp.status_code in {400, 422}
        record(
            name="Invalid k rejected",
            passed=invalid_k_ok,
            latency_ms=invalid_k_latency,
            details=("ok" if invalid_k_ok else invalid_k_resp.text),
        )

        repeat_a_resp, repeat_a_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload=known_payload,
        )
        repeat_b_resp, repeat_b_latency = _request(
            client,
            method="POST",
            path="/recommendations",
            payload=known_payload,
        )
        deterministic_ok = False
        if repeat_a_resp.status_code == 200 and repeat_b_resp.status_code == 200:
            deterministic_ok = (
                repeat_a_resp.json().get("recommendations", [])
                == repeat_b_resp.json().get("recommendations", [])
            )
        record(
            name="Deterministic repeated recommendations",
            passed=deterministic_ok,
            latency_ms=(repeat_a_latency + repeat_b_latency) / 2.0,
            details=("ok" if deterministic_ok else "responses differ"),
        )

        exclude_seen_ok = False
        if rec_ok and history_ok:
            rec_movie_ids = {
                int(row["movieId"])
                for row in rec_body.get("recommendations", [])
                if isinstance(row.get("movieId"), int)
            }
            history_movie_ids = {
                int(row["movieId"])
                for row in history_body.get("history", [])
                if isinstance(row.get("movieId"), int)
            }
            exclude_seen_ok = len(rec_movie_ids.intersection(history_movie_ids)) == 0
        record(
            name="exclude_seen filters seen items",
            passed=exclude_seen_ok,
            latency_ms=None,
            details=("ok" if exclude_seen_ok else "seen items present in recommendations"),
        )

        no_nan_inf_ok = True
        for payload in [rec_body, v1_body]:
            rows = payload.get("recommendations", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                no_nan_inf_ok = False
                break
            for row in rows:
                for key in ["final_score", "residual_score", "ranker_score", "popularity_score"]:
                    if not _is_finite(row.get(key)):
                        no_nan_inf_ok = False
                        break
                if not no_nan_inf_ok:
                    break
            if not no_nan_inf_ok:
                break
        record(
            name="No NaN/inf recommendation scores",
            passed=no_nan_inf_ok,
            latency_ms=None,
            details=("ok" if no_nan_inf_ok else "non-finite score detected"),
        )

        count_ok = False
        if rec_ok:
            count_ok = len(rec_body.get("recommendations", [])) == requested_k
            if not count_ok:
                debug_payload = rec_body.get("debug") if isinstance(rec_body, dict) else None
                count_ok = bool(
                    isinstance(debug_payload, dict)
                    and debug_payload.get("returned_k") is not None
                )
        record(
            name="Recommendation count meets requested k",
            passed=count_ok,
            latency_ms=None,
            details=("ok" if count_ok else "returned rows differ from requested k"),
        )

        explain_request_payload: dict[str, Any] = {
            "top_k": requested_k,
            "style": "concise",
            "max_explanation_items": explanation_limit,
        }
        if resolved_user_id is not None:
            explain_request_payload["user_id"] = resolved_user_id
        else:
            explain_request_payload["user_idx"] = known_user_idx

        v1_explain_resp, v1_explain_latency = _request(
            client,
            method="POST",
            path="/v1/explain",
            payload=explain_request_payload,
        )
        v1_explain_body: dict[str, Any] = (
            v1_explain_resp.json() if v1_explain_resp.status_code == 200 else {}
        )
        v1_explain_ok = (
            v1_explain_resp.status_code == 200
            and isinstance(v1_explain_body.get("recommendations"), list)
            and v1_explain_body.get("explanation_status") in allowed_statuses
        )
        record(
            name="POST /v1/explain",
            passed=v1_explain_ok,
            latency_ms=v1_explain_latency,
            details=("ok" if v1_explain_ok else v1_explain_resp.text),
        )

        explain_alias_resp, explain_alias_latency = _request(
            client,
            method="POST",
            path="/explanations/recommendations",
            payload=explain_request_payload,
        )
        explain_alias_ok = explain_alias_resp.status_code == 200
        record(
            name="POST /explanations/recommendations",
            passed=explain_alias_ok,
            latency_ms=explain_alias_latency,
            details=("ok" if explain_alias_ok else explain_alias_resp.text),
        )

        explain_alias_schema_ok = False
        if v1_explain_ok and explain_alias_ok:
            explain_alias_body = explain_alias_resp.json()
            explain_alias_schema_ok = set(v1_explain_body.keys()) == set(explain_alias_body.keys())
            v1_rows = v1_explain_body.get("recommendations", [])
            alias_rows = explain_alias_body.get("recommendations", [])
            if (
                isinstance(v1_rows, list)
                and isinstance(alias_rows, list)
                and v1_rows
                and alias_rows
            ):
                explain_alias_schema_ok = explain_alias_schema_ok and set(v1_rows[0].keys()) == set(
                    alias_rows[0].keys()
                )
        record(
            name="Schema parity /v1/explain vs /explanations/recommendations",
            passed=explain_alias_schema_ok,
            latency_ms=None,
            details=("ok" if explain_alias_schema_ok else "explain endpoint schema mismatch"),
        )

    passed_checks = sum(1 for check in checks if check["passed"])
    failed_checks = len(checks) - passed_checks

    latency_summary: dict[str, dict[str, float]] = {}
    for name, values in latencies.items():
        if not values:
            continue
        ordered = sorted(values)
        p50_index = min(len(ordered) - 1, int(0.5 * (len(ordered) - 1)))
        p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
        latency_summary[name] = {
            "count": float(len(ordered)),
            "p50_ms": float(ordered[p50_index]),
            "p95_ms": float(ordered[p95_index]),
            "max_ms": float(max(ordered)),
        }

    report = {
        "base_url": base_url,
        "requested_k": requested_k,
        "known_user_idx": known_user_idx,
        "known_user_id": known_user_id,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "latency_summary": latency_summary,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    typer.echo("Step 7B Serving API Validation")
    typer.echo("check | result | latency_ms | details")
    for check in checks:
        latency_text = "-"
        if check["latency_ms"] is not None:
            latency_text = f"{check['latency_ms']:.2f}"
        result = "PASS" if check["passed"] else "FAIL"
        typer.echo(f"{check['name']} | {result} | {latency_text} | {check['details']}")

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
