"""Basic API smoke test script for local development."""

from __future__ import annotations

import json

import httpx
import typer

app = typer.Typer(add_completion=False)


def _fail(message: str, *, payload: dict[str, object] | None = None) -> None:
    body = {"ok": False, "message": message}
    if payload is not None:
        body["payload"] = payload
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    raise typer.Exit(code=1)


def _ensure_status(response: httpx.Response, *, expected: int, name: str) -> None:
    if response.status_code != expected:
        _fail(
            f"{name} returned unexpected status",
            payload={
                "expected": expected,
                "actual": response.status_code,
                "body": response.text,
            },
        )


@app.command()
def main(
    base_url: str = typer.Option("http://127.0.0.1:8000", "--base-url"),
    user_idx: int = typer.Option(0, "--user-idx"),
    top_k: int = typer.Option(20, "--top-k"),
    timeout_seconds: float = typer.Option(10.0, "--timeout-seconds"),
    require_ready: bool = typer.Option(True, "--require-ready/--allow-not-ready"),
) -> None:
    """Execute health, readiness, and recommendation smoke checks."""

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds) as client:
        health = client.get("/healthz")
        _ensure_status(health, expected=200, name="healthz")
        health_body = health.json()
        if health_body.get("status") != "ok":
            _fail("healthz payload missing status=ok", payload={"body": health_body})

        ready = client.get("/readyz")
        _ensure_status(ready, expected=200, name="readyz")
        ready_body = ready.json()
        if require_ready and not bool(ready_body.get("ready", False)):
            _fail("readyz reported not ready", payload={"body": ready_body})

        recommend = client.post(
            "/v1/recommend",
            json={"user_idx": user_idx, "top_k": top_k},
        )
        _ensure_status(recommend, expected=200, name="recommend")
        recommend_body = recommend.json()
        recommendations = recommend_body.get("recommendations", [])
        if not isinstance(recommendations, list):
            _fail("recommend payload has invalid recommendations field", payload=recommend_body)
        if len(recommendations) > top_k:
            _fail(
                "recommend returned more rows than requested top_k",
                payload={"top_k": top_k, "returned": len(recommendations)},
            )

    typer.echo(
        json.dumps(
            {
                "ok": True,
                "base_url": base_url,
                "ready": ready_body,
                "returned_rows": len(recommendations),
                "user_idx": user_idx,
                "top_k": top_k,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
