"""Run local FastAPI serving app for recommendation inference."""

from __future__ import annotations

import os
from pathlib import Path

import typer
import uvicorn

from movie_recsys.serving.config import load_serving_config

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/serving.yaml"), "--config"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    log_level: str | None = typer.Option(None, "--log-level"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start FastAPI app using serving config with optional CLI overrides."""

    serving_config = load_serving_config(config)
    resolved_host = host or serving_config.api.host
    resolved_port = port or serving_config.api.port
    resolved_log_level = log_level or serving_config.api.log_level

    os.environ["MOVIE_RECSYS_SERVING_CONFIG"] = str(config.resolve())
    uvicorn.run(
        "movie_recsys.serving.app:create_app",
        factory=True,
        host=resolved_host,
        port=resolved_port,
        log_level=resolved_log_level,
        reload=reload,
    )


if __name__ == "__main__":
    app()
