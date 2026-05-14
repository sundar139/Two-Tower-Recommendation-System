"""Print or launch local MLflow UI configured for SQLite backend metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from movie_recsys.training.config import load_retrieval_config

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config"),
    run: bool = typer.Option(False, "--run"),
) -> None:
    cfg = load_retrieval_config(config)
    command = [
        "uvx",
        "mlflow",
        "ui",
        "--backend-store-uri",
        cfg.mlflow_tracking_uri,
        "--host",
        cfg.mlflow_ui_host,
        "--port",
        str(cfg.mlflow_ui_port),
    ]
    command_str = " ".join(command)
    typer.echo(command_str)
    typer.echo(f"mlflow_ui_url: {cfg.mlflow_ui_url}")

    if run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    app()
