"""Print or launch local MLflow UI configured for SQLite backend metadata."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config"),
    run: bool = typer.Option(False, "--run"),
) -> None:
    tracking_uri = "sqlite:///mlflow.db"
    ui_host = "127.0.0.1"
    ui_port = 5000
    ui_url = "http://127.0.0.1:5000"

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        tracking_uri,
        "--host",
        ui_host,
        "--port",
        str(ui_port),
    ]
    command_str = " ".join(command)
    typer.echo(f"mlflow_tracking_uri: {tracking_uri}")
    typer.echo(f"mlflow_ui_url: {ui_url}")
    typer.echo(f"command: {command_str}")
    typer.echo(
        "windows_note: If uvx mlflow ui shows WinError 10022/worker noise, "
        "use 'uv run python scripts/start_mlflow_ui.py --run'."
    )

    if run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    app()
