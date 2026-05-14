"""Train plain two-tower retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from movie_recsys.modeling.trainer import train_retriever
from movie_recsys.training.config import load_retrieval_config

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config"),
    sample: bool = typer.Option(False, "--sample"),
    model_type: Literal["baseline", "transformer"] | None = typer.Option(
        None,
        "--model-type",
    ),
) -> None:
    cfg = load_retrieval_config(config, sample=sample)
    result = train_retriever(cfg, sample=sample, model_type=model_type)
    typer.echo("Training complete")
    typer.echo(f"model_type: {result.model_type}")
    typer.echo(f"best_checkpoint: {result.best_checkpoint}")
    typer.echo(f"best_metrics: {result.best_metrics}")
    typer.echo(f"final_train_loss: {result.final_train_loss:.6f}")
    typer.echo(f"mlflow_run_id: {result.mlflow_run_id}")
    typer.echo(f"mlflow_run_url: {result.mlflow_run_url}")


if __name__ == "__main__":
    app()
