"""Train the neural ranker."""

from __future__ import annotations

from pathlib import Path

import typer

from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.trainer import train_ranker

app = typer.Typer(add_completion=False)


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	sample: bool = typer.Option(False, "--sample"),
	run_name: str | None = typer.Option(None, "--run-name"),
) -> None:
	ranker_cfg = load_ranker_config(config)
	result = train_ranker(ranker_cfg, sample=sample, run_name=run_name)
	typer.echo(f"best_checkpoint: {result.best_checkpoint}")
	typer.echo(f"last_checkpoint: {result.last_checkpoint}")
	typer.echo(f"best_val_metrics: {result.best_val_metrics}")
	typer.echo(f"final_train_loss: {result.final_train_loss:.6f}")
	typer.echo(f"mlflow_run_id: {result.mlflow_run_id}")
	typer.echo(f"mlflow_run_url: {result.mlflow_run_url}")
	typer.echo(f"elapsed_seconds: {result.elapsed_seconds:.2f}")


if __name__ == "__main__":
	app()
