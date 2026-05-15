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
    model_type: Literal["baseline", "transformer", "residual_transformer"] | None = typer.Option(
        None,
        "--model-type",
    ),
    init_from_baseline: Path | None = typer.Option(None, "--init-from-baseline"),
    allow_random_init: bool = typer.Option(False, "--allow-random-init"),
    resume_from: Path | None = typer.Option(None, "--resume-from"),
    checkpoint_every_epoch: bool = typer.Option(False, "--checkpoint-every-epoch"),
    max_runtime_hours: float | None = typer.Option(None, "--max-runtime-hours"),
    eval_every_epoch: bool = typer.Option(True, "--eval-every-epoch/--no-eval-every-epoch"),
    save_last: bool = typer.Option(False, "--save-last"),
    run_name: str | None = typer.Option(None, "--run-name"),
) -> None:
    cfg = load_retrieval_config(config, sample=sample)
    result = train_retriever(
        cfg,
        sample=sample,
        model_type=model_type,
        init_from_baseline=init_from_baseline,
        allow_random_init=allow_random_init,
        resume_from=resume_from,
        checkpoint_every_epoch=checkpoint_every_epoch,
        max_runtime_hours=max_runtime_hours,
        eval_every_epoch=eval_every_epoch,
        save_last=save_last,
        run_name=run_name,
    )
    typer.echo("Training complete")
    typer.echo(f"model_type: {result.model_type}")
    typer.echo(f"best_checkpoint: {result.best_checkpoint}")
    typer.echo(f"best_metrics: {result.best_metrics}")
    typer.echo(f"best_epoch: {result.best_epoch}")
    typer.echo(f"start_epoch: {result.start_epoch}")
    typer.echo(f"completed_epochs: {result.completed_epochs}")
    typer.echo(f"elapsed_seconds: {result.elapsed_seconds:.2f}")
    typer.echo(f"stopped_due_to_runtime: {result.stopped_due_to_runtime}")
    typer.echo(f"last_checkpoint: {result.last_checkpoint}")
    typer.echo(f"resumed_from: {result.resumed_from}")
    typer.echo(f"final_train_loss: {result.final_train_loss:.6f}")
    typer.echo(f"mlflow_run_id: {result.mlflow_run_id}")
    typer.echo(f"mlflow_run_url: {result.mlflow_run_url}")


if __name__ == "__main__":
    app()
