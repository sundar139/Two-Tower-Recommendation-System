"""Train the neural ranker."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.trainer import train_ranker

app = typer.Typer(add_completion=False)


def _configure_logging(log_file: Path) -> None:
	log_file.parent.mkdir(parents=True, exist_ok=True)
	stream_handler = logging.StreamHandler(sys.stdout)
	file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		handlers=[stream_handler, file_handler],
		force=True,
	)
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(line_buffering=True)


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	sample: bool = typer.Option(False, "--sample"),
	run_name: str | None = typer.Option(None, "--run-name"),
	resume_from: Path | None = typer.Option(None, "--resume-from"),
	checkpoint_every_epoch: bool = typer.Option(False, "--checkpoint-every-epoch"),
	max_runtime_hours: float | None = typer.Option(None, "--max-runtime-hours"),
	eval_every_epoch: bool = typer.Option(False, "--eval-every-epoch"),
	save_last: bool = typer.Option(True, "--save-last/--no-save-last"),
	rebuild_features: bool = typer.Option(False, "--rebuild-features"),
	full_smoke: bool = typer.Option(False, "--full-smoke"),
	max_train_rows: int | None = typer.Option(None, "--max-train-rows"),
	max_val_queries: int | None = typer.Option(None, "--max-val-queries"),
	log_file: Path = typer.Option(
		Path("artifacts/logs/train_ranker_full.log"),
		"--log-file",
	),
) -> None:
	_configure_logging(log_file)
	ranker_cfg = load_ranker_config(config)
	result = train_ranker(
		ranker_cfg,
		sample=sample,
		run_name=run_name,
		resume_from=resume_from,
		checkpoint_every_epoch=checkpoint_every_epoch,
		max_runtime_hours=max_runtime_hours,
		eval_every_epoch=eval_every_epoch,
		save_last=save_last,
		rebuild_features=rebuild_features,
		full_smoke=full_smoke,
		max_train_rows=max_train_rows,
		max_val_queries=max_val_queries,
		log_file_path=log_file,
	)
	typer.echo(f"log_file: {log_file}")
	typer.echo(f"best_checkpoint: {result.best_checkpoint}")
	typer.echo(f"last_checkpoint: {result.last_checkpoint}")
	typer.echo(f"last_epoch_checkpoint: {result.last_epoch_checkpoint}")
	typer.echo(f"best_val_metrics: {result.best_val_metrics}")
	typer.echo(f"best_epoch: {result.best_epoch}")
	typer.echo(f"start_epoch: {result.start_epoch}")
	typer.echo(f"completed_epochs: {result.completed_epochs}")
	typer.echo(f"feature_count: {result.feature_count}")
	typer.echo(f"train_query_count: {result.train_query_count}")
	typer.echo(f"val_query_count: {result.val_query_count}")
	typer.echo(f"train_row_count: {result.train_row_count}")
	typer.echo(f"val_row_count: {result.val_row_count}")
	typer.echo(f"resumed_from: {result.resumed_from}")
	typer.echo(f"stopped_due_to_runtime: {result.stopped_due_to_runtime}")
	typer.echo(f"stopped_due_to_memory: {result.stopped_due_to_memory}")
	typer.echo(f"memory_status: {result.memory_status}")
	typer.echo(f"full_smoke: {result.full_smoke}")
	typer.echo(f"final_train_loss: {result.final_train_loss:.6f}")
	typer.echo(f"mlflow_run_id: {result.mlflow_run_id}")
	typer.echo(f"mlflow_run_url: {result.mlflow_run_url}")
	typer.echo(f"elapsed_seconds: {result.elapsed_seconds:.2f}")


if __name__ == "__main__":
	app()
