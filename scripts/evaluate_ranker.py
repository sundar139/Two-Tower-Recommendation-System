"""Evaluate neural ranker performance by split."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal

import typer

from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.evaluator import evaluate_ranker_split
from movie_recsys.ranking.features import (
	build_ranker_features,
	resolve_feature_columns_from_artifacts,
	resolve_feature_shard_paths,
)

app = typer.Typer(add_completion=False)


def _configure_logging(log_file: Path) -> None:
	log_file.parent.mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, "a", "utf-8")],
		force=True,
	)
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(line_buffering=True)


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	split: Literal["val", "test"] = typer.Option("val", "--split"),
	sample: bool = typer.Option(False, "--sample"),
	checkpoint: Path | None = typer.Option(None, "--checkpoint"),
	max_queries: int | None = typer.Option(None, "--max-queries"),
	log_file: Path = typer.Option(
		Path("artifacts/logs/evaluate_ranker_full.log"),
		"--log-file",
	),
) -> None:
	_configure_logging(log_file)
	ranker_cfg = load_ranker_config(config)
	feature_paths = resolve_feature_shard_paths(ranker_cfg, split=split, sample=sample)
	if not feature_paths:
		build_ranker_features(
			ranker_cfg,
			sample=sample,
			feature_shard_size=ranker_cfg.feature_shard_size,
			resume=True,
			overwrite=False,
			splits=(split,),
		)
		feature_paths = resolve_feature_shard_paths(ranker_cfg, split=split, sample=sample)

	feature_columns = resolve_feature_columns_from_artifacts(
		ranker_cfg,
		sample=sample,
		fallback_split=split,
	)

	eval_result = evaluate_ranker_split(
		feature_paths=feature_paths,
		feature_columns=feature_columns,
		checkpoint_path=checkpoint or ranker_cfg.best_checkpoint,
		split=split,
		report_dir=ranker_cfg.paths.ranker_report_dir,
		batch_size=(ranker_cfg.batch_size if sample else ranker_cfg.full_eval_batch_size),
		max_queries=max_queries,
		log_every_queries=max(ranker_cfg.log_every_batches, 1),
		log_file_path=log_file,
		logger=logging.getLogger("ranker.evaluate"),
	)

	report_payload = {
		"split": eval_result.split,
		"ranker": eval_result.ranker_metrics,
		"residual": eval_result.residual_metrics,
		"popularity": eval_result.popularity_metrics,
		"query_count": eval_result.query_count,
		"row_count": eval_result.row_count,
		"report_path": str(eval_result.report_path),
		"scored_candidates_path": str(eval_result.scored_candidates_path),
		"log_file": str(log_file),
	}
	typer.echo(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
	app()
