"""Evaluate neural ranker performance by split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer

from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.evaluator import evaluate_ranker_split
from movie_recsys.ranking.features import build_ranker_features

app = typer.Typer(add_completion=False)


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	split: Literal["val", "test"] = typer.Option("val", "--split"),
	sample: bool = typer.Option(False, "--sample"),
	checkpoint: Path | None = typer.Option(None, "--checkpoint"),
) -> None:
	ranker_cfg = load_ranker_config(config)
	feature_path = ranker_cfg.features_path(split=split, sample=sample)
	if not feature_path.exists():
		build_ranker_features(ranker_cfg, sample=sample)

	eval_result = evaluate_ranker_split(
		feature_path=feature_path,
		checkpoint_path=checkpoint or ranker_cfg.best_checkpoint,
		split=split,
		report_dir=ranker_cfg.paths.ranker_report_dir,
		batch_size=ranker_cfg.batch_size,
	)

	report_payload = {
		"split": eval_result.split,
		"ranker": eval_result.ranker_metrics,
		"residual": eval_result.residual_metrics,
		"popularity": eval_result.popularity_metrics,
		"report_path": str(eval_result.report_path),
		"scored_candidates_path": str(eval_result.scored_candidates_path),
	}
	typer.echo(json.dumps(report_payload, indent=2))


if __name__ == "__main__":
	app()
