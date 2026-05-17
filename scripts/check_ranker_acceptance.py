"""Check neural ranker acceptance criteria and guard checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from movie_recsys.modeling.artifacts import save_json
from movie_recsys.ranking.acceptance import (
	evaluate_acceptance,
	guard_candidate_deterministic,
	guard_finite_scores,
	guard_mlflow_logged,
	guard_no_candidate_leakage,
	guard_no_duplicate_candidates,
	guard_one_positive_per_query,
	load_json,
)
from movie_recsys.ranking.config import load_ranker_config

app = typer.Typer(add_completion=False)


def _metrics(payload: dict[str, Any], key: str) -> dict[str, float]:
	value = payload.get(key)
	if not isinstance(value, dict):
		msg = f"Missing metric section '{key}'"
		raise ValueError(msg)
	required = ["hr@10", "mrr@10", "ndcg@10", "recall@50"]
	if any(metric not in value for metric in required):
		msg = f"Metric section '{key}' is missing required ranking metrics"
		raise ValueError(msg)
	return {name: float(value[name]) for name in required}


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	sample: bool = typer.Option(False, "--sample"),
	val_report: Path | None = typer.Option(None, "--val-report"),
	test_report: Path | None = typer.Option(None, "--test-report"),
) -> None:
	ranker_cfg = load_ranker_config(config)
	report_dir = ranker_cfg.paths.ranker_report_dir

	val_payload = load_json(val_report or (report_dir / "ranker_eval_val.json"))
	test_payload = load_json(test_report or (report_dir / "ranker_eval_test.json"))

	ranker_val = _metrics(val_payload, "ranker")
	ranker_test = _metrics(test_payload, "ranker")
	residual_val = _metrics(val_payload, "residual")
	residual_test = _metrics(test_payload, "residual")

	train_candidates = ranker_cfg.candidate_path(split="train", sample=sample)
	val_candidates = ranker_cfg.candidate_path(split="val", sample=sample)
	test_candidates = ranker_cfg.candidate_path(split="test", sample=sample)

	scored_val = report_dir / "ranker_eval_val_scored_candidates.parquet"
	scored_test = report_dir / "ranker_eval_test_scored_candidates.parquet"

	guards = {
		"no_candidate_leakage": guard_no_candidate_leakage(
			train_candidates=train_candidates,
			val_candidates=val_candidates,
			test_candidates=test_candidates,
		),
		"exactly_one_positive_per_query": all(
			[
				guard_one_positive_per_query(train_candidates),
				guard_one_positive_per_query(val_candidates),
				guard_one_positive_per_query(test_candidates),
			]
		),
		"no_duplicate_candidates_per_query": all(
			[
				guard_no_duplicate_candidates(train_candidates),
				guard_no_duplicate_candidates(val_candidates),
				guard_no_duplicate_candidates(test_candidates),
			]
		),
		"no_nan_or_inf_scores": (
			guard_finite_scores(scored_val) and guard_finite_scores(scored_test)
		),
		"candidate_generation_deterministic": all(
			[
				guard_candidate_deterministic(train_candidates.with_suffix(".meta.json")),
				guard_candidate_deterministic(val_candidates.with_suffix(".meta.json")),
				guard_candidate_deterministic(test_candidates.with_suffix(".meta.json")),
			]
		),
		"mlflow_run_logged": guard_mlflow_logged(ranker_cfg.best_checkpoint),
	}

	result = evaluate_acceptance(
		ranker_val=ranker_val,
		ranker_test=ranker_test,
		residual_val=residual_val,
		residual_test=residual_test,
		guards=guards,
	)

	output_path = report_dir / (
		"ranker_acceptance_sample.json" if sample else "ranker_acceptance_full.json"
	)
	save_json(
		output_path,
		{
			"sample_mode": sample,
			"val_report": str(val_report or (report_dir / "ranker_eval_val.json")),
			"test_report": str(test_report or (report_dir / "ranker_eval_test.json")),
			"result": result,
		},
	)

	typer.echo(f"acceptance_path: {output_path}")
	typer.echo(result)


if __name__ == "__main__":
	app()
