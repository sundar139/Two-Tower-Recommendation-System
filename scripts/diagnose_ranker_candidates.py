"""Diagnose ranker candidate quality, leakage risks, and suspicious label correlations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer

from movie_recsys.modeling.artifacts import save_json
from movie_recsys.ranking.candidates import REQUIRED_CANDIDATE_COLUMNS
from movie_recsys.ranking.config import RankerConfig, load_ranker_config
from movie_recsys.ranking.features import METADATA_ONLY_COLUMNS

app = typer.Typer(add_completion=False)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
	if x.size == 0 or y.size == 0:
		return 0.0
	if np.std(x) == 0.0 or np.std(y) == 0.0:
		return 0.0
	return float(np.corrcoef(x, y)[0, 1])


def _to_float(value: Any, *, default: float = 0.0) -> float:
	if value is None:
		return default
	if isinstance(value, (int, float, np.floating, np.integer)):
		return float(value)
	return default


def _to_int(value: Any, *, default: int = 0) -> int:
	if value is None:
		return default
	if isinstance(value, (int, np.integer)):
		return int(value)
	if isinstance(value, float):
		return int(value)
	return default


def _feature_list_check(
	*,
	ranker_cfg: RankerConfig,
	split: str,
	sample: bool,
) -> dict[str, Any]:
	meta_path = ranker_cfg.features_path(split=split, sample=sample).with_suffix(".meta.json")
	if not meta_path.exists():
		return {
			"feature_meta_found": False,
			"metadata_columns_excluded": True,
			"checked_columns": sorted(METADATA_ONLY_COLUMNS),
		}

	with meta_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	feature_columns = payload.get("feature_columns", [])
	if not isinstance(feature_columns, list):
		feature_columns = []
	feature_set = {str(name) for name in feature_columns}
	return {
		"feature_meta_found": True,
		"metadata_columns_excluded": METADATA_ONLY_COLUMNS.isdisjoint(feature_set),
		"checked_columns": sorted(METADATA_ONLY_COLUMNS),
		"feature_column_count": len(feature_columns),
	}


def _diagnose_split(
	*,
	candidate_path: Path,
	ranker_cfg: RankerConfig,
	split: str,
	sample: bool,
) -> dict[str, Any]:
	frame = pl.read_parquet(candidate_path)
	query_group = frame.group_by("query_id").agg(pl.len().alias("candidate_count"))
	positive = frame.filter(pl.col("label") == 1)

	rows = int(frame.height)
	queries = int(query_group.height)
	avg_candidates = _to_float(query_group.get_column("candidate_count").mean())
	min_candidates = _to_int(query_group.get_column("candidate_count").min())
	max_candidates = _to_int(query_group.get_column("candidate_count").max())

	target_injected_positive_rate = _to_float(
		positive.get_column("target_injected").cast(pl.Float64).mean()
	)
	residual_topk_hit_rate = float(1.0 - target_injected_positive_rate)

	positive_count_violations = int(
		frame.group_by("query_id")
		.agg(pl.col("label").sum().alias("pos_count"))
		.filter(pl.col("pos_count") != 1)
		.height
	)
	duplicate_candidate_count = int(
		frame.group_by(["query_id", "item_idx"]).len().filter(pl.col("len") > 1).height
	)

	null_counts = {
		name: int(frame.select(pl.col(name).is_null().sum()).item())
		for name in REQUIRED_CANDIDATE_COLUMNS
		if name in frame.columns
	}

	labels = frame.get_column("label").cast(pl.Float64).to_numpy()
	target_injected = frame.get_column("target_injected").cast(pl.Int8).to_numpy()
	residual_rank = frame.get_column("residual_rank").cast(pl.Float64).to_numpy()
	residual_score = frame.get_column("residual_score").cast(pl.Float64).to_numpy()

	candidate_source_group = frame.group_by("candidate_source").agg(
		pl.col("label").mean().alias("label_mean"),
		pl.len().alias("count"),
	)
	candidate_source_label_mean = {
		str(row["candidate_source"]): {
			"label_mean": float(row["label_mean"]),
			"count": int(row["count"]),
		}
		for row in candidate_source_group.iter_rows(named=True)
	}

	feature_guard = _feature_list_check(
		ranker_cfg=ranker_cfg,
		split=split,
		sample=sample,
	)

	return {
		"split": split,
		"candidate_path": str(candidate_path),
		"queries": queries,
		"rows": rows,
		"avg_candidates_per_query": avg_candidates,
		"min_candidates_per_query": min_candidates,
		"max_candidates_per_query": max_candidates,
		"percent_target_injected": float(target_injected_positive_rate * 100.0),
		"residual_top200_hit_rate": residual_topk_hit_rate,
		"label_distribution": {
			"positive": int(positive.height),
			"negative": int(rows - positive.height),
		},
		"duplicate_candidate_count": duplicate_candidate_count,
		"positive_count_violations": positive_count_violations,
		"null_counts_required_columns": null_counts,
		"feature_guard": feature_guard,
		"suspicious_correlations": {
			"target_injected_vs_label": {
				"pearson": _pearson(target_injected.astype(np.float64), labels),
				"label_mean_when_target_injected": float(
					frame.filter(pl.col("target_injected"))
					.select(pl.col("label").mean())
					.item()
					or 0.0
				),
				"label_mean_when_not_target_injected": float(
					frame.filter(~pl.col("target_injected"))
					.select(pl.col("label").mean())
					.item()
					or 0.0
				),
			},
			"candidate_source_vs_label": candidate_source_label_mean,
			"residual_rank_vs_label": {
				"pearson": _pearson(residual_rank, labels),
			},
			"residual_score_vs_label": {
				"pearson": _pearson(residual_score, labels),
			},
		},
	}


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	sample: bool = typer.Option(False, "--sample"),
) -> None:
	ranker_cfg = load_ranker_config(config)
	diagnostics: dict[str, Any] = {
		"sample": sample,
		"splits": {},
	}

	for split in ["train", "val", "test"]:
		candidate_path = ranker_cfg.candidate_path(split=split, sample=sample)
		if not candidate_path.exists():
			diagnostics["splits"][split] = {
				"split": split,
				"candidate_path": str(candidate_path),
				"missing": True,
			}
			continue
		diagnostics["splits"][split] = _diagnose_split(
			candidate_path=candidate_path,
			ranker_cfg=ranker_cfg,
			split=split,
			sample=sample,
		)

	output_path = ranker_cfg.paths.ranker_report_dir / (
		"ranker_candidate_diagnostics_sample.json"
		if sample
		else "ranker_candidate_diagnostics_full.json"
	)
	save_json(output_path, diagnostics)
	typer.echo(f"diagnostics_path: {output_path}")


if __name__ == "__main__":
	app()
