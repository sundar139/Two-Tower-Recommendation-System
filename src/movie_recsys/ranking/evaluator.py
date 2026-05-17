"""Offline evaluation for neural ranker reranking quality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch

from movie_recsys.modeling.artifacts import load_checkpoint, save_json, save_markdown
from movie_recsys.modeling.metrics import aggregate_ranking_metrics
from movie_recsys.ranking.dataset import infer_feature_columns
from movie_recsys.ranking.model import NeuralRanker


@dataclass(slots=True)
class RankerEvaluationResult:
	split: str
	ranker_metrics: dict[str, float]
	residual_metrics: dict[str, float]
	popularity_metrics: dict[str, float] | None
	report_path: Path
	scored_candidates_path: Path


def _select_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_ranker(checkpoint_path: Path, *, input_dim: int, device: torch.device) -> NeuralRanker:
	payload = load_checkpoint(checkpoint_path)
	model = NeuralRanker(
		input_dim=input_dim,
		hidden_dims=[int(v) for v in payload["hidden_dims"]],
		dropout=float(payload["dropout"]),
		use_layer_norm=bool(payload.get("use_layer_norm", True)),
	)
	model.load_state_dict(cast(dict[str, Any], payload["model_state_dict"]))
	model.to(device)
	model.eval()
	return model


def _score_frame(
	frame: pl.DataFrame,
	*,
	feature_columns: list[str],
	checkpoint_path: Path,
	batch_size: int,
) -> pl.DataFrame:
	features = frame.select(feature_columns).to_numpy().astype(np.float32, copy=False)
	device = _select_device()
	model = _load_ranker(checkpoint_path, input_dim=features.shape[1], device=device)

	scores: list[np.ndarray] = []
	with torch.no_grad():
		for start in range(0, features.shape[0], batch_size):
			end = min(start + batch_size, features.shape[0])
			batch = torch.from_numpy(features[start:end]).to(device)
			logits = model(batch)
			scores.append(cast(np.ndarray, logits.detach().cpu().numpy().astype(np.float32)))

	score_values = np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)

	return frame.with_columns(pl.Series(name="ranker_score", values=score_values))


def _build_predictions(
	scored: pl.DataFrame,
) -> tuple[
	dict[int, list[int]],
	dict[int, list[int]],
	dict[int, list[int]] | None,
	dict[int, set[int]],
]:
	ranker_predictions: dict[int, list[int]] = {}
	residual_predictions: dict[int, list[int]] = {}
	popularity_predictions: dict[int, list[int]] | None = (
		{} if "popularity_score" in scored.columns else None
	)
	targets: dict[int, set[int]] = {}

	query_groups = scored.partition_by("query_id", maintain_order=True)
	for query_idx, group in enumerate(query_groups):
		target_item = int(group.get_column("target_item_idx")[0])
		targets[query_idx] = {target_item}

		ranker_sorted = group.sort(["ranker_score", "residual_rank"], descending=[True, False])
		residual_sorted = group.sort(["residual_score", "residual_rank"], descending=[True, False])

		ranker_predictions[query_idx] = [
			int(v) for v in ranker_sorted.get_column("item_idx").to_list()
		]
		residual_predictions[query_idx] = [
			int(v) for v in residual_sorted.get_column("item_idx").to_list()
		]

		if popularity_predictions is not None:
			popularity_sorted = group.sort(
				["popularity_score", "residual_rank"],
				descending=[True, False],
			)
			popularity_predictions[query_idx] = [
				int(v) for v in popularity_sorted.get_column("item_idx").to_list()
			]

	return ranker_predictions, residual_predictions, popularity_predictions, targets


def _assert_finite_scores(scored: pl.DataFrame) -> None:
	finite = scored.select(
		[
			pl.col("ranker_score").is_finite().all().alias("ranker_ok"),
			pl.col("residual_score").is_finite().all().alias("residual_ok"),
		]
	).row(0)
	if not bool(finite[0]) or not bool(finite[1]):
		msg = "Detected non-finite scores during ranker evaluation"
		raise ValueError(msg)


def evaluate_ranker_split(
	*,
	feature_path: Path,
	checkpoint_path: Path,
	split: str,
	report_dir: Path,
	batch_size: int,
) -> RankerEvaluationResult:
	frame = pl.read_parquet(feature_path).sort(["query_id", "residual_rank", "item_idx"])
	feature_columns = infer_feature_columns(frame)
	scored = _score_frame(
		frame,
		feature_columns=feature_columns,
		checkpoint_path=checkpoint_path,
		batch_size=batch_size,
	)
	_assert_finite_scores(scored)

	ranker_pred, residual_pred, popularity_pred, targets = _build_predictions(scored)
	ranker_metrics = aggregate_ranking_metrics(ranker_pred, targets)
	residual_metrics = aggregate_ranking_metrics(residual_pred, targets)
	popularity_metrics = (
		aggregate_ranking_metrics(popularity_pred, targets)
		if popularity_pred is not None
		else None
	)

	delta_vs_residual = {
		metric: float(ranker_metrics[metric] - residual_metrics[metric])
		for metric in ranker_metrics
	}

	report_payload: dict[str, Any] = {
		"split": split,
		"ranker": ranker_metrics,
		"residual": residual_metrics,
		"delta_vs_residual": delta_vs_residual,
		"beats_residual_ndcg@10": ranker_metrics["ndcg@10"] > residual_metrics["ndcg@10"],
		"feature_path": str(feature_path),
		"checkpoint_path": str(checkpoint_path),
	}
	if popularity_metrics is not None:
		report_payload["popularity"] = popularity_metrics
		report_payload["delta_vs_popularity"] = {
			metric: float(ranker_metrics[metric] - popularity_metrics[metric])
			for metric in ranker_metrics
		}

	report_path = report_dir / f"ranker_eval_{split}.json"
	scored_path = report_dir / f"ranker_eval_{split}_scored_candidates.parquet"
	report_dir.mkdir(parents=True, exist_ok=True)
	save_json(report_path, report_payload)
	scored.write_parquet(scored_path)

	return RankerEvaluationResult(
		split=split,
		ranker_metrics=ranker_metrics,
		residual_metrics=residual_metrics,
		popularity_metrics=popularity_metrics,
		report_path=report_path,
		scored_candidates_path=scored_path,
	)


def write_comparison_markdown(
	*,
	val_report: dict[str, Any],
	test_report: dict[str, Any],
	output_path: Path,
) -> None:
	lines = [
		"# Neural Ranker Comparison",
		"",
		"| Split | Metric | Ranker | Residual | Delta |",
		"|---|---|---:|---:|---:|",
	]
	for split_name, payload in [("val", val_report), ("test", test_report)]:
		ranker = cast(dict[str, float], payload["ranker"])
		residual = cast(dict[str, float], payload["residual"])
		for metric in ["hr@10", "mrr@10", "ndcg@10", "recall@50"]:
			delta = ranker[metric] - residual[metric]
			lines.append(
				f"| {split_name} | {metric} | {ranker[metric]:.6f} | "
				f"{residual[metric]:.6f} | {delta:.6f} |"
			)

	save_markdown(output_path, "\n".join(lines) + "\n")
