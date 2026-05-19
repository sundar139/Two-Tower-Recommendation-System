"""Offline evaluation for neural ranker reranking quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from movie_recsys.modeling.artifacts import load_checkpoint, save_json, save_markdown
from movie_recsys.modeling.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from movie_recsys.ranking.dataset import iter_ranker_query_groups
from movie_recsys.ranking.model import NeuralRanker
from movie_recsys.utils.system import log_memory_status


@dataclass(slots=True)
class RankerEvaluationResult:
	split: str
	ranker_metrics: dict[str, float]
	residual_metrics: dict[str, float]
	popularity_metrics: dict[str, float] | None
	query_count: int
	row_count: int
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


@dataclass(slots=True)
class _MetricAccumulator:
	hr_sum: float = 0.0
	mrr_sum: float = 0.0
	ndcg_sum: float = 0.0
	recall_sum: float = 0.0
	query_count: int = 0

	def update(self, ranked_items: list[int], targets: set[int]) -> None:
		self.hr_sum += hit_rate_at_k(ranked_items, targets, 10)
		self.mrr_sum += mrr_at_k(ranked_items, targets, 10)
		self.ndcg_sum += ndcg_at_k(ranked_items, targets, 10)
		self.recall_sum += recall_at_k(ranked_items, targets, 50)
		self.query_count += 1

	def as_dict(self) -> dict[str, float]:
		if self.query_count <= 0:
			return {"hr@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}
		n = float(self.query_count)
		return {
			"hr@10": float(self.hr_sum / n),
			"mrr@10": float(self.mrr_sum / n),
			"ndcg@10": float(self.ndcg_sum / n),
			"recall@50": float(self.recall_sum / n),
		}


def _score_query_group(
	group: pl.DataFrame,
	*,
	model: NeuralRanker,
	device: torch.device,
	feature_columns: list[str],
	batch_size: int,
) -> pl.DataFrame:
	features = group.select(feature_columns).to_numpy().astype(np.float32, copy=False)
	score_parts: list[np.ndarray] = []
	with torch.no_grad():
		for start in range(0, features.shape[0], batch_size):
			end = min(start + batch_size, features.shape[0])
			batch = torch.from_numpy(features[start:end]).to(device)
			logits = model(batch)
			score_parts.append(cast(np.ndarray, logits.detach().cpu().numpy().astype(np.float32)))

	score_values = np.concatenate(score_parts) if score_parts else np.zeros((0,), dtype=np.float32)
	scored = group.with_columns(pl.Series(name="ranker_score", values=score_values))
	finite = scored.select(
		[
			pl.col("ranker_score").is_finite().all().alias("ranker_ok"),
			pl.col("residual_score").is_finite().all().alias("residual_ok"),
		]
	).row(0)
	if not bool(finite[0]) or not bool(finite[1]):
		msg = "Detected non-finite scores during ranker evaluation"
		raise ValueError(msg)
	return scored


def _rank_items(group: pl.DataFrame, *, score_column: str) -> list[int]:
	sorted_group = group.sort(
		[score_column, "residual_rank", "item_idx"],
		descending=[True, False, False],
	)
	return [int(value) for value in sorted_group.get_column("item_idx").to_list()]


def _target_items(group: pl.DataFrame) -> set[int]:
	positives = {
		int(value)
		for value in group.filter(pl.col("label") == 1).get_column("item_idx").to_list()
	}
	if positives:
		return positives
	return {int(group.get_column("target_item_idx")[0])}


def evaluate_ranker_split(
	*,
	feature_paths: list[Path],
	feature_columns: list[str],
	checkpoint_path: Path,
	split: str,
	report_dir: Path,
	batch_size: int,
	max_queries: int | None = None,
	log_every_queries: int = 1000,
	log_file_path: Path | None = None,
	logger: logging.Logger | None = None,
) -> RankerEvaluationResult:
	if not feature_paths:
		msg = f"No feature shards available for split '{split}'"
		raise FileNotFoundError(msg)

	device = _select_device()
	model = _load_ranker(checkpoint_path, input_dim=len(feature_columns), device=device)

	columns = [
		"query_id",
		"item_idx",
		"target_item_idx",
		"label",
		"residual_score",
		"residual_rank",
		*feature_columns,
	]
	include_popularity = False
	first_frame = pl.read_parquet(feature_paths[0], n_rows=1024)
	if "popularity_score" in first_frame.columns:
		include_popularity = True
		if "popularity_score" not in columns:
			columns.insert(6, "popularity_score")

	ranker_acc = _MetricAccumulator()
	residual_acc = _MetricAccumulator()
	popularity_acc = _MetricAccumulator() if include_popularity else None

	queries_processed = 0
	rows_processed = 0
	buffered_frames: list[pl.DataFrame] = []
	buffered_rows = 0

	report_dir.mkdir(parents=True, exist_ok=True)
	scored_path = report_dir / f"ranker_eval_{split}_scored_candidates.parquet"
	if scored_path.exists():
		scored_path.unlink()
	writer: pq.ParquetWriter | None = None

	def _flush_scored_frames() -> None:
		nonlocal buffered_rows, writer
		if not buffered_frames:
			return
		frame = pl.concat(buffered_frames, how="vertical")
		table = frame.to_arrow()
		if writer is None:
			writer = pq.ParquetWriter(str(scored_path), table.schema)
		writer.write_table(table)
		buffered_frames.clear()
		buffered_rows = 0

	for query_group in iter_ranker_query_groups(
		[str(path) for path in feature_paths],
		columns=columns,
		batch_size_rows=max(batch_size * 8, 4096),
		max_queries=max_queries,
	):
		scored = _score_query_group(
			query_group,
			model=model,
			device=device,
			feature_columns=feature_columns,
			batch_size=batch_size,
		)

		targets = _target_items(scored)
		ranker_acc.update(_rank_items(scored, score_column="ranker_score"), targets)
		residual_acc.update(_rank_items(scored, score_column="residual_score"), targets)
		if popularity_acc is not None and "popularity_score" in scored.columns:
			popularity_acc.update(_rank_items(scored, score_column="popularity_score"), targets)

		queries_processed += 1
		rows_processed += int(scored.height)
		buffered_rows += int(scored.height)
		buffered_frames.append(scored)
		if buffered_rows >= 20000:
			_flush_scored_frames()

		if log_every_queries > 0 and queries_processed % log_every_queries == 0:
			prefix = (
				"[ranker-eval]"
				f" split={split}"
				f" queries={queries_processed}"
				f" rows={rows_processed}"
				f" log_file={log_file_path if log_file_path is not None else '-'}"
			)
			if logger is not None:
				log_memory_status(prefix, logger=logger, disk_path=report_dir)
			else:
				log_memory_status(prefix, disk_path=report_dir)

	_flush_scored_frames()
	if writer is not None:
		writer.close()

	ranker_metrics = ranker_acc.as_dict()
	residual_metrics = residual_acc.as_dict()
	popularity_metrics = popularity_acc.as_dict() if popularity_acc is not None else None

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
		"feature_paths": [str(path) for path in feature_paths],
		"feature_columns": feature_columns,
		"query_count": queries_processed,
		"row_count": rows_processed,
		"checkpoint_path": str(checkpoint_path),
	}
	if popularity_metrics is not None:
		report_payload["popularity"] = popularity_metrics
		report_payload["delta_vs_popularity"] = {
			metric: float(ranker_metrics[metric] - popularity_metrics[metric])
			for metric in ranker_metrics
		}

	report_path = report_dir / f"ranker_eval_{split}.json"
	save_json(report_path, report_payload)

	return RankerEvaluationResult(
		split=split,
		ranker_metrics=ranker_metrics,
		residual_metrics=residual_metrics,
		popularity_metrics=popularity_metrics,
		query_count=queries_processed,
		row_count=rows_processed,
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
