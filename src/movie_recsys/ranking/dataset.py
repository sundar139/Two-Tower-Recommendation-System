"""Dataset and dataloader helpers for neural ranker training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from movie_recsys.ranking.features import resolve_feature_allowlist

METADATA_COLUMNS = {
	"query_id",
	"user_idx",
	"item_idx",
	"split",
	"label",
	"target_item_idx",
	"candidate_source",
	"target_injected",
	"timestamp_context",
	"user_history_length",
}


def infer_feature_columns(frame: pl.DataFrame) -> list[str]:
	"""Infer model feature columns from the explicit feature allowlist."""

	use_frozen_features = any(
		name in frame.columns
		for name in [
			"frozen_emb_dot",
			"frozen_emb_prod_mean",
			"frozen_emb_prod_max",
			"frozen_emb_prod_min",
			"frozen_emb_prod_std",
			"frozen_emb_absdiff_mean",
			"frozen_emb_absdiff_max",
			"frozen_emb_absdiff_min",
			"frozen_emb_absdiff_std",
		]
	)
	return resolve_feature_allowlist(
		frame,
		use_frozen_features=use_frozen_features,
	)


class RankerDataset(Dataset[dict[str, Any]]):
	"""Pointwise ranker dataset backed by feature parquet files."""

	def __init__(
		self,
		parquet_path: str,
		*,
		feature_columns: list[str] | None = None,
		sort_by_query: bool = False,
	) -> None:
		super().__init__()
		frame = pl.read_parquet(parquet_path)
		if sort_by_query:
			frame = frame.sort(["query_id", "residual_rank", "item_idx"])

		selected_feature_columns = feature_columns or infer_feature_columns(frame)
		if not selected_feature_columns:
			msg = "No numeric feature columns were detected in ranker feature table"
			raise ValueError(msg)

		missing = [name for name in selected_feature_columns if name not in frame.columns]
		if missing:
			msg = f"Missing requested feature columns: {', '.join(missing)}"
			raise ValueError(msg)

		self.feature_columns = selected_feature_columns
		self.features = frame.select(self.feature_columns).to_numpy().astype(np.float32, copy=False)
		self.labels = frame.get_column("label").to_numpy().astype(np.float32, copy=False)
		self.query_ids = [str(value) for value in frame.get_column("query_id").to_list()]
		self.item_indices = frame.get_column("item_idx").to_numpy().astype(
			np.int64,
			copy=False,
		)
		self.target_indices = frame.get_column("target_item_idx").to_numpy().astype(
			np.int64,
			copy=False,
		)
		self.residual_scores = frame.get_column("residual_score").to_numpy().astype(
			np.float32,
			copy=False,
		)
		self.residual_ranks = frame.get_column("residual_rank").to_numpy().astype(
			np.int64,
			copy=False,
		)

	@property
	def feature_dim(self) -> int:
		return int(self.features.shape[1])

	def __len__(self) -> int:
		return int(self.features.shape[0])

	def __getitem__(self, index: int) -> dict[str, Any]:
		return {
			"features": torch.tensor(self.features[index], dtype=torch.float32),
			"label": torch.tensor(self.labels[index], dtype=torch.float32),
			"query_id": self.query_ids[index],
			"item_idx": int(self.item_indices[index]),
			"target_item_idx": int(self.target_indices[index]),
			"residual_score": float(self.residual_scores[index]),
			"residual_rank": int(self.residual_ranks[index]),
		}


def collate_ranker_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
	features = torch.stack([row["features"] for row in batch], dim=0)
	labels = torch.stack([row["label"] for row in batch], dim=0)
	return {
		"features": features,
		"labels": labels,
		"query_ids": [str(row["query_id"]) for row in batch],
		"item_idx": torch.tensor([int(row["item_idx"]) for row in batch], dtype=torch.long),
		"target_item_idx": torch.tensor(
			[int(row["target_item_idx"]) for row in batch],
			dtype=torch.long,
		),
		"residual_score": torch.tensor(
			[float(row["residual_score"]) for row in batch],
			dtype=torch.float32,
		),
		"residual_rank": torch.tensor(
			[int(row["residual_rank"]) for row in batch],
			dtype=torch.long,
		),
	}


def make_ranker_dataloader(
	dataset: RankerDataset,
	*,
	batch_size: int,
	shuffle: bool,
	num_workers: int,
	pin_memory: bool,
	prefetch_factor: int | None,
	persistent_workers: bool,
	seed: int,
) -> DataLoader[dict[str, Any]]:
	generator = torch.Generator().manual_seed(seed)
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": batch_size,
		"shuffle": shuffle,
		"num_workers": num_workers,
		"collate_fn": collate_ranker_batch,
		"generator": generator,
		"drop_last": False,
		"pin_memory": pin_memory,
	}
	if num_workers > 0:
		loader_kwargs["persistent_workers"] = persistent_workers
		if prefetch_factor is not None:
			loader_kwargs["prefetch_factor"] = int(prefetch_factor)
	return DataLoader(
		**loader_kwargs,
	)


def iter_ranker_feature_batches(
	feature_paths: list[str],
	*,
	feature_columns: list[str],
	batch_size: int,
	shuffle: bool,
	seed: int,
	max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
	"""Yield mini-batches directly from parquet shards without full in-memory materialization."""

	if batch_size <= 0:
		msg = "batch_size must be positive"
		raise ValueError(msg)

	columns = [
		"query_id",
		"item_idx",
		"target_item_idx",
		"residual_score",
		"residual_rank",
		"label",
		*feature_columns,
	]
	rows_yielded = 0
	batch_seed = int(seed)

	for feature_path in feature_paths:
		parquet_file = pq.ParquetFile(feature_path)
		for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
			frame = cast(pl.DataFrame, pl.from_arrow(record_batch))
			if frame.height == 0:
				continue

			if max_rows is not None:
				remaining = int(max_rows) - rows_yielded
				if remaining <= 0:
					return
				if frame.height > remaining:
					frame = frame.slice(0, remaining)

			if shuffle and frame.height > 1:
				frame = frame.sample(
					n=frame.height,
					with_replacement=False,
					shuffle=True,
					seed=batch_seed,
				)

			rows_yielded += int(frame.height)
			batch_seed += 1

			feature_values = frame.select(feature_columns).to_numpy().astype(np.float32, copy=False)
			labels = frame.get_column("label").to_numpy().astype(np.float32, copy=False)
			item_idx = frame.get_column("item_idx").to_numpy().astype(np.int64, copy=False)
			target_item_idx = frame.get_column("target_item_idx").to_numpy().astype(
				np.int64,
				copy=False,
			)
			residual_score = frame.get_column("residual_score").to_numpy().astype(
				np.float32,
				copy=False,
			)
			residual_rank = frame.get_column("residual_rank").to_numpy().astype(
				np.int64,
				copy=False,
			)
			yield {
				"features": torch.from_numpy(feature_values.copy()),
				"labels": torch.from_numpy(labels.copy()),
				"query_ids": [str(value) for value in frame.get_column("query_id").to_list()],
				"item_idx": torch.from_numpy(item_idx.copy()),
				"target_item_idx": torch.from_numpy(target_item_idx.copy()),
				"residual_score": torch.from_numpy(residual_score.copy()),
				"residual_rank": torch.from_numpy(residual_rank.copy()),
			}

			if max_rows is not None and rows_yielded >= int(max_rows):
				return


def iter_ranker_query_groups(
	feature_paths: list[str],
	*,
	columns: list[str],
	batch_size_rows: int,
	max_rows: int | None = None,
	max_queries: int | None = None,
) -> Iterator[pl.DataFrame]:
	"""Yield query-complete frames from sorted shard files in a streaming manner."""

	if batch_size_rows <= 0:
		msg = "batch_size_rows must be positive"
		raise ValueError(msg)

	query_count = 0
	rows_seen = 0
	carryover: pl.DataFrame | None = None

	for feature_path in feature_paths:
		parquet_file = pq.ParquetFile(feature_path)
		for record_batch in parquet_file.iter_batches(
			batch_size=batch_size_rows,
			columns=columns,
		):
			frame = cast(pl.DataFrame, pl.from_arrow(record_batch))
			if frame.height == 0:
				continue

			if max_rows is not None:
				remaining = int(max_rows) - rows_seen
				if remaining <= 0:
					break
				if frame.height > remaining:
					frame = frame.slice(0, remaining)

			rows_seen += int(frame.height)

			if carryover is not None and carryover.height > 0:
				frame = pl.concat([carryover, frame], how="vertical")
				carryover = None

			groups = frame.partition_by("query_id", maintain_order=True)
			if not groups:
				continue

			for group in groups[:-1]:
				yield group
				query_count += 1
				if max_queries is not None and query_count >= int(max_queries):
					return

			carryover = groups[-1]

		if max_rows is not None and rows_seen >= int(max_rows):
			break

	if carryover is not None and carryover.height > 0:
		yield carryover
