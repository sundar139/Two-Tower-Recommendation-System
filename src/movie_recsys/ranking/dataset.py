"""Dataset and dataloader helpers for neural ranker training."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
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
	seed: int,
) -> DataLoader[dict[str, Any]]:
	generator = torch.Generator().manual_seed(seed)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate_ranker_batch,
		generator=generator,
		drop_last=False,
	)
