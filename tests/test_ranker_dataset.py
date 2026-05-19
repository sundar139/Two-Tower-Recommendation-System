"""Tests for ranker dataset behavior."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from movie_recsys.ranking.dataset import (
	RankerDataset,
	collate_ranker_batch,
	iter_ranker_feature_batches,
)


def _feature_frame() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"query_id": ["q1", "q1", "q2"],
			"user_idx": [1, 1, 2],
			"item_idx": [10, 20, 30],
			"split": ["train", "train", "train"],
			"label": [1, 0, 1],
			"target_item_idx": [10, 10, 30],
			"candidate_source": ["retrieved", "retrieved", "target_injected"],
			"target_injected": [False, False, True],
			"residual_score": [0.9, 0.4, 0.8],
			"residual_rank": [1, 2, 1],
			"feature_a": [0.1, 0.2, 0.3],
			"feature_b": [1.0, 2.0, 3.0],
		}
	)


def test_ranker_dataset_loads_and_returns_finite_tensors(tmp_path) -> None:
	feature_path = tmp_path / "features.parquet"
	_feature_frame().write_parquet(feature_path)

	dataset = RankerDataset(str(feature_path))
	assert len(dataset) == 3
	assert dataset.feature_dim == 2
	assert dataset.feature_columns == ["residual_rank", "residual_score"]

	sample = dataset[0]
	assert sample["features"].shape[0] == 2
	assert np.isfinite(sample["features"].numpy()).all()
	assert float(sample["label"]) in {0.0, 1.0}


def test_ranker_dataset_collate_shapes(tmp_path) -> None:
	feature_path = tmp_path / "features.parquet"
	_feature_frame().write_parquet(feature_path)
	dataset = RankerDataset(str(feature_path))

	batch = collate_ranker_batch([dataset[0], dataset[1]])
	assert tuple(batch["features"].shape) == (2, 2)
	assert tuple(batch["labels"].shape) == (2,)
	assert len(batch["query_ids"]) == 2


def test_streaming_feature_batches_do_not_depend_on_polars_read_parquet(
	tmp_path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	shard_a = tmp_path / "part_000000.parquet"
	shard_b = tmp_path / "part_000001.parquet"

	pl.DataFrame(
		{
			"query_id": ["q1", "q1"],
			"item_idx": [11, 12],
			"target_item_idx": [11, 11],
			"residual_score": [0.9, 0.8],
			"residual_rank": [1, 2],
			"label": [1, 0],
		}
	).write_parquet(shard_a)
	pl.DataFrame(
		{
			"query_id": ["q2"],
			"item_idx": [21],
			"target_item_idx": [21],
			"residual_score": [0.7],
			"residual_rank": [1],
			"label": [1],
		}
	).write_parquet(shard_b)

	def _fail_read_parquet(*_args, **_kwargs):
		raise AssertionError("iter_ranker_feature_batches should not call polars.read_parquet")

	monkeypatch.setattr(pl, "read_parquet", _fail_read_parquet)

	rows = 0
	for batch in iter_ranker_feature_batches(
		[str(shard_a), str(shard_b)],
		feature_columns=["residual_score", "residual_rank"],
		batch_size=2,
		shuffle=False,
		seed=7,
	):
		rows += int(batch["features"].shape[0])

	assert rows == 3
