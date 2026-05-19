"""Tests for ranker dataset behavior."""

from __future__ import annotations

import numpy as np
import polars as pl

from movie_recsys.ranking.dataset import RankerDataset, collate_ranker_batch


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
