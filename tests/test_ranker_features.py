"""Tests for ranker feature engineering."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from movie_recsys.ranking.config import RankerConfig, RankerPathsConfig
from movie_recsys.ranking.features import (
	MANDATORY_FEATURE_COLUMNS,
	_build_split_features,
	_limit_train_negatives,
	build_ranker_features,
)


def _candidate_fixture() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"query_id": ["train_u00000001", "train_u00000001"],
			"user_idx": [1, 1],
			"item_idx": [10, 20],
			"split": ["train", "train"],
			"label": [1, 0],
			"target_item_idx": [10, 10],
			"residual_score": [0.7, 0.4],
			"residual_rank": [1, 2],
			"target_injected": [False, False],
			"user_history_length": [5, 5],
			"timestamp_context": [1000, 1000],
			"candidate_source": ["retrieved", "retrieved"],
		}
	)


def _users_fixture() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"user_idx": [1],
			"positive_rating_count": [30],
			"total_rating_count": [60],
			"mean_rating": [4.0],
			"activity_span_days": [200.0],
			"tag_count": [5],
			"genre_affinity_action": [0.8],
			"genre_affinity_comedy": [0.2],
		}
	)


def _items_fixture() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"item_idx": [10, 20],
			"rating_count": [100, 50],
			"positive_count": [80, 30],
			"mean_rating": [4.2, 3.8],
			"popularity_score": [0.9, 0.5],
			"release_year": [2000, 2010],
			"genome_tag_count": [4, 2],
			"genome_relevance_mean": [0.6, 0.3],
			"genome_relevance_max": [0.9, 0.5],
			"item_genre_action": [1, 0],
			"item_genre_comedy": [0, 1],
		}
	)


def test_feature_generation_required_columns_and_no_nulls() -> None:
	frame, feature_columns = _build_split_features(
		candidates=_candidate_fixture(),
		users_df=_users_fixture(),
		items_df=_items_fixture(),
		use_frozen_features=False,
	)
	for column in MANDATORY_FEATURE_COLUMNS:
		assert column in frame.columns

	null_checks = frame.select(
		[
			pl.col(column).is_null().any().alias(column)
			for column in MANDATORY_FEATURE_COLUMNS
			if column not in {"query_id", "split"}
		]
	).row(0)
	assert all(not bool(value) for value in null_checks)

	for blocked in [
		"target_injected",
		"candidate_source",
		"target_item_idx",
		"label",
		"query_id",
		"split",
	]:
		assert blocked not in feature_columns


def test_feature_generation_genre_overlap_fixture() -> None:
	frame, _feature_columns = _build_split_features(
		candidates=_candidate_fixture(),
		users_df=_users_fixture(),
		items_df=_items_fixture(),
		use_frozen_features=False,
	)
	row_action = frame.filter(pl.col("item_idx") == 10).row(0, named=True)
	row_comedy = frame.filter(pl.col("item_idx") == 20).row(0, named=True)

	assert float(row_action["genre_affinity_dot"]) == pytest.approx(0.8)
	assert float(row_action["genre_overlap_count"]) == pytest.approx(1.0)
	assert float(row_action["max_genre_affinity"]) == pytest.approx(0.8)

	assert float(row_comedy["genre_affinity_dot"]) == pytest.approx(0.2)
	assert float(row_comedy["genre_overlap_count"]) == pytest.approx(1.0)
	assert float(row_comedy["max_genre_affinity"]) == pytest.approx(0.2)


def test_feature_generation_is_deterministic() -> None:
	first, first_features = _build_split_features(
		candidates=_candidate_fixture(),
		users_df=_users_fixture(),
		items_df=_items_fixture(),
		use_frozen_features=False,
	)
	second, second_features = _build_split_features(
		candidates=_candidate_fixture(),
		users_df=_users_fixture(),
		items_df=_items_fixture(),
		use_frozen_features=False,
	)
	assert first.equals(second)
	assert first_features == second_features


def test_feature_allowlist_is_explicit_and_stable() -> None:
	frame, feature_columns = _build_split_features(
		candidates=_candidate_fixture(),
		users_df=_users_fixture(),
		items_df=_items_fixture(),
		use_frozen_features=False,
	)

	assert feature_columns == sorted(feature_columns)
	assert "residual_score" in feature_columns
	assert "residual_rank" in feature_columns
	assert "genre_affinity_dot" in feature_columns
	assert "score_rank_interaction" in feature_columns

	metadata_only = {
		"query_id",
		"split",
		"label",
		"target_item_idx",
		"candidate_source",
		"target_injected",
	}
	assert metadata_only.isdisjoint(set(feature_columns))
	assert metadata_only.issubset(set(frame.columns))


def test_limit_train_negatives_keeps_positive_plus_top_k_negatives() -> None:
	candidates = pl.DataFrame(
		{
			"query_id": ["q1", "q1", "q1", "q1", "q1", "q2", "q2", "q2", "q2"],
			"item_idx": [10, 20, 30, 40, 50, 11, 21, 31, 41],
			"label": [1, 0, 0, 0, 0, 1, 0, 0, 0],
			"residual_rank": [1, 2, 3, 4, 5, 1, 2, 3, 4],
		}
	)

	limited = _limit_train_negatives(candidates, negatives_per_positive=2)
	counts = limited.group_by("query_id").len().sort("query_id")
	assert counts.get_column("len").to_list() == [3, 3]

	q1_items = (
		limited.filter(pl.col("query_id") == "q1")
		.sort("residual_rank")
		.get_column("item_idx")
		.to_list()
	)
	q2_items = (
		limited.filter(pl.col("query_id") == "q2")
		.sort("residual_rank")
		.get_column("item_idx")
		.to_list()
	)
	assert q1_items == [10, 20, 30]
	assert q2_items == [11, 21, 31]


def _test_ranker_cfg(tmp_path: Path) -> RankerConfig:
	paths = RankerPathsConfig(
		ranker_candidate_dir=tmp_path / "candidates",
		ranker_feature_dir=tmp_path / "features",
		ranker_model_dir=tmp_path / "models",
		ranker_report_dir=tmp_path / "reports",
	)
	paths.ranker_candidate_dir.mkdir(parents=True, exist_ok=True)
	paths.ranker_feature_dir.mkdir(parents=True, exist_ok=True)
	paths.ranker_model_dir.mkdir(parents=True, exist_ok=True)
	paths.ranker_report_dir.mkdir(parents=True, exist_ok=True)

	return RankerConfig(
		retriever_config=tmp_path / "retriever.yaml",
		retriever_checkpoint=tmp_path / "retriever.pt",
		mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
		mlflow_artifact_root=str(tmp_path / "mlruns"),
		mlflow_experiment_name="ranker-tests",
		mlflow_ui_host="127.0.0.1",
		mlflow_ui_port=5000,
		mlflow_ui_url="http://127.0.0.1:5000",
		paths=paths,
		negative_samples_per_positive=1,
		feature_shard_size=2,
	)


def test_full_feature_manifest_writes_shards(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	cfg = _test_ranker_cfg(tmp_path)

	candidate_chunk = (
		cfg.candidate_dir_for_scope(sample=False) / "chunks" / "train" / "chunk_000000.parquet"
	)
	candidate_chunk.parent.mkdir(parents=True, exist_ok=True)
	pl.DataFrame(
		{
			"query_id": ["train_u00000001", "train_u00000001", "train_u00000002"],
			"user_idx": [1, 1, 1],
			"item_idx": [10, 20, 30],
			"split": ["train", "train", "train"],
			"label": [1, 0, 1],
			"target_item_idx": [10, 10, 30],
			"residual_score": [0.9, 0.6, 0.8],
			"residual_rank": [1, 2, 1],
			"target_injected": [False, False, False],
			"user_history_length": [5, 5, 7],
			"timestamp_context": [1000, 1000, 2000],
			"candidate_source": ["retrieved", "retrieved", "retrieved"],
		}
	).write_parquet(candidate_chunk)

	users_path = tmp_path / "users.parquet"
	items_path = tmp_path / "items.parquet"
	pl.DataFrame(
		{
			"user_idx": [1],
			"positive_rating_count": [10],
			"total_rating_count": [20],
			"mean_rating": [4.0],
			"activity_span_days": [100.0],
			"tag_count": [2],
			"genre_affinity_action": [0.8],
		}
	).write_parquet(users_path)
	pl.DataFrame(
		{
			"item_idx": [10, 20, 30],
			"rating_count": [100, 90, 80],
			"positive_count": [70, 60, 50],
			"mean_rating": [4.1, 3.8, 4.3],
			"popularity_score": [0.9, 0.7, 0.85],
			"release_year": [2000, 2010, 2005],
			"genome_tag_count": [3, 2, 1],
			"genome_relevance_mean": [0.6, 0.5, 0.4],
			"genome_relevance_max": [0.9, 0.8, 0.7],
			"item_genre_action": [1, 0, 1],
		}
	).write_parquet(items_path)

	monkeypatch.setattr(
		"movie_recsys.ranking.features.load_retrieval_config",
		lambda *_args, **_kwargs: SimpleNamespace(
			users_path=users_path,
			items_path=items_path,
		),
	)

	outputs = build_ranker_features(
		cfg,
		sample=False,
		feature_shard_size=2,
		resume=False,
		overwrite=True,
		splits=("train",),
	)

	manifest_path = outputs["manifest"]
	assert manifest_path.exists()
	with manifest_path.open("r", encoding="utf-8") as handle:
		manifest = json.load(handle)

	assert manifest["completed"] is True
	assert manifest["config_hash"]
	assert manifest["feature_columns"]
	assert manifest["metadata_only_excluded_columns"]
	assert manifest["schema"]

	train_payload = manifest["splits"]["train"]
	assert train_payload["split"] == "train"
	assert train_payload["completed"] is True
	assert train_payload["row_count"] > 0
	assert train_payload["query_count"] > 0
	assert len(train_payload["shard_paths"]) >= 1
	for shard_path in train_payload["shard_paths"]:
		assert Path(shard_path).exists()
