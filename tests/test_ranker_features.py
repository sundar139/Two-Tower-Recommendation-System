"""Tests for ranker feature engineering."""

from __future__ import annotations

import polars as pl
import pytest

from movie_recsys.ranking.features import MANDATORY_FEATURE_COLUMNS, _build_split_features


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
	frame, _feature_columns = _build_split_features(
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
	first, _ = _build_split_features(
		candidates=_candidate_fixture(),
		users_df=_users_fixture(),
		items_df=_items_fixture(),
		use_frozen_features=False,
	)
	second, _ = _build_split_features(
		candidates=_candidate_fixture(),
		users_df=_users_fixture(),
		items_df=_items_fixture(),
		use_frozen_features=False,
	)
	assert first.equals(second)
