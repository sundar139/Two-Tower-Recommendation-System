from __future__ import annotations

from pathlib import Path

import polars as pl

from movie_recsys.data.preprocessing import (
	build_positive_interactions,
	filter_users_by_positive_count,
)
from movie_recsys.data.splits import apply_id_maps, build_id_maps, chronological_user_split

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _positives() -> pl.DataFrame:
	ratings = pl.read_csv(FIXTURES_DIR / "tiny_ratings.csv")
	positives = build_positive_interactions(ratings, 4.0)
	return filter_users_by_positive_count(positives, 3)


def test_chronological_split_correctness() -> None:
	train, val, test = chronological_user_split(_positives())
	assert train.height == 2
	assert val.height == 2
	assert test.height == 2
	assert train.filter(pl.col("userId") == 10).get_column("timestamp").max() < val.filter(
		pl.col("userId") == 10
	).get_column("timestamp").min()


def test_id_mapping_contiguity() -> None:
	train, val, test = chronological_user_split(_positives())
	user_map, item_map = build_id_maps(train, val, test)
	assert user_map.get_column("user_idx").to_list() == [0, 1]
	assert item_map.get_column("item_idx").to_list() == [0, 1, 2, 3]


def test_no_leakage_across_splits_for_exact_interaction() -> None:
	train, val, test = chronological_user_split(_positives())
	user_map, item_map = build_id_maps(train, val, test)
	train_m = apply_id_maps(train, user_map, item_map)
	val_m = apply_id_maps(val, user_map, item_map)
	test_m = apply_id_maps(test, user_map, item_map)

	train_keys = set(train_m.select(["user_idx", "item_idx", "timestamp"]).iter_rows())
	val_keys = set(val_m.select(["user_idx", "item_idx", "timestamp"]).iter_rows())
	test_keys = set(test_m.select(["user_idx", "item_idx", "timestamp"]).iter_rows())
	assert train_keys.isdisjoint(val_keys)
	assert train_keys.isdisjoint(test_keys)
	assert val_keys.isdisjoint(test_keys)
