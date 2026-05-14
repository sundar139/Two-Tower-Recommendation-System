"""Chronological splitting and ID mapping utilities."""

from __future__ import annotations

import polars as pl


def chronological_user_split(
	positives: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
	ranked = positives.sort(["userId", "timestamp", "movieId"]).with_columns(
		pl.int_range(0, pl.len()).over("userId").alias("event_idx"),
		pl.len().over("userId").alias("event_count"),
	)
	train = ranked.filter(pl.col("event_idx") < pl.col("event_count") - 2).drop(
		["event_idx", "event_count"]
	)
	val = ranked.filter(pl.col("event_idx") == pl.col("event_count") - 2).drop(
		["event_idx", "event_count"]
	)
	test = ranked.filter(pl.col("event_idx") == pl.col("event_count") - 1).drop(
		["event_idx", "event_count"]
	)
	return train, val, test


def build_id_maps(
	train: pl.DataFrame,
	val: pl.DataFrame,
	test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
	merged = pl.concat(
		[
			train.select(["userId", "movieId"]),
			val.select(["userId", "movieId"]),
			test.select(["userId", "movieId"]),
		]
	)
	user_map = merged.select("userId").unique().sort("userId").with_row_index("user_idx")
	item_map = merged.select("movieId").unique().sort("movieId").with_row_index("item_idx")
	return user_map, item_map


def apply_id_maps(
	frame: pl.DataFrame,
	user_map: pl.DataFrame,
	item_map: pl.DataFrame,
) -> pl.DataFrame:
	return (
		frame.join(user_map, on="userId", how="inner")
		.join(item_map, on="movieId", how="inner")
		.sort(["user_idx", "timestamp", "item_idx"])
	)


def build_user_histories(train: pl.DataFrame) -> pl.DataFrame:
	return (
		train.sort(["user_idx", "timestamp", "item_idx"])
		.group_by(["user_idx", "userId"])
		.agg(pl.col("item_idx").alias("train_history_item_idx"))
		.sort("user_idx")
	)
