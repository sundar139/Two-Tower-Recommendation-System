"""Validation checks for processed recommendation datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(slots=True)
class ValidationResult:
	name: str
	passed: bool
	details: str


MANDATORY_INTERACTION_COLUMNS = [
	"userId",
	"movieId",
	"explicit_rating",
	"timestamp",
	"user_idx",
	"item_idx",
]


def _ok(name: str, details: str) -> ValidationResult:
	return ValidationResult(name=name, passed=True, details=details)


def _fail(name: str, details: str) -> ValidationResult:
	return ValidationResult(name=name, passed=False, details=details)


def validate_required_columns(
	frame: pl.DataFrame,
	required_columns: list[str],
	name: str,
) -> ValidationResult:
	missing = [col for col in required_columns if col not in frame.columns]
	if missing:
		return _fail(name, f"Missing columns: {missing}")
	return _ok(name, "All required columns present")


def validate_no_nulls(
	frame: pl.DataFrame,
	mandatory_columns: list[str],
	name: str,
) -> ValidationResult:
	null_cols = [c for c in mandatory_columns if frame.get_column(c).null_count() > 0]
	if null_cols:
		return _fail(name, f"Null values found in: {null_cols}")
	return _ok(name, "No nulls in mandatory columns")


def validate_contiguous_ids(frame: pl.DataFrame, column: str) -> ValidationResult:
	values = frame.select(column).unique().sort(column).get_column(column).to_list()
	expected = list(range(len(values)))
	if values != expected:
		return _fail(f"contiguous_{column}", "ID values are not contiguous")
	return _ok(f"contiguous_{column}", "IDs are contiguous")


def validate_processed_files_loadable(output_dir: Path, file_names: list[str]) -> ValidationResult:
	for file_name in file_names:
		path = output_dir / file_name
		if not path.exists():
			return _fail("loadable_files", f"Missing file: {file_name}")
		_ = pl.read_parquet(path)
	return _ok("loadable_files", "Processed files are loadable")


def validate_split_chronology(
	train: pl.DataFrame,
	val: pl.DataFrame,
	test: pl.DataFrame,
) -> ValidationResult:
	per_user = (
		train.group_by("user_idx").agg(pl.col("timestamp").max().alias("train_max_ts"))
		.join(
			val.select(["user_idx", "timestamp"]).rename({"timestamp": "val_ts"}),
			on="user_idx",
		)
		.join(
			test.select(["user_idx", "timestamp"]).rename({"timestamp": "test_ts"}),
			on="user_idx",
		)
	)
	violations = per_user.filter(
		(pl.col("train_max_ts") >= pl.col("val_ts")) | (pl.col("val_ts") >= pl.col("test_ts"))
	)
	if violations.height > 0:
		return _fail("split_chronology", f"Chronology violations for {violations.height} users")
	return _ok("split_chronology", "Per-user chronology is strictly train < val < test")


def validate_no_duplicate_interactions_across_splits(
	train: pl.DataFrame,
	val: pl.DataFrame,
	test: pl.DataFrame,
) -> ValidationResult:
	key_cols = ["user_idx", "item_idx", "timestamp"]
	with_split = pl.concat(
		[
			train.select(key_cols).with_columns(pl.lit("train").alias("split")),
			val.select(key_cols).with_columns(pl.lit("val").alias("split")),
			test.select(key_cols).with_columns(pl.lit("test").alias("split")),
		]
	)
	dupes = with_split.group_by(key_cols).agg(pl.n_unique("split").alias("split_count"))
	dupes = dupes.filter(pl.col("split_count") > 1)
	if dupes.height > 0:
		return _fail(
			"split_duplicate_interactions",
			f"Found {dupes.height} interactions duplicated across splits",
		)
	return _ok("split_duplicate_interactions", "No duplicated interactions across splits")


def validate_no_history_leakage(
	users: pl.DataFrame,
	val: pl.DataFrame,
	test: pl.DataFrame,
) -> ValidationResult:
	if "train_history_item_idx" not in users.columns:
		return _fail("history_leakage", "users frame missing train_history_item_idx")

	user_history = users.select(["user_idx", "train_history_item_idx"])

	def _check(frame: pl.DataFrame) -> int:
		joined = frame.select(["user_idx", "item_idx"]).join(
			user_history,
			on="user_idx",
			how="left",
		)
		leaked = joined.filter(pl.col("train_history_item_idx").list.contains(pl.col("item_idx")))
		return leaked.height

	val_leaks = _check(val)
	test_leaks = _check(test)
	if val_leaks or test_leaks:
		return _fail(
			"history_leakage",
			f"Found leakage rows: val={val_leaks}, test={test_leaks}",
		)
	return _ok("history_leakage", "No val/test target appears in per-user train histories")


def validate_genre_multi_hot_consistency(items: pl.DataFrame) -> ValidationResult:
	genre_cols = [col for col in items.columns if col.startswith("item_genre_")]
	if "genres_list" not in items.columns or not genre_cols:
		return _fail("genre_multi_hot", "Missing genres_list or item_genre_* columns")

	computed = items.select(
		pl.sum_horizontal([pl.col(col) for col in genre_cols]).alias("multi_hot_sum"),
		pl.col("genres_list").list.len().alias("genre_count"),
		pl.col("genres_list").list.contains("(no genres listed)").alias("has_no_genres"),
	)
	violations = computed.filter(
		(pl.col("has_no_genres") & (pl.col("multi_hot_sum") != 1))
		| ((~pl.col("has_no_genres")) & (pl.col("multi_hot_sum") != pl.col("genre_count")))
	)
	if violations.height > 0:
		return _fail("genre_multi_hot", f"Found {violations.height} genre encoding inconsistencies")
	return _ok("genre_multi_hot", "Genre multi-hot encoding is consistent with genres_list")


def run_processed_dataset_validation(
	output_dir: Path,
	parquet_files: Iterable[str],
) -> list[ValidationResult]:
	results: list[ValidationResult] = []
	results.append(validate_processed_files_loadable(output_dir, list(parquet_files)))
	if not results[-1].passed:
		return results

	train = pl.read_parquet(output_dir / "interactions_train.parquet")
	val = pl.read_parquet(output_dir / "interactions_val.parquet")
	test = pl.read_parquet(output_dir / "interactions_test.parquet")
	users = pl.read_parquet(output_dir / "users.parquet")
	items = pl.read_parquet(output_dir / "items.parquet")
	user_map = pl.read_parquet(output_dir / "user_id_map.parquet")
	item_map = pl.read_parquet(output_dir / "item_id_map.parquet")

	for split_name, frame in (("train", train), ("val", val), ("test", test)):
		results.append(
			validate_required_columns(
				frame,
				MANDATORY_INTERACTION_COLUMNS,
				f"required_columns_{split_name}",
			)
		)
		results.append(
			validate_no_nulls(
				frame,
				["user_idx", "item_idx", "timestamp", "explicit_rating"],
				f"null_check_{split_name}",
			)
		)

	results.append(validate_contiguous_ids(user_map, "user_idx"))
	results.append(validate_contiguous_ids(item_map, "item_idx"))
	results.append(validate_split_chronology(train, val, test))
	results.append(validate_no_duplicate_interactions_across_splits(train, val, test))
	results.append(validate_no_history_leakage(users, val, test))
	results.append(validate_genre_multi_hot_consistency(items))

	return results
