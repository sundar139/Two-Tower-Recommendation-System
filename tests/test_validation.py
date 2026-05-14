from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from movie_recsys.data.validation import (
	run_processed_dataset_validation,
	validate_contiguous_ids,
	validate_genre_multi_hot_consistency,
)


def test_validation_failure_for_non_contiguous_ids() -> None:
	bad_map = pl.DataFrame({"user_idx": [0, 2], "userId": [10, 11]})
	result = validate_contiguous_ids(bad_map, "user_idx")
	assert result.passed is False


def test_validation_failure_for_bad_genre_multihot() -> None:
	items = pl.DataFrame(
		{
			"item_idx": [0],
			"genres_list": [["Action", "Crime"]],
			"item_genre_action": [1],
			"item_genre_crime": [0],
		}
	)
	result = validate_genre_multi_hot_consistency(items)
	assert result.passed is False


def test_processed_validation_detects_corrupted_split(tmp_path: Path) -> None:
	train = pl.DataFrame(
		{
			"userId": [10],
			"movieId": [1],
			"explicit_rating": [5.0],
			"timestamp": [100],
			"user_idx": [0],
			"item_idx": [0],
		}
	)
	val = pl.DataFrame(
		{
			"userId": [10],
			"movieId": [2],
			"explicit_rating": [4.0],
			"timestamp": [90],
			"user_idx": [0],
			"item_idx": [1],
		}
	)
	test = pl.DataFrame(
		{
			"userId": [10],
			"movieId": [3],
			"explicit_rating": [4.0],
			"timestamp": [110],
			"user_idx": [0],
			"item_idx": [2],
		}
	)
	users = pl.DataFrame(
		{
			"user_idx": [0],
			"original_userId": [10],
			"train_history_item_idx": [[0]],
		}
	)
	items = pl.DataFrame(
		{
			"item_idx": [0, 1, 2],
			"genres_list": [["Action"], ["Comedy"], ["Drama"]],
			"item_genre_action": [1, 0, 0],
			"item_genre_comedy": [0, 1, 0],
			"item_genre_drama": [0, 0, 1],
		}
	)
	user_map = pl.DataFrame({"user_idx": [0], "userId": [10]})
	item_map = pl.DataFrame({"item_idx": [0, 1, 2], "movieId": [1, 2, 3]})

	train.write_parquet(tmp_path / "interactions_train.parquet")
	val.write_parquet(tmp_path / "interactions_val.parquet")
	test.write_parquet(tmp_path / "interactions_test.parquet")
	users.write_parquet(tmp_path / "users.parquet")
	items.write_parquet(tmp_path / "items.parquet")
	user_map.write_parquet(tmp_path / "user_id_map.parquet")
	item_map.write_parquet(tmp_path / "item_id_map.parquet")

	results = run_processed_dataset_validation(
		tmp_path,
		[
			"interactions_train.parquet",
			"interactions_val.parquet",
			"interactions_test.parquet",
			"users.parquet",
			"items.parquet",
			"user_id_map.parquet",
			"item_id_map.parquet",
		],
	)
	chronology_result = [r for r in results if r.name == "split_chronology"][0]
	assert chronology_result.passed is False


def test_verify_environment_structured_output_exists() -> None:
	root = Path(__file__).resolve().parents[1]
	script_path = root / "scripts" / "verify_environment.py"
	spec = importlib.util.spec_from_file_location("verify_environment", script_path)
	assert spec is not None and spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)

	results = module.collect_results(include_ollama=False)
	payload = json.loads(module.orjson.dumps([module.asdict(r) for r in results]))
	assert isinstance(payload, list)
	assert payload[0]["name"] == "python_version"
