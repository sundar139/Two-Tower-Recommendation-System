"""Prepare deterministic MovieLens features and splits for retrieval training."""

from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl
import typer

from movie_recsys.config import load_data_config, load_project_config
from movie_recsys.data.features import build_item_features, build_user_features
from movie_recsys.data.preprocessing import (
	all_genres,
	build_positive_interactions,
	enrich_movies,
	filter_users_by_positive_count,
	load_movielens_tables,
	select_sample_users,
)
from movie_recsys.data.splits import (
	apply_id_maps,
	build_id_maps,
	build_user_histories,
	chronological_user_split,
)
from movie_recsys.utils.logging import get_logger
from movie_recsys.utils.reproducibility import set_global_seed

logger = get_logger(__name__)
app = typer.Typer(add_completion=False)


def _ensure_output_dir(output_dir: Path, force: bool) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	if force:
		for existing in output_dir.glob("*.parquet"):
			existing.unlink(missing_ok=True)
		for existing in output_dir.glob("*.json"):
			existing.unlink(missing_ok=True)


def _write_stats(path: Path, payload: dict) -> None:
	path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


@app.command()
def main(
	config: Path = typer.Option(Path("configs/data.yaml"), "--config"),
	sample_users: int | None = typer.Option(None, "--sample-users"),
	force: bool = typer.Option(False, "--force"),
	seed: int | None = typer.Option(None, "--seed"),
) -> None:
	project_cfg = load_project_config()
	data_cfg = load_data_config(config)
	effective_seed = seed if seed is not None else project_cfg.random_seed
	set_global_seed(effective_seed)

	raw_ml_dir = data_cfg.raw_data_dir / "ml-25m"
	if not raw_ml_dir.exists():
		raise typer.BadParameter(
			f"Raw dataset directory not found: {raw_ml_dir}. Run download script first."
		)

	tables = load_movielens_tables(raw_ml_dir)
	ratings = tables["ratings"]
	movies = enrich_movies(tables["movies"])
	tags = tables["tags"]
	genome_scores = tables["genome_scores"]
	genome_tags = tables["genome_tags"]

	raw_ratings_count = ratings.height

	positives = build_positive_interactions(ratings, data_cfg.positive_rating_threshold)
	filtered_positives = filter_users_by_positive_count(
		positives, data_cfg.min_positive_interactions_per_user
	)

	eligible_users = (
		filtered_positives.select("userId").unique().sort("userId").get_column("userId")
	)
	sample_size = sample_users if sample_users is not None else data_cfg.sample_users
	output_dir = data_cfg.processed_data_dir
	if sample_size:
		selected_users = select_sample_users(eligible_users, sample_size, effective_seed)
		filtered_positives = filtered_positives.join(
			pl.DataFrame({"userId": selected_users}), on="userId", how="inner"
		)
		ratings = ratings.join(pl.DataFrame({"userId": selected_users}), on="userId", how="inner")
		tags = tags.join(pl.DataFrame({"userId": selected_users}), on="userId", how="inner")
		output_dir = data_cfg.processed_data_dir / "sample"

	_ensure_output_dir(output_dir, force)

	train_raw, val_raw, test_raw = chronological_user_split(filtered_positives)
	user_map, item_map = build_id_maps(train_raw, val_raw, test_raw)

	train = apply_id_maps(train_raw, user_map, item_map)
	val = apply_id_maps(val_raw, user_map, item_map)
	test = apply_id_maps(test_raw, user_map, item_map)

	histories = build_user_histories(train)

	all_items = item_map.join(movies, on="movieId", how="left")
	genre_vocab = all_genres(movies)

	users = build_user_features(
		ratings,
		train,
		movies,
		tags,
		histories,
		genre_vocab,
		data_cfg.positive_rating_threshold,
	)
	items = build_item_features(
		ratings,
		filtered_positives,
		all_items,
		genome_scores,
		genome_tags,
		genre_vocab,
	)

	train.write_parquet(output_dir / data_cfg.output_files.interactions_train)
	val.write_parquet(output_dir / data_cfg.output_files.interactions_val)
	test.write_parquet(output_dir / data_cfg.output_files.interactions_test)
	users.write_parquet(output_dir / data_cfg.output_files.users)
	items.write_parquet(output_dir / data_cfg.output_files.items)
	user_map.write_parquet(output_dir / data_cfg.output_files.user_id_map)
	item_map.write_parquet(output_dir / data_cfg.output_files.item_id_map)
	histories.write_parquet(output_dir / data_cfg.output_files.user_histories)

	stats = {
		"raw_ratings_count": raw_ratings_count,
		"positive_interactions_count": positives.height,
		"users_before_filtering": positives.select("userId").unique().height,
		"users_after_filtering": filtered_positives.select("userId").unique().height,
		"movies_before_filtering": positives.select("movieId").unique().height,
		"movies_after_filtering": filtered_positives.select("movieId").unique().height,
		"train_count": train.height,
		"val_count": val.height,
		"test_count": test.height,
		"user_positive_interactions": {
			"min": int(filtered_positives.group_by("userId").len().get_column("len").min() or 0),
			"median": float(
				filtered_positives.group_by("userId").len().get_column("len").median() or 0
			),
			"max": int(filtered_positives.group_by("userId").len().get_column("len").max() or 0),
		},
		"timestamp_range": {
			"min": int(filtered_positives.get_column("timestamp").min() or 0),
			"max": int(filtered_positives.get_column("timestamp").max() or 0),
		},
		"genres": genre_vocab,
		"missing_values": {
			"train": {col: train.get_column(col).null_count() for col in train.columns},
			"users": {col: users.get_column(col).null_count() for col in users.columns},
			"items": {col: items.get_column(col).null_count() for col in items.columns},
		},
	}
	_write_stats(output_dir / data_cfg.output_files.dataset_stats, stats)

	logger.info("Prepared data written to %s", output_dir)
	logger.info("Rows: train=%s val=%s test=%s", train.height, val.height, test.height)


if __name__ == "__main__":
	app()
