from __future__ import annotations

from pathlib import Path

import polars as pl

from movie_recsys.data.features import (
    build_genre_multi_hot_columns,
    build_item_features,
    build_user_features,
)
from movie_recsys.data.preprocessing import (
    all_genres,
    build_positive_interactions,
    enrich_movies,
    filter_users_by_positive_count,
    parse_genres,
)
from movie_recsys.data.splits import (
    apply_id_maps,
    build_id_maps,
    build_user_histories,
    chronological_user_split,
)
from movie_recsys.data.validation import (
    validate_contiguous_ids,
    validate_no_duplicate_interactions_across_splits,
    validate_no_history_leakage,
)


def _build_fixture_500() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    rating_rows: list[dict[str, int | float]] = []
    users = list(range(1, 101))
    for user_id in users:
        for idx in range(5):
            movie_id = ((user_id * 7 + idx) % 80) + 1
            rating = [5.0, 4.5, 4.0, 3.5, 2.0][idx]
            rating_rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": rating,
                    "timestamp": user_id * 1_000 + idx,
                }
            )

    ratings = pl.DataFrame(rating_rows)

    movies = pl.DataFrame(
        {
            "movieId": list(range(1, 81)),
            "title": [f"Movie {movie_id} ({1990 + (movie_id % 30)})" for movie_id in range(1, 81)],
            "genres": [
                ["Action|Comedy", "Drama|Thriller", "Sci-Fi", "(no genres listed)"][movie_id % 4]
                for movie_id in range(1, 81)
            ],
        }
    )

    tags = pl.DataFrame(
        {
            "userId": users,
            "movieId": [((user_id * 7) % 80) + 1 for user_id in users],
            "tag": ["tagged"] * len(users),
            "timestamp": [user_id * 1_000 + 10 for user_id in users],
        }
    )

    genome_tags = pl.DataFrame(
        {
            "tagId": [1, 2, 3],
            "tag": ["action", "character", "mood"],
        }
    )

    genome_rows: list[dict[str, int | float]] = []
    for movie_id in range(1, 81):
        for tag_id in [1, 2, 3]:
            genome_rows.append(
                {
                    "movieId": movie_id,
                    "tagId": tag_id,
                    "relevance": float(((movie_id + tag_id) % 10) / 10.0),
                }
            )
    genome_scores = pl.DataFrame(genome_rows)

    return ratings, movies, tags, genome_scores, genome_tags


def test_pipeline_smoke_500_rows(tmp_path: Path) -> None:
    ratings, movies_raw, tags, genome_scores, genome_tags = _build_fixture_500()

    positives = build_positive_interactions(ratings, 4.0)
    assert ratings.height == 500
    assert positives.height == 300
    assert parse_genres("Action|Comedy") == ["Action", "Comedy"]

    eligible_positives = filter_users_by_positive_count(positives, 3)
    train_raw, val_raw, test_raw = chronological_user_split(eligible_positives)

    chronology = (
        train_raw.group_by("userId").agg(pl.col("timestamp").max().alias("train_max_ts"))
        .join(
            val_raw.select(["userId", "timestamp"]).rename({"timestamp": "val_ts"}),
            on="userId",
        )
        .join(
            test_raw.select(["userId", "timestamp"]).rename({"timestamp": "test_ts"}),
            on="userId",
        )
    )
    assert chronology.filter(
        (pl.col("train_max_ts") >= pl.col("val_ts")) | (pl.col("val_ts") >= pl.col("test_ts"))
    ).height == 0

    user_map, item_map = build_id_maps(train_raw, val_raw, test_raw)
    train = apply_id_maps(train_raw, user_map, item_map)
    val = apply_id_maps(val_raw, user_map, item_map)
    test = apply_id_maps(test_raw, user_map, item_map)

    assert validate_contiguous_ids(user_map, "user_idx").passed is True
    assert validate_contiguous_ids(item_map, "item_idx").passed is True

    # Per-user recommendation splitting expects overlap of users/items across splits.
    # Leakage checks must be at exact-interaction and same-user target levels.
    assert validate_no_duplicate_interactions_across_splits(train, val, test).passed is True

    histories = build_user_histories(train)
    users_min = (
        train.select(["user_idx", "userId"])
        .unique()
        .rename({"userId": "original_userId"})
        .sort("user_idx")
        .join(histories.select(["user_idx", "train_history_item_idx"]), on="user_idx", how="left")
    )
    assert validate_no_history_leakage(users_min, val, test).passed is True

    movies = enrich_movies(movies_raw)
    genres = all_genres(movies)
    encoded_movies = build_genre_multi_hot_columns(movies, "genres_list", genres, "item")

    genre_cols = [name for name in encoded_movies.columns if name.startswith("item_genre_")]
    sums = encoded_movies.select(
        pl.sum_horizontal([pl.col(name) for name in genre_cols]).alias("sum_hot"),
        pl.col("genres_list").list.len().alias("genre_len"),
        pl.col("genres_list").list.contains("(no genres listed)").alias("no_genres"),
    )
    assert sums.filter(
        (pl.col("no_genres") & (pl.col("sum_hot") != 1))
        | ((~pl.col("no_genres")) & (pl.col("sum_hot") != pl.col("genre_len")))
    ).height == 0

    users = build_user_features(ratings, train, movies, tags, histories, genres, 4.0)
    items_with_idx = item_map.join(movies, on="movieId", how="left")
    items = build_item_features(
        ratings,
        eligible_positives,
        items_with_idx,
        genome_scores,
        genome_tags,
        genres,
    )

    user_mandatory = ["user_idx", "original_userId", "total_rating_count", "positive_rating_count"]
    item_mandatory = ["item_idx", "original_movieId", "rating_count", "positive_count"]
    assert all(users.get_column(col).null_count() == 0 for col in user_mandatory)
    assert all(items.get_column(col).null_count() == 0 for col in item_mandatory)

    train_path = tmp_path / "interactions_train.parquet"
    val_path = tmp_path / "interactions_val.parquet"
    test_path = tmp_path / "interactions_test.parquet"
    users_path = tmp_path / "users.parquet"
    items_path = tmp_path / "items.parquet"
    user_map_path = tmp_path / "user_id_map.parquet"
    item_map_path = tmp_path / "item_id_map.parquet"

    train.write_parquet(train_path)
    val.write_parquet(val_path)
    test.write_parquet(test_path)
    users.write_parquet(users_path)
    items.write_parquet(items_path)
    user_map.write_parquet(user_map_path)
    item_map.write_parquet(item_map_path)

    train_loaded = pl.read_parquet(train_path)
    users_loaded = pl.read_parquet(users_path)
    items_loaded = pl.read_parquet(items_path)

    assert train_loaded.schema["explicit_rating"] == pl.Float64
    assert train_loaded.schema["timestamp"] in {pl.Int64, pl.Int32}
    assert train_loaded.schema["user_idx"] in {pl.UInt32, pl.UInt64, pl.Int64, pl.Int32}
    assert users_loaded.schema["train_history_item_idx"] == pl.List(train_loaded.schema["item_idx"])
    assert items_loaded.schema["top_genome_tags"].is_nested()
