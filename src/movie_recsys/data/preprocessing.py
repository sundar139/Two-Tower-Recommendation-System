"""Deterministic preprocessing for MovieLens interactions and metadata."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

YEAR_PATTERN = re.compile(r"\((\d{4})\)$")


def parse_release_year(title: str) -> int | None:
	match = YEAR_PATTERN.search(title.strip())
	if not match:
		return None
	return int(match.group(1))


def parse_genres(raw_genres: str) -> list[str]:
	if raw_genres == "(no genres listed)":
		return ["(no genres listed)"]
	return [genre.strip() for genre in raw_genres.split("|") if genre.strip()]


def year_bucket(year: int | None) -> str:
	if year is None:
		return "unknown"
	decade = (year // 10) * 10
	return f"{decade}s"


def load_movielens_tables(raw_ml_dir: Path) -> dict[str, pl.DataFrame]:
	return {
		"ratings": pl.read_csv(raw_ml_dir / "ratings.csv"),
		"movies": pl.read_csv(raw_ml_dir / "movies.csv"),
		"tags": pl.read_csv(raw_ml_dir / "tags.csv"),
		"genome_scores": pl.read_csv(raw_ml_dir / "genome-scores.csv"),
		"genome_tags": pl.read_csv(raw_ml_dir / "genome-tags.csv"),
		"links": pl.read_csv(raw_ml_dir / "links.csv"),
	}


def build_positive_interactions(ratings: pl.DataFrame, threshold: float) -> pl.DataFrame:
	return (
		ratings.with_columns(
			(pl.col("rating") >= threshold).alias("is_positive"),
			pl.col("rating").cast(pl.Float64).alias("explicit_rating"),
		)
		.filter(pl.col("is_positive"))
		.sort(["userId", "timestamp", "movieId"])
		.with_columns(pl.int_range(0, pl.len()).over("userId").alias("_event_seq"))
		.with_columns(
			(pl.col("timestamp").cast(pl.Int64) * 1000 + pl.col("_event_seq")).alias("timestamp")
		)
		.drop("_event_seq")
	)


def filter_users_by_positive_count(
	positives: pl.DataFrame, min_positive_interactions_per_user: int
) -> pl.DataFrame:
	counts = positives.group_by("userId").agg(pl.len().alias("positive_count"))
	eligible = counts.filter(pl.col("positive_count") >= min_positive_interactions_per_user)
	return positives.join(eligible.select("userId"), on="userId", how="inner")


def select_sample_users(eligible_user_ids: pl.Series, sample_size: int, seed: int) -> pl.Series:
	all_users = sorted(eligible_user_ids.unique().to_list())
	if sample_size >= len(all_users):
		return pl.Series("userId", all_users)
	rng = np.random.default_rng(seed)
	sample = sorted(rng.choice(np.array(all_users), size=sample_size, replace=False).tolist())
	return pl.Series("userId", sample)


def enrich_movies(movies: pl.DataFrame) -> pl.DataFrame:
	titles = movies.get_column("title").to_list()
	genres_raw = movies.get_column("genres").to_list()

	years = [parse_release_year(str(title)) for title in titles]
	buckets = [year_bucket(year) for year in years]
	genre_lists = [parse_genres(str(g)) for g in genres_raw]

	return movies.with_columns(
		pl.Series("release_year", years, dtype=pl.Int32),
		pl.Series("year_bucket", buckets, dtype=pl.Utf8),
		pl.Series("genres_list", genre_lists, dtype=pl.List(pl.Utf8)),
	)


def all_genres(movies: pl.DataFrame) -> list[str]:
	genre_values = movies.get_column("genres_list").to_list()
	values: set[str] = set()
	for genres in genre_values:
		values.update(genres)
	return sorted(values)
