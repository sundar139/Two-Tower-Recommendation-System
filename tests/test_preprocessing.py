from __future__ import annotations

from pathlib import Path

import polars as pl

from movie_recsys.data.preprocessing import (
	build_positive_interactions,
	filter_users_by_positive_count,
	parse_genres,
	parse_release_year,
	select_sample_users,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ratings() -> pl.DataFrame:
	return pl.read_csv(FIXTURES_DIR / "tiny_ratings.csv")


def test_rating_to_positive_conversion() -> None:
	positives = build_positive_interactions(_ratings(), 4.0)
	assert positives.height == 8
	assert positives.select(pl.col("is_positive").all()).item() is True


def test_release_year_parsing() -> None:
	assert parse_release_year("Toy Story (1995)") == 1995
	assert parse_release_year("Unknown Title") is None


def test_genre_parsing() -> None:
	assert parse_genres("Action|Crime|Thriller") == ["Action", "Crime", "Thriller"]
	assert parse_genres("(no genres listed)") == ["(no genres listed)"]


def test_min_positive_user_filter() -> None:
	positives = build_positive_interactions(_ratings(), 4.0)
	filtered = filter_users_by_positive_count(positives, 3)
	assert set(filtered.get_column("userId").unique().to_list()) == {10, 11}


def test_deterministic_sample_selection() -> None:
	users = pl.Series("userId", [1, 2, 3, 4, 5, 6])
	a = select_sample_users(users, sample_size=3, seed=42)
	b = select_sample_users(users, sample_size=3, seed=42)
	assert a.to_list() == b.to_list()
