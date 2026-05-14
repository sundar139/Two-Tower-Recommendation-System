from __future__ import annotations

from pathlib import Path

import polars as pl

from movie_recsys.data.features import build_genre_multi_hot_columns
from movie_recsys.data.preprocessing import all_genres, enrich_movies

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_genre_multihot_encoding() -> None:
	movies = enrich_movies(pl.read_csv(FIXTURES_DIR / "tiny_movies.csv"))
	genres = all_genres(movies)
	encoded = build_genre_multi_hot_columns(movies, "genres_list", genres, "item")

	row = encoded.filter(pl.col("movieId") == 1).row(0, named=True)
	assert row["item_genre_adventure"] == 1
	assert row["item_genre_action"] == 0


def test_no_genres_multihot_column() -> None:
	movies = enrich_movies(pl.read_csv(FIXTURES_DIR / "tiny_movies.csv"))
	genres = all_genres(movies)
	encoded = build_genre_multi_hot_columns(movies, "genres_list", genres, "item")
	row = encoded.filter(pl.col("movieId") == 4).row(0, named=True)
	assert row["item_genre_(no_genres_listed)"] == 1
