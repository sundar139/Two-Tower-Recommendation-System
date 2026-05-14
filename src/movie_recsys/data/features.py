"""Feature engineering for users and items."""

from __future__ import annotations

import polars as pl


def build_genre_multi_hot_columns(
	frame: pl.DataFrame, genres_column: str, all_genres: list[str], prefix: str
) -> pl.DataFrame:
	expressions: list[pl.Expr] = []
	for genre in all_genres:
		safe_name = genre.lower().replace(" ", "_").replace("-", "_")
		col_name = f"{prefix}_genre_{safe_name}"
		expressions.append(pl.col(genres_column).list.contains(genre).cast(pl.Int8).alias(col_name))
	return frame.with_columns(expressions)


def build_user_features(
	ratings: pl.DataFrame,
	train: pl.DataFrame,
	movies: pl.DataFrame,
	tags: pl.DataFrame,
	histories: pl.DataFrame,
	all_genres: list[str],
	positive_rating_threshold: float,
) -> pl.DataFrame:
	rating_stats = ratings.group_by("userId").agg(
		pl.len().alias("total_rating_count"),
		pl.col("rating").mean().alias("mean_rating"),
		(pl.col("rating") >= positive_rating_threshold).sum().alias("positive_rating_count"),
		pl.col("timestamp").min().alias("first_timestamp"),
		pl.col("timestamp").max().alias("last_timestamp"),
	)

	span_expr = ((pl.col("last_timestamp") - pl.col("first_timestamp")) / 86400).alias(
		"activity_span_days"
	)

	movie_genres = movies.select(["movieId", "genres_list"])
	genre_affinity = (
		train.select(["userId", "movieId"])
		.join(movie_genres, on="movieId", how="left")
		.explode("genres_list")
		.group_by(["userId", "genres_list"])
		.agg(pl.len().alias("genre_count"))
	)

	user_base = train.select(["user_idx", "userId"]).unique().sort("user_idx")
	features = user_base.join(rating_stats, on="userId", how="left").with_columns(span_expr)

	for genre in all_genres:
		safe_name = genre.lower().replace(" ", "_").replace("-", "_")
		feature_name = f"genre_affinity_{safe_name}"
		per_genre = genre_affinity.filter(pl.col("genres_list") == genre).select(
			["userId", pl.col("genre_count").alias(feature_name)]
		)
		features = features.join(per_genre, on="userId", how="left").with_columns(
			pl.col(feature_name).fill_null(0)
		)

	tag_counts = tags.group_by("userId").agg(pl.len().alias("tag_count"))
	features = features.join(tag_counts, on="userId", how="left").with_columns(
		pl.col("tag_count").fill_null(0)
	)

	return (
		features.join(histories, on=["user_idx", "userId"], how="left")
		.rename({"userId": "original_userId"})
		.sort("user_idx")
	)


def build_item_features(
	ratings: pl.DataFrame,
	positives: pl.DataFrame,
	items_with_idx: pl.DataFrame,
	genome_scores: pl.DataFrame,
	genome_tags: pl.DataFrame,
	all_genres: list[str],
) -> pl.DataFrame:
	rating_stats = ratings.group_by("movieId").agg(
		pl.len().alias("rating_count"),
		pl.col("rating").mean().alias("mean_rating"),
	)
	positive_stats = positives.group_by("movieId").agg(pl.len().alias("positive_count"))

	max_rating_count_raw = rating_stats.select(pl.col("rating_count").max()).item()
	if isinstance(max_rating_count_raw, (int, float)):
		max_rating_count = max(float(max_rating_count_raw), 1.0)
	else:
		max_rating_count = 1.0
	popularity = rating_stats.with_columns(
		(pl.col("rating_count") / max_rating_count).alias("popularity_score")
	)

	tagged = genome_scores.join(genome_tags, on="tagId", how="left")
	genome_summary = tagged.group_by("movieId").agg(
		pl.len().alias("genome_tag_count"),
		pl.col("relevance").mean().alias("genome_relevance_mean"),
		pl.col("relevance").max().alias("genome_relevance_max"),
		pl.struct(["tag", "relevance"])
		.sort_by("relevance", descending=True)
		.head(5)
		.alias("top_genome_tags"),
	)

	base = (
		items_with_idx.join(rating_stats, on="movieId", how="left")
		.join(positive_stats, on="movieId", how="left")
		.join(popularity.select(["movieId", "popularity_score"]), on="movieId", how="left")
		.join(genome_summary, on="movieId", how="left")
		.with_columns(
			pl.col("rating_count").fill_null(0),
			pl.col("positive_count").fill_null(0),
			pl.col("mean_rating").fill_null(0.0),
			pl.col("popularity_score").fill_null(0.0),
			pl.col("genome_tag_count").fill_null(0),
			pl.col("genome_relevance_mean").fill_null(0.0),
			pl.col("genome_relevance_max").fill_null(0.0),
		)
		.rename({"movieId": "original_movieId"})
	)

	base = build_genre_multi_hot_columns(base, "genres_list", all_genres, "item")
	return base.sort("item_idx")
