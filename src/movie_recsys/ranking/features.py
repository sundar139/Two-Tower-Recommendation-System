"""Feature engineering utilities for ranking candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from movie_recsys.modeling.artifacts import save_json
from movie_recsys.ranking.candidates import EMBEDDING_FEATURE_COLUMNS
from movie_recsys.ranking.config import RankerConfig
from movie_recsys.training.config import load_retrieval_config

MANDATORY_FEATURE_COLUMNS = [
	"query_id",
	"user_idx",
	"item_idx",
	"split",
	"label",
	"target_item_idx",
	"residual_score",
	"residual_rank",
	"reciprocal_rank",
	"score_rank_interaction",
	"genre_affinity_dot",
	"genre_overlap_count",
	"max_genre_affinity",
	"user_item_popularity_gap",
	"user_item_year_distance",
]


def _candidate_paths(cfg: RankerConfig, *, sample: bool) -> dict[str, Path]:
	return {
		"train": cfg.candidate_path(split="train", sample=sample),
		"val": cfg.candidate_path(split="val", sample=sample),
		"test": cfg.candidate_path(split="test", sample=sample),
	}


def _feature_paths(cfg: RankerConfig, *, sample: bool) -> dict[str, Path]:
	return {
		"train": cfg.features_path(split="train", sample=sample),
		"val": cfg.features_path(split="val", sample=sample),
		"test": cfg.features_path(split="test", sample=sample),
	}


def _user_feature_columns(users_df: pl.DataFrame) -> list[str]:
	base = [
		"positive_rating_count",
		"total_rating_count",
		"mean_rating",
		"activity_span_days",
		"tag_count",
	]
	genre_cols = [name for name in users_df.columns if name.startswith("genre_affinity_")]
	passthrough = [name for name in base if name in users_df.columns]
	return ["user_idx", *passthrough, *sorted(genre_cols)]


def _item_feature_columns(items_df: pl.DataFrame) -> list[str]:
	base = [
		"rating_count",
		"positive_count",
		"mean_rating",
		"popularity_score",
		"release_year",
		"release_year_norm",
		"genome_tag_count",
		"genome_relevance_mean",
		"genome_relevance_max",
	]
	genre_cols = [name for name in items_df.columns if name.startswith("item_genre_")]
	passthrough = [name for name in base if name in items_df.columns]
	return ["item_idx", *passthrough, *sorted(genre_cols)]


def _year_bucket_expr(release_year_expr: pl.Expr) -> pl.Expr:
	return (
		pl.when(release_year_expr < 1950)
		.then(0)
		.when(release_year_expr < 1960)
		.then(1)
		.when(release_year_expr < 1970)
		.then(2)
		.when(release_year_expr < 1980)
		.then(3)
		.when(release_year_expr < 1990)
		.then(4)
		.when(release_year_expr < 2000)
		.then(5)
		.when(release_year_expr < 2010)
		.then(6)
		.when(release_year_expr < 2020)
		.then(7)
		.otherwise(8)
	).cast(pl.Int32)


def _genre_interaction_exprs(frame: pl.DataFrame) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
	user_genre_cols = [name for name in frame.columns if name.startswith("genre_affinity_")]
	item_genre_cols = [name for name in frame.columns if name.startswith("item_genre_")]
	item_lookup = {name.removeprefix("item_genre_"): name for name in item_genre_cols}

	pairs: list[tuple[str, str]] = []
	for user_col in user_genre_cols:
		suffix = user_col.removeprefix("genre_affinity_")
		item_col = item_lookup.get(suffix)
		if item_col is not None:
			pairs.append((user_col, item_col))

	if not pairs:
		zero = pl.lit(0.0)
		return zero.alias("genre_affinity_dot"), zero.alias("genre_overlap_count"), zero.alias(
			"max_genre_affinity"
		)

	dot_terms = [pl.col(u).cast(pl.Float32) * pl.col(i).cast(pl.Float32) for u, i in pairs]
	overlap_terms = [
		(pl.col(i).cast(pl.Float32) * (pl.col(u) > 0).cast(pl.Float32)) for u, i in pairs
	]
	affinity_terms = [pl.col(u).cast(pl.Float32) * pl.col(i).cast(pl.Float32) for u, i in pairs]

	dot_expr = pl.sum_horizontal(dot_terms).alias("genre_affinity_dot")
	overlap_expr = pl.sum_horizontal(overlap_terms).alias("genre_overlap_count")
	max_expr = pl.max_horizontal(affinity_terms).alias("max_genre_affinity")
	return dot_expr, overlap_expr, max_expr


def _sanitize_numeric(frame: pl.DataFrame) -> pl.DataFrame:
	numeric_cols = [
		name
		for name, dtype in frame.schema.items()
		if dtype.is_numeric() and name not in {"label", "user_idx", "item_idx", "target_item_idx"}
	]
	expressions = [
		pl.col(name)
		.cast(pl.Float32)
		.fill_nan(0.0)
		.fill_null(0.0)
		.alias(name)
		for name in numeric_cols
	]
	return frame.with_columns(expressions)


def validate_feature_frame(frame: pl.DataFrame, *, split: str) -> None:
	missing = [name for name in MANDATORY_FEATURE_COLUMNS if name not in frame.columns]
	if missing:
		msg = f"Missing mandatory feature columns for split '{split}': {', '.join(missing)}"
		raise ValueError(msg)

	labels_ok = frame.filter(~pl.col("label").is_in([0, 1])).height == 0
	if not labels_ok:
		msg = f"Split '{split}' contains non-binary labels"
		raise ValueError(msg)

	mandatory_numeric = [
		name
		for name in MANDATORY_FEATURE_COLUMNS
		if name not in {"query_id", "split"}
	]
	has_nulls = frame.select(
		[pl.col(name).is_null().any().alias(name) for name in mandatory_numeric]
	).row(0)
	if any(bool(value) for value in has_nulls):
		msg = f"Split '{split}' has null values in mandatory numeric feature columns"
		raise ValueError(msg)


def _signature(frame: pl.DataFrame) -> str:
	payload = (
		frame.select(["query_id", "item_idx", "label", "residual_rank"]).sort(
			["query_id", "residual_rank", "item_idx"]
		)
	).write_json()
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_split_features(
	*,
	candidates: pl.DataFrame,
	users_df: pl.DataFrame,
	items_df: pl.DataFrame,
	use_frozen_features: bool,
) -> tuple[pl.DataFrame, list[str]]:
	user_cols = _user_feature_columns(users_df)
	item_cols = _item_feature_columns(items_df)

	users_small = users_df.select(user_cols)
	items_small = items_df.select(item_cols)

	frame = candidates.join(users_small, on="user_idx", how="left").join(
		items_small,
		on="item_idx",
		how="left",
	)

	release_year_expr = (
		pl.col("release_year").cast(pl.Float32)
		if "release_year" in frame.columns
		else (pl.col("release_year_norm").cast(pl.Float32) * 200.0 + 1900.0)
	)
	release_year_norm_expr = (
		pl.col("release_year_norm").cast(pl.Float32)
		if "release_year_norm" in frame.columns
		else ((release_year_expr - 1900.0) / 200.0).cast(pl.Float32)
	)

	dot_expr, overlap_expr, max_expr = _genre_interaction_exprs(frame)

	popularity_gap_expr = (
		(pl.col("popularity_score") - pl.col("avg_liked_popularity_score")).cast(pl.Float32)
		if "avg_liked_popularity_score" in frame.columns
		else pl.lit(0.0).cast(pl.Float32)
	)
	year_distance_expr = (
		(release_year_expr - pl.col("avg_liked_release_year")).abs().cast(pl.Float32)
		if "avg_liked_release_year" in frame.columns
		else pl.lit(0.0).cast(pl.Float32)
	)

	feature_frame = frame.with_columns(
		[
			(1.0 / pl.col("residual_rank").cast(pl.Float32)).alias("reciprocal_rank"),
			(
				pl.col("residual_score").cast(pl.Float32)
				* pl.col("residual_rank").cast(pl.Float32)
			).alias("score_rank_interaction"),
			release_year_norm_expr.alias("release_year_norm"),
			_year_bucket_expr(release_year_expr).alias("year_bucket"),
			dot_expr,
			overlap_expr,
			max_expr,
			popularity_gap_expr.alias("user_item_popularity_gap"),
			year_distance_expr.alias("user_item_year_distance"),
		]
	)

	if not use_frozen_features:
		present = [name for name in EMBEDDING_FEATURE_COLUMNS if name in feature_frame.columns]
		if present:
			feature_frame = feature_frame.drop(present)

	feature_frame = feature_frame.with_columns(
		[
			pl.col("label").cast(pl.Int8),
			pl.col("user_idx").cast(pl.Int32),
			pl.col("item_idx").cast(pl.Int32),
			pl.col("target_item_idx").cast(pl.Int32),
			pl.col("residual_rank").cast(pl.Int32),
			pl.col("user_history_length").cast(pl.Int32),
			pl.col("timestamp_context").cast(pl.Int64),
		]
	)
	feature_frame = _sanitize_numeric(feature_frame)
	feature_frame = feature_frame.sort(["query_id", "residual_rank", "item_idx"])

	metadata_cols = {
		"query_id",
		"user_idx",
		"item_idx",
		"split",
		"label",
		"target_item_idx",
		"candidate_source",
		"target_injected",
		"timestamp_context",
		"user_history_length",
	}
	feature_columns = [
		name
		for name, dtype in feature_frame.schema.items()
		if dtype.is_numeric() and name not in metadata_cols
	]

	return feature_frame, sorted(feature_columns)


def build_ranker_features(
	ranker_cfg: RankerConfig,
	*,
	sample: bool,
) -> dict[str, Path]:
	retrieval_cfg = load_retrieval_config(ranker_cfg.retriever_config, sample=sample)
	users_df = pl.read_parquet(retrieval_cfg.users_path)
	items_df = pl.read_parquet(retrieval_cfg.items_path)

	candidate_paths = _candidate_paths(ranker_cfg, sample=sample)
	feature_paths = _feature_paths(ranker_cfg, sample=sample)

	outputs: dict[str, Path] = {}
	for split in ["train", "val", "test"]:
		candidates = pl.read_parquet(candidate_paths[split])
		feature_frame, feature_columns = _build_split_features(
			candidates=candidates,
			users_df=users_df,
			items_df=items_df,
			use_frozen_features=ranker_cfg.use_frozen_retrieval_embeddings,
		)

		validate_feature_frame(feature_frame, split=split)
		output_path = feature_paths[split]
		output_path.parent.mkdir(parents=True, exist_ok=True)
		feature_frame.write_parquet(output_path)

		meta_path = output_path.with_suffix(".meta.json")
		save_json(
			meta_path,
			{
				"split": split,
				"sample": sample,
				"row_count": int(feature_frame.height),
				"query_count": int(feature_frame.select(pl.col("query_id").n_unique()).item()),
				"feature_columns": feature_columns,
				"deterministic_signature": _signature(feature_frame),
				"mandatory_columns": MANDATORY_FEATURE_COLUMNS,
			},
		)

		outputs[f"{split}_features"] = output_path
		outputs[f"{split}_meta"] = meta_path

	return outputs
