"""Feature engineering utilities for ranking candidates."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl

from movie_recsys.modeling.artifacts import save_json
from movie_recsys.ranking.candidates import EMBEDDING_FEATURE_COLUMNS
from movie_recsys.ranking.config import RankerConfig
from movie_recsys.training.config import load_retrieval_config
from movie_recsys.utils.system import log_memory_status, should_stop_for_memory

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

METADATA_ONLY_COLUMNS = {
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

FIXED_MODEL_FEATURE_COLUMNS = {
	"residual_score",
	"residual_rank",
	"reciprocal_rank",
	"score_rank_interaction",
	"user_item_popularity_gap",
	"user_item_year_distance",
	"positive_rating_count",
	"total_rating_count",
	"mean_rating",
	"activity_span_days",
	"tag_count",
	"rating_count",
	"positive_count",
	"popularity_score",
	"release_year",
	"release_year_norm",
	"year_bucket",
	"genome_tag_count",
	"genome_relevance_mean",
	"genome_relevance_max",
	"genre_affinity_dot",
	"genre_overlap_count",
	"max_genre_affinity",
}

PREFIX_MODEL_FEATURE_ALLOWLIST = (
	"genre_affinity_",
	"item_genre_",
)

UTC_TZ = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


def _utc_now_iso() -> str:
	return datetime.now(UTC_TZ).isoformat()


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


def _normalize_splits(splits: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
	if splits is None:
		return ("train", "val", "test")
	allowed = {"train", "val", "test"}
	normalized = tuple(split.strip().lower() for split in splits if split.strip())
	if not normalized:
		msg = "At least one split must be requested"
		raise ValueError(msg)
	unknown = sorted(set(normalized) - allowed)
	if unknown:
		msg = f"Unsupported split values: {', '.join(unknown)}"
		raise ValueError(msg)
	return tuple(split for split in ("train", "val", "test") if split in normalized)


def _feature_config_hash(
	*,
	ranker_cfg: RankerConfig,
	feature_shard_size: int,
	sample: bool,
	splits: tuple[str, ...],
) -> str:
	payload = {
		"sample": sample,
		"splits": list(splits),
		"feature_shard_size": feature_shard_size,
		"negative_samples_per_positive": ranker_cfg.negative_samples_per_positive,
		"use_frozen_retrieval_embeddings": ranker_cfg.use_frozen_retrieval_embeddings,
		"retriever_config": str(ranker_cfg.retriever_config),
		"retriever_checkpoint": str(ranker_cfg.retriever_checkpoint),
	}
	encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def _feature_manifest_path(ranker_cfg: RankerConfig, *, sample: bool) -> Path:
	return ranker_cfg.feature_manifest_path(sample=sample)


def _load_manifest(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if not isinstance(payload, dict):
		msg = f"Invalid feature manifest format in {path}"
		raise ValueError(msg)
	return cast(dict[str, Any], payload)


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)
		handle.write("\n")


def load_feature_manifest(ranker_cfg: RankerConfig, *, sample: bool) -> dict[str, Any]:
	"""Load ranker feature manifest for sample/full scope."""

	return _load_manifest(_feature_manifest_path(ranker_cfg, sample=sample))


def resolve_feature_shard_paths(
	ranker_cfg: RankerConfig,
	*,
	split: str,
	sample: bool,
) -> list[Path]:
	"""Return ranker feature shard paths for the requested split."""

	if sample:
		path = ranker_cfg.features_path(split=split, sample=True)
		return [path] if path.exists() else []

	manifest = load_feature_manifest(ranker_cfg, sample=False)
	splits_payload = manifest.get("splits", {})
	if not isinstance(splits_payload, dict):
		return []
	split_payload = splits_payload.get(split, {})
	if not isinstance(split_payload, dict):
		return []
	shard_paths = split_payload.get("shard_paths", [])
	if not isinstance(shard_paths, list):
		return []

	paths = [Path(str(value)) for value in shard_paths]
	existing = [path for path in paths if path.exists()]
	if existing:
		return existing

	legacy = ranker_cfg.features_path(split=split, sample=False)
	return [legacy] if legacy.exists() else []


def resolve_feature_columns_from_artifacts(
	ranker_cfg: RankerConfig,
	*,
	sample: bool,
	fallback_split: str = "train",
) -> list[str]:
	manifest = load_feature_manifest(ranker_cfg, sample=sample)
	feature_columns = manifest.get("feature_columns")
	if isinstance(feature_columns, list) and feature_columns:
		return [str(name) for name in feature_columns]

	paths = resolve_feature_shard_paths(ranker_cfg, split=fallback_split, sample=sample)
	if not paths:
		return []
	frame = pl.read_parquet(paths[0], n_rows=2048)
	use_frozen_features = any(
		name in frame.columns
		for name in [
			"frozen_emb_dot",
			"frozen_emb_prod_mean",
			"frozen_emb_prod_max",
			"frozen_emb_prod_min",
			"frozen_emb_prod_std",
			"frozen_emb_absdiff_mean",
			"frozen_emb_absdiff_max",
			"frozen_emb_absdiff_min",
			"frozen_emb_absdiff_std",
		]
	)
	return resolve_feature_allowlist(frame, use_frozen_features=use_frozen_features)


def _candidate_chunk_paths(ranker_cfg: RankerConfig, *, split: str, sample: bool) -> list[Path]:
	chunk_dir = ranker_cfg.candidate_dir_for_scope(sample=sample) / "chunks" / split
	if chunk_dir.exists():
		chunk_paths = sorted(chunk_dir.glob("chunk_*.parquet"))
		if chunk_paths:
			return chunk_paths
	candidate_path = ranker_cfg.candidate_path(split=split, sample=sample)
	return [candidate_path] if candidate_path.exists() else []


def _frame_schema_payload(frame: pl.DataFrame) -> dict[str, str]:
	return {name: str(dtype) for name, dtype in frame.schema.items()}


def _write_feature_shard(
	*,
	frame: pl.DataFrame,
	split_dir: Path,
	part_index: int,
) -> dict[str, Any]:
	output_path = split_dir / f"part_{part_index:06d}.parquet"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	frame.write_parquet(output_path)
	row_count = int(frame.height)
	query_count = int(frame.select(pl.col("query_id").n_unique()).item())
	signature = _signature(frame)
	return {
		"part_index": part_index,
		"path": str(output_path),
		"row_count": row_count,
		"query_count": query_count,
		"signature": signature,
		"created_at": _utc_now_iso(),
	}


def _flush_pending_groups(
	*,
	pending_groups: list[pl.DataFrame],
	split_dir: Path,
	part_index: int,
	rows_total: int,
	queries_total: int,
	shard_entries: list[dict[str, Any]],
) -> tuple[int, int, int]:
	if not pending_groups:
		return part_index, rows_total, queries_total
	shard_frame = pl.concat(pending_groups, how="vertical")
	shard_entry = _write_feature_shard(
		frame=shard_frame,
		split_dir=split_dir,
		part_index=part_index,
	)
	pending_groups.clear()
	shard_entries.append(shard_entry)
	return (
		part_index + 1,
		rows_total + int(shard_entry["row_count"]),
		queries_total + int(shard_entry["query_count"]),
	)


def _init_manifest(
	*,
	ranker_cfg: RankerConfig,
	feature_shard_size: int,
	sample: bool,
	splits: tuple[str, ...],
) -> dict[str, Any]:
	return {
		"sample": sample,
		"created_at": _utc_now_iso(),
		"feature_shard_size": int(feature_shard_size),
		"config_hash": _feature_config_hash(
			ranker_cfg=ranker_cfg,
			feature_shard_size=feature_shard_size,
			sample=sample,
			splits=splits,
		),
		"feature_columns": [],
		"metadata_only_excluded_columns": sorted(METADATA_ONLY_COLUMNS),
		"schema": {},
		"splits": {},
		"completed": False,
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


def _is_allowed_model_feature(name: str, *, use_frozen_features: bool) -> bool:
	if name in METADATA_ONLY_COLUMNS:
		return False
	if name in FIXED_MODEL_FEATURE_COLUMNS:
		return True
	if any(name.startswith(prefix) for prefix in PREFIX_MODEL_FEATURE_ALLOWLIST):
		return True
	return use_frozen_features and name in EMBEDDING_FEATURE_COLUMNS


def resolve_feature_allowlist(
	frame: pl.DataFrame,
	*,
	use_frozen_features: bool,
) -> list[str]:
	"""Return deterministic model feature columns from an explicit allowlist."""

	selected = [
		name
		for name, dtype in frame.schema.items()
		if dtype.is_numeric()
		and _is_allowed_model_feature(name, use_frozen_features=use_frozen_features)
	]
	return sorted(selected)


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
	hash_a = pl.struct(["query_id", "item_idx", "label", "residual_rank"]).hash(seed=17)
	hash_b = pl.struct(["query_id", "item_idx", "label", "residual_rank"]).hash(seed=53)
	summary = frame.select(
		[
			pl.len().alias("row_count"),
			pl.col("query_id").n_unique().alias("query_count"),
			hash_a.sum().alias("hash_a_sum"),
			hash_b.sum().alias("hash_b_sum"),
			hash_a.min().alias("hash_a_min"),
			hash_a.max().alias("hash_a_max"),
		]
	).row(0, named=True)
	payload = json.dumps(summary, sort_keys=True)
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _limit_train_negatives(
	candidates: pl.DataFrame,
	*,
	negatives_per_positive: int,
) -> pl.DataFrame:
	if negatives_per_positive <= 0:
		return candidates

	positives = candidates.filter(pl.col("label") == 1)
	negatives = (
		candidates.filter(pl.col("label") == 0)
		.sort(["query_id", "residual_rank", "item_idx"])
		.group_by("query_id", maintain_order=True)
		.head(negatives_per_positive)
	)
	return pl.concat([positives, negatives], how="vertical").sort(
		["query_id", "residual_rank", "item_idx"]
	)


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
	feature_columns = resolve_feature_allowlist(
		feature_frame,
		use_frozen_features=use_frozen_features,
	)

	# Defensive check: explicit allowlist should never include metadata-only columns.
	invalid_feature_columns = sorted(set(feature_columns).intersection(metadata_cols))
	if invalid_feature_columns:
		msg = (
			"Feature allowlist unexpectedly included metadata-only columns: "
			f"{', '.join(invalid_feature_columns)}"
		)
		raise ValueError(msg)

	return feature_frame, sorted(feature_columns)


def build_ranker_features(
	ranker_cfg: RankerConfig,
	*,
	sample: bool,
	feature_shard_size: int | None = None,
	resume: bool = False,
	overwrite: bool = False,
	splits: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Path]:
	selected_splits = _normalize_splits(splits)
	requested_shard_size = (
		feature_shard_size
		if feature_shard_size is not None
		else ranker_cfg.feature_shard_size
	)
	resolved_shard_size = max(
		int(requested_shard_size),
		1,
	)

	retrieval_cfg = load_retrieval_config(ranker_cfg.retriever_config, sample=sample)
	users_df = pl.read_parquet(retrieval_cfg.users_path)
	items_df = pl.read_parquet(retrieval_cfg.items_path)

	candidate_paths = _candidate_paths(ranker_cfg, sample=sample)
	feature_paths = _feature_paths(ranker_cfg, sample=sample)
	manifest_path = _feature_manifest_path(ranker_cfg, sample=sample)

	if sample:
		if overwrite:
			for split in selected_splits:
				feature_path = feature_paths[split]
				meta_path = feature_path.with_suffix(".meta.json")
				if feature_path.exists():
					feature_path.unlink()
				if meta_path.exists():
					meta_path.unlink()

		sample_outputs: dict[str, Path] = {}
		for split in selected_splits:
			if resume and not overwrite:
				feature_path = feature_paths[split]
				meta_path = feature_path.with_suffix(".meta.json")
				if feature_path.exists() and meta_path.exists():
					sample_outputs[f"{split}_features"] = feature_path
					sample_outputs[f"{split}_meta"] = meta_path
					continue

			candidates = pl.read_parquet(candidate_paths[split])
			rows_before_sampling = int(candidates.height)
			if split == "train":
				candidates = _limit_train_negatives(
					candidates,
					negatives_per_positive=ranker_cfg.negative_samples_per_positive,
				)
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
					"rows_before_sampling": rows_before_sampling,
					"row_count": int(feature_frame.height),
					"query_count": int(feature_frame.select(pl.col("query_id").n_unique()).item()),
					"negative_samples_per_positive": (
						ranker_cfg.negative_samples_per_positive if split == "train" else None
					),
					"feature_columns": feature_columns,
					"deterministic_signature": _signature(feature_frame),
					"mandatory_columns": MANDATORY_FEATURE_COLUMNS,
				},
			)

			sample_outputs[f"{split}_features"] = output_path
			sample_outputs[f"{split}_meta"] = meta_path

		manifest_payload = _init_manifest(
			ranker_cfg=ranker_cfg,
			feature_shard_size=resolved_shard_size,
			sample=True,
			splits=selected_splits,
		)
		manifest_payload["completed"] = True
		_save_manifest(manifest_path, manifest_payload)
		sample_outputs["manifest"] = manifest_path
		return sample_outputs

	if overwrite:
		for split in selected_splits:
			split_dir = ranker_cfg.feature_split_dir(split=split, sample=False)
			if split_dir.exists():
				shutil.rmtree(split_dir)

	manifest = _load_manifest(manifest_path) if resume else {}
	if not manifest:
		manifest = _init_manifest(
			ranker_cfg=ranker_cfg,
			feature_shard_size=resolved_shard_size,
			sample=False,
			splits=selected_splits,
		)
	else:
		manifest["created_at"] = _utc_now_iso()
		manifest["feature_shard_size"] = resolved_shard_size
		manifest["completed"] = False
		manifest.setdefault("splits", {})
		manifest.setdefault("metadata_only_excluded_columns", sorted(METADATA_ONLY_COLUMNS))

	outputs: dict[str, Path] = {}
	feature_columns_ref: list[str] | None = None
	schema_ref: dict[str, str] | None = None

	for split in selected_splits:
		source_paths = _candidate_chunk_paths(ranker_cfg, split=split, sample=False)
		if not source_paths:
			msg = f"No candidate source files found for split '{split}'"
			raise FileNotFoundError(msg)

		split_dir = ranker_cfg.feature_split_dir(split=split, sample=False)
		split_dir.mkdir(parents=True, exist_ok=True)

		splits_payload = cast(dict[str, Any], manifest.setdefault("splits", {}))
		split_manifest = cast(dict[str, Any], splits_payload.setdefault(split, {}))
		split_manifest.setdefault("split", split)
		split_manifest.setdefault("shards", [])

		if overwrite and split_dir.exists():
			shutil.rmtree(split_dir)
			split_dir.mkdir(parents=True, exist_ok=True)
			split_manifest["shards"] = []
			split_manifest["source_chunks_processed"] = 0
			split_manifest["completed"] = False

		existing_shards = cast(list[dict[str, Any]], split_manifest.get("shards", []))
		valid_existing_shards = [
			entry
			for entry in existing_shards
			if isinstance(entry, dict) and Path(str(entry.get("path", ""))).exists()
		]
		split_manifest["shards"] = valid_existing_shards
		part_index = len(valid_existing_shards)
		rows_total = sum(int(entry.get("row_count", 0)) for entry in valid_existing_shards)
		queries_total = sum(int(entry.get("query_count", 0)) for entry in valid_existing_shards)
		processed_source_chunks = int(split_manifest.get("source_chunks_processed", 0))

		if (
			resume
			and bool(split_manifest.get("completed", False))
			and len(valid_existing_shards) > 0
		):
			outputs[f"{split}_feature_dir"] = split_dir
			continue

		pending_groups: list[pl.DataFrame] = []
		pending_rows = 0

		for source_index, source_path in enumerate(source_paths):
			if source_index < processed_source_chunks:
				continue

			candidates = pl.read_parquet(source_path)
			if split == "train":
				candidates = _limit_train_negatives(
					candidates,
					negatives_per_positive=ranker_cfg.negative_samples_per_positive,
				)

			feature_frame, split_feature_columns = _build_split_features(
				candidates=candidates,
				users_df=users_df,
				items_df=items_df,
				use_frozen_features=ranker_cfg.use_frozen_retrieval_embeddings,
			)
			validate_feature_frame(feature_frame, split=split)

			if feature_columns_ref is None:
				feature_columns_ref = split_feature_columns
				schema_ref = _frame_schema_payload(feature_frame)
				manifest["feature_columns"] = feature_columns_ref
				manifest["schema"] = schema_ref

			groups = feature_frame.partition_by("query_id", maintain_order=True)
			for group in groups:
				group_rows = int(group.height)
				if pending_rows > 0 and pending_rows + group_rows > resolved_shard_size:
					part_index, rows_total, queries_total = _flush_pending_groups(
						pending_groups=pending_groups,
						split_dir=split_dir,
						part_index=part_index,
						rows_total=rows_total,
						queries_total=queries_total,
						shard_entries=valid_existing_shards,
					)
					pending_rows = 0
				pending_groups.append(group)
				pending_rows += group_rows

			split_manifest["source_chunks_processed"] = source_index + 1
			split_manifest["completed"] = False
			split_manifest["shards"] = valid_existing_shards
			split_manifest["row_count"] = rows_total
			split_manifest["query_count"] = queries_total
			_save_manifest(manifest_path, manifest)

			chunk_progress = (
				"[ranker-features]"
				f" split={split}"
				f" source_chunk={source_index + 1}/{len(source_paths)}"
			)
			log_memory_status(
				chunk_progress,
				disk_path=ranker_cfg.paths.ranker_feature_dir,
			)
			if should_stop_for_memory(
				ranker_cfg.max_ram_percent,
				ranker_cfg.max_pagefile_percent,
				disk_path=ranker_cfg.paths.ranker_feature_dir,
			):
				part_index, rows_total, queries_total = _flush_pending_groups(
					pending_groups=pending_groups,
					split_dir=split_dir,
					part_index=part_index,
					rows_total=rows_total,
					queries_total=queries_total,
					shard_entries=valid_existing_shards,
				)
				pending_rows = 0
				split_manifest["shards"] = valid_existing_shards
				split_manifest["row_count"] = rows_total
				split_manifest["query_count"] = queries_total
				split_manifest["completed"] = False
				manifest["completed"] = False
				_save_manifest(manifest_path, manifest)
				outputs["manifest"] = manifest_path
				outputs[f"{split}_feature_dir"] = split_dir
				return outputs

		part_index, rows_total, queries_total = _flush_pending_groups(
			pending_groups=pending_groups,
			split_dir=split_dir,
			part_index=part_index,
			rows_total=rows_total,
			queries_total=queries_total,
			shard_entries=valid_existing_shards,
		)
		pending_rows = 0

		split_manifest["shards"] = valid_existing_shards
		split_manifest["split"] = split
		split_manifest["shard_paths"] = [entry["path"] for entry in valid_existing_shards]
		split_manifest["row_count"] = rows_total
		split_manifest["query_count"] = queries_total
		split_manifest["feature_columns"] = feature_columns_ref or []
		split_manifest["metadata_only_excluded_columns"] = sorted(METADATA_ONLY_COLUMNS)
		split_manifest["schema"] = schema_ref or {}
		split_manifest["created_at"] = _utc_now_iso()
		split_manifest["completed"] = True
		_save_manifest(manifest_path, manifest)

		outputs[f"{split}_feature_dir"] = split_dir

	splits_payload = cast(dict[str, Any], manifest.get("splits", {}))
	manifest["completed"] = all(
		bool(cast(dict[str, Any], splits_payload.get(split, {})).get("completed", False))
		for split in selected_splits
	)
	manifest["created_at"] = _utc_now_iso()
	_save_manifest(manifest_path, manifest)
	outputs["manifest"] = manifest_path
	return outputs
