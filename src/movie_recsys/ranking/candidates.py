"""Candidate generation from residual retrieval outputs.

Test-time history intentionally matches the approved retrieval evaluator behavior,
which uses train history only (`users.train_history_item_idx`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch

from movie_recsys.modeling.artifacts import load_checkpoint, save_json
from movie_recsys.modeling.datasets import FeatureTables, load_feature_tables
from movie_recsys.modeling.faiss_index import build_flat_ip_index, search_index
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.ranking.config import RankerConfig
from movie_recsys.training.config import RetrievalConfig, load_retrieval_config

REQUIRED_CANDIDATE_COLUMNS = [
	"query_id",
	"user_idx",
	"item_idx",
	"split",
	"label",
	"target_item_idx",
	"residual_score",
	"residual_rank",
	"target_injected",
	"user_history_length",
	"timestamp_context",
	"candidate_source",
]

EMBEDDING_FEATURE_COLUMNS = [
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


@dataclass(slots=True)
class QueryContext:
	query_id: str
	split: str
	user_idx: int
	target_item_idx: int
	history_item_idx: list[int]
	timestamp_context: int


def _select_device(device_name: str) -> torch.device:
	if device_name == "cuda":
		return torch.device("cuda")
	if device_name == "cpu":
		return torch.device("cpu")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_frames(
	cfg: RetrievalConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
	train_df = pl.read_parquet(cfg.train_path).sort(["user_idx", "timestamp", "item_idx"])
	val_df = pl.read_parquet(cfg.val_path).sort(["user_idx", "timestamp", "item_idx"])
	test_df = pl.read_parquet(cfg.test_path).sort(["user_idx", "timestamp", "item_idx"])
	users_df = pl.read_parquet(cfg.users_path).sort("user_idx")
	return train_df, val_df, test_df, users_df


def _train_queries(train_df: pl.DataFrame, *, history_length: int) -> list[QueryContext]:
	grouped = train_df.group_by("user_idx", maintain_order=True).agg(
		pl.col("item_idx").alias("item_idx_seq"),
		pl.col("timestamp").alias("timestamp_seq"),
	)
	queries: list[QueryContext] = []
	for row in grouped.iter_rows(named=True):
		user_idx = int(row["user_idx"])
		items = [int(v) for v in row["item_idx_seq"]]
		timestamps = [int(v) for v in row["timestamp_seq"]]
		if len(items) < 2:
			continue
		target_item = items[-1]
		target_timestamp = timestamps[-1]
		history = items[:-1]
		if len(history) > history_length:
			history = history[-history_length:]
		queries.append(
			QueryContext(
				query_id=f"train_u{user_idx:08d}",
				split="train",
				user_idx=user_idx,
				target_item_idx=target_item,
				history_item_idx=history,
				timestamp_context=target_timestamp,
			)
		)
	return queries


def _history_lookup(users_df: pl.DataFrame) -> dict[int, list[int]]:
	mapping: dict[int, list[int]] = {}
	for row in users_df.select(["user_idx", "train_history_item_idx"]).iter_rows(named=True):
		mapping[int(row["user_idx"])] = [int(v) for v in (row["train_history_item_idx"] or [])]
	return mapping


def _eval_queries(
	split_df: pl.DataFrame,
	*,
	split_name: str,
	history_lookup: dict[int, list[int]],
	history_length: int,
) -> list[QueryContext]:
	queries: list[QueryContext] = []
	for row in split_df.select(["user_idx", "item_idx", "timestamp"]).iter_rows(named=True):
		user_idx = int(row["user_idx"])
		target_item = int(row["item_idx"])
		target_timestamp = int(row["timestamp"])
		history = history_lookup.get(user_idx, [])
		if len(history) > history_length:
			history = history[-history_length:]
		queries.append(
			QueryContext(
				query_id=f"{split_name}_u{user_idx:08d}",
				split=split_name,
				user_idx=user_idx,
				target_item_idx=target_item,
				history_item_idx=history,
				timestamp_context=target_timestamp,
			)
		)
	return queries


def _build_model(
	ranker_cfg: RankerConfig,
	retrieval_cfg: RetrievalConfig,
	tables: FeatureTables,
) -> ResidualTransformerRetriever:
	model = ResidualTransformerRetriever(
		config=retrieval_cfg,
		num_users=tables.user_features.shape[0],
		num_items_with_padding=tables.item_features.shape[0] + 1,
		user_feature_dim=tables.user_features.shape[1],
		item_feature_dim=tables.item_features.shape[1],
	)
	checkpoint = load_checkpoint(ranker_cfg.retriever_checkpoint)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model


def _encode_items(
	model: ResidualTransformerRetriever,
	tables: FeatureTables,
	*,
	device: torch.device,
) -> np.ndarray:
	with torch.no_grad():
		item_features = torch.tensor(tables.item_features, dtype=torch.float32, device=device)
		item_idx_shifted = torch.arange(
			1,
			tables.item_features.shape[0] + 1,
			dtype=torch.long,
			device=device,
		)
		item_emb = model.item_tower(item_idx_shifted, item_features)
	return cast(np.ndarray, item_emb.cpu().numpy().astype(np.float32))


def _encode_query_user(
	query: QueryContext,
	model: ResidualTransformerRetriever,
	tables: FeatureTables,
	*,
	history_length: int,
	device: torch.device,
) -> np.ndarray:
	history = [item + 1 for item in query.history_item_idx[-history_length:]]
	history_tensor = torch.zeros((1, history_length), dtype=torch.long, device=device)
	history_mask = torch.zeros((1, history_length), dtype=torch.bool, device=device)
	if history:
		history_tensor[0, -len(history) :] = torch.tensor(history, dtype=torch.long, device=device)
		history_mask[0, -len(history) :] = True

	user_features = torch.tensor(
		tables.user_features[query.user_idx],
		dtype=torch.float32,
		device=device,
	).unsqueeze(0)
	with torch.no_grad():
		user_emb = model.user_tower(
			torch.tensor([query.user_idx], dtype=torch.long, device=device),
			history_tensor,
			history_mask,
			user_features,
		)
	return cast(np.ndarray, user_emb.cpu().numpy().astype(np.float32))


def _embedding_stats(user_emb: np.ndarray, item_emb: np.ndarray) -> dict[str, float]:
	product = user_emb * item_emb
	abs_diff = np.abs(user_emb - item_emb)
	return {
		"frozen_emb_dot": float(np.dot(user_emb, item_emb)),
		"frozen_emb_prod_mean": float(product.mean()),
		"frozen_emb_prod_max": float(product.max()),
		"frozen_emb_prod_min": float(product.min()),
		"frozen_emb_prod_std": float(product.std()),
		"frozen_emb_absdiff_mean": float(abs_diff.mean()),
		"frozen_emb_absdiff_max": float(abs_diff.max()),
		"frozen_emb_absdiff_min": float(abs_diff.min()),
		"frozen_emb_absdiff_std": float(abs_diff.std()),
	}


def _dedupe_retrieved(items: list[int], scores: list[float]) -> tuple[list[int], list[float]]:
	seen: set[int] = set()
	dedup_items: list[int] = []
	dedup_scores: list[float] = []
	for item, score in zip(items, scores, strict=False):
		if item in seen:
			continue
		seen.add(item)
		dedup_items.append(item)
		dedup_scores.append(score)
	return dedup_items, dedup_scores


def _query_rows(
	query: QueryContext,
	*,
	top_k: int,
	index: Any,
	item_embeddings: np.ndarray,
	model: ResidualTransformerRetriever,
	tables: FeatureTables,
	history_length: int,
	use_embedding_features: bool,
	device: torch.device,
) -> list[dict[str, object]]:
	user_emb_2d = _encode_query_user(
		query,
		model,
		tables,
		history_length=history_length,
		device=device,
	)
	user_emb = user_emb_2d[0]
	item_mapping = np.arange(item_embeddings.shape[0], dtype=np.int64)
	retrieved_items_np, retrieved_scores_np, _latency = search_index(
		index,
		user_emb_2d,
		item_mapping,
		top_k,
	)

	retrieved_items = [int(v) for v in retrieved_items_np[0].tolist()]
	retrieved_scores = [float(v) for v in retrieved_scores_np[0].tolist()]
	retrieved_items, retrieved_scores = _dedupe_retrieved(retrieved_items, retrieved_scores)

	target_in_retrieved = query.target_item_idx in set(retrieved_items)
	if not target_in_retrieved:
		target_score = float(np.dot(user_emb, item_embeddings[query.target_item_idx]))
		retrieved_items.append(query.target_item_idx)
		retrieved_scores.append(target_score)

	rows: list[dict[str, object]] = []
	for rank_idx, (item_idx, residual_score) in enumerate(
		zip(retrieved_items, retrieved_scores, strict=False),
		start=1,
	):
		is_target = int(item_idx == query.target_item_idx)
		target_injected = bool((is_target == 1) and (not target_in_retrieved))
		source = "target_injected" if target_injected else "retrieved"
		row: dict[str, object] = {
			"query_id": query.query_id,
			"user_idx": query.user_idx,
			"item_idx": item_idx,
			"split": query.split,
			"label": is_target,
			"target_item_idx": query.target_item_idx,
			"residual_score": float(residual_score),
			"residual_rank": rank_idx,
			"target_injected": target_injected,
			"user_history_length": len(query.history_item_idx),
			"timestamp_context": query.timestamp_context,
			"candidate_source": source,
		}
		if use_embedding_features:
			row.update(_embedding_stats(user_emb, item_embeddings[item_idx]))
		rows.append(row)
	return rows


def _build_candidate_frame(
	queries: list[QueryContext],
	*,
	split: str,
	top_k: int,
	index: Any,
	item_embeddings: np.ndarray,
	model: ResidualTransformerRetriever,
	tables: FeatureTables,
	history_length: int,
	use_embedding_features: bool,
	device: torch.device,
) -> pl.DataFrame:
	rows: list[dict[str, object]] = []
	for query in queries:
		rows.extend(
			_query_rows(
				query,
				top_k=top_k,
				index=index,
				item_embeddings=item_embeddings,
				model=model,
				tables=tables,
				history_length=history_length,
				use_embedding_features=use_embedding_features,
				device=device,
			)
		)

	frame = pl.DataFrame(rows)
	sort_columns = ["query_id", "residual_rank", "item_idx"]
	frame = frame.sort(sort_columns)
	return frame


def _signature(frame: pl.DataFrame) -> str:
	signature_frame = frame.select(
		["query_id", "user_idx", "item_idx", "label", "residual_rank", "target_injected"]
	).sort(["query_id", "residual_rank", "item_idx"])
	payload = signature_frame.write_json()
	digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
	return digest


def validate_candidate_frame(
	frame: pl.DataFrame,
	*,
	split: str,
	leaked_targets: set[int] | None = None,
) -> dict[str, bool]:
	if frame.height == 0:
		msg = f"Candidate frame for split '{split}' is empty"
		raise ValueError(msg)

	positives = frame.group_by("query_id").agg(pl.col("label").sum().alias("positive_count"))
	one_positive = positives.filter(pl.col("positive_count") != 1).height == 0

	duplicates = (
		frame.group_by(["query_id", "item_idx"]).len().filter(pl.col("len") > 1).height == 0
	)

	max_size_ok = (
		frame.group_by("query_id").len().filter(pl.col("len") > 201).height == 0
	)

	rank_ok = (
		frame.group_by("query_id")
		.agg(
			pl.col("residual_rank").min().alias("min_rank"),
			pl.col("residual_rank").max().alias("max_rank"),
			pl.col("residual_rank").n_unique().alias("unique_rank"),
			pl.len().alias("count"),
		)
		.filter(
			(pl.col("min_rank") != 1)
			| (pl.col("max_rank") != pl.col("count"))
			| (pl.col("unique_rank") != pl.col("count"))
		)
		.height
		== 0
	)

	deterministic_query_ids = (
		frame.select(
			(
				pl.col("query_id")
				== (
					pl.col("split")
					+ pl.lit("_u")
					+ pl.col("user_idx").cast(pl.Utf8).str.zfill(8)
				)
			).all()
		)
		.item()
	)

	no_leakage = True
	if leaked_targets is not None:
		train_targets = set(
			frame.filter(pl.col("label") == 1).get_column("target_item_idx").to_list()
		)
		no_leakage = len(train_targets.intersection(leaked_targets)) == 0

	checks = {
		"one_positive_per_query": bool(one_positive),
		"no_duplicate_candidate_item_per_query": bool(duplicates),
		"candidate_set_size_le_201": bool(max_size_ok),
		"residual_rank_contiguous_and_stable": bool(rank_ok),
		"deterministic_query_id": bool(deterministic_query_ids),
		"no_val_test_target_leakage": bool(no_leakage),
	}

	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		msg = f"Candidate validation failed for split '{split}': {', '.join(failed)}"
		raise ValueError(msg)
	return checks


def _save_frame(path: Path, frame: pl.DataFrame) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	frame.write_parquet(path)


def generate_ranker_candidates(
	ranker_cfg: RankerConfig,
	*,
	sample: bool,
) -> dict[str, Path]:
	"""Generate ranker candidates for train/val/test.

	Returns a map containing candidate parquet paths for each split plus a
	metadata JSON path for each split.
	"""

	retrieval_cfg = load_retrieval_config(ranker_cfg.retriever_config, sample=sample)
	retrieval_cfg.model.model_type = "residual_transformer"

	train_df, val_df, test_df, users_df = _load_frames(retrieval_cfg)
	history_lookup = _history_lookup(users_df)
	tables = load_feature_tables(retrieval_cfg)

	model = _build_model(ranker_cfg, retrieval_cfg, tables)
	device = _select_device(retrieval_cfg.train.device)
	model.to(device)

	item_embeddings = _encode_items(model, tables, device=device)
	index = build_flat_ip_index(item_embeddings)

	train_queries = _train_queries(train_df, history_length=retrieval_cfg.train.history_length)
	val_queries = _eval_queries(
		val_df,
		split_name="val",
		history_lookup=history_lookup,
		history_length=retrieval_cfg.train.history_length,
	)
	# Match the approved retrieval evaluator exactly: test uses train history only.
	test_queries = _eval_queries(
		test_df,
		split_name="test",
		history_lookup=history_lookup,
		history_length=retrieval_cfg.train.history_length,
	)

	val_targets = set(val_df.get_column("item_idx").to_list())
	test_targets = set(test_df.get_column("item_idx").to_list())

	train_frame = _build_candidate_frame(
		train_queries,
		split="train",
		top_k=ranker_cfg.candidate_top_k,
		index=index,
		item_embeddings=item_embeddings,
		model=model,
		tables=tables,
		history_length=retrieval_cfg.train.history_length,
		use_embedding_features=ranker_cfg.use_frozen_retrieval_embeddings,
		device=device,
	)
	val_frame = _build_candidate_frame(
		val_queries,
		split="val",
		top_k=ranker_cfg.candidate_top_k,
		index=index,
		item_embeddings=item_embeddings,
		model=model,
		tables=tables,
		history_length=retrieval_cfg.train.history_length,
		use_embedding_features=ranker_cfg.use_frozen_retrieval_embeddings,
		device=device,
	)
	test_frame = _build_candidate_frame(
		test_queries,
		split="test",
		top_k=ranker_cfg.candidate_top_k,
		index=index,
		item_embeddings=item_embeddings,
		model=model,
		tables=tables,
		history_length=retrieval_cfg.train.history_length,
		use_embedding_features=ranker_cfg.use_frozen_retrieval_embeddings,
		device=device,
	)

	train_checks = validate_candidate_frame(
		train_frame,
		split="train",
		leaked_targets=val_targets.union(test_targets),
	)
	val_checks = validate_candidate_frame(val_frame, split="val")
	test_checks = validate_candidate_frame(test_frame, split="test")

	train_path = ranker_cfg.candidate_path(split="train", sample=sample)
	val_path = ranker_cfg.candidate_path(split="val", sample=sample)
	test_path = ranker_cfg.candidate_path(split="test", sample=sample)
	_save_frame(train_path, train_frame)
	_save_frame(val_path, val_frame)
	_save_frame(test_path, test_frame)

	train_meta_path = train_path.with_suffix(".meta.json")
	val_meta_path = val_path.with_suffix(".meta.json")
	test_meta_path = test_path.with_suffix(".meta.json")

	save_json(
		train_meta_path,
		{
			"split": "train",
			"sample": sample,
			"query_count": int(train_frame.select(pl.col("query_id").n_unique()).item()),
			"row_count": int(train_frame.height),
			"deterministic_signature": _signature(train_frame),
			"validation_checks": train_checks,
			"required_columns": REQUIRED_CANDIDATE_COLUMNS,
			"embedding_feature_columns": EMBEDDING_FEATURE_COLUMNS,
		},
	)
	save_json(
		val_meta_path,
		{
			"split": "val",
			"sample": sample,
			"query_count": int(val_frame.select(pl.col("query_id").n_unique()).item()),
			"row_count": int(val_frame.height),
			"deterministic_signature": _signature(val_frame),
			"validation_checks": val_checks,
			"required_columns": REQUIRED_CANDIDATE_COLUMNS,
			"embedding_feature_columns": EMBEDDING_FEATURE_COLUMNS,
		},
	)
	save_json(
		test_meta_path,
		{
			"split": "test",
			"sample": sample,
			"query_count": int(test_frame.select(pl.col("query_id").n_unique()).item()),
			"row_count": int(test_frame.height),
			"deterministic_signature": _signature(test_frame),
			"validation_checks": test_checks,
			"required_columns": REQUIRED_CANDIDATE_COLUMNS,
			"embedding_feature_columns": EMBEDDING_FEATURE_COLUMNS,
			"test_history_policy": "train_history_only_matches_retrieval_evaluator",
		},
	)

	return {
		"train_candidates": train_path,
		"val_candidates": val_path,
		"test_candidates": test_path,
		"train_meta": train_meta_path,
		"val_meta": val_meta_path,
		"test_meta": test_meta_path,
	}
