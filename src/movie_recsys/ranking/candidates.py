"""Candidate generation from residual retrieval outputs.

Test-time history intentionally matches the approved retrieval evaluator behavior,
which uses train history only (`users.train_history_item_idx`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
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


@dataclass(slots=True)
class CandidateGenerationOptions:
	chunk_size: int = 5000
	resume: bool = False
	max_users: int | None = None
	splits: tuple[str, ...] = ("train", "val", "test")
	overwrite: bool = False
	progress_every: int = 1000


UTC_TZ = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


def _utc_now_iso() -> str:
	return datetime.now(UTC_TZ).isoformat()


def _memory_usage_mb() -> float | None:
	return None


def _emit_progress(
	*,
	split: str,
	processed_queries: int,
	total_queries: int,
	candidates_written: int,
	elapsed_seconds: float,
	output_path: Path,
) -> None:
	rate = float(processed_queries / elapsed_seconds) if elapsed_seconds > 0 else 0.0
	remaining_queries = max(total_queries - processed_queries, 0)
	eta_seconds = float(remaining_queries / rate) if rate > 0 else 0.0
	memory_mb = _memory_usage_mb()
	memory_text = f" memory_mb={memory_mb:.1f}" if memory_mb is not None else ""
	print(
		"[ranker-candidates]"
		f" split={split}"
		f" processed_queries={processed_queries}/{total_queries}"
		f" candidates_written={candidates_written}"
		f" elapsed_s={elapsed_seconds:.1f}"
		f" eta_s={eta_seconds:.1f}"
		f" output_path={output_path}"
		f"{memory_text}",
		flush=True,
	)


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
	# Keep deterministic ordering for reproducible outputs and chunk names.
	ordered = tuple(split for split in ("train", "val", "test") if split in normalized)
	return ordered


def _config_hash(
	*,
	ranker_cfg: RankerConfig,
	retrieval_cfg: RetrievalConfig,
	options: CandidateGenerationOptions,
	sample: bool,
) -> str:
	payload = {
		"sample": sample,
		"splits": list(options.splits),
		"chunk_size": options.chunk_size,
		"candidate_top_k": ranker_cfg.candidate_top_k,
		"use_frozen_retrieval_embeddings": ranker_cfg.use_frozen_retrieval_embeddings,
		"retriever_config": str(ranker_cfg.retriever_config),
		"retriever_checkpoint": str(ranker_cfg.retriever_checkpoint),
		"history_length": retrieval_cfg.train.history_length,
		"random_seed": ranker_cfg.random_seed,
	}
	encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def _manifest_path(ranker_cfg: RankerConfig, *, sample: bool) -> Path:
	return ranker_cfg.candidate_dir_for_scope(sample=sample) / "ranker_candidates_manifest.json"


def _load_manifest(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if not isinstance(payload, dict):
		msg = f"Invalid manifest format in {path}"
		raise ValueError(msg)
	return cast(dict[str, Any], payload)


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(manifest, handle, indent=2, sort_keys=True)
		handle.write("\n")


def _chunk_dir(ranker_cfg: RankerConfig, *, sample: bool, split: str) -> Path:
	return ranker_cfg.candidate_dir_for_scope(sample=sample) / "chunks" / split


def _merge_chunks(chunk_files: list[Path], *, output_path: Path) -> None:
	if not chunk_files:
		msg = "No chunk files found for merge"
		raise ValueError(msg)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if output_path.exists():
		output_path.unlink()
	pl.concat([pl.scan_parquet(path) for path in chunk_files]).sink_parquet(output_path)


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
	candidate_checkpoints = [
		ranker_cfg.retriever_checkpoint,
		retrieval_cfg.paths.model_output_dir / "last_residual_transformer_retriever.pt",
		retrieval_cfg.paths.model_output_dir / "best_residual_transformer_retriever.pt",
	]

	seen: set[Path] = set()
	errors: list[str] = []
	for checkpoint_path in candidate_checkpoints:
		if checkpoint_path in seen or not checkpoint_path.exists():
			continue
		seen.add(checkpoint_path)
		checkpoint = load_checkpoint(checkpoint_path)
		try:
			model.load_state_dict(checkpoint["model_state_dict"])
			model.eval()
			return model
		except RuntimeError as exc:
			errors.append(f"{checkpoint_path}: {exc}")

	msg = "Failed to load residual retriever checkpoint for candidate generation"
	if errors:
		msg = f"{msg}. Tried checkpoints with mismatched shapes: {' | '.join(errors)}"
	raise RuntimeError(msg)


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
	progress_every: int,
	output_path: Path,
) -> pl.DataFrame:
	rows: list[dict[str, object]] = []
	start = monotonic()
	for query_idx, query in enumerate(queries, start=1):
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

		if progress_every > 0 and query_idx % progress_every == 0:
			_emit_progress(
				split=split,
				processed_queries=query_idx,
				total_queries=len(queries),
				candidates_written=len(rows),
				elapsed_seconds=monotonic() - start,
				output_path=output_path,
			)

	frame = pl.DataFrame(rows)
	sort_columns = ["query_id", "residual_rank", "item_idx"]
	frame = frame.sort(sort_columns)
	_emit_progress(
		split=split,
		processed_queries=len(queries),
		total_queries=len(queries),
		candidates_written=frame.height,
		elapsed_seconds=monotonic() - start,
		output_path=output_path,
	)
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
	leaked_targets: set[int] | set[tuple[int, int]] | None = None,
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

	required_non_null = frame.select(
		[pl.col(name).is_null().any().alias(name) for name in REQUIRED_CANDIDATE_COLUMNS]
	).row(0)
	non_null_required_columns = all(not bool(value) for value in required_non_null)
	finite_residual_score = bool(frame.select(pl.col("residual_score").is_finite().all()).item())

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
		train_positive = frame.filter(pl.col("label") == 1)
		if leaked_targets and all(
			isinstance(value, tuple) and len(value) == 2 for value in leaked_targets
		):
			pair_targets = cast(set[tuple[int, int]], leaked_targets)
			train_pairs = {
				(int(user_idx), int(item_idx))
				for user_idx, item_idx in train_positive.select(
					["user_idx", "target_item_idx"]
				).iter_rows()
			}
			heldout_pairs = {
				(int(user_idx), int(item_idx))
				for user_idx, item_idx in pair_targets
				if isinstance(user_idx, int) and isinstance(item_idx, int)
			}
			no_leakage = len(train_pairs.intersection(heldout_pairs)) == 0
		else:
			train_targets = {
				int(item_idx)
				for item_idx in train_positive.get_column("target_item_idx").to_list()
			}
			heldout_targets = {
				int(item_idx)
				for item_idx in leaked_targets
				if isinstance(item_idx, int)
			}
			no_leakage = len(train_targets.intersection(heldout_targets)) == 0

	checks = {
		"one_positive_per_query": bool(one_positive),
		"no_duplicate_candidate_item_per_query": bool(duplicates),
		"candidate_set_size_le_201": bool(max_size_ok),
		"residual_rank_contiguous_and_stable": bool(rank_ok),
		"required_columns_non_null": bool(non_null_required_columns),
		"residual_score_finite": bool(finite_residual_score),
		"deterministic_query_id": bool(deterministic_query_ids),
		"no_val_test_target_leakage": bool(no_leakage),
	}

	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		msg = f"Candidate validation failed for split '{split}': {', '.join(failed)}"
		raise ValueError(msg)
	return checks


def validate_candidate_path(
	path: Path,
	*,
	split: str,
	leaked_targets: set[int] | set[tuple[int, int]] | None = None,
) -> dict[str, bool]:
	if not path.exists():
		msg = f"Candidate parquet not found for split '{split}': {path}"
		raise FileNotFoundError(msg)
	frame = pl.scan_parquet(path)
	row_count = int(frame.select(pl.len()).collect().item())
	if row_count == 0:
		msg = f"Candidate frame for split '{split}' is empty"
		raise ValueError(msg)

	positives_ok = (
		frame.group_by("query_id")
		.agg(pl.col("label").sum().alias("positive_count"))
		.filter(pl.col("positive_count") != 1)
		.select(pl.len())
		.collect()
		.item()
		== 0
	)

	duplicates_ok = (
		frame.group_by(["query_id", "item_idx"])
		.len()
		.filter(pl.col("len") > 1)
		.select(pl.len())
		.collect()
		.item()
		== 0
	)

	size_ok = (
		frame.group_by("query_id")
		.len()
		.filter(pl.col("len") > 201)
		.select(pl.len())
		.collect()
		.item()
		== 0
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
		.select(pl.len())
		.collect()
		.item()
		== 0
	)

	required_non_null = frame.select(
		[pl.col(name).is_null().any().alias(name) for name in REQUIRED_CANDIDATE_COLUMNS]
	).collect()
	non_null_required_columns = all(not bool(value) for value in required_non_null.row(0))
	finite_residual_score = bool(
		frame.select(pl.col("residual_score").is_finite().all()).collect().item()
	)

	deterministic_query_ids = bool(
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
		.collect()
		.item()
	)

	no_leakage = True
	if leaked_targets is not None:
		positive_pairs = {
			(int(user_idx), int(item_idx))
			for user_idx, item_idx in frame.filter(pl.col("label") == 1)
			.select(["user_idx", "target_item_idx"])
			.collect()
			.iter_rows()
		}
		if leaked_targets and all(
			isinstance(value, tuple) and len(value) == 2 for value in leaked_targets
		):
			pair_targets = cast(set[tuple[int, int]], leaked_targets)
			no_leakage = len(positive_pairs.intersection(pair_targets)) == 0
		else:
			target_items = {item_idx for _user_idx, item_idx in positive_pairs}
			heldout_items = {
				int(item_idx)
				for item_idx in leaked_targets
				if isinstance(item_idx, int)
			}
			no_leakage = len(target_items.intersection(heldout_items)) == 0

	checks = {
		"one_positive_per_query": bool(positives_ok),
		"no_duplicate_candidate_item_per_query": bool(duplicates_ok),
		"candidate_set_size_le_201": bool(size_ok),
		"residual_rank_contiguous_and_stable": bool(rank_ok),
		"required_columns_non_null": bool(non_null_required_columns),
		"residual_score_finite": bool(finite_residual_score),
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
	chunk_size: int = 5000,
	resume: bool = False,
	max_users: int | None = None,
	splits: tuple[str, ...] = ("train", "val", "test"),
	overwrite: bool = False,
	progress_every: int = 1000,
) -> dict[str, Path]:
	"""Generate ranker candidates for train/val/test.

	Returns a map containing candidate parquet paths for each split plus a
	metadata JSON path for each split.
	"""
	options = CandidateGenerationOptions(
		chunk_size=max(int(chunk_size), 1),
		resume=bool(resume),
		max_users=max_users,
		splits=_normalize_splits(splits),
		overwrite=bool(overwrite),
		progress_every=max(int(progress_every), 0),
	)

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
	if options.max_users is not None:
		limit = max(int(options.max_users), 0)
		train_queries = train_queries[:limit]
		val_queries = val_queries[:limit]
		test_queries = test_queries[:limit]

	queries_by_split: dict[str, list[QueryContext]] = {
		"train": train_queries,
		"val": val_queries,
		"test": test_queries,
	}

	val_targets = {
		(int(user_idx), int(item_idx))
		for user_idx, item_idx in val_df.select(["user_idx", "item_idx"]).iter_rows()
	}
	test_targets = {
		(int(user_idx), int(item_idx))
		for user_idx, item_idx in test_df.select(["user_idx", "item_idx"]).iter_rows()
	}
	leaked_targets = val_targets.union(test_targets)

	scope_dir = ranker_cfg.candidate_dir_for_scope(sample=sample)
	scope_dir.mkdir(parents=True, exist_ok=True)
	manifest_path = _manifest_path(ranker_cfg, sample=sample)

	if options.overwrite:
		for split in options.splits:
			chunk_dir = _chunk_dir(ranker_cfg, sample=sample, split=split)
			if chunk_dir.exists():
				shutil.rmtree(chunk_dir)
			candidate_path = ranker_cfg.candidate_path(split=split, sample=sample)
			meta_path = candidate_path.with_suffix(".meta.json")
			if candidate_path.exists():
				candidate_path.unlink()
			if meta_path.exists():
				meta_path.unlink()
		if manifest_path.exists():
			manifest_path.unlink()

	manifest = _load_manifest(manifest_path) if options.resume else {}
	if not manifest:
		manifest = {
			"sample": sample,
			"generated_at": _utc_now_iso(),
			"config_hash": _config_hash(
				ranker_cfg=ranker_cfg,
				retrieval_cfg=retrieval_cfg,
				options=options,
				sample=sample,
			),
			"options": {
				"chunk_size": options.chunk_size,
				"resume": options.resume,
				"max_users": options.max_users,
				"splits": list(options.splits),
				"overwrite": options.overwrite,
				"progress_every": options.progress_every,
			},
			"splits": {},
		}
	else:
		manifest["generated_at"] = _utc_now_iso()

	outputs: dict[str, Path] = {}
	for split in options.splits:
		queries = queries_by_split[split]
		output_path = ranker_cfg.candidate_path(split=split, sample=sample)
		chunk_dir = _chunk_dir(ranker_cfg, sample=sample, split=split)
		chunk_dir.mkdir(parents=True, exist_ok=True)

		splits_manifest = cast(dict[str, Any], manifest.setdefault("splits", {}))
		split_manifest = cast(dict[str, Any], splits_manifest.setdefault(split, {}))
		split_manifest.setdefault("split", split)
		split_manifest["completed"] = False
		chunk_entries = cast(list[dict[str, Any]], split_manifest.get("chunks", []))
		entry_by_index = {
			int(entry.get("chunk_index", idx)): entry
			for idx, entry in enumerate(chunk_entries)
			if isinstance(entry, dict)
		}

		chunk_files: list[Path] = []
		chunk_signatures: list[str] = []
		total_rows = 0
		processed_queries = 0
		new_entries: list[dict[str, Any]] = []

		for chunk_index, start_idx in enumerate(range(0, len(queries), options.chunk_size)):
			end_idx = min(start_idx + options.chunk_size, len(queries))
			chunk_queries = queries[start_idx:end_idx]
			if not chunk_queries:
				continue

			chunk_file = chunk_dir / f"chunk_{chunk_index:06d}.parquet"
			existing_entry = entry_by_index.get(chunk_index)
			if (
				options.resume
				and existing_entry is not None
				and bool(existing_entry.get("completed", False))
				and chunk_file.exists()
			):
				row_count = int(existing_entry.get("row_count", 0))
				query_count = int(existing_entry.get("query_count", len(chunk_queries)))
				signature = str(existing_entry.get("signature", ""))
				if not signature:
					signature = _signature(pl.read_parquet(chunk_file))
				total_rows += row_count
				processed_queries += query_count
				chunk_files.append(chunk_file)
				chunk_signatures.append(signature)
				new_entries.append(existing_entry)
				if options.progress_every > 0 and processed_queries % options.progress_every == 0:
					_emit_progress(
						split=split,
						processed_queries=processed_queries,
						total_queries=len(queries),
						candidates_written=total_rows,
						elapsed_seconds=1.0,
						output_path=output_path,
					)
				continue

			chunk_frame = _build_candidate_frame(
				chunk_queries,
				split=split,
				top_k=ranker_cfg.candidate_top_k,
				index=index,
				item_embeddings=item_embeddings,
				model=model,
				tables=tables,
				history_length=retrieval_cfg.train.history_length,
				use_embedding_features=ranker_cfg.use_frozen_retrieval_embeddings,
				device=device,
				progress_every=options.progress_every,
				output_path=chunk_file,
			)
			_save_frame(chunk_file, chunk_frame)

			row_count = int(chunk_frame.height)
			query_count = len(chunk_queries)
			signature = _signature(chunk_frame)
			total_rows += row_count
			processed_queries += query_count
			chunk_files.append(chunk_file)
			chunk_signatures.append(signature)

			start_user_idx = min(query.user_idx for query in chunk_queries)
			end_user_idx = max(query.user_idx for query in chunk_queries)
			new_entries.append(
				{
					"chunk_index": chunk_index,
					"file": str(chunk_file),
					"row_count": row_count,
					"query_count": query_count,
					"start_user_idx": int(start_user_idx),
					"end_user_idx": int(end_user_idx),
					"signature": signature,
					"completed": True,
					"generated_at": _utc_now_iso(),
				}
			)

			split_manifest["chunks"] = sorted(
				new_entries,
				key=lambda entry: int(entry["chunk_index"]),
			)
			split_manifest["row_count"] = int(total_rows)
			split_manifest["query_count"] = int(processed_queries)
			split_manifest["completed"] = False
			_save_manifest(manifest_path, manifest)

		if not chunk_files:
			msg = f"No chunk files generated for split '{split}'"
			raise ValueError(msg)

		chunk_files_sorted = sorted(chunk_files, key=lambda path: path.name)
		_merge_chunks(chunk_files_sorted, output_path=output_path)

		checks = validate_candidate_path(
			output_path,
			split=split,
			leaked_targets=leaked_targets if split == "train" else None,
		)
		split_signature = hashlib.sha256(
			"".join(chunk_signatures).encode("utf-8")
		).hexdigest()
		meta_path = output_path.with_suffix(".meta.json")
		meta_payload: dict[str, Any] = {
			"split": split,
			"sample": sample,
			"query_count": int(len(queries)),
			"row_count": int(total_rows),
			"deterministic_signature": split_signature,
			"validation_checks": checks,
			"required_columns": REQUIRED_CANDIDATE_COLUMNS,
			"embedding_feature_columns": EMBEDDING_FEATURE_COLUMNS,
		}
		if split == "test":
			meta_payload["test_history_policy"] = (
				"train_history_only_matches_retrieval_evaluator"
			)
		save_json(meta_path, meta_payload)

		split_manifest["chunks"] = sorted(
			new_entries,
			key=lambda entry: int(entry["chunk_index"]),
		)
		split_manifest["row_count"] = int(total_rows)
		split_manifest["query_count"] = int(len(queries))
		split_manifest["output_path"] = str(output_path)
		split_manifest["meta_path"] = str(meta_path)
		split_manifest["deterministic_signature"] = split_signature
		split_manifest["completed"] = True
		split_manifest["generated_at"] = _utc_now_iso()
		_save_manifest(manifest_path, manifest)

		outputs[f"{split}_candidates"] = output_path
		outputs[f"{split}_meta"] = meta_path

	outputs["manifest"] = manifest_path
	return outputs
