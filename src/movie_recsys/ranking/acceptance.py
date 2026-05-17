"""Acceptance rules and guard checks for neural ranker promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def load_json(path: Path) -> dict[str, Any]:
	if not path.exists():
		msg = f"Required file not found: {path}"
		raise FileNotFoundError(msg)
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if not isinstance(payload, dict):
		msg = f"Expected JSON object in {path}"
		raise ValueError(msg)
	return payload


def relative_drop(reference: float, value: float) -> float:
	if reference <= 0.0:
		return 0.0
	return max((reference - value) / reference, 0.0)


def primary_rules(
	*,
	ranker_val: dict[str, float],
	ranker_test: dict[str, float],
	residual_val: dict[str, float],
	residual_test: dict[str, float],
) -> dict[str, bool]:
	rule_one = ranker_val["ndcg@10"] > residual_val["ndcg@10"]
	rel_gap = relative_drop(residual_val["ndcg@10"], ranker_val["ndcg@10"])
	rule_two = rel_gap <= 0.01 and (
		ranker_val["mrr@10"] > residual_val["mrr@10"]
		or ranker_val["hr@10"] > residual_val["hr@10"]
	)
	rule_three = (
		ranker_test["ndcg@10"] > residual_test["ndcg@10"]
		and rel_gap < 0.01
	)
	return {
		"rule_1_val_ndcg_improves": rule_one,
		"rule_2_val_within_1pct_and_mrr_or_hr_improves": rule_two,
		"rule_3_test_ndcg_improves_and_val_drop_lt_1pct": rule_three,
		"any_primary_rule_passed": bool(rule_one or rule_two or rule_three),
	}


def guard_one_positive_per_query(candidate_path: Path) -> bool:
	frame = pl.read_parquet(candidate_path)
	grouped = frame.group_by("query_id").agg(pl.col("label").sum().alias("pos_count"))
	return grouped.filter(pl.col("pos_count") != 1).height == 0


def guard_no_duplicate_candidates(candidate_path: Path) -> bool:
	frame = pl.read_parquet(candidate_path)
	duplicates = frame.group_by(["query_id", "item_idx"]).len().filter(pl.col("len") > 1)
	return duplicates.height == 0


def guard_no_candidate_leakage(
	*,
	train_candidates: Path,
	val_candidates: Path,
	test_candidates: Path,
) -> bool:
	train_frame = pl.read_parquet(train_candidates)
	val_frame = pl.read_parquet(val_candidates)
	test_frame = pl.read_parquet(test_candidates)

	train_pairs = {
		(int(user_idx), int(item_idx))
		for user_idx, item_idx in train_frame.filter(pl.col("label") == 1)
		.select(["user_idx", "target_item_idx"])
		.iter_rows()
	}
	heldout_pairs = {
		(int(user_idx), int(item_idx))
		for user_idx, item_idx in val_frame.filter(pl.col("label") == 1)
		.select(["user_idx", "target_item_idx"])
		.iter_rows()
	}.union(
		{
			(int(user_idx), int(item_idx))
			for user_idx, item_idx in test_frame.filter(pl.col("label") == 1)
			.select(["user_idx", "target_item_idx"])
			.iter_rows()
		}
	)
	return len(train_pairs.intersection(heldout_pairs)) == 0


def guard_finite_scores(scored_candidates_path: Path) -> bool:
	frame = pl.read_parquet(scored_candidates_path)
	if "ranker_score" not in frame.columns:
		return False
	checks = frame.select(
		[
			pl.col("ranker_score").is_finite().all().alias("ranker_ok"),
			pl.col("residual_score").is_finite().all().alias("residual_ok"),
		]
	).row(0)
	return bool(checks[0]) and bool(checks[1])


def guard_candidate_deterministic(meta_path: Path) -> bool:
	payload = load_json(meta_path)
	signature = payload.get("deterministic_signature")
	return isinstance(signature, str) and len(signature) > 0


def guard_mlflow_logged(best_checkpoint_path: Path) -> bool:
	if not best_checkpoint_path.exists():
		return False
	import torch

	payload = torch.load(best_checkpoint_path, map_location="cpu")
	run_id = payload.get("mlflow_run_id")
	return isinstance(run_id, str) and len(run_id) > 0


def evaluate_acceptance(
	*,
	ranker_val: dict[str, float],
	ranker_test: dict[str, float],
	residual_val: dict[str, float],
	residual_test: dict[str, float],
	guards: dict[str, bool],
) -> dict[str, Any]:
	rules = primary_rules(
		ranker_val=ranker_val,
		ranker_test=ranker_test,
		residual_val=residual_val,
		residual_test=residual_test,
	)

	failed_reasons: list[str] = []
	if not rules["any_primary_rule_passed"]:
		failed_reasons.append("No primary acceptance rule passed")

	recall_drop = relative_drop(residual_val["recall@50"], ranker_val["recall@50"])
	recall_guard = recall_drop <= 0.05
	if not recall_guard:
		failed_reasons.append("Recall@50 dropped by more than 5% relative to residual")

	for name, passed in guards.items():
		if not passed:
			failed_reasons.append(f"Guard failed: {name}")

	acceptance_passed = len(failed_reasons) == 0
	return {
		"acceptance_passed": acceptance_passed,
		"full_data_ranker_allowed": acceptance_passed,
		"primary_rules": rules,
		"guards": {**guards, "recall50_relative_drop_le_5pct": recall_guard},
		"failed_reasons": failed_reasons,
		"delta_val_ndcg@10": float(ranker_val["ndcg@10"] - residual_val["ndcg@10"]),
		"delta_test_ndcg@10": float(ranker_test["ndcg@10"] - residual_test["ndcg@10"]),
		"recall50_relative_drop_vs_residual": float(recall_drop),
	}
