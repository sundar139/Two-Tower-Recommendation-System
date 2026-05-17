"""Tests for neural ranker acceptance logic."""

from __future__ import annotations

from movie_recsys.ranking.acceptance import evaluate_acceptance


def _residual_metrics() -> dict[str, float]:
	return {"hr@10": 0.10, "mrr@10": 0.05, "ndcg@10": 0.08, "recall@50": 0.20}


def test_acceptance_passes_when_primary_rule_is_true() -> None:
	residual_val = _residual_metrics()
	residual_test = _residual_metrics()
	ranker_val = {**residual_val, "ndcg@10": 0.09}
	ranker_test = {**residual_test, "ndcg@10": 0.09}

	result = evaluate_acceptance(
		ranker_val=ranker_val,
		ranker_test=ranker_test,
		residual_val=residual_val,
		residual_test=residual_test,
		guards={
			"no_candidate_leakage": True,
			"exactly_one_positive_per_query": True,
			"no_duplicate_candidates_per_query": True,
			"no_nan_or_inf_scores": True,
			"candidate_generation_deterministic": True,
			"mlflow_run_logged": True,
		},
	)

	assert result["acceptance_passed"] is True
	assert result["full_data_ranker_allowed"] is True


def test_acceptance_fails_when_rules_and_guard_fail() -> None:
	residual_val = _residual_metrics()
	residual_test = _residual_metrics()
	ranker_val = {**residual_val, "ndcg@10": 0.06, "recall@50": 0.17}
	ranker_test = {**residual_test, "ndcg@10": 0.06}

	result = evaluate_acceptance(
		ranker_val=ranker_val,
		ranker_test=ranker_test,
		residual_val=residual_val,
		residual_test=residual_test,
		guards={
			"no_candidate_leakage": False,
			"exactly_one_positive_per_query": True,
			"no_duplicate_candidates_per_query": True,
			"no_nan_or_inf_scores": True,
			"candidate_generation_deterministic": True,
			"mlflow_run_logged": True,
		},
	)

	assert result["acceptance_passed"] is False
	assert result["full_data_ranker_allowed"] is False
	assert any("No primary acceptance rule passed" in text for text in result["failed_reasons"])
