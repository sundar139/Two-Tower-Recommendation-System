"""Tests for ranker metrics and evaluator math."""

from __future__ import annotations

import math

from movie_recsys.modeling.metrics import aggregate_ranking_metrics


def test_evaluator_ndcg_matches_handcrafted_fixture() -> None:
	predictions = {
		0: [1, 2, 3],
		1: [3, 1, 2],
	}
	targets = {
		0: {1},
		1: {1},
	}

	metrics = aggregate_ranking_metrics(predictions, targets)

	expected_ndcg_q1 = 1.0
	expected_ndcg_q2 = 1.0 / math.log2(3)
	expected_ndcg = (expected_ndcg_q1 + expected_ndcg_q2) / 2.0

	expected_mrr = (1.0 + 0.5) / 2.0
	expected_hr = 1.0

	assert math.isclose(metrics["ndcg@10"], expected_ndcg, rel_tol=1e-6)
	assert math.isclose(metrics["mrr@10"], expected_mrr, rel_tol=1e-6)
	assert math.isclose(metrics["hr@10"], expected_hr, rel_tol=1e-6)
