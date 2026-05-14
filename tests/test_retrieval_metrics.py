from __future__ import annotations

from movie_recsys.modeling.metrics import (
    aggregate_ranking_metrics,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_basic_metrics_behavior() -> None:
    ranked = [10, 20, 30, 40, 50]
    targets = {30}
    assert hit_rate_at_k(ranked, targets, 10) == 1.0
    assert hit_rate_at_k(ranked, targets, 2) == 0.0
    assert mrr_at_k(ranked, targets, 10) == 1.0 / 3.0
    assert recall_at_k(ranked, targets, 50) == 1.0


def test_ndcg_handcrafted_fixture() -> None:
    ranked = [4, 1, 2, 3, 5]
    targets = {2, 3}
    score = ndcg_at_k(ranked, targets, 5)
    assert 0.0 < score < 1.0


def test_aggregate_metrics() -> None:
    predictions = {0: [1, 2, 3], 1: [4, 5, 6]}
    targets = {0: {2}, 1: {7}}
    metrics = aggregate_ranking_metrics(predictions, targets)
    assert set(metrics.keys()) == {"hr@10", "mrr@10", "ndcg@10", "recall@50"}
