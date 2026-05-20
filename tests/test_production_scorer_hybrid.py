"""Tests for production scorer hybrid utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from movie_recsys.ranking.hybrid import (
    PolicyEvaluation,
    PolicySpec,
    metadata_feature_leakage_audit,
    normalize_query_scores,
    rank_positive_item,
    score_columns_are_finite,
    select_best_policy_result,
)


def test_querywise_minmax_normalization_maps_bounds() -> None:
    values = np.asarray([2.0, 4.0, 6.0], dtype=np.float64)
    normalized = normalize_query_scores(values, method="query_minmax")

    assert np.allclose(normalized, np.asarray([0.0, 0.5, 1.0], dtype=np.float64))


def test_hybrid_rank_is_deterministic_under_ties() -> None:
    scores = np.asarray([0.75, 0.75, 0.10], dtype=np.float64)
    residual_rank = np.asarray([2, 1, 3], dtype=np.int64)
    item_idx = np.asarray([100, 200, 300], dtype=np.int64)

    first_rank = rank_positive_item(
        scores=scores,
        residual_rank=residual_rank,
        item_idx=item_idx,
        positive_index=0,
    )
    second_rank = rank_positive_item(
        scores=scores,
        residual_rank=residual_rank,
        item_idx=item_idx,
        positive_index=0,
    )

    assert first_rank == 2
    assert second_rank == 2


def test_metadata_columns_are_excluded_from_score_features() -> None:
    audit = metadata_feature_leakage_audit(
        used_score_columns=["ranker_score", "popularity_score", "residual_score"],
        available_columns=["query_id", "label", "ranker_score"],
    )
    assert audit["metadata_feature_leakage_passed"] is True

    failing_audit = metadata_feature_leakage_audit(
        used_score_columns=["ranker_score", "label"],
        available_columns=["query_id", "label", "ranker_score"],
    )
    assert failing_audit["metadata_feature_leakage_passed"] is False
    assert failing_audit["disallowed_columns_used"] == ["label"]


def test_validation_split_policy_selection_uses_validation_metrics_only() -> None:
    policy_a = PolicySpec(
        policy_name="ranker_plus_popularity",
        ranker_weight=1.0,
        popularity_weight=0.1,
        residual_weight=0.0,
        alpha=1.0,
        beta=0.1,
        gamma=0.0,
    )
    policy_b = PolicySpec(
        policy_name="ranker_plus_popularity",
        ranker_weight=0.7,
        popularity_weight=0.5,
        residual_weight=0.0,
        alpha=0.7,
        beta=0.5,
        gamma=0.0,
    )

    selected = select_best_policy_result(
        [
            PolicyEvaluation(
                policy=policy_a,
                metrics={"hr@10": 0.20, "mrr@10": 0.15, "ndcg@10": 0.18, "recall@50": 0.50},
            ),
            PolicyEvaluation(
                policy=policy_b,
                metrics={"hr@10": 0.19, "mrr@10": 0.14, "ndcg@10": 0.17, "recall@50": 0.51},
            ),
        ]
    )

    assert selected.policy.policy_id == policy_a.policy_id


def test_no_nan_inf_score_handling_reports_failure(tmp_path: Path) -> None:
    finite_path = tmp_path / "finite.parquet"
    non_finite_path = tmp_path / "non_finite.parquet"

    finite_frame = pl.DataFrame(
        {
            "ranker_score": [0.1, 0.2],
            "residual_score": [0.3, 0.4],
            "popularity_score": [0.5, 0.6],
        }
    )
    finite_frame.write_parquet(finite_path)

    non_finite_frame = pl.DataFrame(
        {
            "ranker_score": [0.1, float("nan")],
            "residual_score": [0.3, 0.4],
            "popularity_score": [0.5, float("inf")],
        }
    )
    non_finite_frame.write_parquet(non_finite_path)

    assert score_columns_are_finite(finite_path) is True
    assert score_columns_are_finite(non_finite_path) is False
