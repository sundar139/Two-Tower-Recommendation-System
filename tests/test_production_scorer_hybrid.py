"""Tests for production scorer hybrid utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from movie_recsys.ranking.hybrid import (
    BASELINE_POLICY_NAMES,
    PolicyEvaluation,
    PolicySpec,
    build_policy_grid,
    evaluate_policy_grid,
    is_hybrid_policy,
    metadata_feature_leakage_audit,
    normalize_query_scores,
    rank_positive_item,
    score_columns_are_finite,
    select_best_policy_result,
    select_recall_constrained_policy,
)


def _policy(
    name: str,
    *,
    ndcg: float,
    recall: float,
    mrr: float = 0.2,
    hr: float = 0.3,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> PolicyEvaluation:
    spec = PolicySpec(
        policy_name=name,
        ranker_weight=1.0 if name != "popularity_only" else 0.0,
        popularity_weight=1.0 if name == "popularity_only" else 0.2,
        residual_weight=0.0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    return PolicyEvaluation(
        policy=spec,
        metrics={
            "hr@10": hr,
            "mrr@10": mrr,
            "ndcg@10": ndcg,
            "recall@50": recall,
        },
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


def test_recall_constrained_selector_skips_higher_ndcg_if_recall_invalid() -> None:
    popularity = _policy("popularity_only", ndcg=0.26, recall=0.71, mrr=0.21, hr=0.35)
    hybrid_high_ndcg_bad_recall = _policy(
        "ranker_plus_popularity",
        ndcg=0.31,
        recall=0.60,
        alpha=1.0,
        beta=0.1,
        gamma=0.0,
    )
    hybrid_valid = _policy(
        "ranker_plus_popularity",
        ndcg=0.30,
        recall=0.69,
        alpha=0.5,
        beta=1.0,
        gamma=0.0,
    )

    (
        selected,
        passing,
        fallback_used,
        experimental,
        reason,
        floor,
    ) = select_recall_constrained_policy(
        results=[popularity, hybrid_high_ndcg_bad_recall, hybrid_valid],
        popularity_metrics=popularity.metrics,
        retention_ratio=0.95,
    )

    assert floor == popularity.metrics["recall@50"] * 0.95
    assert fallback_used is False
    assert experimental is False
    assert selected.policy.policy_id == hybrid_valid.policy.policy_id
    assert hybrid_high_ndcg_bad_recall.policy.policy_id not in [
        result.policy.policy_id for result in passing
    ]
    assert "highest validation NDCG@10 among recall-valid" in reason


def test_recall_constrained_selector_falls_back_to_popularity_when_no_hybrid_passes() -> None:
    popularity = _policy("popularity_only", ndcg=0.26, recall=0.71, mrr=0.21, hr=0.35)
    failing_hybrid = _policy(
        "ranker_plus_popularity",
        ndcg=0.32,
        recall=0.60,
        alpha=1.0,
        beta=0.1,
        gamma=0.0,
    )
    residual = _policy("residual_only", ndcg=0.04, recall=0.25, mrr=0.03, hr=0.09)

    selected, _passing, fallback_used, experimental, reason, _floor = (
        select_recall_constrained_policy(
            results=[popularity, failing_hybrid, residual],
            popularity_metrics=popularity.metrics,
            retention_ratio=0.95,
        )
    )

    assert selected.policy.policy_name == "popularity_only"
    assert fallback_used is True
    assert experimental is True
    assert "selected popularity_only safe fallback" in reason


def test_expanded_grid_is_deterministic() -> None:
    kwargs = {
        "ranker_plus_popularity_alpha_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
        "ranker_plus_popularity_beta_values": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
        "ranker_plus_popularity_plus_residual_alpha_values": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        "ranker_plus_popularity_plus_residual_beta_values": [0.2, 0.5, 1.0, 1.5, 2.0],
        "ranker_plus_popularity_plus_residual_gamma_values": [0.0, 0.05, 0.1, 0.2],
        "two_stage_top_k_focus_values": [10, 20, 30, 50],
        "two_stage_alpha": 1.0,
        "two_stage_beta": 0.1,
        "two_stage_gamma": 0.0,
    }
    first = build_policy_grid(**kwargs)
    second = build_policy_grid(**kwargs)

    first_ids = [policy.policy_id for policy in first]
    second_ids = [policy.policy_id for policy in second]
    assert first_ids == second_ids
    assert len(first_ids) == 199

    # Ensure baseline and two-stage policies are both present.
    assert BASELINE_POLICY_NAMES.issubset({policy.policy_name for policy in first})
    assert any(policy.policy_name == "ranker_topk_popularity_backfill" for policy in first)


def test_two_stage_policy_evaluation_is_deterministic(tmp_path: Path) -> None:
    scored_path = tmp_path / "scored.parquet"
    frame = pl.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "item_idx": [1, 2, 3, 4, 5, 6],
            "target_item_idx": [2, 2, 2, 6, 6, 6],
            "label": [0, 1, 0, 0, 0, 1],
            "residual_rank": [1, 2, 3, 1, 2, 3],
            "ranker_score": [0.7, 0.6, 0.5, 0.9, 0.8, 0.1],
            "residual_score": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "popularity_score": [0.4, 0.9, 0.3, 0.7, 0.6, 0.5],
        }
    )
    frame.write_parquet(scored_path)

    policy = PolicySpec(
        policy_name="ranker_topk_popularity_backfill",
        ranker_weight=1.0,
        popularity_weight=0.1,
        residual_weight=0.0,
        alpha=1.0,
        beta=0.1,
        gamma=0.0,
        top_k_focus=10,
    )

    first, _ = evaluate_policy_grid(
        scored_candidates_path=scored_path,
        policies=[policy],
        normalization_method="query_minmax",
    )
    second, _ = evaluate_policy_grid(
        scored_candidates_path=scored_path,
        policies=[policy],
        normalization_method="query_minmax",
    )

    assert first[0].metrics == second[0].metrics
    assert is_hybrid_policy(policy) is True


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
