"""Tests for production scorer selection payload and acceptance logic."""

from __future__ import annotations

from pathlib import Path

from scripts.check_production_scorer_acceptance import evaluate_production_scorer_acceptance
from scripts.select_production_scorer import build_selection_payload

from movie_recsys.ranking.hybrid import PolicyEvaluation, PolicySpec


def _policy_eval() -> PolicyEvaluation:
    policy = PolicySpec(
        policy_name="ranker_plus_popularity",
        ranker_weight=1.0,
        popularity_weight=0.2,
        residual_weight=0.0,
        alpha=1.0,
        beta=0.2,
        gamma=0.0,
    )
    return PolicyEvaluation(
        policy=policy,
        metrics={
            "hr@10": 0.31,
            "mrr@10": 0.17,
            "ndcg@10": 0.20,
            "recall@50": 0.59,
        },
    )


def test_selected_scorer_report_includes_val_and_test_metrics(tmp_path: Path) -> None:
    selected = _policy_eval()
    payload = build_selection_payload(
        normalization_method="query_minmax",
        metadata_audit={"metadata_feature_leakage_passed": True},
        val_results=[selected],
        recall_valid_results=[selected],
        selected_val_result=selected,
        selected_test_metrics={
            "hr@10": 0.30,
            "mrr@10": 0.16,
            "ndcg@10": 0.19,
            "recall@50": 0.58,
        },
        selection_reason="validation-only recall-constrained best",
        recall_constraint_value=0.475,
        selected_by_validation_only=True,
        popularity_safe_fallback_used=False,
        ranker_hybrid_experimental=False,
        val_counts={"query_count": 2, "row_count": 6},
        test_counts={"query_count": 2, "row_count": 6},
        baseline_val={
            "ranker": {"hr@10": 0.2, "mrr@10": 0.1, "ndcg@10": 0.1, "recall@50": 0.4},
            "residual": {"hr@10": 0.1, "mrr@10": 0.05, "ndcg@10": 0.06, "recall@50": 0.3},
            "popularity": {"hr@10": 0.25, "mrr@10": 0.12, "ndcg@10": 0.15, "recall@50": 0.5},
        },
        baseline_test={
            "ranker": {"hr@10": 0.2, "mrr@10": 0.1, "ndcg@10": 0.1, "recall@50": 0.4},
            "residual": {"hr@10": 0.1, "mrr@10": 0.05, "ndcg@10": 0.06, "recall@50": 0.3},
            "popularity": {"hr@10": 0.25, "mrr@10": 0.12, "ndcg@10": 0.15, "recall@50": 0.5},
        },
        val_scores_finite=True,
        test_scores_finite=True,
        output_json=tmp_path / "selection.json",
        output_md=tmp_path / "selection.md",
        val_scored_path=tmp_path / "val.parquet",
        test_scored_path=tmp_path / "test.parquet",
        candidate_diagnostics=None,
    )

    selected_payload = payload["selected_scorer"]
    assert "validation_metrics" in selected_payload
    assert "test_metrics" in selected_payload
    assert payload["selected_by_validation_only"] is True
    assert payload["test_split_used_for_weight_selection"] is False


def test_payload_fallback_flag_is_serialized_for_safe_popularity_mode(tmp_path: Path) -> None:
    selected = _policy_eval()
    payload = build_selection_payload(
        normalization_method="query_minmax",
        metadata_audit={"metadata_feature_leakage_passed": True},
        val_results=[selected],
        recall_valid_results=[selected],
        selected_val_result=selected,
        selected_test_metrics=selected.metrics,
        selection_reason="No hybrid met recall; selected popularity_only safe fallback",
        recall_constraint_value=0.65,
        selected_by_validation_only=True,
        popularity_safe_fallback_used=True,
        ranker_hybrid_experimental=True,
        val_counts={"query_count": 2, "row_count": 6},
        test_counts={"query_count": 2, "row_count": 6},
        baseline_val={
            "ranker": {"hr@10": 0.2, "mrr@10": 0.1, "ndcg@10": 0.1, "recall@50": 0.4},
            "residual": {"hr@10": 0.1, "mrr@10": 0.05, "ndcg@10": 0.06, "recall@50": 0.3},
            "popularity": {"hr@10": 0.25, "mrr@10": 0.12, "ndcg@10": 0.15, "recall@50": 0.7},
        },
        baseline_test={
            "ranker": {"hr@10": 0.2, "mrr@10": 0.1, "ndcg@10": 0.1, "recall@50": 0.4},
            "residual": {"hr@10": 0.1, "mrr@10": 0.05, "ndcg@10": 0.06, "recall@50": 0.3},
            "popularity": {"hr@10": 0.25, "mrr@10": 0.12, "ndcg@10": 0.15, "recall@50": 0.7},
        },
        val_scores_finite=True,
        test_scores_finite=True,
        output_json=tmp_path / "selection.json",
        output_md=tmp_path / "selection.md",
        val_scored_path=tmp_path / "val.parquet",
        test_scored_path=tmp_path / "test.parquet",
        candidate_diagnostics=None,
    )
    assert payload["popularity_safe_fallback_used"] is True
    assert payload["ranker_hybrid_experimental"] is True


def test_acceptance_checker_passes_with_valid_rules_and_guards() -> None:
    result = evaluate_production_scorer_acceptance(
        selected_val={"hr@10": 0.30, "mrr@10": 0.18, "ndcg@10": 0.21, "recall@50": 0.60},
        selected_test={"hr@10": 0.31, "mrr@10": 0.19, "ndcg@10": 0.22, "recall@50": 0.61},
        popularity_val={"hr@10": 0.29, "mrr@10": 0.17, "ndcg@10": 0.20, "recall@50": 0.60},
        popularity_test={"hr@10": 0.30, "mrr@10": 0.18, "ndcg@10": 0.21, "recall@50": 0.61},
        popularity_safe_fallback_used=False,
        guards={
            "no_metadata_leakage": True,
            "no_candidate_leakage": True,
            "no_duplicate_candidates_per_query": True,
            "exactly_one_positive_per_query": True,
            "no_nan_or_inf_scores": True,
            "test_split_not_used_for_weight_selection": True,
            "recall50_relative_drop_vs_popularity_le_5pct": True,
            "mlflow_run_logged_or_comparison_artifact_saved": True,
        },
        recall50_relative_drop_vs_popularity_val=0.0,
        recall50_relative_drop_vs_popularity_test=0.0,
    )

    assert result["acceptance_passed"] is True
    assert result["step6_fastapi_unblocked"] is True


def test_acceptance_checker_passes_in_popularity_safe_fallback_mode() -> None:
    result = evaluate_production_scorer_acceptance(
        selected_val={"hr@10": 0.29, "mrr@10": 0.17, "ndcg@10": 0.20, "recall@50": 0.60},
        selected_test={"hr@10": 0.30, "mrr@10": 0.18, "ndcg@10": 0.21, "recall@50": 0.61},
        popularity_val={"hr@10": 0.29, "mrr@10": 0.17, "ndcg@10": 0.20, "recall@50": 0.60},
        popularity_test={"hr@10": 0.30, "mrr@10": 0.18, "ndcg@10": 0.21, "recall@50": 0.61},
        popularity_safe_fallback_used=True,
        guards={
            "no_metadata_leakage": True,
            "no_candidate_leakage": True,
            "no_duplicate_candidates_per_query": True,
            "exactly_one_positive_per_query": True,
            "no_nan_or_inf_scores": True,
            "test_split_not_used_for_weight_selection": True,
            "recall50_relative_drop_vs_popularity_le_5pct": True,
            "mlflow_run_logged_or_comparison_artifact_saved": True,
        },
        recall50_relative_drop_vs_popularity_val=0.0,
        recall50_relative_drop_vs_popularity_test=0.0,
    )
    assert result["acceptance_passed"] is True
    assert result["step6_unblocked_mode"] == "popularity_baseline_only"


def test_acceptance_checker_fails_with_bad_rules_or_guards() -> None:
    result = evaluate_production_scorer_acceptance(
        selected_val={"hr@10": 0.20, "mrr@10": 0.10, "ndcg@10": 0.10, "recall@50": 0.40},
        selected_test={"hr@10": 0.21, "mrr@10": 0.11, "ndcg@10": 0.11, "recall@50": 0.39},
        popularity_val={"hr@10": 0.30, "mrr@10": 0.20, "ndcg@10": 0.25, "recall@50": 0.60},
        popularity_test={"hr@10": 0.31, "mrr@10": 0.21, "ndcg@10": 0.26, "recall@50": 0.61},
        popularity_safe_fallback_used=False,
        guards={
            "no_metadata_leakage": False,
            "no_candidate_leakage": True,
            "no_duplicate_candidates_per_query": True,
            "exactly_one_positive_per_query": True,
            "no_nan_or_inf_scores": True,
            "test_split_not_used_for_weight_selection": True,
            "recall50_relative_drop_vs_popularity_le_5pct": False,
            "mlflow_run_logged_or_comparison_artifact_saved": True,
        },
        recall50_relative_drop_vs_popularity_val=0.33,
        recall50_relative_drop_vs_popularity_test=0.36,
    )

    assert result["acceptance_passed"] is False
    assert any(
        "No primary production-scorer acceptance rule passed" in text
        for text in result["failed_reasons"]
    )
    assert any("no_metadata_leakage" in text for text in result["failed_reasons"])
