"""Acceptance checks for production scorer policy selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer

from movie_recsys.modeling.artifacts import save_json
from movie_recsys.ranking.acceptance import (
    guard_mlflow_logged,
    guard_no_candidate_leakage,
    guard_no_duplicate_candidates,
    guard_one_positive_per_query,
    relative_drop,
)
from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.hybrid import metrics_are_finite

app = typer.Typer(add_completion=False)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Required file not found: {path}"
        raise FileNotFoundError(msg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def _required_metrics(payload: dict[str, Any], label: str) -> dict[str, float]:
    required = ["hr@10", "mrr@10", "ndcg@10", "recall@50"]
    if any(name not in payload for name in required):
        msg = f"{label} metrics missing required keys"
        raise ValueError(msg)
    return {name: float(payload[name]) for name in required}


def evaluate_production_scorer_acceptance(
    *,
    selected_val: dict[str, float],
    selected_test: dict[str, float],
    popularity_val: dict[str, float],
    popularity_test: dict[str, float],
    popularity_safe_fallback_used: bool,
    guards: dict[str, bool],
    recall50_relative_drop_vs_popularity_val: float,
    recall50_relative_drop_vs_popularity_test: float,
) -> dict[str, Any]:
    """Evaluate acceptance rules for production scorer promotion."""

    val_ndcg_drop = relative_drop(popularity_val["ndcg@10"], selected_val["ndcg@10"])

    rule_one = selected_val["ndcg@10"] > popularity_val["ndcg@10"]
    rule_two = val_ndcg_drop <= 0.01 and (
        selected_val["mrr@10"] > popularity_val["mrr@10"]
        or selected_val["hr@10"] > popularity_val["hr@10"]
    )
    rule_three = (
        selected_test["ndcg@10"] > popularity_test["ndcg@10"]
        and val_ndcg_drop < 0.01
    )
    rule_four = bool(popularity_safe_fallback_used)

    rules = {
        "rule_1_val_ndcg_beats_popularity": rule_one,
        "rule_2_val_within_1pct_and_mrr_or_hr_improves": rule_two,
        "rule_3_test_ndcg_beats_popularity_and_val_drop_lt_1pct": rule_three,
        "rule_4_popularity_safe_fallback_selected": rule_four,
        "any_primary_rule_passed": bool(rule_one or rule_two or rule_three or rule_four),
    }

    failed_reasons: list[str] = []
    if not rules["any_primary_rule_passed"]:
        failed_reasons.append("No primary production-scorer acceptance rule passed")

    for guard_name, passed in guards.items():
        if not passed:
            failed_reasons.append(f"Guard failed: {guard_name}")

    accepted = len(failed_reasons) == 0
    unblocked_mode = (
        "popularity_baseline_only" if popularity_safe_fallback_used else "selected_scorer"
    )
    return {
        "acceptance_passed": accepted,
        "step6_fastapi_unblocked": accepted,
        "step6_unblocked_mode": (unblocked_mode if accepted else "blocked"),
        "primary_rules": rules,
        "guards": guards,
        "failed_reasons": failed_reasons,
        "delta_val_ndcg@10_vs_popularity": float(
            selected_val["ndcg@10"] - popularity_val["ndcg@10"]
        ),
        "delta_test_ndcg@10_vs_popularity": float(
            selected_test["ndcg@10"] - popularity_test["ndcg@10"]
        ),
        "recall50_relative_drop_vs_popularity_val": float(
            recall50_relative_drop_vs_popularity_val
        ),
        "recall50_relative_drop_vs_popularity_test": float(
            recall50_relative_drop_vs_popularity_test
        ),
    }


@app.command()
def main(
    selection: Path = typer.Option(
        Path("artifacts/reports/production_scorer_selection.json"),
        "--selection",
    ),
    config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
    output: Path = typer.Option(
        Path("artifacts/reports/production_scorer_acceptance.json"),
        "--output",
    ),
) -> None:
    selection_payload = _load_json(selection)

    selected_payload = cast(dict[str, Any], selection_payload.get("selected_scorer", {}))
    selected_val = _required_metrics(
        cast(dict[str, Any], selected_payload.get("validation_metrics", {})),
        "selected validation",
    )
    selected_test = _required_metrics(
        cast(dict[str, Any], selected_payload.get("test_metrics", {})),
        "selected test",
    )

    baseline_metrics = cast(dict[str, Any], selection_payload.get("baseline_metrics", {}))
    baseline_val = cast(dict[str, Any], baseline_metrics.get("val", {}))
    baseline_test = cast(dict[str, Any], baseline_metrics.get("test", {}))

    popularity_val = _required_metrics(
        cast(dict[str, Any], baseline_val.get("popularity", {})),
        "popularity validation",
    )
    popularity_test = _required_metrics(
        cast(dict[str, Any], baseline_test.get("popularity", {})),
        "popularity test",
    )

    ranker_cfg = load_ranker_config(config)
    train_candidates = ranker_cfg.candidate_path(split="train", sample=False)
    val_candidates = ranker_cfg.candidate_path(split="val", sample=False)
    test_candidates = ranker_cfg.candidate_path(split="test", sample=False)

    score_sanity = cast(dict[str, Any], selection_payload.get("score_sanity", {}))
    selected_metrics_finite = metrics_are_finite(selected_val) and metrics_are_finite(selected_test)
    popularity_safe_fallback_used = bool(
        selection_payload.get("popularity_safe_fallback_used", False)
    )
    ranker_hybrid_experimental = bool(
        selection_payload.get("ranker_hybrid_experimental", False)
    )

    recall_drop_val = relative_drop(popularity_val["recall@50"], selected_val["recall@50"])
    recall_drop_test = relative_drop(popularity_test["recall@50"], selected_test["recall@50"])
    recall_guard = recall_drop_val <= 0.05

    artifacts = cast(dict[str, Any], selection_payload.get("artifacts", {}))
    selection_markdown = artifacts.get("selection_markdown")
    selection_markdown_exists = (
        isinstance(selection_markdown, str)
        and len(selection_markdown) > 0
        and Path(selection_markdown).exists()
    )

    guards = {
        "no_metadata_leakage": bool(
            selection_payload.get("metadata_feature_leakage_passed", False)
        ),
        "no_candidate_leakage": guard_no_candidate_leakage(
            train_candidates=train_candidates,
            val_candidates=val_candidates,
            test_candidates=test_candidates,
        ),
        "no_duplicate_candidates_per_query": all(
            [
                guard_no_duplicate_candidates(train_candidates),
                guard_no_duplicate_candidates(val_candidates),
                guard_no_duplicate_candidates(test_candidates),
            ]
        ),
        "exactly_one_positive_per_query": all(
            [
                guard_one_positive_per_query(train_candidates),
                guard_one_positive_per_query(val_candidates),
                guard_one_positive_per_query(test_candidates),
            ]
        ),
        "no_nan_or_inf_scores": bool(score_sanity.get("val_input_scores_finite", False))
        and bool(score_sanity.get("test_input_scores_finite", False))
        and selected_metrics_finite,
        "test_split_not_used_for_weight_selection": not bool(
            selection_payload.get("test_split_used_for_weight_selection", True)
        ),
        "recall50_relative_drop_vs_popularity_le_5pct": recall_guard,
        "mlflow_run_logged_or_comparison_artifact_saved": guard_mlflow_logged(
            ranker_cfg.best_checkpoint
        )
        or selection_markdown_exists,
    }

    result = evaluate_production_scorer_acceptance(
        selected_val=selected_val,
        selected_test=selected_test,
        popularity_val=popularity_val,
        popularity_test=popularity_test,
        popularity_safe_fallback_used=popularity_safe_fallback_used,
        guards=guards,
        recall50_relative_drop_vs_popularity_val=recall_drop_val,
        recall50_relative_drop_vs_popularity_test=recall_drop_test,
    )

    output_payload = {
        "selection": str(selection),
        "selected_policy": {
            "policy_name": selected_payload.get("policy_name"),
            "policy_id": selected_payload.get("policy_id"),
            "alpha": selected_payload.get("alpha"),
            "beta": selected_payload.get("beta"),
            "gamma": selected_payload.get("gamma"),
        },
        "popularity_safe_fallback_used": popularity_safe_fallback_used,
        "ranker_hybrid_experimental": ranker_hybrid_experimental,
        "result": result,
    }

    save_json(output, output_payload)
    typer.echo(f"production_scorer_acceptance_path: {output}")
    typer.echo(result)


if __name__ == "__main__":
    app()
