"""Select a production scoring policy from ranker/popularity/residual hybrids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import typer

from movie_recsys.modeling.artifacts import save_json, save_markdown
from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.hybrid import (
    PolicyEvaluation,
    build_policy_grid,
    evaluate_policy_grid,
    metadata_feature_leakage_audit,
    metrics_are_finite,
    query_row_counts,
    score_columns_are_finite,
    select_best_policy_result,
)

app = typer.Typer(add_completion=False)

ALPHA_VALUES = [0.5, 0.7, 0.85, 1.0]
BETA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]
GAMMA_VALUES = [0.0, 0.1, 0.2, 0.3]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Required report not found: {path}"
        raise FileNotFoundError(msg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def _extract_metrics(payload: dict[str, Any], key: str) -> dict[str, float]:
    value = payload.get(key)
    if not isinstance(value, dict):
        msg = f"Missing metrics section '{key}'"
        raise ValueError(msg)
    required = ["hr@10", "mrr@10", "ndcg@10", "recall@50"]
    if any(name not in value for name in required):
        msg = f"Metrics section '{key}' is missing required keys"
        raise ValueError(msg)
    return {name: float(value[name]) for name in required}


def _candidate_diagnostics_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    payload = _load_json(path)
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        return None

    summary: dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        split_payload = splits.get(split_name)
        if not isinstance(split_payload, dict):
            continue
        feature_guard = split_payload.get("feature_guard")
        feature_guard_dict = (
            cast(dict[str, Any], feature_guard)
            if isinstance(feature_guard, dict)
            else {}
        )
        summary[split_name] = {
            "queries": int(split_payload.get("queries", 0)),
            "rows": int(split_payload.get("rows", 0)),
            "duplicate_candidate_count": int(
                split_payload.get("duplicate_candidate_count", 0)
            ),
            "positive_count_violations": int(split_payload.get("positive_count_violations", 0)),
            "metadata_columns_excluded": bool(
                feature_guard_dict.get("metadata_columns_excluded", False)
            ),
        }
    return summary


def _serialize_evaluations(results: list[PolicyEvaluation]) -> list[dict[str, Any]]:
    return [
        {
            **result.policy.as_dict(),
            "metrics": result.metrics,
        }
        for result in sorted(
            results,
            key=lambda entry: (
                entry.metrics["ndcg@10"],
                entry.metrics["mrr@10"],
                entry.metrics["hr@10"],
                entry.metrics["recall@50"],
                entry.policy.policy_id,
            ),
            reverse=True,
        )
    ]


def build_selection_payload(
    *,
    normalization_method: str,
    metadata_audit: dict[str, Any],
    val_results: list[PolicyEvaluation],
    selected_val_result: PolicyEvaluation,
    selected_test_metrics: dict[str, float],
    val_counts: dict[str, int],
    test_counts: dict[str, int],
    baseline_val: dict[str, dict[str, float]],
    baseline_test: dict[str, dict[str, float]],
    val_scores_finite: bool,
    test_scores_finite: bool,
    output_json: Path,
    output_md: Path,
    val_scored_path: Path,
    test_scored_path: Path,
    candidate_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_payload = {
        **selected_val_result.policy.as_dict(),
        "validation_metrics": selected_val_result.metrics,
        "test_metrics": selected_test_metrics,
    }

    return {
        "selection_split": "val",
        "test_split": "test",
        "normalization_method": normalization_method,
        "selection_method": "max_val_ndcg_then_mrr_hr_recall",
        "test_split_used_for_weight_selection": False,
        "metadata_feature_leakage_passed": bool(
            metadata_audit.get("metadata_feature_leakage_passed", False)
        ),
        "metadata_leakage_audit": metadata_audit,
        "score_sanity": {
            "val_input_scores_finite": val_scores_finite,
            "test_input_scores_finite": test_scores_finite,
            "selected_val_metrics_finite": metrics_are_finite(selected_val_result.metrics),
            "selected_test_metrics_finite": metrics_are_finite(selected_test_metrics),
        },
        "grid": {
            "alpha_values": ALPHA_VALUES,
            "beta_values": BETA_VALUES,
            "gamma_values": GAMMA_VALUES,
        },
        "validation_results": _serialize_evaluations(val_results),
        "selected_scorer": selected_payload,
        "baseline_metrics": {
            "val": baseline_val,
            "test": baseline_test,
        },
        "query_row_counts": {
            "val": val_counts,
            "test": test_counts,
        },
        "candidate_diagnostics_summary": candidate_diagnostics,
        "artifacts": {
            "selection_json": str(output_json),
            "selection_markdown": str(output_md),
            "val_scored_candidates": str(val_scored_path),
            "test_scored_candidates": str(test_scored_path),
        },
    }


def _selection_markdown(payload: dict[str, Any]) -> str:
    selected = cast(dict[str, Any], payload["selected_scorer"])
    selected_val = cast(dict[str, float], selected["validation_metrics"])
    selected_test = cast(dict[str, float], selected["test_metrics"])
    baseline_test = cast(dict[str, Any], payload["baseline_metrics"])["test"]
    popularity_test = cast(dict[str, float], baseline_test["popularity"])

    lines = [
        "# Production Scorer Selection",
        "",
        "## Summary",
        "",
        f"- normalization_method: {payload['normalization_method']}",
        f"- metadata_feature_leakage_passed: {payload['metadata_feature_leakage_passed']}",
        "- test_split_used_for_weight_selection: "
        f"{payload['test_split_used_for_weight_selection']}",
        "",
        "## Selected Scorer",
        "",
        f"- policy: {selected['policy_name']}",
        f"- policy_id: {selected['policy_id']}",
        f"- alpha: {selected['alpha']}",
        f"- beta: {selected['beta']}",
        f"- gamma: {selected['gamma']}",
        "",
        "### Validation Metrics",
        "",
        f"- hr@10: {selected_val['hr@10']:.6f}",
        f"- mrr@10: {selected_val['mrr@10']:.6f}",
        f"- ndcg@10: {selected_val['ndcg@10']:.6f}",
        f"- recall@50: {selected_val['recall@50']:.6f}",
        "",
        "### Test Metrics",
        "",
        f"- hr@10: {selected_test['hr@10']:.6f}",
        f"- mrr@10: {selected_test['mrr@10']:.6f}",
        f"- ndcg@10: {selected_test['ndcg@10']:.6f}",
        f"- recall@50: {selected_test['recall@50']:.6f}",
        "",
        "### Test Delta vs Popularity",
        "",
        f"- ndcg@10: {selected_test['ndcg@10'] - popularity_test['ndcg@10']:+.6f}",
        "",
        "## Validation Selection Table",
        "",
        "| Policy | alpha | beta | gamma | hr@10 | mrr@10 | ndcg@10 | recall@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in cast(list[dict[str, Any]], payload["validation_results"]):
        metrics = cast(dict[str, float], row["metrics"])
        lines.append(
            "| "
            f"{row['policy_name']} | "
            f"{row['alpha'] if row['alpha'] is not None else '-'} | "
            f"{row['beta'] if row['beta'] is not None else '-'} | "
            f"{row['gamma'] if row['gamma'] is not None else '-'} | "
            f"{metrics['hr@10']:.6f} | "
            f"{metrics['mrr@10']:.6f} | "
            f"{metrics['ndcg@10']:.6f} | "
            f"{metrics['recall@50']:.6f} |"
        )

    return "\n".join(lines) + "\n"


@app.command()
def main(
    config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
    normalization_method: Literal["query_minmax", "query_zscore"] = typer.Option(
        "query_minmax",
        "--normalization-method",
    ),
    output_json: Path | None = typer.Option(None, "--output-json"),
    output_md: Path | None = typer.Option(None, "--output-md"),
    batch_size_rows: int = typer.Option(8192, "--batch-size-rows"),
    max_queries: int | None = typer.Option(None, "--max-queries"),
) -> None:
    ranker_cfg = load_ranker_config(config)
    report_dir = ranker_cfg.paths.ranker_report_dir

    val_scored_path = report_dir / "ranker_eval_val_scored_candidates.parquet"
    test_scored_path = report_dir / "ranker_eval_test_scored_candidates.parquet"

    if not val_scored_path.exists() or not test_scored_path.exists():
        msg = (
            "Missing scored candidate artifacts. Run ranker evaluation first: "
            "scripts/evaluate_ranker.py --split val and --split test"
        )
        raise FileNotFoundError(msg)

    val_report = _load_json(report_dir / "ranker_eval_val.json")
    test_report = _load_json(report_dir / "ranker_eval_test.json")

    baseline_val = {
        "ranker": _extract_metrics(val_report, "ranker"),
        "residual": _extract_metrics(val_report, "residual"),
        "popularity": _extract_metrics(val_report, "popularity"),
    }
    baseline_test = {
        "ranker": _extract_metrics(test_report, "ranker"),
        "residual": _extract_metrics(test_report, "residual"),
        "popularity": _extract_metrics(test_report, "popularity"),
    }

    used_score_columns = ["ranker_score", "popularity_score", "residual_score"]
    schema_names = pl.scan_parquet(val_scored_path).collect_schema().names()
    metadata_audit = metadata_feature_leakage_audit(
        used_score_columns=used_score_columns,
        available_columns=schema_names,
    )

    val_scores_finite = score_columns_are_finite(val_scored_path)
    test_scores_finite = score_columns_are_finite(test_scored_path)
    if not val_scores_finite or not test_scores_finite:
        msg = "Detected NaN/Inf score values in scored candidate artifacts"
        raise ValueError(msg)

    policies = build_policy_grid(
        alpha_values=ALPHA_VALUES,
        beta_values=BETA_VALUES,
        gamma_values=GAMMA_VALUES,
    )

    val_results, val_counts = evaluate_policy_grid(
        scored_candidates_path=val_scored_path,
        policies=policies,
        normalization_method=normalization_method,
        batch_size_rows=batch_size_rows,
        max_queries=max_queries,
    )

    selected_val_result = select_best_policy_result(val_results)
    selected_test_result, test_counts_stream = evaluate_policy_grid(
        scored_candidates_path=test_scored_path,
        policies=[selected_val_result.policy],
        normalization_method=normalization_method,
        batch_size_rows=batch_size_rows,
        max_queries=max_queries,
    )
    selected_test_metrics = selected_test_result[0].metrics

    if max_queries is None:
        val_counts_exact = dict(
            zip(["query_count", "row_count"], query_row_counts(val_scored_path), strict=True)
        )
        test_counts_exact = dict(
            zip(["query_count", "row_count"], query_row_counts(test_scored_path), strict=True)
        )
    else:
        val_counts_exact = val_counts
        test_counts_exact = test_counts_stream

    output_json_path = output_json or (report_dir / "production_scorer_selection.json")
    output_md_path = output_md or (report_dir / "production_scorer_selection.md")

    candidate_diagnostics = _candidate_diagnostics_summary(
        report_dir / "ranker_candidate_diagnostics_full.json"
    )

    payload = build_selection_payload(
        normalization_method=normalization_method,
        metadata_audit=metadata_audit,
        val_results=val_results,
        selected_val_result=selected_val_result,
        selected_test_metrics=selected_test_metrics,
        val_counts=val_counts_exact,
        test_counts=test_counts_exact,
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        val_scores_finite=val_scores_finite,
        test_scores_finite=test_scores_finite,
        output_json=output_json_path,
        output_md=output_md_path,
        val_scored_path=val_scored_path,
        test_scored_path=test_scored_path,
        candidate_diagnostics=candidate_diagnostics,
    )

    save_json(output_json_path, payload)
    save_markdown(output_md_path, _selection_markdown(payload))

    typer.echo(f"selection_json: {output_json_path}")
    typer.echo(f"selection_markdown: {output_md_path}")
    typer.echo(
        {
            "selected_policy": selected_val_result.policy.policy_name,
            "selected_policy_id": selected_val_result.policy.policy_id,
            "selected_weights": {
                "ranker": selected_val_result.policy.ranker_weight,
                "popularity": selected_val_result.policy.popularity_weight,
                "residual": selected_val_result.policy.residual_weight,
            },
            "validation_metrics": selected_val_result.metrics,
            "test_metrics": selected_test_metrics,
            "metadata_feature_leakage_passed": payload["metadata_feature_leakage_passed"],
            "validation_rows": len(payload["validation_results"]),
        }
    )


if __name__ == "__main__":
    app()
