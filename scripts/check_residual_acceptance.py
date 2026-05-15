"""Check full-data residual transformer acceptance against baseline and popularity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from movie_recsys.modeling.artifacts import save_json, save_markdown

app = typer.Typer(add_completion=False)


def _extract_metrics(payload: dict[str, Any], keys: list[str]) -> dict[str, float] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
            return {k: float(v) for k, v in value.items()}
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Required file not found: {path}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline_or_popularity_metrics(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    metrics = _extract_metrics(payload, ["metrics", "baseline", "popularity", "two_tower"])
    if metrics is None:
        msg = f"Unable to find metric dictionary in report: {path}"
        raise ValueError(msg)
    return metrics


def evaluate_acceptance(
    *,
    residual_val: dict[str, float],
    residual_test: dict[str, float],
    baseline_val: dict[str, float],
    baseline_test: dict[str, float],
    popularity_val: dict[str, float],
    popularity_test: dict[str, float],
) -> dict[str, Any]:
    residual_val_ndcg = residual_val["ndcg@10"]
    baseline_val_ndcg = baseline_val["ndcg@10"]

    relative_gap = (
        (baseline_val_ndcg - residual_val_ndcg) / max(abs(baseline_val_ndcg), 1e-12)
        if baseline_val_ndcg >= residual_val_ndcg
        else 0.0
    )

    rule_one = residual_val_ndcg > baseline_val_ndcg
    rule_two = (
        relative_gap <= 0.05
        and (
            residual_val["recall@50"] > baseline_val["recall@50"]
            or residual_val["hr@10"] > baseline_val["hr@10"]
        )
    )

    acceptance_passed = bool(rule_one or rule_two)

    def _delta(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
        return {metric: float(lhs[metric] - rhs[metric]) for metric in lhs}

    result = {
        "acceptance_passed": acceptance_passed,
        "rule_one_residual_ndcg_beats_baseline": rule_one,
        "rule_two_within_5pct_and_recall_or_hr_improves": rule_two,
        "relative_ndcg_gap_to_baseline": float(relative_gap),
        "residual_vs_baseline_val": _delta(residual_val, baseline_val),
        "residual_vs_baseline_test": _delta(residual_test, baseline_test),
        "residual_vs_popularity_val": _delta(residual_val, popularity_val),
        "residual_vs_popularity_test": _delta(residual_test, popularity_test),
        "cl_epidtn_unblocked": acceptance_passed,
        "residual_should_remain_experimental": not acceptance_passed,
    }
    return result


def _to_markdown(
    *,
    summary_path: Path,
    baseline_val_path: Path,
    baseline_test_path: Path,
    popularity_val_path: Path,
    popularity_test_path: Path,
    result: dict[str, Any],
) -> str:
    lines = [
        "# Residual Acceptance Check",
        "",
        f"Summary: {summary_path}",
        f"Baseline val: {baseline_val_path}",
        f"Baseline test: {baseline_test_path}",
        f"Popularity val: {popularity_val_path}",
        f"Popularity test: {popularity_test_path}",
        "",
        f"acceptance_passed: {result['acceptance_passed']}",
        f"cl_epidtn_unblocked: {result['cl_epidtn_unblocked']}",
        f"residual_should_remain_experimental: {result['residual_should_remain_experimental']}",
        "",
        "## Val Deltas vs Baseline",
        "",
        f"- hr@10: {result['residual_vs_baseline_val']['hr@10']:.6f}",
        f"- mrr@10: {result['residual_vs_baseline_val']['mrr@10']:.6f}",
        f"- ndcg@10: {result['residual_vs_baseline_val']['ndcg@10']:.6f}",
        f"- recall@50: {result['residual_vs_baseline_val']['recall@50']:.6f}",
        "",
        "## Test Deltas vs Baseline",
        "",
        f"- hr@10: {result['residual_vs_baseline_test']['hr@10']:.6f}",
        f"- mrr@10: {result['residual_vs_baseline_test']['mrr@10']:.6f}",
        f"- ndcg@10: {result['residual_vs_baseline_test']['ndcg@10']:.6f}",
        f"- recall@50: {result['residual_vs_baseline_test']['recall@50']:.6f}",
        "",
    ]
    return "\n".join(lines) + "\n"


@app.command()
def main(
    summary: Path = typer.Option(
        Path("artifacts/reports/full_residual_transformer_summary.json"),
        "--summary",
    ),
    baseline_val_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_baseline_val.json"),
        "--baseline-val-report",
    ),
    baseline_test_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_baseline_test.json"),
        "--baseline-test-report",
    ),
    popularity_val_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_popularity_val.json"),
        "--popularity-val-report",
    ),
    popularity_test_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_popularity_test.json"),
        "--popularity-test-report",
    ),
    output_json: Path = typer.Option(
        Path("artifacts/reports/full_residual_acceptance.json"),
        "--output-json",
    ),
    output_md: Path = typer.Option(
        Path("artifacts/reports/full_residual_acceptance.md"),
        "--output-md",
    ),
) -> None:
    summary_payload = _load_json(summary)
    residual_val = _extract_metrics(summary_payload, ["val_metrics", "residual_val"])
    residual_test = _extract_metrics(summary_payload, ["test_metrics", "residual_test"])
    if residual_val is None or residual_test is None:
        msg = "Summary must include val_metrics and test_metrics entries"
        raise ValueError(msg)

    baseline_val = _load_baseline_or_popularity_metrics(baseline_val_report)
    baseline_test = _load_baseline_or_popularity_metrics(baseline_test_report)
    popularity_val = _load_baseline_or_popularity_metrics(popularity_val_report)
    popularity_test = _load_baseline_or_popularity_metrics(popularity_test_report)

    result = evaluate_acceptance(
        residual_val=residual_val,
        residual_test=residual_test,
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        popularity_val=popularity_val,
        popularity_test=popularity_test,
    )

    payload = {
        "summary": str(summary),
        "baseline_val_report": str(baseline_val_report),
        "baseline_test_report": str(baseline_test_report),
        "popularity_val_report": str(popularity_val_report),
        "popularity_test_report": str(popularity_test_report),
        "result": result,
    }

    save_json(output_json, payload)
    save_markdown(
        output_md,
        _to_markdown(
            summary_path=summary,
            baseline_val_path=baseline_val_report,
            baseline_test_path=baseline_test_report,
            popularity_val_path=popularity_val_report,
            popularity_test_path=popularity_test_report,
            result=result,
        ),
    )

    typer.echo(payload)


if __name__ == "__main__":
    app()
