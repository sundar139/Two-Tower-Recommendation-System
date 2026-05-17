"""Check sample CL acceptance before allowing full-data CL training."""

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


def _load_metric_report(path: Path, keys: list[str]) -> dict[str, float]:
    payload = _load_json(path)
    metrics = _extract_metrics(payload, keys)
    if metrics is None:
        msg = f"Unable to find metric dictionary in report: {path}"
        raise ValueError(msg)
    return metrics


def evaluate_acceptance(
    *,
    cl_val: dict[str, float],
    cl_test: dict[str, float],
    residual_val: dict[str, float],
    residual_test: dict[str, float],
    baseline_val: dict[str, float],
    baseline_test: dict[str, float],
    popularity_val: dict[str, float],
    popularity_test: dict[str, float],
    faiss_top10_match: bool,
) -> dict[str, Any]:
    rule_one = cl_val["ndcg@10"] > popularity_val["ndcg@10"]
    rule_two = cl_val["ndcg@10"] >= (residual_val["ndcg@10"] - 0.001)
    rule_three = cl_test["ndcg@10"] >= (baseline_test["ndcg@10"] - 0.002)
    faiss_rule = bool(faiss_top10_match)

    acceptance_passed = bool(rule_one and rule_two and rule_three and faiss_rule)

    def _delta(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
        return {metric: float(lhs[metric] - rhs[metric]) for metric in lhs}

    return {
        "acceptance_passed": acceptance_passed,
        "rule_one_val_ndcg_beats_popularity": rule_one,
        "rule_two_val_ndcg_within_0p001_of_residual": rule_two,
        "rule_three_test_ndcg_not_worse_than_baseline_by_0p002": rule_three,
        "faiss_top10_match_rule": faiss_rule,
        "cl_vs_popularity_val": _delta(cl_val, popularity_val),
        "cl_vs_popularity_test": _delta(cl_test, popularity_test),
        "cl_vs_baseline_val": _delta(cl_val, baseline_val),
        "cl_vs_baseline_test": _delta(cl_test, baseline_test),
        "cl_vs_residual_val": _delta(cl_val, residual_val),
        "cl_vs_residual_test": _delta(cl_test, residual_test),
        "full_data_cl_allowed": acceptance_passed,
    }


def _to_markdown(
    *,
    ablation_summary: Path,
    cl_val_report: Path,
    cl_test_report: Path,
    residual_val_report: Path,
    residual_test_report: Path,
    baseline_val_report: Path,
    baseline_test_report: Path,
    popularity_val_report: Path,
    popularity_test_report: Path,
    faiss_report: Path,
    result: dict[str, Any],
) -> str:
    lines = [
        "# Contrastive Acceptance Check",
        "",
        f"Ablation summary: {ablation_summary}",
        f"CL val report: {cl_val_report}",
        f"CL test report: {cl_test_report}",
        f"Residual val report: {residual_val_report}",
        f"Residual test report: {residual_test_report}",
        f"Baseline val report: {baseline_val_report}",
        f"Baseline test report: {baseline_test_report}",
        f"Popularity val report: {popularity_val_report}",
        f"Popularity test report: {popularity_test_report}",
        f"FAISS report: {faiss_report}",
        "",
        f"acceptance_passed: {result['acceptance_passed']}",
        f"full_data_cl_allowed: {result['full_data_cl_allowed']}",
        "",
        "## Rules",
        "",
        f"- val_ndcg_beats_popularity: {result['rule_one_val_ndcg_beats_popularity']}",
        (
            "- val_ndcg_within_0p001_of_residual: "
            f"{result['rule_two_val_ndcg_within_0p001_of_residual']}"
        ),
        (
            "- test_ndcg_not_worse_than_baseline_by_0p002: "
            f"{result['rule_three_test_ndcg_not_worse_than_baseline_by_0p002']}"
        ),
        f"- faiss_top10_match: {result['faiss_top10_match_rule']}",
        "",
        "## Val Deltas",
        "",
        f"- cl_vs_baseline_ndcg@10: {result['cl_vs_baseline_val']['ndcg@10']:.6f}",
        f"- cl_vs_residual_ndcg@10: {result['cl_vs_residual_val']['ndcg@10']:.6f}",
        f"- cl_vs_popularity_ndcg@10: {result['cl_vs_popularity_val']['ndcg@10']:.6f}",
        "",
        "## Test Deltas",
        "",
        f"- cl_vs_baseline_ndcg@10: {result['cl_vs_baseline_test']['ndcg@10']:.6f}",
        f"- cl_vs_residual_ndcg@10: {result['cl_vs_residual_test']['ndcg@10']:.6f}",
        f"- cl_vs_popularity_ndcg@10: {result['cl_vs_popularity_test']['ndcg@10']:.6f}",
        "",
    ]
    return "\n".join(lines)


@app.command()
def main(
    ablation_summary: Path = typer.Option(
        Path("artifacts/reports/contrastive_ablation_sample.json"),
        "--ablation-summary",
    ),
    cl_val_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_cl_residual_transformer_val.json"),
        "--cl-val-report",
    ),
    cl_test_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_cl_residual_transformer_test.json"),
        "--cl-test-report",
    ),
    residual_val_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_residual_transformer_val.json"),
        "--residual-val-report",
    ),
    residual_test_report: Path = typer.Option(
        Path("artifacts/reports/retrieval_eval_residual_transformer_test.json"),
        "--residual-test-report",
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
    faiss_report: Path = typer.Option(
        Path("artifacts/reports/faiss_export_report_cl_residual_transformer.json"),
        "--faiss-report",
    ),
    output_json: Path = typer.Option(
        Path("artifacts/reports/contrastive_acceptance_sample.json"),
        "--output-json",
    ),
    output_md: Path = typer.Option(
        Path("artifacts/reports/contrastive_acceptance_sample.md"),
        "--output-md",
    ),
) -> None:
    ablation_payload = _load_json(ablation_summary)

    cl_val = _load_metric_report(cl_val_report, ["cl_residual_transformer"])
    cl_test = _load_metric_report(cl_test_report, ["cl_residual_transformer"])
    residual_val = _load_metric_report(residual_val_report, ["residual_transformer"])
    residual_test = _load_metric_report(residual_test_report, ["residual_transformer"])
    baseline_val = _load_metric_report(baseline_val_report, ["baseline", "two_tower", "metrics"])
    baseline_test = _load_metric_report(
        baseline_test_report,
        ["baseline", "two_tower", "metrics"],
    )
    popularity_val = _load_metric_report(popularity_val_report, ["popularity", "metrics"])
    popularity_test = _load_metric_report(popularity_test_report, ["popularity", "metrics"])

    faiss_payload = _load_json(faiss_report)
    faiss_match = bool(faiss_payload.get("top10_match", False))

    result = evaluate_acceptance(
        cl_val=cl_val,
        cl_test=cl_test,
        residual_val=residual_val,
        residual_test=residual_test,
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        popularity_val=popularity_val,
        popularity_test=popularity_test,
        faiss_top10_match=faiss_match,
    )

    payload = {
        "ablation_summary": str(ablation_summary),
        "best_cl_name": ablation_payload.get("best_cl_name", ""),
        "best_cl_run_id": ablation_payload.get("best_cl_run_id", ""),
        "best_cl_run_url": ablation_payload.get("best_cl_run_url", ""),
        "cl_val_report": str(cl_val_report),
        "cl_test_report": str(cl_test_report),
        "residual_val_report": str(residual_val_report),
        "residual_test_report": str(residual_test_report),
        "baseline_val_report": str(baseline_val_report),
        "baseline_test_report": str(baseline_test_report),
        "popularity_val_report": str(popularity_val_report),
        "popularity_test_report": str(popularity_test_report),
        "faiss_report": str(faiss_report),
        "faiss_top10_match": faiss_match,
        "result": result,
    }

    save_json(output_json, payload)
    save_markdown(
        output_md,
        _to_markdown(
            ablation_summary=ablation_summary,
            cl_val_report=cl_val_report,
            cl_test_report=cl_test_report,
            residual_val_report=residual_val_report,
            residual_test_report=residual_test_report,
            baseline_val_report=baseline_val_report,
            baseline_test_report=baseline_test_report,
            popularity_val_report=popularity_val_report,
            popularity_test_report=popularity_test_report,
            faiss_report=faiss_report,
            result=result,
        ),
    )

    typer.echo(payload)


if __name__ == "__main__":
    app()
