"""Evaluate contrastive acceptance gates for sample/full CL workflows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import typer

from movie_recsys.modeling.artifacts import save_json, save_markdown

app = typer.Typer(add_completion=False)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Required file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return payload


def _is_metric_dict(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"hr@10", "mrr@10", "ndcg@10", "recall@50"}
    if not required.issubset(payload.keys()):
        return False
    return all(isinstance(payload[key], (int, float)) for key in required)


def _to_float_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "hr@10": float(payload["hr@10"]),
        "mrr@10": float(payload["mrr@10"]),
        "ndcg@10": float(payload["ndcg@10"]),
        "recall@50": float(payload["recall@50"]),
    }


def _summary_metric(summary: dict[str, Any], key: str) -> dict[str, float] | None:
    value = summary.get(key)
    if isinstance(value, dict) and _is_metric_dict(value):
        return _to_float_metrics(value)
    return None


def _compare_metric(
    compare_payload: dict[str, Any] | None,
    *,
    split: str,
    model: str,
) -> dict[str, float] | None:
    if compare_payload is None:
        return None
    split_payload = compare_payload.get(split)
    if not isinstance(split_payload, dict):
        return None
    model_payload = split_payload.get(model)
    if isinstance(model_payload, dict) and _is_metric_dict(model_payload):
        return _to_float_metrics(model_payload)
    return None


def _resolve_metrics(
    *,
    summary: dict[str, Any],
    compare_payload: dict[str, Any] | None,
    sample: bool,
) -> dict[str, dict[str, float]]:
    cl_val = _summary_metric(summary, "best_cl_val") or _summary_metric(summary, "cl_val")
    cl_test = _summary_metric(summary, "best_cl_test") or _summary_metric(summary, "cl_test")

    residual_val = _summary_metric(summary, "residual_val")
    residual_test = _summary_metric(summary, "residual_test")
    popularity_val = _summary_metric(summary, "popularity_val")
    popularity_test = _summary_metric(summary, "popularity_test")
    baseline_val = _summary_metric(summary, "baseline_val")
    baseline_test = _summary_metric(summary, "baseline_test")

    residual_val = residual_val or _compare_metric(
        compare_payload,
        split="val",
        model="residual_transformer",
    )
    residual_test = residual_test or _compare_metric(
        compare_payload,
        split="test",
        model="residual_transformer",
    )
    popularity_val = popularity_val or _compare_metric(
        compare_payload,
        split="val",
        model="popularity",
    )
    popularity_test = popularity_test or _compare_metric(
        compare_payload,
        split="test",
        model="popularity",
    )
    baseline_val = baseline_val or _compare_metric(compare_payload, split="val", model="baseline")
    baseline_test = baseline_test or _compare_metric(
        compare_payload,
        split="test",
        model="baseline",
    )

    required = {
        "cl_val": cl_val,
        "cl_test": cl_test,
        "residual_val": residual_val,
        "residual_test": residual_test,
        "popularity_val": popularity_val,
        "popularity_test": popularity_test,
    }
    if sample:
        required["baseline_val"] = baseline_val
        required["baseline_test"] = baseline_test

    missing = [name for name, metrics in required.items() if metrics is None]
    if missing:
        msg = (
            "Missing required metrics for acceptance evaluation: "
            + ", ".join(sorted(missing))
        )
        raise ValueError(msg)

    resolved = {
        "cl_val": cl_val,
        "cl_test": cl_test,
        "residual_val": residual_val,
        "residual_test": residual_test,
        "popularity_val": popularity_val,
        "popularity_test": popularity_test,
    }
    if baseline_val is not None:
        resolved["baseline_val"] = baseline_val
    if baseline_test is not None:
        resolved["baseline_test"] = baseline_test
    return {key: value for key, value in resolved.items() if value is not None}


def _best_trial_losses(summary: dict[str, Any]) -> dict[str, float] | None:
    trials = summary.get("contrastive_trials")
    if not isinstance(trials, list):
        return None
    best_name = str(summary.get("best_cl_name", ""))
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        if str(trial.get("name", "")) != best_name:
            continue
        keys = [
            "final_retrieval_loss",
            "final_user_contrastive_loss",
            "final_item_contrastive_loss",
            "final_alignment_contrastive_loss",
            "final_residual_anchor_loss",
            "final_total_loss",
        ]
        losses: dict[str, float] = {}
        for key in keys:
            if key in trial and isinstance(trial[key], (int, float)):
                losses[key] = float(trial[key])
        return losses
    return None


def _relative_drop(reference: float, value: float) -> float:
    if reference <= 0.0:
        return 0.0
    return max((reference - value) / reference, 0.0)


def _sample_rules(metrics: dict[str, dict[str, float]]) -> dict[str, bool]:
    cl_val = metrics["cl_val"]
    cl_test = metrics["cl_test"]
    residual_val = metrics["residual_val"]
    popularity_val = metrics["popularity_val"]
    popularity_test = metrics["popularity_test"]

    rule_one = cl_val["ndcg@10"] > popularity_val["ndcg@10"]
    rule_two = (
        cl_val["ndcg@10"] > residual_val["ndcg@10"]
        and cl_val["ndcg@10"] >= (popularity_val["ndcg@10"] - 0.001)
    )
    rule_three = (
        cl_test["ndcg@10"] > popularity_test["ndcg@10"]
        and cl_val["ndcg@10"] >= (popularity_val["ndcg@10"] - 0.0015)
    )
    return {
        "rule_1_val_beats_popularity": rule_one,
        "rule_2_val_beats_residual_and_close_to_popularity": rule_two,
        "rule_3_test_beats_popularity_and_val_close": rule_three,
        "any_primary_rule_passed": bool(rule_one or rule_two or rule_three),
    }


def _full_rules(metrics: dict[str, dict[str, float]]) -> dict[str, bool]:
    cl_val = metrics["cl_val"]
    cl_test = metrics["cl_test"]
    residual_val = metrics["residual_val"]
    residual_test = metrics["residual_test"]

    within_two_percent = cl_val["ndcg@10"] >= (residual_val["ndcg@10"] * 0.98)
    improved_recall_or_hr = (
        cl_val["recall@50"] > residual_val["recall@50"]
        or cl_val["hr@10"] > residual_val["hr@10"]
    )
    rule_one = cl_val["ndcg@10"] > residual_val["ndcg@10"]
    rule_two = within_two_percent and improved_recall_or_hr
    rule_three = cl_test["ndcg@10"] > residual_test["ndcg@10"] and within_two_percent

    return {
        "rule_1_val_beats_residual": rule_one,
        "rule_2_within_2pct_and_improves_recall_or_hr": rule_two,
        "rule_3_test_improves_and_val_drop_lt_2pct": rule_three,
        "any_primary_rule_passed": bool(rule_one or rule_two or rule_three),
    }


def evaluate_acceptance(
    *,
    summary: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    sample: bool,
    faiss_top10_match: bool,
    losses: dict[str, float] | None,
) -> dict[str, Any]:
    primary_rules = _sample_rules(metrics) if sample else _full_rules(metrics)

    cl_val = metrics["cl_val"]
    residual_val = metrics["residual_val"]
    popularity_val = metrics["popularity_val"]

    recall_relative_drop = _relative_drop(residual_val["recall@50"], cl_val["recall@50"])
    recall_guard_passed = recall_relative_drop <= 0.05

    losses_available = losses is not None and len(losses) > 0
    losses_finite = False
    if losses_available and losses is not None:
        losses_finite = all(math.isfinite(value) for value in losses.values())

    secondary_checks = {
        "faiss_top10_match": bool(faiss_top10_match),
        "recall50_relative_drop_le_5pct": recall_guard_passed,
        "loss_values_available": losses_available,
        "loss_values_finite": losses_finite,
    }

    failed_reasons: list[str] = []
    if not primary_rules["any_primary_rule_passed"]:
        failed_reasons.append("No primary acceptance rule passed")
    if not secondary_checks["faiss_top10_match"]:
        failed_reasons.append("FAISS top10 parity check failed")
    if sample and not secondary_checks["recall50_relative_drop_le_5pct"]:
        failed_reasons.append("Recall@50 collapsed by more than 5% vs residual")
    if not secondary_checks["loss_values_available"]:
        failed_reasons.append("Loss diagnostics are missing from summary")
    elif not secondary_checks["loss_values_finite"]:
        failed_reasons.append("Non-finite retrieval or component losses detected")

    acceptance_passed = not failed_reasons
    return {
        "sample_mode": sample,
        "acceptance_passed": acceptance_passed,
        "full_data_cl_allowed": acceptance_passed,
        "primary_rules": primary_rules,
        "secondary_checks": secondary_checks,
        "failed_reasons": failed_reasons,
        "gap_to_popularity_val_ndcg": (
            float(popularity_val["ndcg@10"] - cl_val["ndcg@10"])
        ),
        "gap_to_residual_val_ndcg": float(residual_val["ndcg@10"] - cl_val["ndcg@10"]),
        "recall50_relative_drop_vs_residual": float(recall_relative_drop),
        "best_cl_name": str(summary.get("best_cl_name", "")),
        "best_cl_run_id": str(summary.get("best_cl_run_id", "")),
        "best_cl_run_url": str(summary.get("best_cl_run_url", "")),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    result = payload["result"]
    lines = [
        "# Contrastive Acceptance",
        "",
        f"summary: {payload['summary']}",
        f"comparison_report: {payload['comparison_report']}",
        f"faiss_report: {payload['faiss_report']}",
        f"sample_mode: {result['sample_mode']}",
        f"acceptance_passed: {result['acceptance_passed']}",
        f"full_data_cl_allowed: {result['full_data_cl_allowed']}",
        "",
        "## Primary Rules",
        "",
    ]
    for key, value in result["primary_rules"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Secondary Checks",
            "",
        ]
    )
    for key, value in result["secondary_checks"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Gaps",
            "",
            f"- gap_to_popularity_val_ndcg: {result['gap_to_popularity_val_ndcg']:.6f}",
            f"- gap_to_residual_val_ndcg: {result['gap_to_residual_val_ndcg']:.6f}",
            (
                "- recall50_relative_drop_vs_residual: "
                f"{result['recall50_relative_drop_vs_residual']:.6f}"
            ),
            "",
            "## Failure Reasons",
            "",
        ]
    )
    if result["failed_reasons"]:
        for reason in result["failed_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


@app.command()
def main(
    summary: Path = typer.Option(
        Path("artifacts/reports/contrastive_ablation_sample.json"),
        "--summary",
        "--ablation-summary",
    ),
    sample: bool = typer.Option(False, "--sample"),
    comparison_report: Path | None = typer.Option(None, "--comparison-report"),
    faiss_report: Path | None = typer.Option(None, "--faiss-report"),
    output_json: Path | None = typer.Option(None, "--output-json"),
    output_md: Path | None = typer.Option(None, "--output-md"),
) -> None:
    summary_payload = _load_json(summary)

    compare_candidates = []
    if comparison_report is not None:
        compare_candidates.append(comparison_report)
    if sample:
        compare_candidates.append(Path("artifacts/reports/retriever_ablation_sample.json"))
    else:
        compare_candidates.append(Path("artifacts/reports/retriever_ablation_full.json"))

    compare_payload: dict[str, Any] | None = None
    selected_compare = ""
    for candidate in compare_candidates:
        if candidate.exists():
            compare_payload = _load_json(candidate)
            selected_compare = str(candidate)
            break

    metrics = _resolve_metrics(
        summary=summary_payload,
        compare_payload=compare_payload,
        sample=sample,
    )

    faiss_candidates = []
    if faiss_report is not None:
        faiss_candidates.append(faiss_report)
    if sample:
        faiss_candidates.append(
            Path("artifacts/reports/faiss_export_report_cl_residual_transformer.json")
        )
    else:
        faiss_candidates.append(
            Path("artifacts/reports/faiss_export_report_cl_residual_transformer.json")
        )

    faiss_top10_match = False
    selected_faiss = ""
    for candidate in faiss_candidates:
        if candidate.exists():
            faiss_payload = _load_json(candidate)
            faiss_top10_match = bool(faiss_payload.get("top10_match", False))
            selected_faiss = str(candidate)
            break

    losses = _best_trial_losses(summary_payload)
    result = evaluate_acceptance(
        summary=summary_payload,
        metrics=metrics,
        sample=sample,
        faiss_top10_match=faiss_top10_match,
        losses=losses,
    )

    if output_json is None:
        if sample:
            output_json = Path("artifacts/reports/contrastive_acceptance_sample.json")
        else:
            output_json = Path("artifacts/reports/contrastive_acceptance_full.json")
    if output_md is None:
        if sample:
            output_md = Path("artifacts/reports/contrastive_acceptance_sample.md")
        else:
            output_md = Path("artifacts/reports/contrastive_acceptance_full.md")

    payload = {
        "summary": str(summary),
        "comparison_report": selected_compare,
        "faiss_report": selected_faiss,
        "metrics": metrics,
        "losses": losses,
        "result": result,
    }

    save_json(output_json, payload)
    save_markdown(output_md, _render_markdown(payload))
    typer.echo(payload)


if __name__ == "__main__":
    app()
