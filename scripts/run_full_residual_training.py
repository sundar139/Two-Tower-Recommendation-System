"""Safe wrapper for full-data residual transformer training and validation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import Any

import typer

from movie_recsys.modeling.artifacts import save_json, save_markdown
from movie_recsys.training.config import RetrievalConfig, load_retrieval_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer(add_completion=False)


def _to_bool(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"true", "1", "yes"}


def _parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _parse_mlflow_fields(output: str) -> tuple[str, str]:
    kv = _parse_key_values(output)
    return kv.get("mlflow_run_id", ""), kv.get("mlflow_run_url", "")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def build_train_command(
    *,
    config: Path,
    baseline_checkpoint: Path,
    resume_from: Path | None,
    max_runtime_hours: float | None,
    run_name: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_retriever.py",
        "--config",
        str(config),
        "--model-type",
        "residual_transformer",
        "--init-from-baseline",
        str(baseline_checkpoint),
        "--checkpoint-every-epoch",
        "--eval-every-epoch",
        "--save-last",
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    if max_runtime_hours is not None:
        command.extend(["--max-runtime-hours", str(max_runtime_hours)])
    if run_name:
        command.extend(["--run-name", run_name])
    return command


def build_eval_command(
    *,
    config: Path,
    model: str,
    split: str,
    checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluate_retriever.py",
        "--config",
        str(config),
        "--model",
        model,
        "--split",
        split,
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    return command


def build_export_command(*, config: Path, checkpoint: Path | None) -> list[str]:
    command = [
        sys.executable,
        "scripts/export_faiss_index.py",
        "--config",
        str(config),
        "--model-type",
        "residual_transformer",
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    return command


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Required report not found: {path}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(payload: dict[str, Any], keys: list[str]) -> dict[str, float] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and all(
            isinstance(metric_value, (int, float)) for metric_value in value.values()
        ):
            return {k: float(v) for k, v in value.items()}
    return None


def _delta(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
    return {metric: float(lhs[metric] - rhs[metric]) for metric in lhs}


def _evaluate_acceptance(
    *,
    residual_val: dict[str, float],
    baseline_val: dict[str, float],
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
    passed = bool(rule_one or rule_two)
    return {
        "acceptance_passed": passed,
        "rule_one_residual_ndcg_beats_baseline": rule_one,
        "rule_two_within_5pct_and_recall_or_hr_improves": rule_two,
        "relative_ndcg_gap_to_baseline": float(relative_gap),
        "cl_epidtn_unblocked": passed,
        "residual_should_remain_experimental": not passed,
    }


def _load_baseline_metrics(report_dir: Path, split: str) -> dict[str, float]:
    candidates = [
        report_dir / f"retrieval_eval_baseline_{split}.json",
        report_dir / f"two_tower_{split}.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = _load_json(candidate)
        metrics = _extract_metrics(payload, ["baseline", "two_tower", "metrics"])
        if metrics is not None:
            return metrics
    msg = f"Unable to resolve baseline metrics for split={split} in {report_dir}"
    raise FileNotFoundError(msg)


def _ensure_env_copy() -> None:
    env_example = PROJECT_ROOT / "env.example"
    env_file = PROJECT_ROOT / ".env"
    if not env_example.exists():
        msg = f"env.example not found at {env_example}"
        raise FileNotFoundError(msg)
    shutil.copy2(env_example, env_file)


def _verify_full_data_inputs(cfg: RetrievalConfig) -> None:
    required = [
        cfg.train_path,
        cfg.val_path,
        cfg.test_path,
        cfg.users_path,
        cfg.items_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        msg = "Missing required processed full-data files: " + ", ".join(missing)
        raise FileNotFoundError(msg)


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Full Residual Transformer Summary",
        "",
        f"status: {summary['status']}",
        f"train_runtime_seconds: {summary['train_runtime_seconds']:.2f}",
        f"train_run_url: {summary.get('train_run_url', '')}",
        f"resume_command: {summary.get('resume_command', '')}",
        "",
    ]

    if "val_metrics" in summary:
        val = summary["val_metrics"]
        test = summary["test_metrics"]
        lines.extend(
            [
                "## Residual Metrics",
                "",
                f"- val ndcg@10: {val['ndcg@10']:.6f}",
                f"- test ndcg@10: {test['ndcg@10']:.6f}",
                f"- faiss top10_match: {summary['faiss_top10_match']}",
                f"- faiss top200 latency ms: {summary['faiss_top200_latency_ms']:.6f}",
                "",
                "## Acceptance",
                "",
                f"- passes_acceptance: {summary['passes_acceptance_criteria']}",
                f"- cl_epidtn_unblocked: {summary['cl_epidtn_unblocked']}",
                f"- residual_experimental: {summary['residual_should_remain_experimental']}",
                "",
            ]
        )

    return "\n".join(lines)


def _ensure_reference_reports(report_dir: Path) -> None:
    required = [
        (report_dir / "retrieval_eval_baseline_val.json", "baseline", "val"),
        (report_dir / "retrieval_eval_baseline_test.json", "baseline", "test"),
        (report_dir / "retrieval_eval_popularity_val.json", "popularity", "val"),
        (report_dir / "retrieval_eval_popularity_test.json", "popularity", "test"),
    ]
    reference_config = Path("configs/retrieval.yaml")
    for report_path, model_name, split in required:
        if report_path.exists():
            continue
        _run_command(
            build_eval_command(
                config=reference_config,
                model=model_name,
                split=split,
                checkpoint=None,
            )
        )


@app.command()
def main(
    config: Path = typer.Option(Path("configs/transformer_retrieval_residual.yaml"), "--config"),
    resume_from: Path | None = typer.Option(None, "--resume-from"),
    max_runtime_hours: float | None = typer.Option(None, "--max-runtime-hours"),
    skip_train: bool = typer.Option(False, "--skip-train"),
    evaluate_only: bool = typer.Option(False, "--evaluate-only"),
    run_name: str | None = typer.Option("full_residual_transformer_train", "--run-name"),
) -> None:
    _ensure_env_copy()

    cfg = load_retrieval_config(config, sample=False)
    _verify_full_data_inputs(cfg)

    baseline_checkpoint = cfg.paths.model_output_dir / "best_baseline_retriever.pt"
    if not baseline_checkpoint.exists():
        msg = f"Missing baseline checkpoint: {baseline_checkpoint}"
        raise FileNotFoundError(msg)

    train_run_id = ""
    train_run_url = ""
    train_runtime_seconds = 0.0
    best_epoch = -1
    checkpoint_for_eval: Path | None = None
    stopped_due_to_runtime = False
    last_checkpoint = ""

    if not skip_train and not evaluate_only:
        train_command = build_train_command(
            config=config,
            baseline_checkpoint=baseline_checkpoint,
            resume_from=resume_from,
            max_runtime_hours=max_runtime_hours,
            run_name=run_name,
        )
        train_start = monotonic()
        train_result = _run_command(train_command)
        train_runtime_seconds = monotonic() - train_start

        train_kv = _parse_key_values(train_result.stdout)
        train_run_id = train_kv.get("mlflow_run_id", "")
        train_run_url = train_kv.get("mlflow_run_url", "")
        best_epoch = int(train_kv.get("best_epoch", "-1"))
        stopped_due_to_runtime = _to_bool(train_kv.get("stopped_due_to_runtime"))
        last_checkpoint = train_kv.get("last_checkpoint", "")
        best_checkpoint_raw = train_kv.get("best_checkpoint", "")
        checkpoint_for_eval = Path(best_checkpoint_raw) if best_checkpoint_raw else None

        if stopped_due_to_runtime:
            resume_checkpoint = last_checkpoint or (
                str(checkpoint_for_eval) if checkpoint_for_eval else ""
            )
            summary = {
                "status": "stopped_due_to_runtime",
                "train_run_id": train_run_id,
                "train_run_url": train_run_url,
                "train_runtime_seconds": train_runtime_seconds,
                "best_epoch": best_epoch,
                "checkpoint_path": resume_checkpoint,
                "resume_command": (
                    "uv run python scripts/run_full_residual_training.py "
                    f"--resume-from {resume_checkpoint} --max-runtime-hours 4"
                ),
            }
            summary_json = cfg.paths.report_output_dir / "full_residual_transformer_summary.json"
            summary_md = cfg.paths.report_output_dir / "full_residual_transformer_summary.md"
            save_json(summary_json, summary)
            save_markdown(summary_md, _to_markdown(summary))
            typer.echo(summary)
            return

    if checkpoint_for_eval is None:
        if resume_from is not None and resume_from.exists():
            checkpoint_for_eval = resume_from
        else:
            checkpoint_for_eval = (
                cfg.paths.model_output_dir / "best_residual_transformer_retriever.pt"
            )

    if not checkpoint_for_eval.exists():
        msg = f"Residual checkpoint not found for evaluation: {checkpoint_for_eval}"
        raise FileNotFoundError(msg)

    val_eval_result = _run_command(
        build_eval_command(
            config=config,
            model="residual_transformer",
            split="val",
            checkpoint=checkpoint_for_eval,
        )
    )
    val_run_id, val_run_url = _parse_mlflow_fields(val_eval_result.stdout)

    test_eval_result = _run_command(
        build_eval_command(
            config=config,
            model="residual_transformer",
            split="test",
            checkpoint=checkpoint_for_eval,
        )
    )
    test_run_id, test_run_url = _parse_mlflow_fields(test_eval_result.stdout)

    export_result = _run_command(
        build_export_command(
            config=config,
            checkpoint=checkpoint_for_eval,
        )
    )
    faiss_run_id, faiss_run_url = _parse_mlflow_fields(export_result.stdout)

    val_report = _load_json(
        cfg.paths.report_output_dir / "retrieval_eval_residual_transformer_val.json"
    )
    test_report = _load_json(
        cfg.paths.report_output_dir / "retrieval_eval_residual_transformer_test.json"
    )
    faiss_report = _load_json(
        cfg.paths.report_output_dir / "faiss_export_report_residual_transformer.json"
    )

    val_metrics = _extract_metrics(val_report, ["residual_transformer"]) or {}
    test_metrics = _extract_metrics(test_report, ["residual_transformer"]) or {}
    popularity_val = _extract_metrics(val_report, ["popularity"]) or {}
    popularity_test = _extract_metrics(test_report, ["popularity"]) or {}

    _ensure_reference_reports(cfg.paths.report_output_dir)
    baseline_val = _load_baseline_metrics(cfg.paths.report_output_dir, "val")
    baseline_test = _load_baseline_metrics(cfg.paths.report_output_dir, "test")

    acceptance = _evaluate_acceptance(
        residual_val=val_metrics,
        baseline_val=baseline_val,
    )

    summary = {
        "status": "completed",
        "train_run_id": train_run_id,
        "eval_val_run_id": val_run_id,
        "eval_test_run_id": test_run_id,
        "faiss_run_id": faiss_run_id,
        "train_run_url": train_run_url,
        "eval_val_run_url": val_run_url,
        "eval_test_run_url": test_run_url,
        "faiss_run_url": faiss_run_url,
        "train_runtime_seconds": train_runtime_seconds,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "faiss_top10_match": bool(faiss_report.get("top10_match", False)),
        "faiss_top200_latency_ms": float(faiss_report.get("top200_latency_ms", 0.0)),
        "comparison_vs_baseline": {
            "val": _delta(val_metrics, baseline_val),
            "test": _delta(test_metrics, baseline_test),
        },
        "comparison_vs_popularity": {
            "val": _delta(val_metrics, popularity_val),
            "test": _delta(test_metrics, popularity_test),
        },
        "passes_acceptance_criteria": bool(acceptance["acceptance_passed"]),
        "cl_epidtn_unblocked": bool(acceptance["cl_epidtn_unblocked"]),
        "residual_should_remain_experimental": bool(
            acceptance["residual_should_remain_experimental"]
        ),
        "checkpoint_used_for_eval": str(checkpoint_for_eval),
        "stopped_due_to_runtime": stopped_due_to_runtime,
        "last_checkpoint": last_checkpoint,
    }

    summary_json = cfg.paths.report_output_dir / "full_residual_transformer_summary.json"
    summary_md = cfg.paths.report_output_dir / "full_residual_transformer_summary.md"
    save_json(summary_json, summary)
    save_markdown(summary_md, _to_markdown(summary))

    typer.echo(summary)


if __name__ == "__main__":
    app()
