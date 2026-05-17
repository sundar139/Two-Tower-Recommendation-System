"""Evaluate popularity or two-tower retriever."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import polars as pl
import torch
import typer

from movie_recsys.modeling.artifacts import load_checkpoint, save_json, save_markdown
from movie_recsys.modeling.cl_retrieval import CLResidualTransformerRetriever
from movie_recsys.modeling.datasets import load_feature_tables
from movie_recsys.modeling.evaluator import evaluate_popularity_baseline, evaluate_two_tower
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import load_retrieval_config
from movie_recsys.training.mlflow_utils import (
    configure_mlflow,
    log_artifacts,
    log_metrics,
    log_training_params,
    print_mlflow_run_summary,
    set_retrieval_tags,
)

app = typer.Typer(add_completion=False)


def _split_path(config, split: str) -> Path:
    if split == "val":
        return config.val_path
    return config.test_path


def _format_metrics_table(
    popularity_metrics: dict[str, float],
    model_metrics: dict[str, float],
    delta: dict[str, float],
    model_label: str,
) -> str:
    rows = [f"| Metric | Popularity | {model_label} | Delta |", "|---|---:|---:|---:|"]
    for metric in ["hr@10", "mrr@10", "ndcg@10", "recall@50"]:
        rows.append(
            "| "
            f"{metric} | {popularity_metrics[metric]:.6f} | "
            f"{model_metrics[metric]:.6f} | {delta[metric]:.6f} |"
        )
    return "\n".join(rows)


def _normalize_model_name(model: str) -> str:
    if model in {"baseline", "two_tower"}:
        return "baseline"
    if model in {"residual_transformer", "residual"}:
        return "residual_transformer"
    if model in {"cl_residual_transformer", "cl_residual", "cl"}:
        return "cl_residual_transformer"
    if model in {"transformer", "popularity"}:
        return model
    msg = f"Unsupported model name: {model}"
    raise ValueError(msg)


def _default_checkpoint(cfg, normalized_model: str) -> Path:
    return cfg.paths.model_output_dir / f"best_{normalized_model}_retriever.pt"


def _build_retriever(cfg, tables, normalized_model: str):
    common_kwargs = {
        "config": cfg,
        "num_users": tables.user_features.shape[0],
        "num_items_with_padding": tables.item_features.shape[0] + 1,
        "user_feature_dim": tables.user_features.shape[1],
        "item_feature_dim": tables.item_features.shape[1],
    }
    if normalized_model == "transformer":
        return TransformerRetriever(**common_kwargs)
    if normalized_model == "residual_transformer":
        return ResidualTransformerRetriever(**common_kwargs)
    if normalized_model == "cl_residual_transformer":
        return CLResidualTransformerRetriever(**common_kwargs)
    return BaselineRetriever(**common_kwargs)


def _update_summary_report(
    summary_path: Path,
    *,
    split: str,
    payload: dict[str, Any],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        existing = summary_path.read_text(encoding="utf-8")
        summary = {} if not existing else dict(json.loads(existing))
    else:
        summary = {}
    summary[split] = payload
    save_json(summary_path, summary)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config"),
    model: str = typer.Option("popularity", "--model"),
    split: str = typer.Option("val", "--split"),
    sample: bool = typer.Option(False, "--sample"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
) -> None:
    cfg = load_retrieval_config(config, sample=sample)
    normalized_model = _normalize_model_name(model)
    if normalized_model in {
        "baseline",
        "transformer",
        "residual_transformer",
        "cl_residual_transformer",
    }:
        cfg.model.model_type = normalized_model  # type: ignore[assignment]

    train_df = pl.read_parquet(cfg.train_path)
    split_df = pl.read_parquet(_split_path(cfg, split))
    items_df = pl.read_parquet(cfg.items_path)
    users_df = pl.read_parquet(cfg.users_path)

    popularity = evaluate_popularity_baseline(train_df, split_df, items_df)
    experiment = configure_mlflow(cfg)

    with mlflow.start_run(run_name=f"evaluate_{normalized_model}_{split}") as run:
        set_retrieval_tags(model_type=normalized_model, split=split, sample=sample)
        log_training_params(cfg)

        if normalized_model == "popularity":
            metrics = popularity.metrics
            report = {
                "model": "popularity",
                "split": split,
                "metrics": metrics,
            }
            report_path = cfg.paths.report_output_dir / f"retrieval_eval_popularity_{split}.json"
            save_json(report_path, report)
            _update_summary_report(
                cfg.paths.report_output_dir
                / f"popularity_{'sample' if sample else 'full'}_summary.json",
                split=split,
                payload=report,
            )
            log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
            log_artifacts([report_path])
            print_mlflow_run_summary(
                config=cfg,
                run=run,
                experiment_id=experiment.experiment_id,
            )
            typer.echo(report)
            return

        feature_tables = load_feature_tables(cfg)
        model_obj = _build_retriever(cfg, feature_tables, normalized_model)
        ckpt = load_checkpoint(checkpoint or _default_checkpoint(cfg, normalized_model))
        model_obj.load_state_dict(ckpt["model_state_dict"])
        model_obj.eval()

        if torch.cuda.is_available():
            model_obj.cuda()

        eval_result, _embeddings, latency_ms = evaluate_two_tower(
            model_obj,
            train_df,
            split_df,
            users_df,
            feature_tables,
            history_length=cfg.train.history_length,
        )

        delta = {
            key: eval_result.metrics[key] - popularity.metrics[key] for key in eval_result.metrics
        }
        report = {
            "model": normalized_model,
            "split": split,
            normalized_model: eval_result.metrics,
            "popularity": popularity.metrics,
            "delta": delta,
            "avg_query_latency_ms": latency_ms,
            "beats_popularity_ndcg@10": (
                eval_result.metrics["ndcg@10"] > popularity.metrics["ndcg@10"]
            ),
        }

        report_json = (
            cfg.paths.report_output_dir / f"retrieval_eval_{normalized_model}_{split}.json"
        )
        table_md = cfg.paths.report_output_dir / f"retrieval_eval_{normalized_model}_{split}.md"
        metrics_table = _format_metrics_table(
            popularity.metrics,
            eval_result.metrics,
            delta,
            model_label=normalized_model,
        )
        save_json(report_json, report)
        _update_summary_report(
            cfg.paths.report_output_dir
            / f"{normalized_model}_{'sample' if sample else 'full'}_summary.json",
            split=split,
            payload=report,
        )

        if normalized_model == "baseline":
            # Backward-compatible report names used by earlier Step 2 scripts.
            save_json(cfg.paths.report_output_dir / f"two_tower_{split}.json", report)
            _update_summary_report(
                cfg.paths.report_output_dir
                / f"two_tower_{'sample' if sample else 'full'}_summary.json",
                split=split,
                payload=report,
            )

        save_markdown(
            table_md,
            metrics_table
            + "\n\n"
            + f"{normalized_model} beats popularity on NDCG@10: "
            + f"{report['beats_popularity_ndcg@10']}\n",
        )

        log_metrics({f"{split}_{k}": v for k, v in eval_result.metrics.items()})
        log_metrics({f"{split}_pop_{k}": v for k, v in popularity.metrics.items()})
        log_artifacts([report_json, table_md])
        print_mlflow_run_summary(
            config=cfg,
            run=run,
            experiment_id=experiment.experiment_id,
        )

        typer.echo(report)
        typer.echo(metrics_table)
        typer.echo(
            f"{normalized_model} beats popularity on NDCG@10: "
            f"{report['beats_popularity_ndcg@10']}"
        )


if __name__ == "__main__":
    app()
