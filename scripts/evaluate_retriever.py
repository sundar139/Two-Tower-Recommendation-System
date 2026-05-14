"""Evaluate popularity or two-tower retriever."""

from __future__ import annotations

from pathlib import Path

import mlflow
import polars as pl
import torch
import typer

from movie_recsys.modeling.artifacts import load_checkpoint, save_json, save_markdown
from movie_recsys.modeling.datasets import load_feature_tables
from movie_recsys.modeling.evaluator import evaluate_popularity_baseline, evaluate_two_tower
from movie_recsys.modeling.retrieval import TwoTowerRetriever
from movie_recsys.training.config import load_retrieval_config
from movie_recsys.training.mlflow_utils import (
    get_active_run_id,
    log_artifacts,
    log_metrics,
    log_training_params,
    set_retrieval_tags,
    setup_mlflow,
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
) -> str:
    rows = ["| Metric | Popularity | Two-Tower | Delta |", "|---|---:|---:|---:|"]
    for metric in ["hr@10", "mrr@10", "ndcg@10", "recall@50"]:
        rows.append(
            "| "
            f"{metric} | {popularity_metrics[metric]:.6f} | "
            f"{model_metrics[metric]:.6f} | {delta[metric]:.6f} |"
        )
    return "\n".join(rows)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config"),
    model: str = typer.Option("popularity", "--model"),
    split: str = typer.Option("val", "--split"),
    sample: bool = typer.Option(False, "--sample"),
    checkpoint: Path = typer.Option(Path("artifacts/models/best_retriever.pt"), "--checkpoint"),
) -> None:
    cfg = load_retrieval_config(config, sample=sample)

    train_df = pl.read_parquet(cfg.train_path)
    split_df = pl.read_parquet(_split_path(cfg, split))
    items_df = pl.read_parquet(cfg.items_path)
    users_df = pl.read_parquet(cfg.users_path)

    popularity = evaluate_popularity_baseline(train_df, split_df, items_df)
    setup_mlflow(cfg)

    with mlflow.start_run(run_name=f"evaluate_{model}_{split}"):
        set_retrieval_tags(model_type=model, split=split, sample=sample)
        log_training_params(cfg)

        if model == "popularity":
            metrics = popularity.metrics
            report = {
                "model": "popularity",
                "split": split,
                "metrics": metrics,
            }
            report_path = cfg.paths.report_output_dir / f"popularity_{split}.json"
            save_json(report_path, report)
            log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
            log_artifacts([report_path])
            typer.echo(report)
            typer.echo(f"mlflow_run_id: {get_active_run_id()}")
            return

        feature_tables = load_feature_tables(cfg)
        model_obj = TwoTowerRetriever(
            config=cfg,
            num_users=feature_tables.user_features.shape[0],
            num_items_with_padding=feature_tables.item_features.shape[0] + 1,
            user_feature_dim=feature_tables.user_features.shape[1],
            item_feature_dim=feature_tables.item_features.shape[1],
        )
        ckpt = load_checkpoint(checkpoint)
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
            "model": "two_tower",
            "split": split,
            "two_tower": eval_result.metrics,
            "popularity": popularity.metrics,
            "delta": delta,
            "avg_query_latency_ms": latency_ms,
            "beats_popularity_ndcg@10": eval_result.metrics["ndcg@10"]
            > popularity.metrics["ndcg@10"],
        }

        report_json = cfg.paths.report_output_dir / f"two_tower_{split}.json"
        table_md = cfg.paths.report_output_dir / f"two_tower_{split}.md"
        metrics_table = _format_metrics_table(popularity.metrics, eval_result.metrics, delta)
        save_json(report_json, report)
        save_markdown(
            table_md,
            metrics_table
            + f"\n\nTwo-tower beats popularity on NDCG@10: {report['beats_popularity_ndcg@10']}\n",
        )

        log_metrics({f"{split}_{k}": v for k, v in eval_result.metrics.items()})
        log_metrics({f"{split}_pop_{k}": v for k, v in popularity.metrics.items()})
        log_artifacts([report_json, table_md])

        typer.echo(report)
        typer.echo(metrics_table)
        typer.echo(f"Two-tower beats popularity on NDCG@10: {report['beats_popularity_ndcg@10']}")
        typer.echo(f"mlflow_run_id: {get_active_run_id()}")


if __name__ == "__main__":
    app()
