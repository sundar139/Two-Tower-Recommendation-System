"""Run a lightweight manual ablation sweep for plain two-tower retrieval on sample data."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from movie_recsys.modeling.artifacts import load_checkpoint, save_json
from movie_recsys.modeling.datasets import load_feature_tables
from movie_recsys.modeling.evaluator import evaluate_popularity_baseline, evaluate_two_tower
from movie_recsys.modeling.retrieval import TwoTowerRetriever
from movie_recsys.modeling.trainer import train_retriever
from movie_recsys.training.config import RetrievalConfig, load_retrieval_config

app = typer.Typer(add_completion=False)


def _build_model(cfg: RetrievalConfig) -> tuple[TwoTowerRetriever, object]:
    tables = load_feature_tables(cfg)
    model = TwoTowerRetriever(
        config=cfg,
        num_users=tables.user_features.shape[0],
        num_items_with_padding=tables.item_features.shape[0] + 1,
        user_feature_dim=tables.user_features.shape[1],
        item_feature_dim=tables.item_features.shape[1],
    )
    return model, tables


def _evaluate_trial(cfg: RetrievalConfig, checkpoint: Path) -> dict[str, float]:
    train_df = pl.read_parquet(cfg.train_path)
    val_df = pl.read_parquet(cfg.val_path)
    items_df = pl.read_parquet(cfg.items_path)
    users_df = pl.read_parquet(cfg.users_path)

    popularity = evaluate_popularity_baseline(train_df, val_df, items_df)
    model, tables = _build_model(cfg)
    model.load_state_dict(load_checkpoint(checkpoint)["model_state_dict"])
    model.eval()

    eval_result, _emb, _latency = evaluate_two_tower(
        model,
        train_df,
        val_df,
        users_df,
        tables,
        history_length=cfg.train.history_length,
    )
    return {
        "train_loss": float("nan"),
        "val_hr@10": eval_result.metrics["hr@10"],
        "val_mrr@10": eval_result.metrics["mrr@10"],
        "val_ndcg@10": eval_result.metrics["ndcg@10"],
        "val_recall@50": eval_result.metrics["recall@50"],
        "pop_ndcg@10": popularity.metrics["ndcg@10"],
        "beats_popularity": eval_result.metrics["ndcg@10"] > popularity.metrics["ndcg@10"],
    }


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval_sample_stronger.yaml"), "--config"),
) -> None:
    base = load_retrieval_config(config, sample=True)

    trials = [
        {
            "name": "lr1e3_temp007_drop01",
            "learning_rate": 1e-3,
            "temperature": 0.07,
            "dropout": 0.1,
            "epochs": 10,
        },
        {
            "name": "lr5e4_temp01_drop00",
            "learning_rate": 5e-4,
            "temperature": 0.1,
            "dropout": 0.0,
            "epochs": 12,
        },
        {
            "name": "lr2e4_temp005_drop00",
            "learning_rate": 2e-4,
            "temperature": 0.05,
            "dropout": 0.0,
            "epochs": 12,
        },
    ]

    results: list[dict[str, object]] = []
    for trial in trials:
        cfg = base.model_copy(deep=True)
        cfg.train.learning_rate = float(trial["learning_rate"])
        cfg.model.temperature = float(trial["temperature"])
        cfg.model.dropout = float(trial["dropout"])
        cfg.train.epochs = int(trial["epochs"])
        cfg.paths.model_output_dir = cfg.paths.model_output_dir / str(trial["name"])
        cfg.paths.model_output_dir.mkdir(parents=True, exist_ok=True)

        train_result = train_retriever(cfg, sample=True)
        metrics = _evaluate_trial(cfg, train_result.best_checkpoint)
        metrics["train_loss"] = float(train_result.final_train_loss)

        payload = {
            "trial": trial,
            "best_checkpoint": str(train_result.best_checkpoint),
            "best_val_metrics": train_result.best_metrics,
            "summary": metrics,
        }
        results.append(payload)
        typer.echo(payload)

    report_path = base.paths.report_output_dir / "retrieval_ablation_sample.json"
    save_json(report_path, {"results": results})
    typer.echo({"report": str(report_path), "trials": len(results)})


if __name__ == "__main__":
    app()
