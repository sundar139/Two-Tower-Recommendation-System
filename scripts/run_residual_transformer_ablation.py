"""Sample-only ablation for residual transformer retriever."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
import typer

from movie_recsys.modeling.artifacts import load_checkpoint, save_json, save_markdown
from movie_recsys.modeling.datasets import load_feature_tables
from movie_recsys.modeling.evaluator import evaluate_popularity_baseline, evaluate_two_tower
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.trainer import train_retriever
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import RetrievalConfig, load_retrieval_config

app = typer.Typer(add_completion=False)


def _build_model(cfg: RetrievalConfig, feature_tables, model_type: str):
    common_kwargs = {
        "config": cfg,
        "num_users": feature_tables.user_features.shape[0],
        "num_items_with_padding": feature_tables.item_features.shape[0] + 1,
        "user_feature_dim": feature_tables.user_features.shape[1],
        "item_feature_dim": feature_tables.item_features.shape[1],
    }
    if model_type == "transformer":
        return TransformerRetriever(**common_kwargs)
    if model_type == "residual_transformer":
        return ResidualTransformerRetriever(**common_kwargs)
    return BaselineRetriever(**common_kwargs)


def _apply_residual_trial_to_config(cfg: RetrievalConfig, trial: dict) -> RetrievalConfig:
    next_cfg = cfg.model_copy(deep=True)
    next_cfg.model.model_type = "residual_transformer"
    next_cfg.model.dropout = float(trial["dropout"])
    next_cfg.model.sequence_pooling = str(trial["pooling"])
    next_cfg.model.transformer_layers = int(trial["layers"])
    next_cfg.model.transformer_heads = int(trial["heads"])
    next_cfg.model.initial_transformer_gate = float(trial["initial_gate"])
    next_cfg.train.learning_rate = float(trial["lr"])
    next_cfg.train.weight_decay = float(trial["weight_decay"])
    next_cfg.train.epochs = int(trial["epochs"])
    next_cfg.train.scheduler = str(trial["scheduler"])
    next_cfg.train.warmup_steps = int(trial["warmup_steps"])
    next_cfg.paths.model_output_dir = next_cfg.paths.model_output_dir / str(trial["name"])
    next_cfg.paths.model_output_dir.mkdir(parents=True, exist_ok=True)
    return next_cfg


def _evaluate_checkpoint(
    cfg: RetrievalConfig,
    feature_tables,
    *,
    model_type: str,
    split: str,
    checkpoint: Path,
) -> dict[str, float]:
    model = _build_model(cfg, feature_tables, model_type)
    model.load_state_dict(load_checkpoint(checkpoint)["model_state_dict"])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    train_df = pl.read_parquet(cfg.train_path)
    split_df = pl.read_parquet(cfg.val_path if split == "val" else cfg.test_path)
    users_df = pl.read_parquet(cfg.users_path)
    result, _emb, _latency = evaluate_two_tower(
        model,
        train_df,
        split_df,
        users_df,
        feature_tables,
        history_length=cfg.train.history_length,
    )
    return result.metrics


def _to_markdown(summary: dict) -> str:
    lines = [
        "# Residual Transformer Ablation (Sample)",
        "",
        (
            "| Config | LR | Dropout | Pooling | Layers | Heads | Init Gate | "
            "Scheduler | Final Loss | Val NDCG@10 | Run ID |"
        ),
        "|---|---:|---:|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in summary["residual_trials"]:
        lines.append(
            f"| {row['name']} | {row['learning_rate']:.6f} | {row['dropout']:.2f} | "
            f"{row['pooling']} | {row['layers']} | {row['heads']} | "
            f"{row['initial_gate']:.3f} | {row['scheduler']} | {row['final_train_loss']:.6f} | "
            f"{row['best_val_metrics']['ndcg@10']:.6f} | {row['mlflow_run_id']} |"
        )

    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| Model | HR@10 | MRR@10 | NDCG@10 | Recall@50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ["popularity_val", "baseline_val", "transformer_val", "best_residual_val"]:
        metrics = summary[name]
        lines.append(
            f"| {name} | {metrics['hr@10']:.6f} | {metrics['mrr@10']:.6f} | "
            f"{metrics['ndcg@10']:.6f} | {metrics['recall@50']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best Residual Test",
            "",
            f"Run URL: {summary['best_residual_run_url']}",
            "",
            "| HR@10 | MRR@10 | NDCG@10 | Recall@50 |",
            "|---:|---:|---:|---:|",
            f"| {summary['best_residual_test']['hr@10']:.6f} | "
            f"{summary['best_residual_test']['mrr@10']:.6f} | "
            f"{summary['best_residual_test']['ndcg@10']:.6f} | "
            f"{summary['best_residual_test']['recall@50']:.6f} |",
            "",
        ]
    )
    return "\n".join(lines)


@app.command()
def main(
    baseline_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--baseline-config"),
    transformer_config: Path = typer.Option(
        Path("configs/transformer_retrieval_stable.yaml"),
        "--transformer-config",
    ),
    residual_config: Path = typer.Option(
        Path("configs/transformer_retrieval_residual.yaml"),
        "--residual-config",
    ),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    if not sample:
        raise typer.BadParameter(
            "run_residual_transformer_ablation.py is sample-only. Use --sample."
        )

    baseline_cfg = load_retrieval_config(baseline_config, sample=True)
    transformer_cfg = load_retrieval_config(transformer_config, sample=True)
    residual_cfg = load_retrieval_config(residual_config, sample=True)

    baseline_tables = load_feature_tables(baseline_cfg)
    transformer_tables = load_feature_tables(transformer_cfg)
    residual_tables = load_feature_tables(residual_cfg)

    train_df = pl.read_parquet(baseline_cfg.train_path)
    val_df = pl.read_parquet(baseline_cfg.val_path)
    items_df = pl.read_parquet(baseline_cfg.items_path)

    pop_val = evaluate_popularity_baseline(train_df, val_df, items_df).metrics

    baseline_ckpt = baseline_cfg.paths.model_output_dir / "best_baseline_retriever.pt"
    if not baseline_ckpt.exists():
        baseline_result = train_retriever(baseline_cfg, sample=True, model_type="baseline")
        baseline_ckpt = baseline_result.best_checkpoint

    transformer_ckpt = transformer_cfg.paths.model_output_dir / "best_transformer_retriever.pt"
    if not transformer_ckpt.exists():
        transformer_result = train_retriever(
            transformer_cfg,
            sample=True,
            model_type="transformer",
        )
        transformer_ckpt = transformer_result.best_checkpoint

    baseline_val = _evaluate_checkpoint(
        baseline_cfg,
        baseline_tables,
        model_type="baseline",
        split="val",
        checkpoint=baseline_ckpt,
    )
    transformer_val = _evaluate_checkpoint(
        transformer_cfg,
        transformer_tables,
        model_type="transformer",
        split="val",
        checkpoint=transformer_ckpt,
    )

    trials = [
        {
            "name": "res_lr1e4_d10_mean_g294",
            "lr": 1e-4,
            "dropout": 0.10,
            "pooling": "mean",
            "layers": 1,
            "heads": 2,
            "weight_decay": 1e-6,
            "epochs": 10,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
            "initial_gate": -2.944,
        },
        {
            "name": "res_lr1e4_d8_mean_g325",
            "lr": 1e-4,
            "dropout": 0.08,
            "pooling": "mean",
            "layers": 1,
            "heads": 2,
            "weight_decay": 1e-6,
            "epochs": 10,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
            "initial_gate": -3.25,
        },
        {
            "name": "res_lr8e5_d10_mean_g294",
            "lr": 8e-5,
            "dropout": 0.10,
            "pooling": "mean",
            "layers": 1,
            "heads": 2,
            "weight_decay": 1e-6,
            "epochs": 12,
            "scheduler": "warmup_cosine",
            "warmup_steps": 250,
            "initial_gate": -2.944,
        },
        {
            "name": "res_lr1e4_d10_last_g294",
            "lr": 1e-4,
            "dropout": 0.10,
            "pooling": "last",
            "layers": 1,
            "heads": 2,
            "weight_decay": 1e-6,
            "epochs": 10,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
            "initial_gate": -2.944,
        },
        {
            "name": "res_lr1e4_d10_mean_g250",
            "lr": 1e-4,
            "dropout": 0.10,
            "pooling": "mean",
            "layers": 1,
            "heads": 2,
            "weight_decay": 1e-6,
            "epochs": 10,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
            "initial_gate": -2.50,
        },
    ]

    trial_results: list[dict] = []
    best_idx = -1
    best_val_ndcg = -1.0

    for idx, trial in enumerate(trials):
        trial_cfg = _apply_residual_trial_to_config(residual_cfg, trial)
        train_result = train_retriever(
            trial_cfg,
            sample=True,
            model_type="residual_transformer",
            init_from_baseline=baseline_ckpt,
            allow_random_init=False,
        )

        row = {
            "name": str(trial["name"]),
            "trial_index": idx,
            "learning_rate": float(trial["lr"]),
            "dropout": float(trial["dropout"]),
            "pooling": str(trial["pooling"]),
            "layers": int(trial["layers"]),
            "heads": int(trial["heads"]),
            "initial_gate": float(trial["initial_gate"]),
            "weight_decay": float(trial["weight_decay"]),
            "epochs": int(trial["epochs"]),
            "scheduler": str(trial["scheduler"]),
            "warmup_steps": int(trial["warmup_steps"]),
            "final_train_loss": float(train_result.final_train_loss),
            "best_val_metrics": train_result.best_metrics,
            "mlflow_run_id": train_result.mlflow_run_id,
            "mlflow_run_url": train_result.mlflow_run_url,
            "best_checkpoint": str(train_result.best_checkpoint),
        }
        trial_results.append(row)

        val_ndcg = float(train_result.best_metrics.get("ndcg@10", -1.0))
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_idx = idx

    if best_idx < 0:
        raise RuntimeError("No residual transformer trial succeeded")

    best_trial = trial_results[best_idx]
    best_checkpoint = Path(best_trial["best_checkpoint"])
    best_trial_cfg = _apply_residual_trial_to_config(
        residual_cfg,
        {
            "name": best_trial["name"],
            "lr": best_trial["learning_rate"],
            "dropout": best_trial["dropout"],
            "pooling": best_trial["pooling"],
            "layers": best_trial["layers"],
            "heads": best_trial["heads"],
            "initial_gate": best_trial["initial_gate"],
            "weight_decay": best_trial["weight_decay"],
            "epochs": best_trial["epochs"],
            "scheduler": best_trial["scheduler"],
            "warmup_steps": best_trial["warmup_steps"],
        },
    )

    best_residual_val = _evaluate_checkpoint(
        best_trial_cfg,
        residual_tables,
        model_type="residual_transformer",
        split="val",
        checkpoint=best_checkpoint,
    )
    best_residual_test = _evaluate_checkpoint(
        best_trial_cfg,
        residual_tables,
        model_type="residual_transformer",
        split="test",
        checkpoint=best_checkpoint,
    )

    summary = {
        "sample": True,
        "popularity_val": pop_val,
        "baseline_val": baseline_val,
        "transformer_val": transformer_val,
        "residual_trials": trial_results,
        "best_residual_name": best_trial["name"],
        "best_residual_run_id": best_trial["mlflow_run_id"],
        "best_residual_run_url": best_trial["mlflow_run_url"],
        "best_residual_val": best_residual_val,
        "best_residual_test": best_residual_test,
        "meets_gate": bool(
            best_residual_val["ndcg@10"] > pop_val["ndcg@10"]
            and best_residual_val["ndcg@10"] >= baseline_val["ndcg@10"] - 0.001
        ),
    }

    report_json = baseline_cfg.paths.report_output_dir / "residual_transformer_ablation_sample.json"
    report_md = baseline_cfg.paths.report_output_dir / "residual_transformer_ablation_sample.md"
    save_json(report_json, summary)
    save_markdown(report_md, _to_markdown(summary))

    typer.echo({"report_json": str(report_json), "report_md": str(report_md), "summary": summary})


if __name__ == "__main__":
    app()
