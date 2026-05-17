"""Sample-only ablation for CL residual transformer retrieval."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
import typer

from movie_recsys.modeling.artifacts import load_checkpoint, save_json, save_markdown
from movie_recsys.modeling.cl_retrieval import CLResidualTransformerRetriever
from movie_recsys.modeling.datasets import load_feature_tables
from movie_recsys.modeling.evaluator import evaluate_popularity_baseline, evaluate_two_tower
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.trainer import train_retriever
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
    if model_type == "cl_residual_transformer":
        return CLResidualTransformerRetriever(**common_kwargs)
    if model_type == "residual_transformer":
        return ResidualTransformerRetriever(**common_kwargs)
    return BaselineRetriever(**common_kwargs)


def _apply_cl_trial_to_config(
    cfg: RetrievalConfig,
    trial: dict,
    *,
    residual_checkpoint: Path,
) -> RetrievalConfig:
    next_cfg = cfg.model_copy(deep=True)
    next_cfg.model.model_type = "cl_residual_transformer"
    next_cfg.model.init_from_residual = str(residual_checkpoint)
    next_cfg.model.contrastive_temperature = float(trial["contrastive_temperature"])
    next_cfg.model.lambda_user_cl = float(trial["lambda_user_cl"])
    next_cfg.model.lambda_item_cl = float(trial["lambda_item_cl"])
    next_cfg.model.lambda_alignment_cl = float(trial["lambda_alignment_cl"])
    next_cfg.model.augmentation_mask_prob = float(trial["augmentation_mask_prob"])
    next_cfg.model.augmentation_dropout_prob = float(trial["augmentation_dropout_prob"])
    next_cfg.model.augmentation_crop_min_ratio = float(trial["augmentation_crop_min_ratio"])
    next_cfg.model.augmentation_reorder_prob = float(trial["augmentation_reorder_prob"])
    next_cfg.model.augmentation_reorder_window = int(trial["augmentation_reorder_window"])

    next_cfg.train.learning_rate = float(trial["learning_rate"])
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
        "# Contrastive Ablation (Sample)",
        "",
        (
            "| Config | Lambda User | Lambda Item | Temp | Mask | Dropout | Crop Min | "
            "Reorder | Final Total Loss | Val NDCG@10 | Run ID |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["contrastive_trials"]:
        lines.append(
            f"| {row['name']} | {row['lambda_user_cl']:.3f} | {row['lambda_item_cl']:.3f} | "
            f"{row['contrastive_temperature']:.3f} | {row['augmentation_mask_prob']:.2f} | "
            f"{row['augmentation_dropout_prob']:.2f} | {row['augmentation_crop_min_ratio']:.2f} | "
            f"{row['augmentation_reorder_prob']:.2f} | {row['final_total_loss']:.6f} | "
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
    for name in ["popularity_val", "baseline_val", "residual_val", "best_cl_val"]:
        metrics = summary[name]
        lines.append(
            f"| {name} | {metrics['hr@10']:.6f} | {metrics['mrr@10']:.6f} | "
            f"{metrics['ndcg@10']:.6f} | {metrics['recall@50']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best CL Test",
            "",
            f"Run URL: {summary['best_cl_run_url']}",
            "",
            "| HR@10 | MRR@10 | NDCG@10 | Recall@50 |",
            "|---:|---:|---:|---:|",
            f"| {summary['best_cl_test']['hr@10']:.6f} | "
            f"{summary['best_cl_test']['mrr@10']:.6f} | "
            f"{summary['best_cl_test']['ndcg@10']:.6f} | "
            f"{summary['best_cl_test']['recall@50']:.6f} |",
            "",
        ]
    )
    return "\n".join(lines)


@app.command()
def main(
    baseline_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--baseline-config"),
    residual_config: Path = typer.Option(
        Path("configs/transformer_retrieval_residual.yaml"),
        "--residual-config",
    ),
    contrastive_config: Path = typer.Option(
        Path("configs/cl_retrieval.yaml"),
        "--contrastive-config",
    ),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    if not sample:
        raise typer.BadParameter("run_contrastive_ablation.py is sample-only. Use --sample.")

    baseline_cfg = load_retrieval_config(baseline_config, sample=True)
    residual_cfg = load_retrieval_config(residual_config, sample=True)
    contrastive_cfg = load_retrieval_config(contrastive_config, sample=True)

    baseline_tables = load_feature_tables(baseline_cfg)
    residual_tables = load_feature_tables(residual_cfg)
    contrastive_tables = load_feature_tables(contrastive_cfg)

    train_df = pl.read_parquet(baseline_cfg.train_path)
    val_df = pl.read_parquet(baseline_cfg.val_path)
    items_df = pl.read_parquet(baseline_cfg.items_path)

    pop_val = evaluate_popularity_baseline(train_df, val_df, items_df).metrics

    baseline_ckpt = baseline_cfg.paths.model_output_dir / "best_baseline_retriever.pt"
    if not baseline_ckpt.exists():
        baseline_result = train_retriever(baseline_cfg, sample=True, model_type="baseline")
        baseline_ckpt = baseline_result.best_checkpoint

    residual_ckpt = residual_cfg.paths.model_output_dir / "best_residual_transformer_retriever.pt"
    if not residual_ckpt.exists():
        residual_result = train_retriever(
            residual_cfg,
            sample=True,
            model_type="residual_transformer",
            init_from_baseline=baseline_ckpt,
            allow_random_init=False,
        )
        residual_ckpt = residual_result.best_checkpoint

    baseline_val = _evaluate_checkpoint(
        baseline_cfg,
        baseline_tables,
        model_type="baseline",
        split="val",
        checkpoint=baseline_ckpt,
    )
    residual_val = _evaluate_checkpoint(
        residual_cfg,
        residual_tables,
        model_type="residual_transformer",
        split="val",
        checkpoint=residual_ckpt,
    )

    trials = [
        {
            "name": "cl_u005_i002_t010_m10_d10",
            "contrastive_temperature": 0.10,
            "lambda_user_cl": 0.05,
            "lambda_item_cl": 0.02,
            "lambda_alignment_cl": 0.0,
            "augmentation_mask_prob": 0.10,
            "augmentation_dropout_prob": 0.10,
            "augmentation_crop_min_ratio": 0.70,
            "augmentation_reorder_prob": 0.10,
            "augmentation_reorder_window": 3,
            "learning_rate": 2e-4,
            "weight_decay": 1e-6,
            "epochs": 8,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
        },
        {
            "name": "cl_u010_i002_t010_m10_d10",
            "contrastive_temperature": 0.10,
            "lambda_user_cl": 0.10,
            "lambda_item_cl": 0.02,
            "lambda_alignment_cl": 0.0,
            "augmentation_mask_prob": 0.10,
            "augmentation_dropout_prob": 0.10,
            "augmentation_crop_min_ratio": 0.70,
            "augmentation_reorder_prob": 0.10,
            "augmentation_reorder_window": 3,
            "learning_rate": 2e-4,
            "weight_decay": 1e-6,
            "epochs": 8,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
        },
        {
            "name": "cl_u005_i005_t010_m10_d10",
            "contrastive_temperature": 0.10,
            "lambda_user_cl": 0.05,
            "lambda_item_cl": 0.05,
            "lambda_alignment_cl": 0.0,
            "augmentation_mask_prob": 0.10,
            "augmentation_dropout_prob": 0.10,
            "augmentation_crop_min_ratio": 0.70,
            "augmentation_reorder_prob": 0.10,
            "augmentation_reorder_window": 3,
            "learning_rate": 2e-4,
            "weight_decay": 1e-6,
            "epochs": 8,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
        },
        {
            "name": "cl_u005_i002_t007_m15_d10",
            "contrastive_temperature": 0.07,
            "lambda_user_cl": 0.05,
            "lambda_item_cl": 0.02,
            "lambda_alignment_cl": 0.0,
            "augmentation_mask_prob": 0.15,
            "augmentation_dropout_prob": 0.10,
            "augmentation_crop_min_ratio": 0.60,
            "augmentation_reorder_prob": 0.15,
            "augmentation_reorder_window": 3,
            "learning_rate": 2e-4,
            "weight_decay": 1e-6,
            "epochs": 8,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
        },
        {
            "name": "cl_u003_i002_t010_m05_d05",
            "contrastive_temperature": 0.10,
            "lambda_user_cl": 0.03,
            "lambda_item_cl": 0.02,
            "lambda_alignment_cl": 0.0,
            "augmentation_mask_prob": 0.05,
            "augmentation_dropout_prob": 0.05,
            "augmentation_crop_min_ratio": 0.80,
            "augmentation_reorder_prob": 0.05,
            "augmentation_reorder_window": 2,
            "learning_rate": 2e-4,
            "weight_decay": 1e-6,
            "epochs": 8,
            "scheduler": "warmup_cosine",
            "warmup_steps": 200,
        },
    ]

    trial_results: list[dict] = []
    best_idx = -1
    best_val_ndcg = -1.0

    for idx, trial in enumerate(trials):
        trial_cfg = _apply_cl_trial_to_config(
            contrastive_cfg,
            trial,
            residual_checkpoint=residual_ckpt,
        )
        train_result = train_retriever(
            trial_cfg,
            sample=True,
            model_type="cl_residual_transformer",
            init_from_residual=residual_ckpt,
            allow_random_init=False,
        )

        row = {
            "name": str(trial["name"]),
            "trial_index": idx,
            "contrastive_temperature": float(trial["contrastive_temperature"]),
            "lambda_user_cl": float(trial["lambda_user_cl"]),
            "lambda_item_cl": float(trial["lambda_item_cl"]),
            "lambda_alignment_cl": float(trial["lambda_alignment_cl"]),
            "augmentation_mask_prob": float(trial["augmentation_mask_prob"]),
            "augmentation_dropout_prob": float(trial["augmentation_dropout_prob"]),
            "augmentation_crop_min_ratio": float(trial["augmentation_crop_min_ratio"]),
            "augmentation_reorder_prob": float(trial["augmentation_reorder_prob"]),
            "augmentation_reorder_window": int(trial["augmentation_reorder_window"]),
            "learning_rate": float(trial["learning_rate"]),
            "weight_decay": float(trial["weight_decay"]),
            "epochs": int(trial["epochs"]),
            "scheduler": str(trial["scheduler"]),
            "warmup_steps": int(trial["warmup_steps"]),
            "final_train_loss": float(train_result.final_train_loss),
            "final_retrieval_loss": float(train_result.final_retrieval_loss),
            "final_user_contrastive_loss": float(train_result.final_user_contrastive_loss),
            "final_item_contrastive_loss": float(train_result.final_item_contrastive_loss),
            "final_alignment_contrastive_loss": float(
                train_result.final_alignment_contrastive_loss
            ),
            "final_total_loss": float(train_result.final_total_loss),
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
        raise RuntimeError("No contrastive trial succeeded")

    best_trial = trial_results[best_idx]
    best_checkpoint = Path(best_trial["best_checkpoint"])
    best_trial_cfg = _apply_cl_trial_to_config(
        contrastive_cfg,
        {
            "name": best_trial["name"],
            "contrastive_temperature": best_trial["contrastive_temperature"],
            "lambda_user_cl": best_trial["lambda_user_cl"],
            "lambda_item_cl": best_trial["lambda_item_cl"],
            "lambda_alignment_cl": best_trial["lambda_alignment_cl"],
            "augmentation_mask_prob": best_trial["augmentation_mask_prob"],
            "augmentation_dropout_prob": best_trial["augmentation_dropout_prob"],
            "augmentation_crop_min_ratio": best_trial["augmentation_crop_min_ratio"],
            "augmentation_reorder_prob": best_trial["augmentation_reorder_prob"],
            "augmentation_reorder_window": best_trial["augmentation_reorder_window"],
            "learning_rate": best_trial["learning_rate"],
            "weight_decay": best_trial["weight_decay"],
            "epochs": best_trial["epochs"],
            "scheduler": best_trial["scheduler"],
            "warmup_steps": best_trial["warmup_steps"],
        },
        residual_checkpoint=residual_ckpt,
    )

    best_cl_val = _evaluate_checkpoint(
        best_trial_cfg,
        contrastive_tables,
        model_type="cl_residual_transformer",
        split="val",
        checkpoint=best_checkpoint,
    )
    best_cl_test = _evaluate_checkpoint(
        best_trial_cfg,
        contrastive_tables,
        model_type="cl_residual_transformer",
        split="test",
        checkpoint=best_checkpoint,
    )

    summary = {
        "sample": True,
        "popularity_val": pop_val,
        "baseline_val": baseline_val,
        "residual_val": residual_val,
        "contrastive_trials": trial_results,
        "best_cl_name": best_trial["name"],
        "best_cl_run_id": best_trial["mlflow_run_id"],
        "best_cl_run_url": best_trial["mlflow_run_url"],
        "best_cl_val": best_cl_val,
        "best_cl_test": best_cl_test,
        "gate_beats_popularity_ndcg": best_cl_val["ndcg@10"] > pop_val["ndcg@10"],
        "gate_within_0p001_ndcg_of_residual": (
            best_cl_val["ndcg@10"] >= (residual_val["ndcg@10"] - 0.001)
        ),
        "meets_gate": bool(
            best_cl_val["ndcg@10"] > pop_val["ndcg@10"]
            and best_cl_val["ndcg@10"] >= (residual_val["ndcg@10"] - 0.001)
        ),
    }

    report_json = baseline_cfg.paths.report_output_dir / "contrastive_ablation_sample.json"
    report_md = baseline_cfg.paths.report_output_dir / "contrastive_ablation_sample.md"
    save_json(report_json, summary)
    save_markdown(report_md, _to_markdown(summary))

    typer.echo({"report_json": str(report_json), "report_md": str(report_md), "summary": summary})


if __name__ == "__main__":
    app()
