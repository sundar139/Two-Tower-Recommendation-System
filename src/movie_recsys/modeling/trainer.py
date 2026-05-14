"""Training loop for plain two-tower retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import mlflow
import numpy as np
import polars as pl
import torch
from torch import nn

from movie_recsys.modeling.artifacts import save_checkpoint, save_config_snapshot, save_json
from movie_recsys.modeling.datasets import (
    FeatureTables,
    RetrievalDataset,
    load_feature_tables,
    make_retrieval_dataloader,
)
from movie_recsys.modeling.evaluator import (
    RetrieverModel,
    evaluate_popularity_baseline,
    evaluate_two_tower,
)
from movie_recsys.modeling.losses import InBatchCrossEntropyLoss
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import RetrievalConfig
from movie_recsys.training.mlflow_utils import (
    configure_mlflow,
    flatten_config_for_snapshot,
    log_artifacts,
    log_metrics,
    log_training_params,
    print_mlflow_run_summary,
    set_retrieval_tags,
)
from movie_recsys.utils.reproducibility import set_global_seed


@dataclass(slots=True)
class TrainingResult:
    best_checkpoint: Path
    best_metrics: dict[str, float]
    final_train_loss: float
    model_type: str
    mlflow_run_id: str
    mlflow_run_url: str


def _build_scheduler(
    config: RetrievalConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = config.train.scheduler
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.train.scheduler_t_max, 1),
            eta_min=config.train.min_learning_rate,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.train.scheduler_factor,
            patience=config.train.scheduler_patience,
            min_lr=config.train.min_learning_rate,
        )
    msg = f"Unsupported scheduler: {name}"
    raise ValueError(msg)


def _select_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _normalize_model_type(model_type: str) -> Literal["baseline", "transformer"]:
    if model_type in {"two_tower", "baseline"}:
        return "baseline"
    if model_type == "transformer":
        return "transformer"
    msg = f"Unsupported model type: {model_type}"
    raise ValueError(msg)


def _build_retriever_model(
    config: RetrievalConfig,
    feature_tables: FeatureTables,
    model_type: Literal["baseline", "transformer"],
) -> nn.Module:
    common_kwargs = {
        "config": config,
        "num_users": feature_tables.user_features.shape[0],
        "num_items_with_padding": feature_tables.item_features.shape[0] + 1,
        "user_feature_dim": feature_tables.user_features.shape[1],
        "item_feature_dim": feature_tables.item_features.shape[1],
    }
    if model_type == "transformer":
        return TransformerRetriever(**common_kwargs)
    return BaselineRetriever(**common_kwargs)


def build_model(
    config: RetrievalConfig,
    *,
    model_type: str | None = None,
) -> tuple[nn.Module, RetrievalDataset, RetrievalDataset]:
    feature_tables = load_feature_tables(config)
    train_ds = RetrievalDataset(
        str(config.train_path),
        feature_tables,
        history_length=config.train.history_length,
    )
    val_ds = RetrievalDataset(
        str(config.val_path),
        feature_tables,
        history_length=config.train.history_length,
    )

    normalized_model_type = _normalize_model_type(model_type or config.model.model_type)
    model = _build_retriever_model(
        config,
        feature_tables,
        model_type=normalized_model_type,
    )
    return model, train_ds, val_ds


def train_retriever(
    config: RetrievalConfig,
    *,
    sample: bool,
    model_type: str | None = None,
) -> TrainingResult:
    set_global_seed(config.train.random_seed)
    normalized_model_type = _normalize_model_type(model_type or config.model.model_type)
    config = config.model_copy(deep=True)
    config.model.model_type = normalized_model_type

    feature_tables = load_feature_tables(config)
    train_ds = RetrievalDataset(
        str(config.train_path),
        feature_tables,
        history_length=config.train.history_length,
    )

    train_loader = make_retrieval_dataloader(
        train_ds,
        batch_size=config.train.train_batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        seed=config.train.random_seed,
    )

    model = _build_retriever_model(
        config,
        feature_tables,
        model_type=normalized_model_type,
    )

    device = _select_device(config.train.device)
    model.to(device)

    criterion = InBatchCrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = _build_scheduler(config, optimizer)

    amp_enabled = bool(config.train.amp_enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_df = pl.read_parquet(config.train_path)
    val_df = pl.read_parquet(config.val_path)
    users_df = pl.read_parquet(config.users_path)
    items_df = pl.read_parquet(config.items_path)

    best_ndcg = -1.0
    best_metrics: dict[str, float] = {}
    best_checkpoint = config.paths.model_output_dir / f"best_{normalized_model_type}_retriever.pt"
    experiment = configure_mlflow(config)

    with mlflow.start_run(run_name=f"{normalized_model_type}_retriever_train") as run:
        set_retrieval_tags(model_type=normalized_model_type, split="val", sample=sample)
        log_training_params(config)

        for epoch in range(config.train.epochs):
            model.train()
            losses: list[float] = []
            for batch in train_loader:
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    output = model(batch)
                    loss = criterion(output["logits"])

                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))

            epoch_loss = float(np.mean(losses)) if losses else 0.0
            log_metrics({"train_loss": epoch_loss}, step=epoch)

            eval_result, _embeddings, _latency = evaluate_two_tower(
                cast(RetrieverModel, model),
                train_df,
                val_df,
                users_df,
                feature_tables,
                history_length=config.train.history_length,
            )
            val_metrics = {f"val_{k}": v for k, v in eval_result.metrics.items()}
            log_metrics(val_metrics, step=epoch)
            log_metrics({"learning_rate": optimizer.param_groups[0]["lr"]}, step=epoch)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(eval_result.metrics["ndcg@10"])
                else:
                    scheduler.step()

            if eval_result.metrics["ndcg@10"] > best_ndcg:
                best_ndcg = eval_result.metrics["ndcg@10"]
                best_metrics = dict(eval_result.metrics)
                save_checkpoint(
                    best_checkpoint,
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(mode="json"),
                        "metrics": best_metrics,
                    },
                )

        popularity_result = evaluate_popularity_baseline(train_df, val_df, items_df)
        report_payload = {
            "model_type": normalized_model_type,
            "best_val_metrics": best_metrics,
            "popularity_val_metrics": popularity_result.metrics,
            "sample": sample,
            "device": str(device),
            "scheduler": config.train.scheduler,
        }
        report_path = config.paths.report_output_dir / f"train_report_{normalized_model_type}.json"
        save_json(report_path, report_payload)

        config_snapshot = (
            config.paths.model_output_dir / f"train_config_snapshot_{normalized_model_type}.json"
        )
        save_config_snapshot(config_snapshot, flatten_config_for_snapshot(config))

        log_artifacts([best_checkpoint, report_path, config_snapshot])
        run_summary = print_mlflow_run_summary(
            config=config,
            run=run,
            experiment_id=experiment.experiment_id,
        )

    return TrainingResult(
        best_checkpoint=best_checkpoint,
        best_metrics=best_metrics,
        final_train_loss=epoch_loss,
        model_type=normalized_model_type,
        mlflow_run_id=run_summary["mlflow_run_id"],
        mlflow_run_url=run_summary["mlflow_run_url"],
    )
