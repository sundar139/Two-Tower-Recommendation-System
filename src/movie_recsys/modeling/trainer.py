"""Training loop for plain two-tower retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast

import mlflow
import numpy as np
import polars as pl
import torch
from torch import nn

from movie_recsys.modeling.artifacts import (
    load_checkpoint,
    save_checkpoint,
    save_config_snapshot,
    save_json,
)
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
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import RetrievalConfig
from movie_recsys.training.mlflow_utils import (
    build_mlflow_run_url,
    configure_mlflow,
    flatten_config_for_snapshot,
    get_mlflow_ui_url,
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
    best_epoch: int
    start_epoch: int
    completed_epochs: int
    elapsed_seconds: float
    stopped_due_to_runtime: bool
    last_checkpoint: Path | None
    resumed_from: Path | None


def _build_scheduler(
    config: RetrievalConfig,
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
) -> tuple[
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    bool,
]:
    name = config.train.scheduler
    if name == "none":
        return None, False
    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.train.scheduler_t_max, 1),
                eta_min=config.train.min_learning_rate,
            ),
            False,
        )
    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=config.train.scheduler_factor,
                patience=config.train.scheduler_patience,
                min_lr=config.train.min_learning_rate,
            ),
            False,
        )
    if name == "warmup_cosine":
        total_steps = max(steps_per_epoch * max(config.train.epochs, 1), 1)
        warmup_steps = min(max(config.train.warmup_steps, 0), total_steps)
        base_lr = max(config.train.learning_rate, 1e-12)
        min_factor = min(config.train.min_learning_rate / base_lr, 1.0)

        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), 1e-8)
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_factor, cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda), True

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


def _normalize_model_type(
    model_type: str,
) -> Literal["baseline", "transformer", "residual_transformer"]:
    if model_type in {"two_tower", "baseline"}:
        return "baseline"
    if model_type == "transformer":
        return "transformer"
    if model_type == "residual_transformer":
        return "residual_transformer"
    msg = f"Unsupported model type: {model_type}"
    raise ValueError(msg)


def _build_retriever_model(
    config: RetrievalConfig,
    feature_tables: FeatureTables,
    model_type: Literal["baseline", "transformer", "residual_transformer"],
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
    if model_type == "residual_transformer":
        return ResidualTransformerRetriever(**common_kwargs)
    return BaselineRetriever(**common_kwargs)


def _load_partial_state(
    model: nn.Module,
    source_checkpoint_path: Path,
) -> dict[str, int]:
    source_checkpoint = load_checkpoint(source_checkpoint_path)
    source_state = source_checkpoint.get("model_state_dict", source_checkpoint)
    if not isinstance(source_state, dict):
        msg = f"Checkpoint at {source_checkpoint_path} does not include a state dict"
        raise ValueError(msg)

    target_state = model.state_dict()
    merged_state = {k: v.clone() for k, v in target_state.items()}

    loaded = 0
    skipped_missing = 0
    skipped_shape = 0

    for key, value in source_state.items():
        if key not in target_state:
            skipped_missing += 1
            continue
        if target_state[key].shape != value.shape:
            skipped_shape += 1
            continue
        merged_state[key] = value
        loaded += 1

    model.load_state_dict(merged_state)
    return {
        "loaded_keys": loaded,
        "skipped_missing_keys": skipped_missing,
        "skipped_shape_mismatch_keys": skipped_shape,
    }


def _epoch_checkpoint_path(
    config: RetrievalConfig,
    *,
    model_type: str,
    epoch: int,
) -> Path:
    checkpoint_dir = config.paths.model_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{model_type}_epoch_{epoch}.pt"


def _build_training_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    ),
    epoch: int,
    best_metric: float,
    best_checkpoint_path: Path,
    config: RetrievalConfig,
    model_type: str,
    mlflow_run_id: str | None,
    best_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_checkpoint_path": str(best_checkpoint_path),
        "config": flatten_config_for_snapshot(config),
        "random_seed": config.train.random_seed,
        "model_type": model_type,
        "mlflow_run_id": mlflow_run_id,
        "best_metrics": best_metrics,
    }


def _load_resume_training_state(
    *,
    resume_from: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    ),
) -> dict[str, Any]:
    payload = load_checkpoint(resume_from)
    if "model_state_dict" not in payload:
        msg = f"Resume checkpoint is missing model_state_dict: {resume_from}"
        raise ValueError(msg)

    model.load_state_dict(cast(dict[str, Any], payload["model_state_dict"]))
    optimizer_state = payload.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)

    start_epoch = int(payload.get("epoch", -1)) + 1
    best_metric = float(payload.get("best_metric", -1.0))
    best_checkpoint_raw = str(payload.get("best_checkpoint_path", ""))
    best_checkpoint_path = Path(best_checkpoint_raw) if best_checkpoint_raw else None
    best_metrics = cast(dict[str, float], payload.get("best_metrics", {}))

    return {
        "start_epoch": max(start_epoch, 0),
        "best_metric": best_metric,
        "best_checkpoint_path": best_checkpoint_path,
        "model_type": payload.get("model_type"),
        "mlflow_run_id": payload.get("mlflow_run_id"),
        "best_metrics": best_metrics,
    }


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
    init_from_baseline: Path | None = None,
    allow_random_init: bool = False,
    resume_from: Path | None = None,
    checkpoint_every_epoch: bool = False,
    max_runtime_hours: float | None = None,
    eval_every_epoch: bool = True,
    save_last: bool = False,
    run_name: str | None = None,
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

    baseline_init_summary: dict[str, int] | None = None
    if init_from_baseline is not None:
        baseline_init_summary = _load_partial_state(model, init_from_baseline)
    elif normalized_model_type == "residual_transformer" and not allow_random_init:
        default_baseline_ckpt = config.paths.model_output_dir / "best_baseline_retriever.pt"
        if not default_baseline_ckpt.exists():
            msg = (
                "Residual transformer requires baseline initialization. "
                "Pass --init-from-baseline or use --allow-random-init."
            )
            raise FileNotFoundError(msg)
        baseline_init_summary = _load_partial_state(model, default_baseline_ckpt)

    device = _select_device(config.train.device)
    model.to(device)

    criterion = InBatchCrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler, scheduler_step_per_batch = _build_scheduler(
        config,
        optimizer,
        steps_per_epoch=max(len(train_loader), 1),
    )

    best_checkpoint = config.paths.model_output_dir / f"best_{normalized_model_type}_retriever.pt"
    best_ndcg = -1.0
    best_metrics: dict[str, float] = {}
    best_epoch = -1
    start_epoch = 0
    resumed_mlflow_run_id: str | None = None

    if resume_from is not None:
        resume_state = _load_resume_training_state(
            resume_from=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        resume_model_type = resume_state["model_type"]
        if resume_model_type and resume_model_type != normalized_model_type:
            msg = (
                "Resume checkpoint model_type does not match requested model_type: "
                f"{resume_model_type} != {normalized_model_type}"
            )
            raise ValueError(msg)

        start_epoch = int(resume_state["start_epoch"])
        best_ndcg = float(resume_state["best_metric"])
        resumed_best_checkpoint = cast(Path | None, resume_state["best_checkpoint_path"])
        if resumed_best_checkpoint is not None:
            best_checkpoint = resumed_best_checkpoint
        resumed_mlflow_run_id = cast(str | None, resume_state["mlflow_run_id"])
        best_metrics = cast(dict[str, float], resume_state["best_metrics"])

    amp_enabled = bool(config.train.amp_enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_df = pl.read_parquet(config.train_path)
    val_df = pl.read_parquet(config.val_path)
    users_df = pl.read_parquet(config.users_path)
    items_df = pl.read_parquet(config.items_path)

    experiment = configure_mlflow(config)
    run_start_time = monotonic()
    max_runtime_seconds = (
        None
        if max_runtime_hours is None or max_runtime_hours <= 0.0
        else float(max_runtime_hours) * 3600.0
    )
    stopped_due_to_runtime = False
    last_checkpoint: Path | None = None
    completed_epochs = start_epoch
    epoch_loss = 0.0

    active_run: Any | None = None
    resumed_existing_run = False
    default_run_name = run_name or f"{normalized_model_type}_retriever_train"

    if resumed_mlflow_run_id is not None:
        try:
            active_run = mlflow.start_run(run_id=resumed_mlflow_run_id)
            resumed_existing_run = True
        except Exception:
            active_run = mlflow.start_run(run_name=default_run_name)
    else:
        active_run = mlflow.start_run(run_name=default_run_name)

    if active_run is None:
        msg = "Failed to start MLflow run"
        raise RuntimeError(msg)

    run = active_run
    run_url = build_mlflow_run_url(
        experiment.experiment_id,
        run.info.run_id,
        ui_url=get_mlflow_ui_url(config),
    )

    try:
        set_retrieval_tags(model_type=normalized_model_type, split="val", sample=sample)
        log_training_params(config)
        mlflow.log_param("run_name", default_run_name)
        mlflow.log_param(
            "init_from_baseline",
            str(init_from_baseline) if init_from_baseline else "",
        )
        mlflow.log_param("allow_random_init", str(allow_random_init).lower())
        mlflow.log_param("checkpoint_every_epoch", str(checkpoint_every_epoch).lower())
        mlflow.log_param("eval_every_epoch", str(eval_every_epoch).lower())
        mlflow.log_param("save_last", str(save_last).lower())
        mlflow.log_param("resume_from", str(resume_from) if resume_from else "")
        mlflow.log_param(
            "max_runtime_hours",
            "" if max_runtime_hours is None else str(max_runtime_hours),
        )
        if resume_from is not None:
            mlflow.set_tag("resumed", "true")
            mlflow.set_tag("resume_checkpoint", str(resume_from))
            if resumed_existing_run:
                mlflow.set_tag("resume_mode", "existing_run")
            else:
                mlflow.set_tag("resume_mode", "new_run")
                if resumed_mlflow_run_id:
                    mlflow.set_tag("resumed_from_run_id", resumed_mlflow_run_id)
        if baseline_init_summary is not None:
            mlflow.log_params(
                {
                    "baseline_init_loaded_keys": baseline_init_summary["loaded_keys"],
                    "baseline_init_skipped_missing_keys": baseline_init_summary[
                        "skipped_missing_keys"
                    ],
                    "baseline_init_skipped_shape_mismatch_keys": baseline_init_summary[
                        "skipped_shape_mismatch_keys"
                    ],
                }
            )

        for epoch in range(start_epoch, config.train.epochs):
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
                clip_norm = config.train.gradient_clip_norm
                if clip_norm <= 0.0:
                    clip_norm = config.train.max_grad_norm
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()
                losses.append(float(loss.item()))

                if max_runtime_seconds is not None:
                    elapsed_seconds = monotonic() - run_start_time
                    if elapsed_seconds >= max_runtime_seconds:
                        stopped_due_to_runtime = True
                        break

            epoch_loss = float(np.mean(losses)) if losses else 0.0
            log_metrics({"train_loss": epoch_loss}, step=epoch)

            eval_result = None
            if eval_every_epoch:
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
                if scheduler_step_per_batch:
                    pass
                elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if eval_result is not None:
                        scheduler.step(eval_result.metrics["ndcg@10"])
                else:
                    scheduler.step()

            if eval_result is not None and eval_result.metrics["ndcg@10"] > best_ndcg:
                best_ndcg = eval_result.metrics["ndcg@10"]
                best_metrics = dict(eval_result.metrics)
                best_epoch = epoch
                save_checkpoint(
                    best_checkpoint,
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(mode="json"),
                        "metrics": best_metrics,
                        "epoch": epoch,
                    },
                )

            if checkpoint_every_epoch:
                epoch_checkpoint = _epoch_checkpoint_path(
                    config,
                    model_type=normalized_model_type,
                    epoch=epoch,
                )
                save_checkpoint(
                    epoch_checkpoint,
                    _build_training_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_metric=best_ndcg,
                        best_checkpoint_path=best_checkpoint,
                        config=config,
                        model_type=normalized_model_type,
                        mlflow_run_id=run.info.run_id,
                        best_metrics=best_metrics,
                    ),
                )
                last_checkpoint = epoch_checkpoint

                elapsed_seconds = monotonic() - run_start_time
                remaining_epochs = max(config.train.epochs - (epoch + 1), 0)
                print(f"checkpoint_path: {epoch_checkpoint}")
                print(f"current_epoch: {epoch}")
                print(f"elapsed_seconds: {elapsed_seconds:.2f}")
                print(f"estimated_remaining_epochs: {remaining_epochs}")
                print(f"mlflow_run_url: {run_url}")

            completed_epochs = epoch + 1

            if stopped_due_to_runtime:
                last_checkpoint = (
                    config.paths.model_output_dir / f"last_{normalized_model_type}_retriever.pt"
                )
                save_checkpoint(
                    last_checkpoint,
                    _build_training_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_metric=best_ndcg,
                        best_checkpoint_path=best_checkpoint,
                        config=config,
                        model_type=normalized_model_type,
                        mlflow_run_id=run.info.run_id,
                        best_metrics=best_metrics,
                    )
                )
                break

        if save_last and not stopped_due_to_runtime:
            last_checkpoint = (
                config.paths.model_output_dir / f"last_{normalized_model_type}_retriever.pt"
            )
            save_checkpoint(
                last_checkpoint,
                _build_training_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=max(completed_epochs - 1, 0),
                    best_metric=best_ndcg,
                    best_checkpoint_path=best_checkpoint,
                    config=config,
                    model_type=normalized_model_type,
                    mlflow_run_id=run.info.run_id,
                    best_metrics=best_metrics,
                ),
            )

        popularity_result = evaluate_popularity_baseline(train_df, val_df, items_df)
        report_payload = {
            "model_type": normalized_model_type,
            "best_val_metrics": best_metrics,
            "popularity_val_metrics": popularity_result.metrics,
            "sample": sample,
            "device": str(device),
            "scheduler": config.train.scheduler,
            "best_epoch": best_epoch,
            "start_epoch": start_epoch,
            "completed_epochs": completed_epochs,
            "stopped_due_to_runtime": stopped_due_to_runtime,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else "",
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
    finally:
        mlflow.end_run()

    elapsed_seconds = monotonic() - run_start_time
    return TrainingResult(
        best_checkpoint=best_checkpoint,
        best_metrics=best_metrics,
        final_train_loss=epoch_loss,
        model_type=normalized_model_type,
        mlflow_run_id=run_summary["mlflow_run_id"],
        mlflow_run_url=run_summary["mlflow_run_url"],
        best_epoch=best_epoch,
        start_epoch=start_epoch,
        completed_epochs=completed_epochs,
        elapsed_seconds=elapsed_seconds,
        stopped_due_to_runtime=stopped_due_to_runtime,
        last_checkpoint=last_checkpoint,
        resumed_from=resume_from,
    )
