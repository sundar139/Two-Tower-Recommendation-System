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
from torch.nn import functional

from movie_recsys.constants import PROJECT_ROOT
from movie_recsys.modeling.artifacts import (
    load_checkpoint,
    save_checkpoint,
    save_config_snapshot,
    save_json,
)
from movie_recsys.modeling.cl_retrieval import CLResidualTransformerRetriever
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
    final_retrieval_loss: float
    final_user_contrastive_loss: float
    final_item_contrastive_loss: float
    final_alignment_contrastive_loss: float
    final_residual_anchor_loss: float
    final_total_loss: float


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
) -> Literal["baseline", "transformer", "residual_transformer", "cl_residual_transformer"]:
    if model_type in {"two_tower", "baseline"}:
        return "baseline"
    if model_type == "transformer":
        return "transformer"
    if model_type == "residual_transformer":
        return "residual_transformer"
    if model_type == "cl_residual_transformer":
        return "cl_residual_transformer"
    msg = f"Unsupported model type: {model_type}"
    raise ValueError(msg)


def _build_retriever_model(
    config: RetrievalConfig,
    feature_tables: FeatureTables,
    model_type: Literal[
        "baseline",
        "transformer",
        "residual_transformer",
        "cl_residual_transformer",
    ],
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
    if model_type == "cl_residual_transformer":
        return CLResidualTransformerRetriever(**common_kwargs)
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


def _load_strict_state(model: nn.Module, source_checkpoint_path: Path) -> None:
    source_checkpoint = load_checkpoint(source_checkpoint_path)
    source_state = source_checkpoint.get("model_state_dict", source_checkpoint)
    if not isinstance(source_state, dict):
        msg = f"Checkpoint at {source_checkpoint_path} does not include a state dict"
        raise ValueError(msg)
    model.load_state_dict(cast(dict[str, Any], source_state))


def _compute_contrastive_weight_scale(
    config: RetrievalConfig,
    *,
    epoch: int,
) -> float:
    warmup_epochs = max(int(config.model.contrastive_warmup_epochs), 0)
    if warmup_epochs > 0:
        warmup_scale = min(max(float(epoch) / float(warmup_epochs), 0.0), 1.0)
    else:
        warmup_scale = 1.0

    scale = warmup_scale
    decay_start = config.model.contrastive_decay_start_epoch
    min_scale = min(max(float(config.model.contrastive_min_weight_scale), 0.0), 1.0)
    if (
        decay_start is not None
        and decay_start >= 0
        and config.train.epochs > 1
        and epoch >= decay_start
    ):
        decay_span = max(config.train.epochs - 1 - decay_start, 1)
        decay_progress = min(max(float(epoch - decay_start) / float(decay_span), 0.0), 1.0)
        decay_scale = 1.0 - (decay_progress * (1.0 - min_scale))
        scale = min(scale, decay_scale)

    return min(max(scale, 0.0), 1.0)


def _residual_anchor_loss(
    *,
    student_user_emb: torch.Tensor,
    student_item_emb: torch.Tensor,
    teacher_user_emb: torch.Tensor,
    teacher_item_emb: torch.Tensor,
) -> torch.Tensor:
    user_anchor = 1.0 - functional.cosine_similarity(
        student_user_emb,
        teacher_user_emb,
        dim=1,
    ).mean()
    item_anchor = 1.0 - functional.cosine_similarity(
        student_item_emb,
        teacher_item_emb,
        dim=1,
    ).mean()
    return 0.5 * (user_anchor + item_anchor)


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
    init_from_residual: Path | None = None,
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

    init_summary: dict[str, int] | None = None
    applied_init_checkpoint: Path | None = None
    init_from_baseline_applied = False
    init_from_residual_applied = False
    residual_teacher_checkpoint: Path | None = None

    def _resolve_optional_checkpoint(raw_path: Path | None) -> Path | None:
        if raw_path is None:
            return None
        if raw_path.is_absolute():
            return raw_path
        return (PROJECT_ROOT / raw_path).resolve()

    init_from_residual_resolved = _resolve_optional_checkpoint(init_from_residual)
    config_init_from_residual = _resolve_optional_checkpoint(
        Path(config.model.init_from_residual) if config.model.init_from_residual else None
    )

    if init_from_baseline is not None:
        init_summary = _load_partial_state(model, init_from_baseline)
        applied_init_checkpoint = init_from_baseline
        init_from_baseline_applied = True
    elif init_from_residual_resolved is not None:
        init_summary = _load_partial_state(model, init_from_residual_resolved)
        applied_init_checkpoint = init_from_residual_resolved
        init_from_residual_applied = True
    elif normalized_model_type == "residual_transformer" and not allow_random_init:
        default_baseline_ckpt = config.paths.model_output_dir / "best_baseline_retriever.pt"
        if not default_baseline_ckpt.exists():
            msg = (
                "Residual transformer requires baseline initialization. "
                "Pass --init-from-baseline or use --allow-random-init."
            )
            raise FileNotFoundError(msg)
        init_summary = _load_partial_state(model, default_baseline_ckpt)
        applied_init_checkpoint = default_baseline_ckpt
        init_from_baseline_applied = True
    elif normalized_model_type == "cl_residual_transformer" and not allow_random_init:
        default_residual_ckpt = (
            config.paths.model_output_dir / "best_residual_transformer_retriever.pt"
        )
        resolved_ckpt = config_init_from_residual or default_residual_ckpt
        if not resolved_ckpt.exists():
            msg = (
                "CL residual transformer requires residual initialization. "
                "Pass --init-from-residual, set model.init_from_residual, "
                "or use --allow-random-init."
            )
            raise FileNotFoundError(msg)
        init_summary = _load_partial_state(model, resolved_ckpt)
        applied_init_checkpoint = resolved_ckpt
        init_from_residual_applied = True

    if normalized_model_type == "cl_residual_transformer":
        default_residual_ckpt = (
            config.paths.model_output_dir / "best_residual_transformer_retriever.pt"
        )
        candidate_teacher_ckpt = (
            init_from_residual_resolved or config_init_from_residual or default_residual_ckpt
        )
        if candidate_teacher_ckpt.exists():
            residual_teacher_checkpoint = candidate_teacher_ckpt

    device = _select_device(config.train.device)
    model.to(device)

    residual_anchor_lambda = float(config.model.lambda_residual_anchor)
    residual_teacher_model: ResidualTransformerRetriever | None = None
    if normalized_model_type == "cl_residual_transformer" and residual_anchor_lambda > 0.0:
        if residual_teacher_checkpoint is None:
            msg = (
                "Residual anchor regularization requires a residual checkpoint. "
                "Set model.init_from_residual or pass --init-from-residual to a valid "
                "best_residual_transformer_retriever.pt checkpoint."
            )
            raise FileNotFoundError(msg)
        residual_teacher_model = cast(
            ResidualTransformerRetriever,
            _build_retriever_model(
                config,
                feature_tables,
                model_type="residual_transformer",
            ),
        )
        _load_strict_state(residual_teacher_model, residual_teacher_checkpoint)
        residual_teacher_model.to(device)
        residual_teacher_model.eval()
        residual_teacher_model.requires_grad_(False)

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
    epoch_retrieval_loss = 0.0
    epoch_user_contrastive_loss = 0.0
    epoch_item_contrastive_loss = 0.0
    epoch_alignment_contrastive_loss = 0.0
    epoch_residual_anchor_loss = 0.0
    epoch_total_loss = 0.0

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
        existing_run_params = dict(run.data.params)

        def _safe_log_param(key: str, value: str) -> None:
            existing_value = existing_run_params.get(key)
            if existing_value is None:
                mlflow.log_param(key, value)
                existing_run_params[key] = value
                return
            if existing_value != value:
                mlflow.set_tag(f"resume_param_{key}", value)

        set_retrieval_tags(model_type=normalized_model_type, split="val", sample=sample)
        if not resumed_existing_run:
            log_training_params(config)
        _safe_log_param("run_name", default_run_name)
        _safe_log_param(
            "init_from_baseline",
            str(init_from_baseline) if init_from_baseline else "",
        )
        _safe_log_param(
            "init_from_residual",
            str(init_from_residual_resolved) if init_from_residual_resolved else "",
        )
        _safe_log_param(
            "residual_teacher_checkpoint",
            str(residual_teacher_checkpoint) if residual_teacher_checkpoint else "",
        )
        _safe_log_param("lambda_residual_anchor", str(residual_anchor_lambda))
        _safe_log_param("init_checkpoint_path", str(applied_init_checkpoint or ""))
        _safe_log_param("init_from_baseline_applied", str(init_from_baseline_applied).lower())
        _safe_log_param("init_from_residual_applied", str(init_from_residual_applied).lower())
        _safe_log_param("allow_random_init", str(allow_random_init).lower())
        _safe_log_param("checkpoint_every_epoch", str(checkpoint_every_epoch).lower())
        _safe_log_param("eval_every_epoch", str(eval_every_epoch).lower())
        _safe_log_param("save_last", str(save_last).lower())
        _safe_log_param("resume_from", str(resume_from) if resume_from else "")
        _safe_log_param(
            "max_runtime_hours",
            "" if max_runtime_hours is None else str(max_runtime_hours),
        )
        _safe_log_param(
            "contrastive_warmup_epochs",
            str(config.model.contrastive_warmup_epochs),
        )
        _safe_log_param(
            "contrastive_decay_start_epoch",
            ""
            if config.model.contrastive_decay_start_epoch is None
            else str(config.model.contrastive_decay_start_epoch),
        )
        _safe_log_param(
            "contrastive_min_weight_scale",
            str(config.model.contrastive_min_weight_scale),
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
        if init_summary is not None:
            _safe_log_param(
                "baseline_init_loaded_keys",
                str(init_summary["loaded_keys"]),
            )
            _safe_log_param(
                "baseline_init_skipped_missing_keys",
                str(init_summary["skipped_missing_keys"]),
            )
            _safe_log_param(
                "baseline_init_skipped_shape_mismatch_keys",
                str(init_summary["skipped_shape_mismatch_keys"]),
            )
            _safe_log_param("loaded_key_count", str(init_summary["loaded_keys"]))
            _safe_log_param(
                "skipped_key_count",
                str(
                    init_summary["skipped_missing_keys"]
                    + init_summary["skipped_shape_mismatch_keys"]
                ),
            )

        for epoch in range(start_epoch, config.train.epochs):
            model.train()

            effective_lambda_user_cl = 0.0
            effective_lambda_item_cl = 0.0
            effective_lambda_alignment_cl = 0.0
            contrastive_weight_scale = 0.0
            if normalized_model_type == "cl_residual_transformer":
                contrastive_weight_scale = _compute_contrastive_weight_scale(config, epoch=epoch)
                effective_lambda_user_cl = (
                    float(config.model.lambda_user_cl) * contrastive_weight_scale
                )
                effective_lambda_item_cl = (
                    float(config.model.lambda_item_cl) * contrastive_weight_scale
                )
                effective_lambda_alignment_cl = (
                    float(config.model.lambda_alignment_cl) * contrastive_weight_scale
                )
                cast(
                    CLResidualTransformerRetriever,
                    model,
                ).set_effective_contrastive_weights(
                    lambda_user_cl=effective_lambda_user_cl,
                    lambda_item_cl=effective_lambda_item_cl,
                    lambda_alignment_cl=effective_lambda_alignment_cl,
                )

            losses: list[float] = []
            retrieval_losses: list[float] = []
            user_cl_losses: list[float] = []
            item_cl_losses: list[float] = []
            alignment_cl_losses: list[float] = []
            residual_anchor_losses: list[float] = []
            total_losses: list[float] = []
            for batch in train_loader:
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    output = model(batch)
                    retrieval_loss = output.get("retrieval_loss")
                    if retrieval_loss is None:
                        retrieval_loss = criterion(output["logits"])

                    user_contrastive_loss = output.get("user_contrastive_loss")
                    if user_contrastive_loss is None:
                        user_contrastive_loss = torch.zeros_like(retrieval_loss)

                    item_contrastive_loss = output.get("item_contrastive_loss")
                    if item_contrastive_loss is None:
                        item_contrastive_loss = torch.zeros_like(retrieval_loss)

                    alignment_contrastive_loss = output.get("alignment_contrastive_loss")
                    if alignment_contrastive_loss is None:
                        alignment_contrastive_loss = torch.zeros_like(retrieval_loss)

                    total_loss = output.get("total_loss")
                    if total_loss is None:
                        total_loss = retrieval_loss

                    residual_anchor_loss = torch.zeros_like(retrieval_loss)
                    if residual_teacher_model is not None and residual_anchor_lambda > 0.0:
                        with torch.no_grad():
                            teacher_user_emb = residual_teacher_model.encode_user(batch)
                            teacher_item_emb = residual_teacher_model.encode_item(batch)
                        residual_anchor_loss = _residual_anchor_loss(
                            student_user_emb=output["user_emb"],
                            student_item_emb=output["item_emb"],
                            teacher_user_emb=teacher_user_emb,
                            teacher_item_emb=teacher_item_emb,
                        )
                        total_loss = total_loss + (residual_anchor_lambda * residual_anchor_loss)

                if (
                    not bool(torch.isfinite(retrieval_loss).item())
                    or not bool(torch.isfinite(user_contrastive_loss).item())
                    or not bool(torch.isfinite(item_contrastive_loss).item())
                    or not bool(torch.isfinite(alignment_contrastive_loss).item())
                    or not bool(torch.isfinite(residual_anchor_loss).item())
                    or not bool(torch.isfinite(total_loss).item())
                ):
                    continue

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                clip_norm = config.train.gradient_clip_norm
                if clip_norm <= 0.0:
                    clip_norm = config.train.max_grad_norm
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()
                losses.append(float(total_loss.item()))
                retrieval_losses.append(float(retrieval_loss.item()))
                user_cl_losses.append(float(user_contrastive_loss.item()))
                item_cl_losses.append(float(item_contrastive_loss.item()))
                alignment_cl_losses.append(float(alignment_contrastive_loss.item()))
                residual_anchor_losses.append(float(residual_anchor_loss.item()))
                total_losses.append(float(total_loss.item()))

                if max_runtime_seconds is not None:
                    elapsed_seconds = monotonic() - run_start_time
                    if elapsed_seconds >= max_runtime_seconds:
                        stopped_due_to_runtime = True
                        break

            epoch_loss = float(np.mean(losses)) if losses else 0.0
            epoch_retrieval_loss = float(np.mean(retrieval_losses)) if retrieval_losses else 0.0
            epoch_user_contrastive_loss = float(np.mean(user_cl_losses)) if user_cl_losses else 0.0
            epoch_item_contrastive_loss = float(np.mean(item_cl_losses)) if item_cl_losses else 0.0
            epoch_alignment_contrastive_loss = (
                float(np.mean(alignment_cl_losses)) if alignment_cl_losses else 0.0
            )
            epoch_residual_anchor_loss = (
                float(np.mean(residual_anchor_losses)) if residual_anchor_losses else 0.0
            )
            epoch_total_loss = float(np.mean(total_losses)) if total_losses else epoch_loss

            log_metrics(
                {
                    "train_loss": epoch_total_loss,
                    "train/retrieval_loss": epoch_retrieval_loss,
                    "train/user_contrastive_loss": epoch_user_contrastive_loss,
                    "train/item_contrastive_loss": epoch_item_contrastive_loss,
                    "train/alignment_contrastive_loss": epoch_alignment_contrastive_loss,
                    "train/residual_anchor_loss": epoch_residual_anchor_loss,
                    "train/contrastive_weight_scale": contrastive_weight_scale,
                    "train/effective_lambda_user_cl": effective_lambda_user_cl,
                    "train/effective_lambda_item_cl": effective_lambda_item_cl,
                    "train/effective_lambda_alignment_cl": effective_lambda_alignment_cl,
                    "train/total_loss": epoch_total_loss,
                },
                step=epoch,
            )

            eval_result = None
            if eval_every_epoch and not stopped_due_to_runtime:
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
            "init_checkpoint_path": str(applied_init_checkpoint or ""),
            "init_from_baseline": init_from_baseline_applied,
            "init_from_residual": init_from_residual_applied,
            "loaded_key_count": 0 if init_summary is None else init_summary["loaded_keys"],
            "skipped_key_count": (
                0
                if init_summary is None
                else init_summary["skipped_missing_keys"]
                + init_summary["skipped_shape_mismatch_keys"]
            ),
            "best_epoch": best_epoch,
            "start_epoch": start_epoch,
            "completed_epochs": completed_epochs,
            "stopped_due_to_runtime": stopped_due_to_runtime,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else "",
            "final_retrieval_loss": epoch_retrieval_loss,
            "final_user_contrastive_loss": epoch_user_contrastive_loss,
            "final_item_contrastive_loss": epoch_item_contrastive_loss,
            "final_alignment_contrastive_loss": epoch_alignment_contrastive_loss,
            "final_residual_anchor_loss": epoch_residual_anchor_loss,
            "final_total_loss": epoch_total_loss,
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
        final_train_loss=epoch_total_loss,
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
        final_retrieval_loss=epoch_retrieval_loss,
        final_user_contrastive_loss=epoch_user_contrastive_loss,
        final_item_contrastive_loss=epoch_item_contrastive_loss,
        final_alignment_contrastive_loss=epoch_alignment_contrastive_loss,
        final_residual_anchor_loss=epoch_residual_anchor_loss,
        final_total_loss=epoch_total_loss,
    )
