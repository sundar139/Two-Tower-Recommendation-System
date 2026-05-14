"""MLflow helpers for retrieval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlflow
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

from movie_recsys.training.config import RetrievalConfig


def _artifact_root_uri(artifact_root: str) -> str:
    if "://" in artifact_root:
        return artifact_root
    return Path(artifact_root).resolve().as_uri()


def get_mlflow_ui_url(config: RetrievalConfig | None = None) -> str:
    if config is not None and config.mlflow_ui_url:
        return config.mlflow_ui_url.rstrip("/")
    return "http://127.0.0.1:5000"


def build_mlflow_run_url(
    experiment_id: str,
    run_id: str,
    *,
    ui_url: str | None = None,
) -> str:
    base_url = (ui_url or get_mlflow_ui_url()).rstrip("/")
    return f"{base_url}/#/experiments/{experiment_id}/runs/{run_id}"


def configure_mlflow(config: RetrievalConfig) -> Experiment:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.mlflow_experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=config.mlflow_experiment_name,
            artifact_location=_artifact_root_uri(config.mlflow_artifact_root),
        )
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(config.mlflow_experiment_name)
    if experiment is None:
        msg = "Failed to resolve MLflow experiment after configuration"
        raise RuntimeError(msg)
    return experiment


def setup_mlflow(config: RetrievalConfig) -> None:
    configure_mlflow(config)


def print_mlflow_run_summary(
    *,
    config: RetrievalConfig,
    run: Run | None = None,
    experiment_id: str | None = None,
) -> dict[str, str]:
    active_run = run or mlflow.active_run()
    if active_run is None:
        msg = "Cannot print MLflow run summary without an active run"
        raise RuntimeError(msg)

    run_id = cast(str, active_run.info.run_id)
    exp_id = experiment_id or cast(str, active_run.info.experiment_id)
    ui_url = get_mlflow_ui_url(config)
    run_url = build_mlflow_run_url(exp_id, run_id, ui_url=ui_url)

    summary = {
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "mlflow_experiment_name": config.mlflow_experiment_name,
        "mlflow_run_id": run_id,
        "mlflow_ui_url": ui_url,
        "mlflow_run_url": run_url,
    }
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def set_retrieval_tags(*, model_type: str, split: str, sample: bool) -> None:
    mlflow.set_tags(
        {
            "step": "plain_two_tower_retriever",
            "model_family": "two_tower",
            "dataset": "movielens_25m",
            "model_type": model_type,
            "split": split,
            "sample": str(sample).lower(),
        }
    )


def log_training_params(config: RetrievalConfig) -> None:
    params = {
        "model_type": config.model.model_type,
        "batch_size": config.train.train_batch_size,
        "embedding_dim": config.model.embedding_dim,
        "user_id_embedding_dim": config.model.user_id_embedding_dim,
        "item_id_embedding_dim": config.model.item_id_embedding_dim,
        "learning_rate": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
        "epochs": config.train.epochs,
        "temperature": config.model.temperature,
        "history_length": config.train.history_length,
        "scheduler": config.train.scheduler,
        "scheduler_t_max": config.train.scheduler_t_max,
        "scheduler_patience": config.train.scheduler_patience,
        "scheduler_factor": config.train.scheduler_factor,
        "min_learning_rate": config.train.min_learning_rate,
        "transformer_layers": config.model.transformer_layers,
        "transformer_heads": config.model.transformer_heads,
        "transformer_ffn_dim": config.model.transformer_ffn_dim,
        "sequence_pooling": config.model.sequence_pooling,
        "attention_type": "scaled_dot_product_attention",
    }
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    def _normalize_metric_name(name: str) -> str:
        return name.replace("@", "_at_")

    for name, value in metrics.items():
        metric_name = _normalize_metric_name(name)
        if step is None:
            mlflow.log_metric(metric_name, float(value))
        else:
            mlflow.log_metric(metric_name, float(value), step=step)


def log_artifacts(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path))


def get_active_run_id() -> str | None:
    active = mlflow.active_run()
    if active is None:
        return None
    return cast(str, active.info.run_id)


def flatten_config_for_snapshot(config: RetrievalConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")
