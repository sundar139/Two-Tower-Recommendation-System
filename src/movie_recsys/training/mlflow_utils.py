"""MLflow helpers for retrieval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlflow

from movie_recsys.training.config import RetrievalConfig


def setup_mlflow(config: RetrievalConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)


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
        "batch_size": config.train.train_batch_size,
        "embedding_dim": config.model.embedding_dim,
        "learning_rate": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
        "epochs": config.train.epochs,
        "temperature": config.model.temperature,
        "history_length": config.train.history_length,
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
