"""Typed configuration for retrieval baselines and two-tower training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from movie_recsys.constants import PROJECT_ROOT


class EnvConfig(BaseSettings):
    """Environment-backed defaults for local development."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MOVIELENS_DATA_DIR: str = "data/processed"
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "movielens-two-tower"
    MODEL_OUTPUT_DIR: str = "artifacts/models"
    INDEX_OUTPUT_DIR: str = "artifacts/faiss"
    REPORT_OUTPUT_DIR: str = "artifacts/reports"
    RANDOM_SEED: int = 42
    DEVICE: str = "auto"
    TRAIN_BATCH_SIZE: int = 256
    EVAL_BATCH_SIZE: int = 512
    NUM_WORKERS: int = 0
    RETRIEVAL_EMBEDDING_DIM: int = 128
    USER_HISTORY_LENGTH: int = 50
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-6
    EPOCHS: int = 3
    AMP_ENABLED: bool = True


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processed_data_dir: Path
    model_output_dir: Path
    index_output_dir: Path
    report_output_dir: Path


class DataFilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interactions_train: str = "interactions_train.parquet"
    interactions_val: str = "interactions_val.parquet"
    interactions_test: str = "interactions_test.parquet"
    users: str = "users.parquet"
    items: str = "items.parquet"
    user_id_map: str = "user_id_map.parquet"
    item_id_map: str = "item_id_map.parquet"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_dim: int = 128
    user_id_embedding_dim: int = 64
    item_id_embedding_dim: int = 64
    feature_hidden_dim: int = 128
    projection_hidden_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.07


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 0
    history_length: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 3
    amp_enabled: bool = True
    max_grad_norm: float = 1.0


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig
    files: DataFilesConfig = Field(default_factory=DataFilesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    @property
    def train_path(self) -> Path:
        return self.paths.processed_data_dir / self.files.interactions_train

    @property
    def val_path(self) -> Path:
        return self.paths.processed_data_dir / self.files.interactions_val

    @property
    def test_path(self) -> Path:
        return self.paths.processed_data_dir / self.files.interactions_test

    @property
    def users_path(self) -> Path:
        return self.paths.processed_data_dir / self.files.users

    @property
    def items_path(self) -> Path:
        return self.paths.processed_data_dir / self.files.items


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        msg = f"Expected dictionary config in {path}"
        raise ValueError(msg)
    return payload


def load_retrieval_config(
    config_path: str | Path = "configs/retrieval.yaml",
    *,
    sample: bool = False,
) -> RetrievalConfig:
    env = EnvConfig()
    raw = _load_yaml(_resolve_path(config_path))

    paths = raw.get("paths", {})
    train = raw.get("train", {})
    model = raw.get("model", {})

    processed_dir = _resolve_path(paths.get("processed_data_dir", env.MOVIELENS_DATA_DIR))
    if sample:
        processed_dir = processed_dir / "sample"

    config = RetrievalConfig(
        paths=PathsConfig(
            processed_data_dir=processed_dir,
            model_output_dir=_resolve_path(paths.get("model_output_dir", env.MODEL_OUTPUT_DIR)),
            index_output_dir=_resolve_path(paths.get("index_output_dir", env.INDEX_OUTPUT_DIR)),
            report_output_dir=_resolve_path(paths.get("report_output_dir", env.REPORT_OUTPUT_DIR)),
        ),
        files=DataFilesConfig.model_validate(raw.get("files", {})),
        model=ModelConfig(
            embedding_dim=model.get("embedding_dim", env.RETRIEVAL_EMBEDDING_DIM),
            user_id_embedding_dim=model.get("user_id_embedding_dim", 64),
            item_id_embedding_dim=model.get("item_id_embedding_dim", 64),
            feature_hidden_dim=model.get("feature_hidden_dim", 128),
            projection_hidden_dim=model.get("projection_hidden_dim", 256),
            dropout=model.get("dropout", 0.1),
            temperature=model.get("temperature", 0.07),
        ),
        train=TrainConfig(
            random_seed=train.get("random_seed", env.RANDOM_SEED),
            device=train.get("device", env.DEVICE),
            train_batch_size=train.get("train_batch_size", env.TRAIN_BATCH_SIZE),
            eval_batch_size=train.get("eval_batch_size", env.EVAL_BATCH_SIZE),
            num_workers=train.get("num_workers", env.NUM_WORKERS),
            history_length=train.get("history_length", env.USER_HISTORY_LENGTH),
            learning_rate=train.get("learning_rate", env.LEARNING_RATE),
            weight_decay=train.get("weight_decay", env.WEIGHT_DECAY),
            epochs=train.get("epochs", env.EPOCHS),
            amp_enabled=train.get("amp_enabled", env.AMP_ENABLED),
            max_grad_norm=train.get("max_grad_norm", 1.0),
        ),
        mlflow_tracking_uri=raw.get("mlflow_tracking_uri", env.MLFLOW_TRACKING_URI),
        mlflow_experiment_name=raw.get("mlflow_experiment_name", env.MLFLOW_EXPERIMENT_NAME),
    )

    config.paths.model_output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.index_output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.report_output_dir.mkdir(parents=True, exist_ok=True)
    return config
