"""Configuration models and loaders for the neural ranker."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from movie_recsys.constants import PROJECT_ROOT
from movie_recsys.training.config import EnvConfig


class RankerPathsConfig(BaseModel):
	"""Filesystem paths used by ranker generation, training, and evaluation."""

	model_config = ConfigDict(extra="forbid")

	ranker_candidate_dir: Path
	ranker_model_dir: Path
	ranker_report_dir: Path


class RankerConfig(BaseModel):
	"""Main ranker configuration payload."""

	model_config = ConfigDict(extra="forbid")

	model_type: Literal["neural_ranker"] = "neural_ranker"
	retrieval_backbone: Literal["residual_transformer"] = "residual_transformer"
	retriever_config: Path
	retriever_checkpoint: Path

	candidate_top_k: int = 200
	ranker_top_k: int = 10
	negative_samples_per_positive: int = 50

	batch_size: int = 1024
	epochs: int = 8
	learning_rate: float = 5e-4
	weight_decay: float = 1e-5
	dropout: float = 0.10
	hidden_dims: list[int] = Field(default_factory=lambda: [512, 256, 128])

	use_frozen_retrieval_embeddings: bool = True
	embedding_interaction_dim: int = 128

	loss_type: Literal["bce", "bpr", "hybrid"] = "bce"
	pairwise_margin: float = 0.2

	amp_enabled: bool = True
	gradient_clip_norm: float = 1.0
	scheduler: Literal["none", "cosine", "plateau", "warmup_cosine"] = "warmup_cosine"
	warmup_steps: int = 200
	random_seed: int = 42

	mlflow_tracking_uri: str
	mlflow_artifact_root: str
	mlflow_experiment_name: str
	mlflow_ui_host: str
	mlflow_ui_port: int
	mlflow_ui_url: str

	paths: RankerPathsConfig

	def artifact_scope(self, *, sample: bool) -> str:
		return "sample" if sample else "full"

	def candidate_dir_for_scope(self, *, sample: bool) -> Path:
		return self.paths.ranker_candidate_dir / self.artifact_scope(sample=sample)

	def candidate_path(self, *, split: str, sample: bool) -> Path:
		return self.candidate_dir_for_scope(sample=sample) / f"ranker_candidates_{split}.parquet"

	def features_path(self, *, split: str, sample: bool) -> Path:
		return self.candidate_dir_for_scope(sample=sample) / f"ranker_features_{split}.parquet"

	@property
	def best_checkpoint(self) -> Path:
		return self.paths.ranker_model_dir / "best_neural_ranker.pt"

	@property
	def last_checkpoint(self) -> Path:
		return self.paths.ranker_model_dir / "last_neural_ranker.pt"


ModelTypeLiteral = Literal["neural_ranker"]
RetrievalBackboneLiteral = Literal["residual_transformer"]
LossTypeLiteral = Literal["bce", "bpr", "hybrid"]
SchedulerLiteral = Literal["none", "cosine", "plateau", "warmup_cosine"]


def _as_str(raw: dict[str, object], key: str, default: str) -> str:
	value = raw.get(key, default)
	if isinstance(value, str):
		return value
	return str(value)


def _as_int(raw: dict[str, object], key: str, default: int) -> int:
	value = raw.get(key, default)
	if isinstance(value, bool):
		return int(value)
	if isinstance(value, int):
		return value
	if isinstance(value, float):
		return int(value)
	if isinstance(value, str):
		return int(value)
	msg = f"Expected integer-like value for {key}, got {type(value).__name__}"
	raise ValueError(msg)


def _as_float(raw: dict[str, object], key: str, default: float) -> float:
	value = raw.get(key, default)
	if isinstance(value, bool):
		return float(value)
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		return float(value)
	msg = f"Expected float-like value for {key}, got {type(value).__name__}"
	raise ValueError(msg)


def _as_bool(raw: dict[str, object], key: str, default: bool) -> bool:
	value = raw.get(key, default)
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		lowered = value.strip().lower()
		if lowered in {"1", "true", "yes", "y"}:
			return True
		if lowered in {"0", "false", "no", "n"}:
			return False
	if isinstance(value, (int, float)):
		return bool(value)
	msg = f"Expected boolean-like value for {key}, got {type(value).__name__}"
	raise ValueError(msg)


def _as_int_list(raw: dict[str, object], key: str, default: list[int]) -> list[int]:
	value = raw.get(key, default)
	if isinstance(value, list):
		parsed: list[int] = []
		for idx, element in enumerate(value):
			fallback = default[idx] if idx < len(default) else 0
			parsed.append(_as_int({"value": element}, "value", fallback))
		return parsed
	msg = f"Expected list value for {key}, got {type(value).__name__}"
	raise ValueError(msg)


def _normalize_model_type(value: str) -> ModelTypeLiteral:
	if value == "neural_ranker":
		return cast(ModelTypeLiteral, value)
	msg = f"Unsupported model_type: {value}"
	raise ValueError(msg)


def _normalize_backbone(value: str) -> RetrievalBackboneLiteral:
	if value == "residual_transformer":
		return cast(RetrievalBackboneLiteral, value)
	msg = f"Unsupported retrieval_backbone: {value}"
	raise ValueError(msg)


def _normalize_loss_type(value: str) -> LossTypeLiteral:
	if value in {"bce", "bpr", "hybrid"}:
		return cast(LossTypeLiteral, value)
	msg = f"Unsupported loss_type: {value}"
	raise ValueError(msg)


def _normalize_scheduler(value: str) -> SchedulerLiteral:
	if value in {"none", "cosine", "plateau", "warmup_cosine"}:
		return cast(SchedulerLiteral, value)
	msg = f"Unsupported scheduler: {value}"
	raise ValueError(msg)


def _resolve_path(path_like: str | Path) -> Path:
	path = Path(path_like)
	if path.is_absolute():
		return path
	return (PROJECT_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, object]:
	with path.open("r", encoding="utf-8") as handle:
		payload = yaml.safe_load(handle) or {}
	if not isinstance(payload, dict):
		msg = f"Expected dictionary config in {path}"
		raise ValueError(msg)
	return payload


def load_ranker_config(
	config_path: str | Path = "configs/ranker.yaml",
) -> RankerConfig:
	"""Load ranker configuration with environment-backed MLflow defaults."""

	env = EnvConfig()
	raw = _load_yaml(_resolve_path(config_path))

	config = RankerConfig(
		model_type=_normalize_model_type(_as_str(raw, "model_type", "neural_ranker")),
		retrieval_backbone=_normalize_backbone(
			_as_str(raw, "retrieval_backbone", "residual_transformer")
		),
		retriever_config=_resolve_path(_as_str(raw, "retriever_config", "configs/retrieval.yaml")),
		retriever_checkpoint=_resolve_path(
			_as_str(
				raw,
				"retriever_checkpoint",
				"artifacts/models/best_residual_transformer_retriever.pt",
			)
		),
		candidate_top_k=_as_int(raw, "candidate_top_k", 200),
		ranker_top_k=_as_int(raw, "ranker_top_k", 10),
		negative_samples_per_positive=_as_int(raw, "negative_samples_per_positive", 50),
		batch_size=_as_int(raw, "batch_size", 1024),
		epochs=_as_int(raw, "epochs", 8),
		learning_rate=_as_float(raw, "learning_rate", 5e-4),
		weight_decay=_as_float(raw, "weight_decay", 1e-5),
		dropout=_as_float(raw, "dropout", 0.10),
		hidden_dims=_as_int_list(raw, "hidden_dims", [512, 256, 128]),
		use_frozen_retrieval_embeddings=_as_bool(raw, "use_frozen_retrieval_embeddings", True),
		embedding_interaction_dim=_as_int(raw, "embedding_interaction_dim", 128),
		loss_type=_normalize_loss_type(_as_str(raw, "loss_type", "bce")),
		pairwise_margin=_as_float(raw, "pairwise_margin", 0.2),
		amp_enabled=_as_bool(raw, "amp_enabled", True),
		gradient_clip_norm=_as_float(raw, "gradient_clip_norm", 1.0),
		scheduler=_normalize_scheduler(_as_str(raw, "scheduler", "warmup_cosine")),
		warmup_steps=_as_int(raw, "warmup_steps", 200),
		random_seed=_as_int(raw, "random_seed", 42),
		mlflow_tracking_uri=_as_str(raw, "mlflow_tracking_uri", env.MLFLOW_TRACKING_URI),
		mlflow_artifact_root=_as_str(raw, "mlflow_artifact_root", env.MLFLOW_ARTIFACT_ROOT),
		mlflow_experiment_name=_as_str(raw, "mlflow_experiment_name", env.MLFLOW_EXPERIMENT_NAME),
		mlflow_ui_host=_as_str(raw, "mlflow_ui_host", env.MLFLOW_UI_HOST),
		mlflow_ui_port=_as_int(raw, "mlflow_ui_port", env.MLFLOW_UI_PORT),
		mlflow_ui_url=_as_str(raw, "mlflow_ui_url", env.MLFLOW_UI_URL),
		paths=RankerPathsConfig(
			ranker_candidate_dir=_resolve_path(
				_as_str(raw, "ranker_candidate_dir", "artifacts/ranker/candidates")
			),
			ranker_model_dir=_resolve_path(
				_as_str(raw, "ranker_model_dir", env.MODEL_OUTPUT_DIR)
			),
			ranker_report_dir=_resolve_path(
				_as_str(raw, "ranker_report_dir", env.REPORT_OUTPUT_DIR)
			),
		),
	)

	config.paths.ranker_candidate_dir.mkdir(parents=True, exist_ok=True)
	config.paths.ranker_model_dir.mkdir(parents=True, exist_ok=True)
	config.paths.ranker_report_dir.mkdir(parents=True, exist_ok=True)
	return config
