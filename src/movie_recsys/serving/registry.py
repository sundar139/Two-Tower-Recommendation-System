"""Artifact registry responsible for serving-time model assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch

from movie_recsys.modeling.artifacts import load_checkpoint
from movie_recsys.modeling.cl_retrieval import CLResidualTransformerRetriever
from movie_recsys.modeling.datasets import FeatureTables, load_feature_tables
from movie_recsys.modeling.faiss_index import load_faiss_bundle
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.ranking.config import RankerConfig, load_ranker_config
from movie_recsys.ranking.model import NeuralRanker
from movie_recsys.serving.config import ServingConfig
from movie_recsys.training.config import RetrievalConfig, load_retrieval_config


@dataclass(slots=True)
class LoadedArtifacts:
    """Container for loaded serving artifacts."""

    retrieval_config: RetrievalConfig
    ranker_config: RankerConfig
    feature_tables: FeatureTables
    index: Any
    item_mapping: np.ndarray
    item_embeddings: np.ndarray
    retrieval_model: Any
    ranker_model: Any
    feature_columns: list[str]
    users_frame: pl.DataFrame
    items_frame: pl.DataFrame
    device: torch.device


def _select_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        msg = "Configured serving device 'cuda' is unavailable on this machine"
        raise RuntimeError(msg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_feature_columns(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Expected object JSON in feature manifest: {manifest_path}"
        raise ValueError(msg)

    raw_columns = payload.get("feature_columns")
    if not isinstance(raw_columns, list):
        msg = f"Missing feature_columns list in feature manifest: {manifest_path}"
        raise ValueError(msg)

    columns = [str(value) for value in raw_columns if str(value)]
    if not columns:
        msg = f"Feature manifest has an empty feature_columns list: {manifest_path}"
        raise ValueError(msg)
    return columns


def _build_retriever_model(
    retrieval_config: RetrievalConfig,
    tables: FeatureTables,
) -> Any:
    common_kwargs = {
        "config": retrieval_config,
        "num_users": tables.user_features.shape[0],
        "num_items_with_padding": tables.item_features.shape[0] + 1,
        "user_feature_dim": tables.user_features.shape[1],
        "item_feature_dim": tables.item_features.shape[1],
    }
    model_type = retrieval_config.model.model_type
    if model_type == "transformer":
        return TransformerRetriever(**common_kwargs)
    if model_type == "residual_transformer":
        return ResidualTransformerRetriever(**common_kwargs)
    if model_type == "cl_residual_transformer":
        return CLResidualTransformerRetriever(**common_kwargs)
    return BaselineRetriever(**common_kwargs)


class ArtifactRegistry:
    """Registry that lazily loads and exposes model artifacts."""

    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._artifacts: LoadedArtifacts | None = None

    @property
    def config(self) -> ServingConfig:
        return self._config

    @property
    def artifacts(self) -> LoadedArtifacts:
        if self._artifacts is None:
            msg = "Serving artifacts are not loaded"
            raise RuntimeError(msg)
        return self._artifacts

    def is_ready(self) -> bool:
        return self._artifacts is not None

    def _required_file_paths(self) -> dict[str, Path]:
        return {
            "faiss_index": self._config.paths.faiss_dir / "index.faiss",
            "faiss_mapping": self._config.paths.faiss_dir / "item_idx_mapping.parquet",
            "faiss_metadata": self._config.paths.faiss_dir / "index_metadata.json",
            "residual_checkpoint": self._config.paths.residual_checkpoint,
            "ranker_checkpoint": self._config.paths.ranker_checkpoint,
            "ranker_feature_manifest": self._config.paths.ranker_feature_manifest,
            "retrieval_config": self._config.paths.retrieval_config,
            "ranker_config": self._config.paths.ranker_config,
        }

    def _validate_required_paths(self) -> None:
        missing = [
            f"{name}={path}"
            for name, path in self._required_file_paths().items()
            if not path.exists()
        ]
        if missing:
            msg = "Missing serving artifacts: " + ", ".join(missing)
            raise FileNotFoundError(msg)

    def load(self) -> None:
        """Load configured assets into memory."""

        if self._artifacts is not None:
            return

        self._validate_required_paths()

        retrieval_config = load_retrieval_config(
            self._config.paths.retrieval_config,
            sample=bool(self._config.runtime.sample_data),
        )
        ranker_config = load_ranker_config(self._config.paths.ranker_config)

        if retrieval_config.model.model_type != "residual_transformer":
            msg = (
                "Serving is configured for residual transformer retrieval, "
                f"received: {retrieval_config.model.model_type}"
            )
            raise ValueError(msg)

        feature_tables = load_feature_tables(retrieval_config)
        users_frame = pl.read_parquet(retrieval_config.users_path).sort("user_idx")
        items_frame = pl.read_parquet(retrieval_config.items_path).sort("item_idx")

        feature_columns = _load_feature_columns(self._config.paths.ranker_feature_manifest)
        device = _select_device(self._config.runtime.device)

        retrieval_model = _build_retriever_model(retrieval_config, feature_tables)
        retrieval_payload = load_checkpoint(self._config.paths.residual_checkpoint)
        retrieval_model.load_state_dict(cast(dict[str, Any], retrieval_payload["model_state_dict"]))
        retrieval_model.to(device)
        retrieval_model.eval()

        with torch.no_grad():
            item_features = torch.tensor(
                feature_tables.item_features,
                dtype=torch.float32,
                device=device,
            )
            item_idx = torch.arange(
                1,
                feature_tables.item_features.shape[0] + 1,
                dtype=torch.long,
                device=device,
            )
            item_embeddings_tensor = retrieval_model.encode_item(
                {"item_idx": item_idx, "item_features": item_features}
            )
        item_embeddings = cast(
            np.ndarray,
            item_embeddings_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
        )

        index, item_mapping = load_faiss_bundle(self._config.paths.faiss_dir)
        if item_mapping.shape[0] != item_embeddings.shape[0]:
            msg = (
                "FAISS item mapping size does not match encoded item embeddings: "
                f"mapping={item_mapping.shape[0]} embeddings={item_embeddings.shape[0]}"
            )
            raise ValueError(msg)

        ranker_payload = load_checkpoint(self._config.paths.ranker_checkpoint)
        ranker_model = NeuralRanker(
            input_dim=len(feature_columns),
            hidden_dims=[int(value) for value in ranker_payload["hidden_dims"]],
            dropout=float(ranker_payload["dropout"]),
            use_layer_norm=bool(ranker_payload.get("use_layer_norm", True)),
        )
        ranker_model.load_state_dict(cast(dict[str, Any], ranker_payload["model_state_dict"]))
        ranker_model.to(device)
        ranker_model.eval()

        self._artifacts = LoadedArtifacts(
            retrieval_config=retrieval_config,
            ranker_config=ranker_config,
            feature_tables=feature_tables,
            index=index,
            item_mapping=item_mapping,
            item_embeddings=item_embeddings,
            retrieval_model=retrieval_model,
            ranker_model=ranker_model,
            feature_columns=feature_columns,
            users_frame=users_frame,
            items_frame=items_frame,
            device=device,
        )

    def validate_paths(self) -> dict[str, Path]:
        """Return required artifact paths for diagnostics and readiness checks."""

        return self._required_file_paths()
