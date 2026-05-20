"""Artifact registry responsible for serving-time model assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from movie_recsys.serving.config import ServingConfig


@dataclass(slots=True)
class LoadedArtifacts:
    """Container for loaded serving artifacts."""

    index: Any
    item_mapping: np.ndarray
    retrieval_model: Any
    ranker_model: Any
    feature_columns: list[str]
    users_frame: Any
    items_frame: Any


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

    def load(self) -> None:
        """Load configured assets into memory."""

        msg = "Artifact loading is not implemented yet"
        raise NotImplementedError(msg)

    def validate_paths(self) -> dict[str, Path]:
        """Return required artifact paths for diagnostics and readiness checks."""

        paths = {
            "faiss_dir": self._config.paths.faiss_dir,
            "residual_checkpoint": self._config.paths.residual_checkpoint,
            "ranker_checkpoint": self._config.paths.ranker_checkpoint,
            "ranker_feature_manifest": self._config.paths.ranker_feature_manifest,
            "retrieval_config": self._config.paths.retrieval_config,
            "ranker_config": self._config.paths.ranker_config,
        }
        return paths
