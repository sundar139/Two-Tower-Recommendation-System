"""Tests for serving artifact registry baseline behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from movie_recsys.serving.config import ServingConfig
from movie_recsys.serving.registry import ArtifactRegistry


def test_registry_not_ready_before_loading() -> None:
    registry = ArtifactRegistry(ServingConfig())
    assert registry.is_ready() is False


def test_registry_load_fails_when_required_artifacts_missing(tmp_path: Path) -> None:
    config = ServingConfig.model_validate(
        {
            "paths": {
                "retrieval_config": str(tmp_path / "missing_retrieval.yaml"),
                "ranker_config": str(tmp_path / "missing_ranker.yaml"),
                "faiss_dir": str(tmp_path / "missing_faiss"),
                "residual_checkpoint": str(tmp_path / "missing_retriever.pt"),
                "ranker_checkpoint": str(tmp_path / "missing_ranker.pt"),
                "ranker_feature_manifest": str(tmp_path / "missing_manifest.json"),
            },
            "scoring": {
                "policy_name": "ranker_topk_popularity_backfill",
                "alpha": 1.0,
                "beta": 0.1,
                "gamma": 0.0,
                "top_k_focus": 20,
            },
            "api": {"host": "127.0.0.1", "port": 8000, "log_level": "info"},
            "runtime": {
                "candidate_top_k": 50,
                "default_top_k": 10,
                "max_top_k": 50,
                "min_top_k": 1,
                "device": "cpu",
            },
        }
    )

    registry = ArtifactRegistry(config)
    with pytest.raises(FileNotFoundError):
        registry.load()

