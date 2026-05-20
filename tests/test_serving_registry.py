"""Tests for serving artifact registry baseline behavior."""

from __future__ import annotations

from movie_recsys.serving.config import ServingConfig
from movie_recsys.serving.registry import ArtifactRegistry


def test_registry_not_ready_before_loading() -> None:
    registry = ArtifactRegistry(ServingConfig())
    assert registry.is_ready() is False
