"""Tests for serving config loading behavior."""

from __future__ import annotations

from movie_recsys.serving.config import ServingConfig


def test_serving_config_defaults_construct() -> None:
    config = ServingConfig()
    assert config.runtime.default_top_k >= 1
