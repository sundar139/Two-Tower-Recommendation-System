"""Tests for serving config loading behavior."""

from __future__ import annotations

from pathlib import Path

from movie_recsys.serving.config import ServingConfig, load_serving_config


def test_serving_config_defaults_construct() -> None:
    config = ServingConfig()
    assert config.runtime.default_top_k >= 1


def test_load_serving_config_reads_yaml(tmp_path: Path) -> None:
        config_path = tmp_path / "serving.yaml"
        config_path.write_text(
                """
paths:
    retrieval_config: configs/transformer_retrieval_residual.yaml
    ranker_config: configs/ranker.yaml
    faiss_dir: artifacts/faiss
    residual_checkpoint: artifacts/models/best_residual_transformer_retriever.pt
    ranker_checkpoint: artifacts/models/best_neural_ranker.pt
    ranker_feature_manifest: artifacts/ranker/features/full/ranker_features_manifest.json
runtime:
    candidate_top_k: 120
    default_top_k: 15
    max_top_k: 120
    min_top_k: 5
    device: cpu
scoring:
    policy_name: ranker_topk_popularity_backfill
    alpha: 1.0
    beta: 0.1
    gamma: 0.0
    top_k_focus: 20
api:
    host: 127.0.0.1
    port: 8010
    log_level: warning
""".strip(),
                encoding="utf-8",
        )

        config = load_serving_config(config_path)
        assert config.runtime.candidate_top_k == 120
        assert config.runtime.default_top_k == 15
        assert config.api.port == 8010
        assert config.runtime.device == "cpu"
