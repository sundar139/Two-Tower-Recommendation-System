"""Typed configuration loader for FastAPI serving."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from movie_recsys.constants import PROJECT_ROOT


class ServingPathsConfig(BaseModel):
    """Artifact and dataset paths used by serving components."""

    model_config = ConfigDict(extra="forbid")

    retrieval_config: Path = Path("configs/transformer_retrieval_residual.yaml")
    ranker_config: Path = Path("configs/ranker.yaml")
    faiss_dir: Path = Path("artifacts/faiss")
    residual_checkpoint: Path = Path(
        "artifacts/models/best_residual_transformer_retriever.pt"
    )
    ranker_checkpoint: Path = Path("artifacts/models/best_neural_ranker.pt")
    ranker_feature_manifest: Path = Path(
        "artifacts/ranker/features/full/ranker_features_manifest.json"
    )


class ServingScoringConfig(BaseModel):
    """Hybrid policy weights aligned to selected production scorer."""

    model_config = ConfigDict(extra="forbid")

    policy_name: str = "ranker_topk_popularity_backfill"
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.0
    top_k_focus: int = 20


class ServingApiConfig(BaseModel):
    """FastAPI runtime settings."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


class ServingRuntimeConfig(BaseModel):
    """Inference and retrieval runtime constraints."""

    model_config = ConfigDict(extra="forbid")

    candidate_top_k: int = 200
    default_top_k: int = 20
    max_top_k: int = 200
    min_top_k: int = 1
    device: str = "auto"
    sample_data: bool = True


class ServingConfig(BaseModel):
    """Top-level serving configuration."""

    model_config = ConfigDict(extra="forbid")

    paths: ServingPathsConfig = ServingPathsConfig()
    scoring: ServingScoringConfig = ServingScoringConfig()
    api: ServingApiConfig = ServingApiConfig()
    runtime: ServingRuntimeConfig = ServingRuntimeConfig()


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


def load_serving_config(config_path: str | Path = "configs/serving.yaml") -> ServingConfig:
    """Load and resolve serving config from YAML."""

    raw = _load_yaml(_resolve_path(config_path))

    paths_raw = raw.get("paths", {})
    if not isinstance(paths_raw, dict):
        msg = "Expected 'paths' object in serving config"
        raise ValueError(msg)
    paths = ServingPathsConfig(
        retrieval_config=_resolve_path(
            paths_raw.get("retrieval_config", "configs/transformer_retrieval_residual.yaml")
        ),
        ranker_config=_resolve_path(paths_raw.get("ranker_config", "configs/ranker.yaml")),
        faiss_dir=_resolve_path(paths_raw.get("faiss_dir", "artifacts/faiss")),
        residual_checkpoint=_resolve_path(
            paths_raw.get(
                "residual_checkpoint",
                "artifacts/models/best_residual_transformer_retriever.pt",
            )
        ),
        ranker_checkpoint=_resolve_path(
            paths_raw.get("ranker_checkpoint", "artifacts/models/best_neural_ranker.pt")
        ),
        ranker_feature_manifest=_resolve_path(
            paths_raw.get(
                "ranker_feature_manifest",
                "artifacts/ranker/features/full/ranker_features_manifest.json",
            )
        ),
    )

    scoring_raw = raw.get("scoring", {})
    if not isinstance(scoring_raw, dict):
        msg = "Expected 'scoring' object in serving config"
        raise ValueError(msg)
    scoring = ServingScoringConfig.model_validate(scoring_raw)

    api_raw = raw.get("api", {})
    if not isinstance(api_raw, dict):
        msg = "Expected 'api' object in serving config"
        raise ValueError(msg)
    api = ServingApiConfig.model_validate(api_raw)

    runtime_raw = raw.get("runtime", {})
    if not isinstance(runtime_raw, dict):
        msg = "Expected 'runtime' object in serving config"
        raise ValueError(msg)
    runtime = ServingRuntimeConfig.model_validate(runtime_raw)

    return ServingConfig(paths=paths, scoring=scoring, api=api, runtime=runtime)
