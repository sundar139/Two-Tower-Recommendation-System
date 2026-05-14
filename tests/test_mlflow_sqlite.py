from __future__ import annotations

from movie_recsys.constants import PROJECT_ROOT
from movie_recsys.training.config import EnvConfig, load_retrieval_config
from movie_recsys.training.mlflow_utils import build_mlflow_run_url


def test_build_mlflow_run_url_expected_format() -> None:
    url = build_mlflow_run_url(
        "123456789",
        "abc123",
        ui_url="http://127.0.0.1:5000",
    )
    assert url == "http://127.0.0.1:5000/#/experiments/123456789/runs/abc123"


def test_sqlite_tracking_uri_default() -> None:
    env = EnvConfig()
    assert env.MLFLOW_TRACKING_URI == "sqlite:///mlflow.db"

    cfg = load_retrieval_config("configs/retrieval.yaml", sample=True)
    assert cfg.mlflow_tracking_uri == "sqlite:///mlflow.db"


def test_gitignore_covers_mlflow_db() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "mlflow.db" in gitignore
