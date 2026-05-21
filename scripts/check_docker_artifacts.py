"""Validate required local artifacts before running Docker Compose serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from movie_recsys.constants import PROJECT_ROOT

app = typer.Typer(add_completion=False)


def _resolve_path(path_like: str | Path, root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        msg = f"Expected mapping in config file: {path}"
        raise ValueError(msg)
    return payload


def _relative_display(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_required_paths(config_path: Path, root: Path = PROJECT_ROOT) -> list[tuple[str, Path]]:
    """Build the full list of required files for local Docker serving."""

    serving_config_path = _resolve_path(config_path, root)
    serving_payload = _load_yaml(serving_config_path)
    serving_paths = serving_payload.get("paths", {})
    if not isinstance(serving_paths, dict):
        msg = "Expected 'paths' object in serving config"
        raise ValueError(msg)

    retrieval_config = _resolve_path(
        serving_paths.get("retrieval_config", "configs/transformer_retrieval_residual.yaml"),
        root,
    )
    ranker_config = _resolve_path(
        serving_paths.get("ranker_config", "configs/ranker.yaml"),
        root,
    )
    faiss_dir = _resolve_path(serving_paths.get("faiss_dir", "artifacts/faiss"), root)
    residual_checkpoint = _resolve_path(
        serving_paths.get(
            "residual_checkpoint",
            "artifacts/models/best_residual_transformer_retriever.pt",
        ),
        root,
    )
    ranker_checkpoint = _resolve_path(
        serving_paths.get("ranker_checkpoint", "artifacts/models/best_neural_ranker.pt"),
        root,
    )

    retrieval_payload = _load_yaml(retrieval_config)
    retrieval_paths = retrieval_payload.get("paths", {})
    retrieval_files = retrieval_payload.get("files", {})
    if not isinstance(retrieval_paths, dict):
        msg = "Expected 'paths' object in retrieval config"
        raise ValueError(msg)
    if not isinstance(retrieval_files, dict):
        msg = "Expected 'files' object in retrieval config"
        raise ValueError(msg)

    processed_data_dir = _resolve_path(
        retrieval_paths.get("processed_data_dir", "data/processed"),
        root,
    )

    return [
        ("serving config", serving_config_path),
        ("ranker config", ranker_config),
        ("retrieval config", retrieval_config),
        ("residual checkpoint", residual_checkpoint),
        ("ranker checkpoint", ranker_checkpoint),
        ("faiss index", faiss_dir / "index.faiss"),
        ("faiss metadata", faiss_dir / "index_metadata.json"),
        ("faiss item mapping", faiss_dir / "item_idx_mapping.parquet"),
        (
            "items parquet",
            processed_data_dir / str(retrieval_files.get("items", "items.parquet")),
        ),
        (
            "users parquet",
            processed_data_dir / str(retrieval_files.get("users", "users.parquet")),
        ),
        (
            "interactions_train parquet",
            processed_data_dir
            / str(retrieval_files.get("interactions_train", "interactions_train.parquet")),
        ),
        (
            "user_id_map parquet",
            processed_data_dir / str(retrieval_files.get("user_id_map", "user_id_map.parquet")),
        ),
        (
            "item_id_map parquet",
            processed_data_dir / str(retrieval_files.get("item_id_map", "item_id_map.parquet")),
        ),
    ]


def build_preflight_report(config_path: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Evaluate required files and return a machine-friendly report."""

    checks = []
    missing_paths: list[str] = []
    for label, path in resolve_required_paths(config_path, root=root):
        exists = path.exists()
        display_path = _relative_display(path, root)
        checks.append(
            {
                "label": label,
                "path": display_path,
                "exists": exists,
                "status": "FOUND" if exists else "MISSING",
            }
        )
        if not exists:
            missing_paths.append(display_path)

    return {
        "ok": len(missing_paths) == 0,
        "checks": checks,
        "missing_paths": missing_paths,
    }


def print_preflight_report(report: dict[str, Any]) -> None:
    """Print a concise FOUND/MISSING table and final status."""

    typer.echo("FOUND/MISSING")
    typer.echo(f"{'status':<8} {'label':<28} path")
    typer.echo("-" * 80)
    for row in report["checks"]:
        typer.echo(f"{row['status']:<8} {row['label']:<28} {row['path']}")

    ok = bool(report["ok"])
    typer.echo("")
    typer.echo(f"final ok: {str(ok).lower()}")
    if not ok:
        typer.echo("missing paths:")
        for path in report["missing_paths"]:
            typer.echo(f"- {path}")


@app.command()
def main(
    config: Path = typer.Option(Path("configs/serving.yaml"), "--config"),
) -> None:
    """Validate local files needed by Docker Compose serving."""

    report = build_preflight_report(config)
    print_preflight_report(report)
    if not report["ok"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
