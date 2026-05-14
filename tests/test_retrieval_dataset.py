from __future__ import annotations

from pathlib import Path

import polars as pl
import torch

from movie_recsys.modeling.datasets import (
    RetrievalDataset,
    collate_retrieval_batch,
    load_feature_tables,
)
from movie_recsys.training.config import load_retrieval_config


def _write_small_tables(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    train = pl.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1],
            "item_idx": [1, 2, 3, 2, 3, 4],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "explicit_rating": [5.0] * 6,
            "userId": [10, 10, 10, 11, 11, 11],
            "movieId": [101, 102, 103, 102, 103, 104],
            "rating": [5.0] * 6,
            "is_positive": [True] * 6,
        }
    )
    train.write_parquet(processed / "interactions_train.parquet")
    train.head(2).write_parquet(processed / "interactions_val.parquet")
    train.head(2).write_parquet(processed / "interactions_test.parquet")

    users = pl.DataFrame(
        {
            "user_idx": [0, 1],
            "original_userId": [10, 11],
            "total_rating_count": [3, 3],
            "positive_rating_count": [3, 3],
            "mean_rating": [5.0, 5.0],
            "tag_count": [0, 0],
            "train_history_item_idx": [[1, 2], [2, 3]],
        }
    )
    users.write_parquet(processed / "users.parquet")

    items = pl.DataFrame(
        {
            "item_idx": [0, 1, 2, 3, 4],
            "original_movieId": [100, 101, 102, 103, 104],
            "release_year": [1999, 2000, 2001, 2002, 2003],
            "positive_count": [1, 2, 3, 4, 5],
            "popularity_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "rating_count": [1, 1, 1, 1, 1],
            "mean_rating": [4.0, 4.0, 4.0, 4.0, 4.0],
        }
    )
    items.write_parquet(processed / "items.parquet")

    pl.DataFrame({"user_idx": [0, 1], "userId": [10, 11]}).write_parquet(
        processed / "user_id_map.parquet"
    )
    pl.DataFrame({"item_idx": [0, 1, 2, 3, 4], "movieId": [100, 101, 102, 103, 104]}).write_parquet(
        processed / "item_id_map.parquet"
    )

    cfg_path = tmp_path / "retrieval.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  processed_data_dir: {processed.as_posix()}",
                "  model_output_dir: artifacts/models",
                "  index_output_dir: artifacts/faiss",
                "  report_output_dir: artifacts/reports",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_retrieval_dataset_history_and_shapes(tmp_path: Path) -> None:
    cfg_path = _write_small_tables(tmp_path)
    cfg = load_retrieval_config(cfg_path)
    tables = load_feature_tables(cfg)

    dataset = RetrievalDataset(str(cfg.train_path), tables, history_length=2)
    sample = dataset[2]
    assert len(sample["history_item_idx"]) <= 2
    assert sample["item_idx"] == 4  # item_idx 3 shifted by +1

    batch = collate_retrieval_batch([dataset[0], dataset[1], dataset[2]], history_length=2)
    assert batch["history_item_idx"].shape == (3, 2)
    assert batch["history_mask"].dtype == torch.bool
    assert batch["item_idx"].dtype == torch.long


def test_history_excludes_target_item(tmp_path: Path) -> None:
    cfg_path = _write_small_tables(tmp_path)
    cfg = load_retrieval_config(cfg_path)
    tables = load_feature_tables(cfg)
    dataset = RetrievalDataset(str(cfg.train_path), tables, history_length=10)

    row = dataset[2]
    target = row["item_idx"] - 1
    history_original = [v - 1 for v in row["history_item_idx"].tolist()]
    assert target not in history_original


def test_item_release_year_feature_is_included(tmp_path: Path) -> None:
    cfg_path = _write_small_tables(tmp_path)
    cfg = load_retrieval_config(cfg_path)
    tables = load_feature_tables(cfg)
    assert "release_year_norm" in tables.item_feature_columns


def test_feature_tables_are_standardized_and_finite(tmp_path: Path) -> None:
    cfg_path = _write_small_tables(tmp_path)
    cfg = load_retrieval_config(cfg_path)
    tables = load_feature_tables(cfg)

    assert torch.isfinite(torch.tensor(tables.user_features)).all()
    assert torch.isfinite(torch.tensor(tables.item_features)).all()
    assert abs(float(tables.user_features.mean())) < 1e-4
    assert abs(float(tables.item_features.mean())) < 1e-4
