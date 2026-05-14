"""Datasets and collators for retrieval training/evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from movie_recsys.training.config import RetrievalConfig


@dataclass(slots=True)
class FeatureTables:
    user_features: np.ndarray
    item_features: np.ndarray
    user_feature_columns: list[str]
    item_feature_columns: list[str]


def _select_numeric_feature_columns(frame: pl.DataFrame, exclude: set[str]) -> list[str]:
    columns: list[str] = []
    for name, dtype in frame.schema.items():
        if name in exclude:
            continue
        if dtype.is_numeric():
            columns.append(name)
    return columns


def load_feature_tables(config: RetrievalConfig) -> FeatureTables:
    users = pl.read_parquet(config.users_path)
    items = pl.read_parquet(config.items_path)

    # Preserve year signal for item MLP while keeping magnitude stable.
    if "release_year" in items.columns:
        items = items.with_columns(
            ((pl.col("release_year") - 1900) / 200.0)
            .cast(pl.Float32)
            .alias("release_year_norm")
        )

    user_feature_cols = _select_numeric_feature_columns(
        users,
        exclude={"user_idx", "original_userId", "first_timestamp", "last_timestamp"},
    )
    item_feature_cols = _select_numeric_feature_columns(
        items,
        exclude={"item_idx", "original_movieId"},
    )

    users_sorted = users.sort("user_idx")
    items_sorted = items.sort("item_idx")

    user_features = users_sorted.select(user_feature_cols).to_numpy().astype(np.float32, copy=False)
    item_features = items_sorted.select(item_feature_cols).to_numpy().astype(np.float32, copy=False)

    def _standardize(values: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean = clean.mean(axis=0, keepdims=True)
        std = clean.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        standardized = (clean - mean) / std
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
        return cast(np.ndarray, standardized)

    user_features = _standardize(user_features).astype(np.float32, copy=False)
    item_features = _standardize(item_features).astype(np.float32, copy=False)

    return FeatureTables(
        user_features=user_features,
        item_features=item_features,
        user_feature_columns=user_feature_cols,
        item_feature_columns=item_feature_cols,
    )


class RetrievalDataset(Dataset[dict[str, Any]]):
    """One example per interaction with strict history-before-target behavior.

    Item indices are shifted by +1 in returned tensors so 0 can be reserved for padding.
    """

    def __init__(
        self,
        interactions_path: str,
        feature_tables: FeatureTables,
        *,
        history_length: int,
    ) -> None:
        super().__init__()
        self.history_length = history_length

        frame = pl.read_parquet(interactions_path).sort(["user_idx", "timestamp", "item_idx"])
        self.user_idx = frame.get_column("user_idx").to_numpy().astype(np.int64, copy=False)
        self.item_idx = frame.get_column("item_idx").to_numpy().astype(np.int64, copy=False)
        self.timestamp = frame.get_column("timestamp").to_numpy().astype(np.int64, copy=False)
        self.explicit_rating = (
            frame.get_column("explicit_rating").to_numpy().astype(np.float32, copy=False)
        )

        self.user_features = feature_tables.user_features
        self.item_features = feature_tables.item_features

        self._user_start: dict[int, int] = {}
        self._pos_in_user = np.zeros(len(self.user_idx), dtype=np.int32)
        prev_user = -1
        current_pos = 0
        for i, user in enumerate(self.user_idx.tolist()):
            if user != prev_user:
                self._user_start[user] = i
                current_pos = 0
                prev_user = user
            self._pos_in_user[i] = current_pos
            current_pos += 1

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, index: int) -> dict[str, Any]:
        user = int(self.user_idx[index])
        item = int(self.item_idx[index])
        start = self._user_start[user]
        history_slice = self.item_idx[start:index]
        if history_slice.size > self.history_length:
            history_slice = history_slice[-self.history_length :]

        history_shifted = (history_slice + 1).astype(np.int64, copy=False)
        target_item_shifted = item + 1

        return {
            "user_idx": user,
            "item_idx": target_item_shifted,
            "history_item_idx": history_shifted,
            "user_features": self.user_features[user],
            "item_features": self.item_features[item],
            "timestamp": int(self.timestamp[index]),
            "explicit_rating": float(self.explicit_rating[index]),
        }


def collate_retrieval_batch(
    batch: list[dict[str, Any]], history_length: int
) -> dict[str, torch.Tensor]:
    batch_size = len(batch)

    user_idx = torch.tensor([row["user_idx"] for row in batch], dtype=torch.long)
    item_idx = torch.tensor([row["item_idx"] for row in batch], dtype=torch.long)

    history = torch.zeros((batch_size, history_length), dtype=torch.long)
    history_mask = torch.zeros((batch_size, history_length), dtype=torch.bool)

    for i, row in enumerate(batch):
        values = row["history_item_idx"]
        if len(values) == 0:
            continue
        take = values[-history_length:]
        seq = torch.tensor(take, dtype=torch.long)
        history[i, -len(take) :] = seq
        history_mask[i, -len(take) :] = True

    user_features = torch.tensor(
        np.stack([row["user_features"] for row in batch]), dtype=torch.float32
    )
    item_features = torch.tensor(
        np.stack([row["item_features"] for row in batch]), dtype=torch.float32
    )

    return {
        "user_idx": user_idx,
        "item_idx": item_idx,
        "history_item_idx": history,
        "history_mask": history_mask,
        "user_features": user_features,
        "item_features": item_features,
    }


def make_retrieval_dataloader(
    dataset: RetrievalDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda rows: collate_retrieval_batch(
            rows, history_length=dataset.history_length
        ),
        generator=generator,
        drop_last=False,
    )
