"""Offline evaluators for popularity and two-tower retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from movie_recsys.modeling.datasets import FeatureTables
from movie_recsys.modeling.faiss_index import build_flat_ip_index, search_index
from movie_recsys.modeling.metrics import aggregate_ranking_metrics
from movie_recsys.modeling.popularity import evaluate_popularity
from movie_recsys.modeling.retrieval import TwoTowerRetriever


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, float]
    predictions: dict[int, list[int]]


def _split_targets(split: pl.DataFrame) -> dict[int, set[int]]:
    targets: dict[int, set[int]] = {}
    for user_idx, item_idx in split.select(["user_idx", "item_idx"]).iter_rows():
        targets.setdefault(int(user_idx), set()).add(int(item_idx))
    return targets


def _train_seen(train: pl.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for user_idx, item_idx in train.select(["user_idx", "item_idx"]).iter_rows():
        seen.setdefault(int(user_idx), set()).add(int(item_idx))
    return seen


def evaluate_popularity_baseline(
    train: pl.DataFrame, split: pl.DataFrame, items: pl.DataFrame
) -> EvaluationResult:
    metrics, predictions = evaluate_popularity(train, split, items)
    return EvaluationResult(metrics=metrics, predictions=predictions)


def _build_user_tensor_inputs(
    user_idx: int,
    users_frame: pl.DataFrame,
    feature_tables: FeatureTables,
    *,
    history_length: int,
) -> dict[str, torch.Tensor]:
    row = users_frame.filter(pl.col("user_idx") == user_idx)
    history = row.get_column("train_history_item_idx").to_list()[0] or []
    history = [int(item) + 1 for item in history[-history_length:]]

    hist_tensor = torch.zeros((1, history_length), dtype=torch.long)
    mask_tensor = torch.zeros((1, history_length), dtype=torch.bool)
    if history:
        hist_tensor[0, -len(history) :] = torch.tensor(history, dtype=torch.long)
        mask_tensor[0, -len(history) :] = True

    return {
        "user_idx": torch.tensor([user_idx], dtype=torch.long),
        "history_item_idx": hist_tensor,
        "history_mask": mask_tensor,
        "user_features": torch.tensor(
            [feature_tables.user_features[user_idx]], dtype=torch.float32
        ),
    }


def evaluate_two_tower(
    model: TwoTowerRetriever,
    train: pl.DataFrame,
    split: pl.DataFrame,
    users: pl.DataFrame,
    feature_tables: FeatureTables,
    *,
    history_length: int,
    top_k: int = 200,
) -> tuple[EvaluationResult, np.ndarray, float]:
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        item_features = torch.tensor(
            feature_tables.item_features, dtype=torch.float32, device=device
        )
        item_idx = torch.arange(
            1, feature_tables.item_features.shape[0] + 1, device=device, dtype=torch.long
        )
        item_batch = {"item_idx": item_idx, "item_features": item_features}
        item_emb = model.encode_item(item_batch).cpu().numpy().astype(np.float32)

    index = build_flat_ip_index(item_emb)
    mapping = np.arange(feature_tables.item_features.shape[0], dtype=np.int64)

    targets = _split_targets(split)
    seen = _train_seen(train)

    predictions: dict[int, list[int]] = {}
    total_latency = 0.0
    for user_idx in sorted(targets.keys()):
        inputs = _build_user_tensor_inputs(
            user_idx,
            users,
            feature_tables,
            history_length=history_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            user_emb = model.user_tower(
                inputs["user_idx"],
                inputs["history_item_idx"],
                inputs["history_mask"],
                inputs["user_features"],
            )
        retrieved, _scores, latency = search_index(index, user_emb.cpu().numpy(), mapping, top_k)
        total_latency += latency
        seen_items = seen.get(user_idx, set())
        filtered = [int(item) for item in retrieved[0].tolist() if int(item) not in seen_items]
        predictions[user_idx] = filtered

    metrics = aggregate_ranking_metrics(predictions, targets)
    avg_latency_ms = total_latency / max(len(targets), 1)
    return EvaluationResult(metrics=metrics, predictions=predictions), item_emb, avg_latency_ms
