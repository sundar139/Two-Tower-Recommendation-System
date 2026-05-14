"""Popularity baseline for retrieval."""

from __future__ import annotations

import polars as pl

from movie_recsys.modeling.metrics import aggregate_ranking_metrics


def popularity_ranking(items: pl.DataFrame) -> list[int]:
    score_col = "positive_count" if "positive_count" in items.columns else "popularity_score"
    ranked = items.sort(score_col, descending=True).get_column("item_idx").to_list()
    return [int(v) for v in ranked]


def evaluate_popularity(
    train: pl.DataFrame,
    split: pl.DataFrame,
    items: pl.DataFrame,
) -> tuple[dict[str, float], dict[int, list[int]]]:
    ranked_all = popularity_ranking(items)

    seen_by_user: dict[int, set[int]] = {}
    for user_idx, item_idx in train.select(["user_idx", "item_idx"]).iter_rows():
        seen_by_user.setdefault(int(user_idx), set()).add(int(item_idx))

    targets: dict[int, set[int]] = {}
    for user_idx, item_idx in split.select(["user_idx", "item_idx"]).iter_rows():
        targets.setdefault(int(user_idx), set()).add(int(item_idx))

    predictions: dict[int, list[int]] = {}
    for user_idx in targets:
        seen = seen_by_user.get(user_idx, set())
        gt_items = targets.get(user_idx, set())
        filtered = [item for item in ranked_all if item not in seen or item in gt_items][:200]
        predictions[user_idx] = filtered

    metrics = aggregate_ranking_metrics(predictions, targets)
    return metrics, predictions
