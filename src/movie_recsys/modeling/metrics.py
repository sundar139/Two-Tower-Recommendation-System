"""Retrieval metrics for offline evaluation."""

from __future__ import annotations

import math


def _ensure_targets(targets: list[int] | set[int] | tuple[int, ...]) -> set[int]:
    if isinstance(targets, set):
        return targets
    return set(targets)


def hit_rate_at_k(ranked_items: list[int], targets: list[int] | set[int], k: int) -> float:
    top_k = ranked_items[:k]
    target_set = _ensure_targets(targets)
    return 1.0 if any(item in target_set for item in top_k) else 0.0


def mrr_at_k(ranked_items: list[int], targets: list[int] | set[int], k: int) -> float:
    target_set = _ensure_targets(targets)
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in target_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_items: list[int], targets: list[int] | set[int], k: int) -> float:
    target_set = _ensure_targets(targets)
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in target_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(target_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def recall_at_k(ranked_items: list[int], targets: list[int] | set[int], k: int) -> float:
    target_set = _ensure_targets(targets)
    if not target_set:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in target_set)
    return hits / len(target_set)


def aggregate_ranking_metrics(
    predictions: dict[int, list[int]],
    targets: dict[int, set[int]],
) -> dict[str, float]:
    if not predictions:
        return {"hr@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}

    hr = []
    mrr = []
    ndcg = []
    recall = []
    for user_idx, ranked in predictions.items():
        user_targets = targets.get(user_idx, set())
        hr.append(hit_rate_at_k(ranked, user_targets, 10))
        mrr.append(mrr_at_k(ranked, user_targets, 10))
        ndcg.append(ndcg_at_k(ranked, user_targets, 10))
        recall.append(recall_at_k(ranked, user_targets, 50))

    n = float(len(hr))
    return {
        "hr@10": float(sum(hr) / n),
        "mrr@10": float(sum(mrr) / n),
        "ndcg@10": float(sum(ndcg) / n),
        "recall@50": float(sum(recall) / n),
    }
