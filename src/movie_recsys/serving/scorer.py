"""Serving-time hybrid scoring policy utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from movie_recsys.ranking.hybrid import normalize_query_scores


@dataclass(slots=True)
class PolicySpec:
    """Weighting policy for final score composition."""

    policy_name: str
    alpha: float
    beta: float
    gamma: float
    top_k_focus: int


def _validate_lengths(
    *,
    ranker_scores: np.ndarray,
    residual_scores: np.ndarray,
    popularity_scores: np.ndarray,
    residual_rank: np.ndarray,
    item_idx: np.ndarray,
) -> None:
    size = ranker_scores.shape[0]
    arrays = {
        "residual_scores": residual_scores,
        "popularity_scores": popularity_scores,
        "residual_rank": residual_rank,
        "item_idx": item_idx,
    }
    mismatched = [name for name, values in arrays.items() if values.shape[0] != size]
    if mismatched:
        msg = "Scorer inputs have mismatched lengths: " + ", ".join(mismatched)
        raise ValueError(msg)


def _hybrid_stage_one_score(
    *,
    ranker_scores: np.ndarray,
    residual_scores: np.ndarray,
    popularity_scores: np.ndarray,
    policy: PolicySpec,
) -> np.ndarray:
    ranker_norm = normalize_query_scores(ranker_scores.astype(np.float64), method="query_minmax")
    residual_norm = normalize_query_scores(
        residual_scores.astype(np.float64),
        method="query_minmax",
    )
    popularity_norm = normalize_query_scores(
        popularity_scores.astype(np.float64),
        method="query_minmax",
    )
    return (
        policy.alpha * ranker_norm
        + policy.beta * popularity_norm
        + policy.gamma * residual_norm
    )


def rank_with_policy(
    *,
    ranker_scores: np.ndarray,
    residual_scores: np.ndarray,
    popularity_scores: np.ndarray,
    residual_rank: np.ndarray,
    item_idx: np.ndarray,
    policy: PolicySpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic candidate order and display scores for one query."""

    _validate_lengths(
        ranker_scores=ranker_scores,
        residual_scores=residual_scores,
        popularity_scores=popularity_scores,
        residual_rank=residual_rank,
        item_idx=item_idx,
    )

    if ranker_scores.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    stage_one_scores = _hybrid_stage_one_score(
        ranker_scores=ranker_scores,
        residual_scores=residual_scores,
        popularity_scores=popularity_scores,
        policy=policy,
    )

    if policy.policy_name == "ranker_topk_popularity_backfill":
        stage_one_order = np.lexsort((item_idx, residual_rank, -stage_one_scores))
        top_k_focus = int(max(policy.top_k_focus, 0))
        top_k_focus = min(top_k_focus, int(stage_one_order.shape[0]))
        focused_indices = stage_one_order[:top_k_focus]

        display_scores = stage_one_scores.copy()
        if top_k_focus < int(stage_one_order.shape[0]):
            remaining_mask = np.ones(int(stage_one_order.shape[0]), dtype=bool)
            if focused_indices.size > 0:
                remaining_mask[focused_indices] = False
            remaining_indices = np.flatnonzero(remaining_mask)
            remaining_sorted = remaining_indices[
                np.lexsort(
                    (
                        item_idx[remaining_indices],
                        residual_rank[remaining_indices],
                        -popularity_scores[remaining_indices],
                    )
                )
            ]
            display_scores[remaining_sorted] = popularity_scores[remaining_sorted]
            final_order = np.concatenate((focused_indices, remaining_sorted))
            return final_order.astype(np.int64), display_scores

        return focused_indices.astype(np.int64), display_scores

    if policy.policy_name == "ranker_only":
        linear_scores = normalize_query_scores(
            ranker_scores.astype(np.float64),
            method="query_minmax",
        )
    elif policy.policy_name == "popularity_only":
        linear_scores = normalize_query_scores(
            popularity_scores.astype(np.float64),
            method="query_minmax",
        )
    elif policy.policy_name == "residual_only":
        linear_scores = normalize_query_scores(
            residual_scores.astype(np.float64),
            method="query_minmax",
        )
    else:
        linear_scores = stage_one_scores

    final_order = np.lexsort((item_idx, residual_rank, -linear_scores))
    return final_order.astype(np.int64), linear_scores
