"""Tests for serving policy scoring and deterministic ordering."""

from __future__ import annotations

import numpy as np
import pytest

from movie_recsys.serving.scorer import PolicySpec, rank_with_policy


def test_rank_with_two_stage_backfill_order() -> None:
    ranker_scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
    residual_scores = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    popularity_scores = np.asarray([0.1, 0.2, 0.9, 0.8], dtype=np.float64)
    residual_rank = np.asarray([1, 2, 3, 4], dtype=np.int64)
    item_idx = np.asarray([10, 20, 30, 40], dtype=np.int64)
    policy = PolicySpec(
        policy_name="ranker_topk_popularity_backfill",
        alpha=1.0,
        beta=0.1,
        gamma=0.0,
        top_k_focus=2,
    )

    order, display_scores = rank_with_policy(
        ranker_scores=ranker_scores,
        residual_scores=residual_scores,
        popularity_scores=popularity_scores,
        residual_rank=residual_rank,
        item_idx=item_idx,
        policy=policy,
    )

    assert order.tolist() == [0, 1, 2, 3]
    assert display_scores.shape == ranker_scores.shape
    assert display_scores[2] >= display_scores[3]


def test_rank_with_policy_rejects_mismatched_lengths() -> None:
    policy = PolicySpec(
        policy_name="ranker_only",
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        top_k_focus=20,
    )
    with pytest.raises(ValueError):
        rank_with_policy(
            ranker_scores=np.asarray([0.1, 0.2], dtype=np.float64),
            residual_scores=np.asarray([0.1], dtype=np.float64),
            popularity_scores=np.asarray([0.1, 0.2], dtype=np.float64),
            residual_rank=np.asarray([1, 2], dtype=np.int64),
            item_idx=np.asarray([1, 2], dtype=np.int64),
            policy=policy,
        )
