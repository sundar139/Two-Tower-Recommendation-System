"""Serving-time hybrid scoring policy utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PolicySpec:
    """Weighting policy for final score composition."""

    alpha: float
    beta: float
    gamma: float
    top_k_focus: int


def combine_scores(
    *,
    ranker_scores: np.ndarray,
    residual_scores: np.ndarray,
    popularity_scores: np.ndarray,
    policy: PolicySpec,
) -> np.ndarray:
    """Compose final scores from ranker, retrieval, and popularity channels."""

    return (
        policy.alpha * ranker_scores
        + policy.beta * residual_scores
        + policy.gamma * popularity_scores
    )
