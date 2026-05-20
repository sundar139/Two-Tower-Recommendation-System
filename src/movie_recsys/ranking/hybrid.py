"""Hybrid scorer utilities for production scorer policy selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from movie_recsys.ranking.dataset import iter_ranker_query_groups

NormalizationMethod = Literal["query_minmax", "query_zscore"]

METADATA_FORBIDDEN_SCORE_COLUMNS = {
    "label",
    "target_item_idx",
    "target_injected",
    "candidate_source",
    "query_id",
    "split",
}

REQUIRED_SCORER_COLUMNS = [
    "query_id",
    "item_idx",
    "target_item_idx",
    "label",
    "residual_rank",
    "ranker_score",
    "residual_score",
    "popularity_score",
]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Single scoring policy in the production selection grid."""

    policy_name: str
    ranker_weight: float
    popularity_weight: float
    residual_weight: float
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None

    @property
    def policy_id(self) -> str:
        if self.alpha is None and self.beta is None and self.gamma is None:
            return self.policy_name

        parts: list[str] = [self.policy_name]
        if self.alpha is not None:
            parts.append(f"a={_format_weight(self.alpha)}")
        if self.beta is not None:
            parts.append(f"b={_format_weight(self.beta)}")
        if self.gamma is not None:
            parts.append(f"g={_format_weight(self.gamma)}")
        return "|".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_id": self.policy_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "ranker_weight": float(self.ranker_weight),
            "popularity_weight": float(self.popularity_weight),
            "residual_weight": float(self.residual_weight),
        }


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Metrics for one policy evaluated on one split."""

    policy: PolicySpec
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.policy.as_dict(),
            "metrics": self.metrics,
        }


def _format_weight(value: float) -> str:
    rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def build_policy_grid(
    *,
    alpha_values: Sequence[float],
    beta_values: Sequence[float],
    gamma_values: Sequence[float],
) -> list[PolicySpec]:
    """Build manual scorer-policy grid (no automated search)."""

    grid: list[PolicySpec] = [
        PolicySpec(
            policy_name="popularity_only",
            ranker_weight=0.0,
            popularity_weight=1.0,
            residual_weight=0.0,
        ),
        PolicySpec(
            policy_name="residual_only",
            ranker_weight=0.0,
            popularity_weight=0.0,
            residual_weight=1.0,
        ),
        PolicySpec(
            policy_name="ranker_only",
            ranker_weight=1.0,
            popularity_weight=0.0,
            residual_weight=0.0,
        ),
    ]

    for alpha in alpha_values:
        for beta in beta_values:
            grid.append(
                PolicySpec(
                    policy_name="ranker_plus_popularity",
                    ranker_weight=float(alpha),
                    popularity_weight=float(beta),
                    residual_weight=0.0,
                    alpha=float(alpha),
                    beta=float(beta),
                    gamma=0.0,
                )
            )

    for alpha in alpha_values:
        for beta in beta_values:
            grid.append(
                PolicySpec(
                    policy_name="ranker_plus_residual",
                    ranker_weight=float(alpha),
                    popularity_weight=0.0,
                    residual_weight=float(beta),
                    alpha=float(alpha),
                    beta=float(beta),
                    gamma=0.0,
                )
            )

    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                grid.append(
                    PolicySpec(
                        policy_name="ranker_plus_popularity_plus_residual",
                        ranker_weight=float(alpha),
                        popularity_weight=float(beta),
                        residual_weight=float(gamma),
                        alpha=float(alpha),
                        beta=float(beta),
                        gamma=float(gamma),
                    )
                )

    return grid


def metadata_feature_leakage_audit(
    *,
    used_score_columns: Sequence[str],
    available_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Confirm metadata fields are not used as scoring features."""

    used = set(used_score_columns)
    forbidden = set(METADATA_FORBIDDEN_SCORE_COLUMNS)
    disallowed = sorted(used & forbidden)

    available = set(available_columns or [])
    forbidden_present = sorted(available & forbidden)

    return {
        "metadata_feature_leakage_passed": len(disallowed) == 0,
        "used_score_columns": sorted(used),
        "forbidden_columns": sorted(forbidden),
        "forbidden_columns_present_in_dataset": forbidden_present,
        "disallowed_columns_used": disallowed,
    }


def score_columns_are_finite(
    scored_candidates_path: Path,
    *,
    score_columns: Sequence[str] = ("ranker_score", "residual_score", "popularity_score"),
) -> bool:
    """Return whether every score value is finite for all requested columns."""

    schema_names = set(pl.scan_parquet(scored_candidates_path).collect_schema().names())
    missing = [name for name in score_columns if name not in schema_names]
    if missing:
        return False

    checks = (
        pl.scan_parquet(scored_candidates_path)
        .select([pl.col(name).is_finite().all().alias(name) for name in score_columns])
        .collect()
        .row(0)
    )
    return all(bool(flag) for flag in checks)


def query_row_counts(scored_candidates_path: Path) -> tuple[int, int]:
    """Return (query_count, row_count) for scored candidate parquet."""

    payload = (
        pl.scan_parquet(scored_candidates_path)
        .select(
            [
                pl.col("query_id").n_unique().alias("query_count"),
                pl.len().alias("row_count"),
            ]
        )
        .collect()
        .row(0)
    )
    return (int(payload[0]), int(payload[1]))


def normalize_query_scores(
    values: npt.NDArray[np.float64],
    *,
    method: NormalizationMethod,
) -> npt.NDArray[np.float64]:
    """Apply query-wise normalization for one score vector."""

    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        msg = "Detected NaN/Inf scores before normalization"
        raise ValueError(msg)

    if method == "query_minmax":
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        denom = maximum - minimum
        if denom <= 1e-12:
            return np.zeros_like(values)
        return (values - minimum) / denom

    if method == "query_zscore":
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        if std_value <= 1e-12:
            return np.zeros_like(values)
        return (values - mean_value) / std_value

    msg = f"Unsupported normalization method: {method}"
    raise ValueError(msg)


def rank_positive_item(
    *,
    scores: npt.NDArray[np.float64],
    residual_rank: npt.NDArray[np.int64],
    item_idx: npt.NDArray[np.int64],
    positive_index: int,
) -> int:
    """Return 1-based rank of the positive item using deterministic tie-breaks."""

    positive_score = float(scores[positive_index])
    positive_residual_rank = int(residual_rank[positive_index])
    positive_item_idx = int(item_idx[positive_index])

    higher = scores > positive_score
    ties = scores == positive_score
    tie_break = (residual_rank < positive_residual_rank) | (
        (residual_rank == positive_residual_rank) & (item_idx < positive_item_idx)
    )
    better = higher | (ties & tie_break)
    return int(np.count_nonzero(better) + 1)


def select_best_policy_result(results: Sequence[PolicyEvaluation]) -> PolicyEvaluation:
    """Select best scorer using validation metrics only."""

    if not results:
        msg = "Cannot select best policy from empty evaluation results"
        raise ValueError(msg)

    return max(
        results,
        key=lambda result: (
            result.metrics["ndcg@10"],
            result.metrics["mrr@10"],
            result.metrics["hr@10"],
            result.metrics["recall@50"],
            result.policy.policy_id,
        ),
    )


def evaluate_policy_grid(
    *,
    scored_candidates_path: Path,
    policies: Sequence[PolicySpec],
    normalization_method: NormalizationMethod,
    batch_size_rows: int = 8192,
    max_queries: int | None = None,
) -> tuple[list[PolicyEvaluation], dict[str, int]]:
    """Evaluate all policies for one split using streamed query groups."""

    if not scored_candidates_path.exists():
        msg = f"Scored candidates path not found: {scored_candidates_path}"
        raise FileNotFoundError(msg)
    if not policies:
        msg = "At least one scoring policy is required"
        raise ValueError(msg)

    schema_names = set(pl.scan_parquet(scored_candidates_path).collect_schema().names())
    missing = [name for name in REQUIRED_SCORER_COLUMNS if name not in schema_names]
    if missing:
        msg = f"Missing required columns in scored candidates: {', '.join(sorted(missing))}"
        raise ValueError(msg)

    weights = np.asarray(
        [
            [policy.ranker_weight, policy.popularity_weight, policy.residual_weight]
            for policy in policies
        ],
        dtype=np.float64,
    )
    n_policies = int(weights.shape[0])

    hr_sum = np.zeros(n_policies, dtype=np.float64)
    mrr_sum = np.zeros(n_policies, dtype=np.float64)
    ndcg_sum = np.zeros(n_policies, dtype=np.float64)
    recall_sum = np.zeros(n_policies, dtype=np.float64)

    query_count = 0
    row_count = 0

    iterator = iter_ranker_query_groups(
        [str(scored_candidates_path)],
        columns=REQUIRED_SCORER_COLUMNS,
        batch_size_rows=batch_size_rows,
        max_queries=max_queries,
    )

    for group in iterator:
        if group.height == 0:
            continue

        query_count += 1
        row_count += int(group.height)

        ranker_raw = np.asarray(group.get_column("ranker_score").to_numpy(), dtype=np.float64)
        popularity_raw = np.asarray(
            group.get_column("popularity_score").to_numpy(),
            dtype=np.float64,
        )
        residual_raw = np.asarray(group.get_column("residual_score").to_numpy(), dtype=np.float64)

        ranker_norm = normalize_query_scores(ranker_raw, method=normalization_method)
        popularity_norm = normalize_query_scores(popularity_raw, method=normalization_method)
        residual_norm = normalize_query_scores(residual_raw, method=normalization_method)

        base_scores = np.column_stack((ranker_norm, popularity_norm, residual_norm))
        combined_scores = np.matmul(base_scores, weights.T)

        labels = np.asarray(group.get_column("label").to_numpy(), dtype=np.int64)
        item_idx = np.asarray(group.get_column("item_idx").to_numpy(), dtype=np.int64)
        target_item_idx = np.asarray(
            group.get_column("target_item_idx").to_numpy(),
            dtype=np.int64,
        )
        residual_rank = np.asarray(group.get_column("residual_rank").to_numpy(), dtype=np.int64)

        positive_locations = np.flatnonzero(labels == 1)
        if positive_locations.size > 0:
            positive_index = int(positive_locations[0])
        else:
            target_matches = np.flatnonzero(item_idx == int(target_item_idx[0]))
            if target_matches.size == 0:
                msg = "Could not locate positive target item in query group"
                raise ValueError(msg)
            positive_index = int(target_matches[0])

        positive_scores = combined_scores[positive_index, :]
        positive_residual_rank = int(residual_rank[positive_index])
        positive_item_idx = int(item_idx[positive_index])

        higher = combined_scores > positive_scores
        ties = combined_scores == positive_scores
        tie_break = (residual_rank[:, None] < positive_residual_rank) | (
            (residual_rank[:, None] == positive_residual_rank)
            & (item_idx[:, None] < positive_item_idx)
        )
        better = higher | (ties & tie_break)
        ranks = np.sum(better, axis=0) + 1

        hit_at_10 = ranks <= 10
        hit_at_50 = ranks <= 50
        hr_sum += hit_at_10.astype(np.float64)
        mrr_sum += np.where(hit_at_10, 1.0 / ranks, 0.0)
        ndcg_sum += np.where(hit_at_10, 1.0 / np.log2(ranks + 1.0), 0.0)
        recall_sum += hit_at_50.astype(np.float64)

    if query_count <= 0:
        msg = "No query groups were evaluated"
        raise ValueError(msg)

    denominator = float(query_count)
    evaluations: list[PolicyEvaluation] = []
    for index, policy in enumerate(policies):
        metrics = {
            "hr@10": float(hr_sum[index] / denominator),
            "mrr@10": float(mrr_sum[index] / denominator),
            "ndcg@10": float(ndcg_sum[index] / denominator),
            "recall@50": float(recall_sum[index] / denominator),
        }
        evaluations.append(PolicyEvaluation(policy=policy, metrics=metrics))

    return evaluations, {"query_count": int(query_count), "row_count": int(row_count)}


def metrics_are_finite(metrics: dict[str, float]) -> bool:
    """Return whether all metric values are finite."""

    return all(np.isfinite(value) for value in metrics.values())
