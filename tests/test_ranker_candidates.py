"""Tests for ranker candidate generation."""

from __future__ import annotations

import polars as pl
import pytest

from movie_recsys.ranking.candidates import validate_candidate_frame


def _valid_candidate_frame() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"query_id": ["train_u00000001", "train_u00000001", "train_u00000001"],
			"user_idx": [1, 1, 1],
			"item_idx": [10, 20, 30],
			"split": ["train", "train", "train"],
			"label": [0, 1, 0],
			"target_item_idx": [20, 20, 20],
			"residual_score": [0.9, 0.8, 0.7],
			"residual_rank": [1, 2, 3],
			"target_injected": [False, False, False],
			"user_history_length": [5, 5, 5],
			"timestamp_context": [1000, 1000, 1000],
			"candidate_source": ["retrieved", "retrieved", "retrieved"],
		}
	)


def test_candidate_validation_accepts_target_inclusion() -> None:
	frame = _valid_candidate_frame()
	checks = validate_candidate_frame(frame, split="train", leaked_targets=set())
	assert checks["one_positive_per_query"]
	assert checks["no_duplicate_candidate_item_per_query"]


def test_candidate_validation_rejects_duplicate_candidate_item() -> None:
	frame = _valid_candidate_frame().with_columns(pl.Series("item_idx", [10, 20, 20]))
	with pytest.raises(ValueError, match="Candidate validation failed"):
		validate_candidate_frame(frame, split="train", leaked_targets=set())


def test_candidate_validation_detects_leakage() -> None:
	frame = _valid_candidate_frame()
	with pytest.raises(ValueError, match="Candidate validation failed"):
		validate_candidate_frame(frame, split="train", leaked_targets={20})
