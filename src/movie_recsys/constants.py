"""Project-wide constants for the MovieLens recommendation pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_MOVIELENS_FILES = (
	"ratings.csv",
	"movies.csv",
	"tags.csv",
	"genome-scores.csv",
	"genome-tags.csv",
	"links.csv",
)
