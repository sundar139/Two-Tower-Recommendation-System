"""Schema constants and column contracts for dataset processing."""

from __future__ import annotations

MANDATORY_INTERACTION_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
MANDATORY_MOVIE_COLUMNS = ["movieId", "title", "genres"]
MANDATORY_TAG_COLUMNS = ["userId", "movieId", "tag", "timestamp"]
MANDATORY_GENOME_SCORES_COLUMNS = ["movieId", "tagId", "relevance"]
MANDATORY_GENOME_TAGS_COLUMNS = ["tagId", "tag"]

INTERACTION_ID_COLUMNS = ["userId", "movieId", "user_idx", "item_idx"]
