# Data Pipeline

## Data Source

- Dataset: MovieLens-25M
- Source: official GroupLens distribution (`https://files.grouplens.org/datasets/movielens/ml-25m.zip`)

## Preprocessing Logic

1. Load `ratings.csv`, `movies.csv`, `tags.csv`, `genome-scores.csv`, `genome-tags.csv`, and `links.csv` with Polars.
2. Build implicit-positive interactions using:
	- positive if `rating >= positive_rating_threshold` (default `4.0`)
3. Keep `explicit_rating` column for later ranker compatibility.
4. Filter users to those with at least `min_positive_interactions_per_user` positives (default `3`).
5. Optionally select deterministic sample users for smoke runs.

## Chronological Split Strategy

Per user (after filtering):

- last positive interaction -> test
- second-to-last positive interaction -> validation
- all earlier positives -> train

This enforces strict per-user chronology and prevents future interactions from entering train histories.

## Generated Feature Tables

- `interactions_train.parquet`
- `interactions_val.parquet`
- `interactions_test.parquet`
- `users.parquet`
- `items.parquet`
- `user_id_map.parquet`
- `item_id_map.parquet`
- `user_histories.parquet`
- `dataset_stats.json`

User features include activity, rating statistics, genre affinities, tag counts, and `train_history_item_idx`.

Item features include metadata, release year bucketing, genre multi-hot columns, rating/positive statistics, normalized popularity, and compact genome summaries (top tags + relevance stats).

## Leakage Prevention Checks

Validation logic checks:

- required columns and null constraints
- contiguous `user_idx` and `item_idx`
- strict chronology (`train < val < test`) per user
- no duplicate exact interactions across splits
- no validation/test target leakage into `train_history_item_idx`
- genre multi-hot consistency with `genres_list`

## Sanity Checks

Use `dataset_stats.json` to validate:

- raw rating counts and positive counts
- user/movie cardinalities before vs after filtering
- train/val/test row counts
- min/median/max user positive interactions
- timestamp range
- genre vocabulary
- missing-value report
