# Neural Ranker Workflow

## Overview

The neural ranker is a second-stage re-ranker that consumes top-200 candidates from the approved residual transformer retriever and reorders them for top-10 recommendation quality.

Backbone policy:

- retrieval stage stays `residual_transformer`
- contrastive residual remains experimental
- ranker promotion uses explicit acceptance gates

## Pipeline

1. Generate residual top-k candidate sets for `train`, `val`, and `test`.
2. Build ranker feature matrices from candidates + user/item tables.
3. Train MLP ranker and save best/last checkpoints.
4. Evaluate ranker vs residual vs popularity on val/test splits.
5. Run acceptance checks before full-data promotion.

## Core Components

- Config and paths: `src/movie_recsys/ranking/config.py`, `configs/ranker.yaml`
- Candidate generation: `src/movie_recsys/ranking/candidates.py`
- Feature engineering: `src/movie_recsys/ranking/features.py`
- Dataset and loaders: `src/movie_recsys/ranking/dataset.py`
- Model and losses: `src/movie_recsys/ranking/model.py`, `src/movie_recsys/ranking/losses.py`
- Training loop: `src/movie_recsys/ranking/trainer.py`
- Evaluation/comparison: `src/movie_recsys/ranking/evaluator.py`
- Acceptance logic: `src/movie_recsys/ranking/acceptance.py`

Scripts:

- `scripts/generate_ranker_candidates.py`
- `scripts/train_ranker.py`
- `scripts/evaluate_ranker.py`
- `scripts/compare_retrieval_ranker.py`
- `scripts/check_ranker_acceptance.py`

## Sample Workflow

```powershell
uv run python scripts/generate_ranker_candidates.py --sample
uv run python scripts/train_ranker.py --sample
uv run python scripts/evaluate_ranker.py --sample --split val
uv run python scripts/evaluate_ranker.py --sample --split test
uv run python scripts/compare_retrieval_ranker.py --sample
uv run python scripts/check_ranker_acceptance.py --sample
```

## Sample Results (Latest)

Validation:

- ranker NDCG@10: `0.566292`
- residual NDCG@10: `0.014706`
- delta: `+0.551586`

Test:

- ranker NDCG@10: `0.575695`
- residual NDCG@10: `0.012702`
- delta: `+0.562994`

Acceptance (`artifacts/reports/ranker_acceptance_sample.json`):

- `acceptance_passed: true`
- `full_data_ranker_allowed: true`
- all guard checks passed

## Full Workflow

Run only after sample acceptance passes:

```powershell
uv run python scripts/generate_ranker_candidates.py
uv run python scripts/train_ranker.py
uv run python scripts/evaluate_ranker.py --split val
uv run python scripts/evaluate_ranker.py --split test
uv run python scripts/compare_retrieval_ranker.py
uv run python scripts/check_ranker_acceptance.py
```

## Full-Data Results (Latest)

Step 5B: Full-Data Neural Ranker Validation completed.

The full-data neural ranker pipeline was validated using residual transformer retrieval candidates. Candidate diagnostics passed integrity checks, and full validation/test ranker evaluation completed successfully.

Full validation results:

- Ranker NDCG@10: `0.181151`
- Residual retriever NDCG@10: `0.045040`
- Delta: `+0.136110`
- Query count: `161,821`
- Row count: `32,452,397`

Full test results:

- Ranker NDCG@10: `0.187112`
- Residual retriever NDCG@10: `0.032480`
- Delta: `+0.154633`
- Query count: `161,821`
- Row count: `32,461,176`

Acceptance passed against the residual retrieval baseline. Popularity still outperformed ranker-only NDCG@10 on both full splits, so a popularity-aware scorer audit was required before serving/API work.

## Step 5C: Production Scorer Selection (Full Data)

Objective:

- determine production scorer policy across `popularity_only`, `residual_only`, `ranker_only`, and ranker hybrids
- use validation-only weight selection (no test leakage)

Scorer selection setup:

- normalization: query-wise min-max (`query_minmax`)
- candidate metadata leakage audit: passed
- weight grid (manual):
	- `alpha = [0.5, 0.7, 0.85, 1.0]`
	- `beta = [0.0, 0.1, 0.2, 0.3, 0.5]`
	- `gamma = [0.0, 0.1, 0.2, 0.3]`
- validation table rows evaluated: `123`

Selected scorer (validation winner):

- policy: `ranker_plus_popularity`
- weights: `alpha=1.0`, `beta=0.1`, `gamma=0.0`
- validation metrics: `hr@10=0.435364`, `mrr@10=0.273246`, `ndcg@10=0.311523`, `recall@50=0.597203`
- test metrics: `hr@10=0.447748`, `mrr@10=0.277436`, `ndcg@10=0.317591`, `recall@50=0.637241`

Acceptance against popularity baseline:

- primary rules: passed
- mandatory guards: failed on `recall50_relative_drop_vs_popularity_le_5pct`
- recall@50 relative drop vs popularity: val `0.160572`, test `0.062205`
- `acceptance_passed: false`
- `step6_fastapi_unblocked: false`

Reports:

- `artifacts/reports/production_scorer_selection.json`
- `artifacts/reports/production_scorer_selection.md`
- `artifacts/reports/production_scorer_acceptance.json`

## Step 5D: Recall-Constrained Production Scorer Tuning

Why Step 5D was needed:

- Step 5C selected the highest validation NDCG scorer, but it failed acceptance due to Recall@50 drop vs popularity.

Step 5D selection policy:

- validation-only selection (`selected_by_validation_only=true`)
- strict recall constraint: `Recall@50 >= 0.95 * popularity Recall@50`
- recall constraint value on val: `0.675868`
- metadata leakage audit: passed

Expanded grid:

- `ranker_plus_popularity`
	- `alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]`
	- `beta=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]`
- `ranker_plus_popularity_plus_residual`
	- `alpha=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`
	- `beta=[0.2, 0.5, 1.0, 1.5, 2.0]`
	- `gamma=[0.0, 0.05, 0.1, 0.2]`
- `ranker_topk_popularity_backfill`
	- `top_k_focus=[10, 20, 30, 50]`

Selection summary:

- candidates passing recall constraint: `64`
- popularity safe fallback used: `false`
- selected scorer: `ranker_topk_popularity_backfill`
- selected weights: `alpha=1.0`, `beta=0.1`, `gamma=0.0`, `top_k_focus=20`

Selected scorer metrics:

- validation: `hr@10=0.435364`, `mrr@10=0.273246`, `ndcg@10=0.311523`, `recall@50=0.729658`
- test: `hr@10=0.447748`, `mrr@10=0.277436`, `ndcg@10=0.317591`, `recall@50=0.712374`

Acceptance outcome:

- `acceptance_passed: true`
- `step6_fastapi_unblocked: true`
- `step6_unblocked_mode: selected_scorer`
- recall guard passed (`recall50_relative_drop_vs_popularity_le_5pct=true`)

Training (`scripts/train_ranker.py`, run id `d1666b7ed1b34e39bc15202273380f95`):

- `completed_epochs: 8`
- `final_train_loss: 0.034609`
- `best_val_metrics`: `hr@10=0.281478`, `mrr@10=0.150470`, `ndcg@10=0.181151`, `recall@50=0.532570`
- `stopped_due_to_runtime: false`
- `stopped_due_to_memory: false`
- checkpoints: `artifacts/models/best_neural_ranker.pt`, `artifacts/models/checkpoints/neural_ranker_epoch_7.pt`

Validation (`artifacts/reports/ranker_eval_val.json`):

- ranker NDCG@10: `0.181151`
- residual NDCG@10: `0.045040`
- popularity NDCG@10: `0.266328`
- ranker delta vs residual NDCG@10: `+0.136110`
- ranker delta vs popularity NDCG@10: `-0.085177`
- evaluated size: `161,821` queries / `32,452,397` rows

Test (`artifacts/reports/ranker_eval_test.json`):

- ranker NDCG@10: `0.187112`
- residual NDCG@10: `0.032480`
- popularity NDCG@10: `0.262527`
- ranker delta vs residual NDCG@10: `+0.154633`
- ranker delta vs popularity NDCG@10: `-0.075414`
- evaluated size: `161,821` queries / `32,461,176` rows

Acceptance (`artifacts/reports/ranker_acceptance_full.json`):

- `acceptance_passed: true`
- `full_data_ranker_allowed: true`
- all primary rules and guard checks passed

## Acceptance Rules

Primary gate uses OR over:

- val NDCG@10 improves over residual
- val NDCG@10 drop <= 1% and val MRR@10 or HR@10 improves
- test NDCG@10 improves and val NDCG@10 drop < 1%

Mandatory guards:

- no candidate leakage
- exactly one positive per query
- no duplicate candidates per query
- finite residual/ranker scores
- deterministic candidate generation signatures
- MLflow run ID persisted in best checkpoint
- recall@50 relative drop vs residual <= 5%

## Artifacts

- Candidates: `artifacts/ranker/candidates/{sample|full}/`
- Features: `artifacts/ranker/features/{sample|full}/`
- Checkpoints: `artifacts/models/best_neural_ranker.pt`, `artifacts/models/last_neural_ranker.pt`
- Reports: `artifacts/reports/ranker_eval_*.json`, `artifacts/reports/ranker_comparison.md`, `artifacts/reports/ranker_acceptance_*.json`
