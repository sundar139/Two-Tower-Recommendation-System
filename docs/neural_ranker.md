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
