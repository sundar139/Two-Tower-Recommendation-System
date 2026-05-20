# Production Scorer Decision (Step 5C)

## Why This Step Was Required

The neural ranker passed Step 5B against the residual retriever baseline, but popularity still outperformed ranker-only NDCG@10 on full validation and test.

Because of that gap, serving/API work remained blocked until scorer-policy selection was completed across popularity, residual, ranker, and popularity-aware hybrids.

## Candidate Policies

1. `popularity_only`
2. `residual_only`
3. `ranker_only`
4. `ranker_plus_popularity`
5. `ranker_plus_residual`
6. `ranker_plus_popularity_plus_residual`

Hybrid forms use query-wise normalized scores:

- `ranker_plus_popularity = alpha * ranker_norm + beta * popularity_norm`
- `ranker_plus_residual = alpha * ranker_norm + beta * residual_norm`
- `ranker_plus_popularity_plus_residual = alpha * ranker_norm + beta * popularity_norm + gamma * residual_norm`

## Selection Protocol

- normalization method: `query_minmax` (query-local, split-local)
- no global normalization leakage across splits
- weight selection split: validation only
- test split is never used for weight search
- manual grid only (no Optuna):
  - `alpha = [0.5, 0.7, 0.85, 1.0]`
  - `beta = [0.0, 0.1, 0.2, 0.3, 0.5]`
  - `gamma = [0.0, 0.1, 0.2, 0.3]`

## Full-Data Inputs

- val queries/rows: `161,821 / 32,452,397`
- test queries/rows: `161,821 / 32,461,176`

## Selection Outcome

Selected scorer (validation winner):

- policy: `ranker_plus_popularity`
- weights: `alpha=1.0`, `beta=0.1`, `gamma=0.0`
- validation metrics:
  - `hr@10=0.435364`
  - `mrr@10=0.273246`
  - `ndcg@10=0.311523`
  - `recall@50=0.597203`
- test metrics:
  - `hr@10=0.447748`
  - `mrr@10=0.277436`
  - `ndcg@10=0.317591`
  - `recall@50=0.637241`

Reference single-policy validation NDCG@10 values:

- popularity_only: `0.266328`
- ranker_only: `0.181151`
- residual_only: `0.045040`

The full validation table (all 123 policy rows) is stored in:

- `artifacts/reports/production_scorer_selection.md`

## Acceptance Outcome

Acceptance report:

- `artifacts/reports/production_scorer_acceptance.json`

Result:

- `acceptance_passed: false`
- primary popularity-comparison rules: passed
- failed mandatory guard: `recall50_relative_drop_vs_popularity_le_5pct`
- recall@50 relative drop vs popularity:
  - val: `0.160572`
  - test: `0.062205`

Decision:

- Step 6 FastAPI serving is **not unblocked**.
- Next required work is recall-preserving scorer design/audit that satisfies the popularity recall guard.

## Reproducibility Commands

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
uv run python -m ruff check .
uv run python -m mypy src
uv run python -m pytest -q
uv run python scripts/select_production_scorer.py --config configs/ranker.yaml
uv run python scripts/check_production_scorer_acceptance.py --selection artifacts/reports/production_scorer_selection.json
```
