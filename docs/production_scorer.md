# Production Scorer Decision (Step 5C -> Step 5D)

## Context

Step 5C selected a high-NDCG scorer (`ranker_plus_popularity`, `alpha=1.0`, `beta=0.1`) but failed acceptance due to Recall@50 drop vs popularity (val `16.0572%`, test `6.2205%`).

Step 5D introduced recall-constrained validation selection to ensure production-safe recall behavior before serving/API work.

## Candidate Policies (Step 5D)

1. `popularity_only`
2. `residual_only`
3. `ranker_only`
4. `ranker_plus_popularity`
5. `ranker_plus_popularity_plus_residual`
6. `ranker_topk_popularity_backfill` (two-stage deterministic policy)

## Selection Protocol (Step 5D)

- normalization: `query_minmax` (query-local, split-local)
- no global normalization leakage across splits
- validation-only selection (`selected_by_validation_only=true`)
- strict recall constraint: `Recall@50 >= 0.95 * popularity Recall@50`
- recall constraint value on val: `0.675868`
- test split not used for weight selection

Expanded manual grid (no Optuna):

- `ranker_plus_popularity`
  - `alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]`
  - `beta=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]`
- `ranker_plus_popularity_plus_residual`
  - `alpha=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`
  - `beta=[0.2, 0.5, 1.0, 1.5, 2.0]`
  - `gamma=[0.0, 0.05, 0.1, 0.2]`
- `ranker_topk_popularity_backfill`
  - `top_k_focus=[10, 20, 30, 50]`
  - stage-1 hybrid weights: `alpha=1.0`, `beta=0.1`, `gamma=0.0`

## Step 5D Outcome

Selected scorer (validation winner under recall constraint):

- policy: `ranker_topk_popularity_backfill`
- weights: `alpha=1.0`, `beta=0.1`, `gamma=0.0`, `top_k_focus=20`
- candidates passing recall constraint: `64`
- popularity safe fallback used: `false`

Validation metrics:

- `hr@10=0.435364`
- `mrr@10=0.273246`
- `ndcg@10=0.311523`
- `recall@50=0.729658`

Test metrics (evaluated only after validation selection):

- `hr@10=0.447748`
- `mrr@10=0.277436`
- `ndcg@10=0.317591`
- `recall@50=0.712374`

## Acceptance Outcome

- `acceptance_passed: true`
- `step6_fastapi_unblocked: true`
- `step6_unblocked_mode: selected_scorer`
- primary rules: passed
- mandatory guards: passed
- recall guard: passed (`recall50_relative_drop_vs_popularity_le_5pct=true`)

## Step 6B Serving Integration

Step 6B serving uses this exact selected scorer policy for known users:

- `ranker_topk_popularity_backfill`
- `alpha=1.0`, `beta=0.1`, `gamma=0.0`, `top_k_focus=20`

Unknown users are handled via serving-only popularity fallback (`scorer_policy=popularity_fallback`) when `allow_cold_start=true`. This fallback does not change the selected Step 5D production scorer decision.

## Reports

- `artifacts/reports/production_scorer_selection.json`
- `artifacts/reports/production_scorer_selection.md`
- `artifacts/reports/production_scorer_acceptance.json`

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
