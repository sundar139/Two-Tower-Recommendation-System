# MovieLens Two-Tower Recommender

Production-grade repository for baseline and transformer two-tower retrieval on MovieLens-25M.

## Current Scope

Implemented in this step:

- Environment setup with Python 3.12 + uv
- Deterministic MovieLens download tooling
- Polars-based preprocessing and feature generation
- Chronological per-user train/validation/test splitting
- Data and environment validation checks
- Unit tests for preprocessing, features, splits, and safety constraints

Implemented in Step 2 (plain retrieval baseline):

- Popularity baseline retrieval evaluator
- Plain two-tower retriever with in-batch negatives
- Retrieval Dataset/DataLoader with leakage-safe history windows
- Offline metrics: HR@10, MRR@10, NDCG@10, Recall@50
- FAISS flat inner-product index export/reload/search
- MLflow logging for training, evaluation, and FAISS export runs

Implemented in Step 3 (transformer retrieval encoder):

- Custom transformer user sequence encoder with causal self-attention
- Padding-aware and causal masking to prevent leakage
- Configurable sequence pooling (`last` or `mean`)
- TransformerRetriever integrated into train/eval/export workflow
- Residual transformer retriever with gated baseline + transformer blending
- Baseline-checkpoint initialization flow for residual retriever
- Residual sample ablation and four-way comparison tooling

Implemented in Step 4 (contrastive residual enhancement):

- CL residual transformer retriever (`cl_residual_transformer`)
- Sequence augmentations for two-view user contrastive learning
- Symmetric InfoNCE helpers for user/item/alignment objectives
- CL training integration with decomposed loss logging
- Contrastive sample ablation and acceptance checker tooling

Implemented in Step 5 (neural re-ranker):

- Residual top-200 candidate generation for train/val/test ranker queries
- Deterministic candidate validation and metadata signature logging
- Ranker feature engineering over retrieval/user/item/interaction signals
- MLP neural ranker with BCE/BPR/hybrid loss support
- Ranker train/eval scripts with MLflow logging and checkpointing
- Ranker-vs-residual comparison and acceptance checker tooling

Implemented in Step 6 (FastAPI serving layer):

- Typed serving config (`configs/serving.yaml`) and runtime app wiring
- Artifact registry for residual retriever, FAISS bundle, and neural ranker
- Recommendation service with deterministic two-stage scorer policy application
- FastAPI endpoints for health, readiness, and recommendations
- Typed API schemas and structured error responses
- Local runtime helper (`scripts/run_api.py`) and smoke test (`scripts/smoke_test_api.py`)
- Serving-focused test coverage for config, registry, scorer, errors, and API behavior

Not yet implemented in this step:

- Ollama explanation endpoints

## Dataset Note

Raw MovieLens archives and extracted CSV files are not committed. Processed artifacts are also excluded from version control.

## Windows PowerShell Setup

```powershell
Copy-Item env.example .env -Force
uv sync --extra dev
```

## Download MovieLens-25M

```powershell
uv run python scripts/download_movielens.py --config configs/data.yaml
```

Checksum validation is supported and automatically enforced when `expected_checksum`
is configured in `configs/data.yaml`.

## Prepare Sample Data

```powershell
uv run python scripts/prepare_data.py --config configs/data.yaml --sample-users 1000 --seed 42 --force
```

## Prepare Full Data

```powershell
uv run python scripts/prepare_data.py --config configs/data.yaml --force
```

## Verify Environment

```powershell
uv run python verify.py
uv run python scripts/verify_environment.py
```

## MLflow UI (SQLite Backend)

Metadata backend: `sqlite:///mlflow.db`

Artifacts: local `./mlruns` (ignored by git)

Start UI directly:

```powershell
uvx mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Or via project helper:

```powershell
uv run python scripts/start_mlflow_ui.py --run
```

Windows note: if `uvx mlflow ui` emits WinError 10022/worker noise, prefer the helper command above.

UI URL:

`http://127.0.0.1:5000`

Training/evaluation/export scripts print:

- `mlflow_tracking_uri`
- `mlflow_experiment_name`
- `mlflow_run_id`
- `mlflow_ui_url`
- `mlflow_run_url`

## Step 2 Retrieval Commands

```powershell
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split val
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --sample --model-type baseline
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model baseline --split val --sample
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml --sample --model-type baseline
```

## Step 3 Transformer Commands

```powershell
uv run python scripts/diagnose_transformer_retriever.py --config configs/transformer_retrieval_stable.yaml --sample
uv run python scripts/run_transformer_ablation.py --sample
uv run python scripts/train_retriever.py --config configs/transformer_retrieval_stable.yaml --sample --model-type transformer
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split test --sample
uv run python scripts/train_retriever.py --config configs/transformer_retrieval_residual.yaml --sample --model-type residual_transformer --init-from-baseline artifacts/models/best_baseline_retriever.pt
uv run python scripts/run_residual_transformer_ablation.py --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_residual.yaml --model residual_transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_residual.yaml --model residual_transformer --split test --sample
uv run python scripts/compare_retrievers.py --sample
uv run python scripts/export_faiss_index.py --config configs/transformer_retrieval_residual.yaml --sample --model-type residual_transformer
```

## Step 4 Contrastive Commands

```powershell
uv run python scripts/run_contrastive_ablation.py --sample
uv run python scripts/train_retriever.py --config configs/cl_retrieval.yaml --sample --model-type cl_residual_transformer --init-from-residual artifacts/models/best_residual_transformer_retriever.pt
uv run python scripts/evaluate_retriever.py --config configs/cl_retrieval.yaml --model cl_residual_transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/cl_retrieval.yaml --model cl_residual_transformer --split test --sample
uv run python scripts/export_faiss_index.py --config configs/cl_retrieval.yaml --sample --model-type cl_residual_transformer
uv run python scripts/compare_retrievers.py --sample --cl-config configs/cl_retrieval.yaml
uv run python scripts/check_contrastive_acceptance.py --summary artifacts/reports/contrastive_ablation_sample.json --sample
```

Run full-data CL only after sample acceptance passes:

```powershell
uv run python scripts/train_retriever.py --config configs/cl_retrieval.yaml --model-type cl_residual_transformer --init-from-residual artifacts/models/best_residual_transformer_retriever.pt
```

## Step 5 Neural Ranker Commands

Sample workflow:

```powershell
uv run python scripts/generate_ranker_candidates.py --sample
uv run python scripts/train_ranker.py --sample
uv run python scripts/evaluate_ranker.py --sample --split val
uv run python scripts/evaluate_ranker.py --sample --split test
uv run python scripts/compare_retrieval_ranker.py --sample
uv run python scripts/check_ranker_acceptance.py --sample
```

Run full-data ranker only after sample acceptance passes:

```powershell
uv run python scripts/generate_ranker_candidates.py
uv run python scripts/train_ranker.py
uv run python scripts/evaluate_ranker.py --split val
uv run python scripts/evaluate_ranker.py --split test
uv run python scripts/compare_retrieval_ranker.py
uv run python scripts/check_ranker_acceptance.py
```

## Step 6 FastAPI Serving Commands

Run API:

```powershell
uv run python scripts/run_api.py --config configs/serving.yaml --host 127.0.0.1 --port 8000
```

Smoke test:

```powershell
uv run python scripts/smoke_test_api.py --base-url http://127.0.0.1:8000 --user-idx 0 --top-k 20 --require-ready
```

Serving workflow details:

- `docs/fastapi_serving.md`

## Full Residual Validation Commands

```powershell
uv run python scripts/run_full_residual_training.py --max-runtime-hours 4
```

Resume from checkpoint:

```powershell
uv run python scripts/run_full_residual_training.py --resume-from artifacts/models/checkpoints/residual_transformer_epoch_3.pt --max-runtime-hours 4
```

Evaluation-only mode (skip train and use latest residual checkpoint):

```powershell
uv run python scripts/run_full_residual_training.py --evaluate-only
```

Acceptance checker:

```powershell
uv run python scripts/check_residual_acceptance.py --summary artifacts/reports/full_residual_transformer_summary.json
```

## Contrastive Stabilization Status

First CL attempt outcome:

- The initial broad CL attempt did not produce a stable promotion decision and was superseded by a focused second-round matrix with stricter acceptance checks.

Second focused CL sample ablation (latest):

- best trial: `focused_proj_warm_anchor_u050_i020_t007_a001`
- best trial val NDCG@10: `0.023174`
- popularity val NDCG@10: `0.024347`
- residual val NDCG@10: `0.020215`

Acceptance result from `scripts/check_contrastive_acceptance.py --summary artifacts/reports/contrastive_ablation_sample.json --sample`:

- `acceptance_passed: false`
- `full_data_cl_allowed: false`
- failure reason: no primary acceptance rule passed
- guard checks: FAISS parity, recall collapse guard, and finite-loss checks all passed

Decision:

- CL remains experimental.
- Residual transformer remains the production retrieval backbone.
- Full-data CL is blocked until a future sample acceptance run passes.
- Neural ranker work should continue using residual-transformer retrieval artifacts.

## Neural Ranker Full-Data Status (Latest)

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

Acceptance passed against the residual retrieval baseline. Popularity still outperformed ranker-only NDCG@10 on both splits, so Step 5C ran a popularity-aware scorer audit before any serving/API work.

Step 5C: Production Scorer Selection and Popularity-Aware Ranker Audit completed.

- normalization: query-wise min-max (split-local; no cross-split leakage)
- validation-only weight selection: test split was not used for weight search
- evaluated policies: `popularity_only`, `residual_only`, `ranker_only`, `ranker_plus_popularity`, `ranker_plus_popularity_plus_residual`, `ranker_topk_popularity_backfill`
- manual grid (Step 5C): `alpha=[0.5, 0.7, 0.85, 1.0]`, `beta=[0.0, 0.1, 0.2, 0.3, 0.5]`, `gamma=[0.0, 0.1, 0.2, 0.3]`

Selected scorer (validation winner):

- policy: `ranker_plus_popularity`
- weights: `alpha=1.0`, `beta=0.1`, `gamma=0.0`
- validation: `hr@10=0.435364`, `mrr@10=0.273246`, `ndcg@10=0.311523`, `recall@50=0.597203`
- test: `hr@10=0.447748`, `mrr@10=0.277436`, `ndcg@10=0.317591`, `recall@50=0.637241`

Production scorer acceptance (`scripts/check_production_scorer_acceptance.py`):

- `acceptance_passed: false`
- primary popularity-comparison rules: passed
- failed guard: `recall50_relative_drop_vs_popularity_le_5pct`
- recall relative drop vs popularity: val `0.160572`, test `0.062205`
- Step 6 FastAPI serving unblocked: `false`

Reports:

- `artifacts/reports/production_scorer_selection.json`
- `artifacts/reports/production_scorer_selection.md`
- `artifacts/reports/production_scorer_acceptance.json`
- `docs/production_scorer.md`

Step 5D: Recall-Constrained Production Scorer Tuning completed.

- objective: enforce validation recall guard before scorer selection (`Recall@50 >= 0.95 * popularity Recall@50`)
- recall constraint value: `0.675868`
- selected_by_validation_only: `true`
- candidates_passing_recall_constraint: `64`
- popularity_safe_fallback_used: `false`

Expanded recall-aware grid:

- `ranker_plus_popularity`: `alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]`, `beta=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]`
- `ranker_plus_popularity_plus_residual`: `alpha=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`, `beta=[0.2, 0.5, 1.0, 1.5, 2.0]`, `gamma=[0.0, 0.05, 0.1, 0.2]`
- `ranker_topk_popularity_backfill`: `top_k_focus=[10, 20, 30, 50]` (deterministic two-stage policy)

Selected scorer (Step 5D validation winner):

- policy: `ranker_topk_popularity_backfill`
- weights: `alpha=1.0`, `beta=0.1`, `gamma=0.0`, `top_k_focus=20`
- validation: `hr@10=0.435364`, `mrr@10=0.273246`, `ndcg@10=0.311523`, `recall@50=0.729658`
- test: `hr@10=0.447748`, `mrr@10=0.277436`, `ndcg@10=0.317591`, `recall@50=0.712374`

Step 5D acceptance (`scripts/check_production_scorer_acceptance.py`):

- `acceptance_passed: true`
- `step6_fastapi_unblocked: true`
- `step6_unblocked_mode: selected_scorer`
- recall guard result: passed (`recall50_relative_drop_vs_popularity_le_5pct=true`, val/test drop `0.0`)

## Step 2 Validation Commands

### Sample Validation Commands

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split val --sample
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --sample --model-type baseline
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model baseline --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split test --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model baseline --split test --sample
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml --sample --model-type baseline
git status --short
git ls-files data artifacts mlruns models .venv mlflow.db
```

### Full-Data Validation Commands

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split val
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split test
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --model-type baseline
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model baseline --split val
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model baseline --split test
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml --model-type baseline
git status --short
git ls-files data artifacts mlruns models .venv mlflow.db
```

## Run Quality Checks

```powershell
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Expected Processed Outputs

Default output root: `data/processed/`.
Sample mode output root: `data/processed/sample/`.

Expected files:

- interactions_train.parquet
- interactions_val.parquet
- interactions_test.parquet
- users.parquet
- items.parquet
- user_id_map.parquet
- item_id_map.parquet
- user_histories.parquet
- dataset_stats.json

## Retrieval Metrics

- HR@10: hit-rate at top-10.
- MRR@10: reciprocal rank at top-10.
- NDCG@10: position-weighted ranking gain at top-10.
- Recall@50: recall at top-50.

## Artifact Policy

Do not commit generated data or experiment artifacts:

- `data/raw/ml-25m.zip`, `data/raw/ml-25m/`
- `data/processed/**` generated outputs
- `artifacts/**`
- `mlruns/**`
- `mlflow.db`
- model checkpoints and FAISS index files

Transformer limitations currently in scope:

- no contrastive learning yet
- no FastAPI endpoints yet
- no Ollama explanation endpoints yet
