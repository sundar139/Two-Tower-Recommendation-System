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

Not yet implemented in this step:

- FastAPI serving layer
- Ollama explanation endpoints
- Neural ranker

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
- Ranker work should continue using residual-transformer retrieval artifacts.

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
- no neural ranker yet
- no FastAPI endpoints yet
- no Ollama explanation endpoints yet
