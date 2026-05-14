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

Not yet implemented in this step:

- Contrastive learning objective
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
uv run python scripts/start_mlflow_ui.py
```

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
uv run python scripts/diagnose_transformer_retriever.py --config configs/transformer_retrieval.yaml --sample
uv run python scripts/run_transformer_ablation.py --sample
uv run python scripts/train_retriever.py --config configs/transformer_retrieval_stable.yaml --sample --model-type transformer
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split test --sample
uv run python scripts/compare_retrievers.py --sample
```

## Transformer Stabilization Status

Latest sample comparison (stabilization attempt):

- popularity val NDCG@10: `0.024347`
- baseline val NDCG@10: `0.025047`
- stable transformer val NDCG@10: `0.018203`

Result: transformer remains below baseline and popularity on sample validation NDCG@10.

Step 3 is not approved yet. CL-EPIDTN contrastive learning remains blocked until transformer baseline quality is recovered.

Example MLflow run URLs from stabilization:

- baseline sample train: `http://127.0.0.1:5000/#/experiments/1/runs/e9bf170a6e324b9ca7917d994cd939f5`
- stable transformer sample train: `http://127.0.0.1:5000/#/experiments/1/runs/4940a9671a354a8fa27c8c9f4cf809fa`
- best ablation trial: `http://127.0.0.1:5000/#/experiments/1/runs/745306f30a1145ef97e7f4bd716a0b68`

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
