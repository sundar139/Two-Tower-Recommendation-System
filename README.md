# MovieLens Two-Tower Recommender

Production-grade repository for a CL-enhanced transformer two-tower recommendation system on MovieLens-25M.

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
- MLflow logging for training and evaluation runs

Not yet implemented in this step:

- Contrastive learning objective
- FastAPI serving layer
- Ollama explanation endpoints
- Custom transformer sequence encoder
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

## Step 2 Retrieval Commands

```powershell
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split val
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split val
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml
```

## Step 2 Validation Commands

### Sample Validation Commands

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split val --sample
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split test --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split test --sample
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml --sample
git status --short
git ls-files data artifacts mlruns models .venv
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
uv run python scripts/train_retriever.py --config configs/retrieval.yaml
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split val
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split test
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml
git status --short
git ls-files data artifacts mlruns models .venv
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
- model checkpoints and FAISS index files

## Next Planned Step

Extend the plain baseline with transformer-based sequence encoding and contrastive learning.
