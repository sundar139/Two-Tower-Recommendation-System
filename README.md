# MovieLens-25M Two-Tower Recommendation System

An industry-style recommender system that combines residual transformer retrieval, neural re-ranking, recall-constrained production scoring, FastAPI serving, Docker packaging, and optional local Ollama explanations.

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-005571)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?logo=ollama&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package%20Manager-4B8BBE)
![Ruff](https://img.shields.io/badge/Ruff-Lint-46A5F5)
![mypy](https://img.shields.io/badge/mypy-Type%20Checked-2A6DB2)
![pytest](https://img.shields.io/badge/pytest-Tested-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Experiment Tracking](#experiment-tracking)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

This repository implements a full recommender engineering workflow on MovieLens-25M: offline data preparation, retrieval modeling, ranking, policy selection, serving, validation, and packaging. It is non-trivial because strong offline recommendation quality requires both architecture choices and deployment-safe decision rules, not only model training.

Popularity is a strong baseline on MovieLens. Claims are weak unless they beat popularity and protect recall behavior, so this project keeps popularity as a first-class benchmark. A two-stage architecture is realistic for production settings: retrieval scales candidate generation, and ranking improves precision on a manageable shortlist.

### Key Outcomes

- Residual transformer retriever approved as the production retrieval backbone.
- CL contrastive retriever implemented but remains experimental.
- Neural ranker improved over residual retrieval baseline.
- Recall-constrained production scorer beat popularity on NDCG@10 while improving Recall@50.
- FastAPI serving validated.
- Docker local serving validated.
- Ollama explanations validated as optional post-processing.

Approved headline numbers:

- Validation: popularity NDCG@10 `0.2663275394` -> selected scorer NDCG@10 `0.3115227229`
- Validation: popularity Recall@50 `0.7114404187` -> selected scorer Recall@50 `0.7296580790`
- Test: popularity NDCG@10 `0.2625267030` -> selected scorer NDCG@10 `0.3175914351`
- Test: popularity Recall@50 `0.6795100759` -> selected scorer Recall@50 `0.7123735486`
- Serving: API validation `18/18`, p50 `29.70 ms`, p95 `35.05 ms`, max `36.64 ms`
- Docker smoke test `6/6`, Ollama validation `9/9`, explanations generated `3`, explanation latency about `70 to 73 seconds`

## Architecture

```mermaid
flowchart LR
    A[MovieLens-25M Raw Data] --> B[Preprocessing and Feature Engineering]
    B --> C[User and Item Towers]
    C --> D[Residual Transformer Retriever]
    D --> E[FAISS Top-200 Candidate Retrieval]
    E --> F[Neural Ranker]
    F --> G[Recall-Constrained Production Scorer]
    G --> H[FastAPI Service]
    H --> I[Optional Ollama Explanations]
```

| Component | Purpose | Technology | Output Artifact |
|---|---|---|---|
| Data pipeline | Build leakage-safe train/val/test data and features | Polars, PyArrow, Python scripts | `data/processed/*.parquet`, `dataset_stats.json` |
| Residual transformer retriever | Generate user/item embeddings and retrieval scores | PyTorch | `artifacts/models/best_residual_transformer_retriever.pt` |
| FAISS index | Fast inner-product candidate lookup | FAISS | `artifacts/faiss/index.faiss` + metadata/mapping |
| Neural ranker | Re-rank top-200 candidates | PyTorch MLP ranker | `artifacts/models/best_neural_ranker.pt` |
| Production scorer | Select serving policy under recall guard | Weighted scoring policy search | `artifacts/reports/production_scorer_selection.json` |
| FastAPI serving | Online inference endpoints | FastAPI, Uvicorn, Pydantic | Live API + `artifacts/reports/serving_api_validation.json` |
| Docker local packaging | Reproducible local service packaging | Docker, docker-compose | `artifacts/reports/docker_smoke_test.json` |
| Ollama explanation layer | Optional post-hoc item explanations | Ollama local models + HTTP integration | `artifacts/reports/ollama_explanation_validation.json` |
| MLflow tracking | Run metadata and artifact tracking | MLflow + SQLite backend | `mlflow.db`, `mlruns/` |

## Results

### A) Model Progression and Promotion Decisions

| Stage | Purpose | Outcome | Promotion Decision |
|---|---|---|---|
| Popularity baseline | Strong non-neural benchmark | Remained highly competitive | Kept as mandatory baseline and fallback component |
| Plain two-tower retriever | Initial neural retrieval baseline | Established retrieval baseline quality | Approved as intermediate baseline |
| Residual transformer retriever | Improve retrieval while preserving baseline signal | Outperformed plain two-tower in approved flow | Promoted as production retrieval backbone |
| CL contrastive retriever | Add auxiliary contrastive objectives | Implemented and validated structurally, sample acceptance failed | Kept experimental, not promoted |
| Neural ranker | Improve ranking over residual retrieval candidates | Improved strongly vs residual retrieval-only metrics | Promoted into production scoring path |
| Recall-constrained production scorer | Beat popularity without recall collapse | Beat popularity on NDCG@10 and improved Recall@50 | Final approved serving scorer |

### B) Final Full-Data Metrics

| Model | Split | HR@10 | MRR@10 | NDCG@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| Popularity baseline | Validation | 0.3853640751 | 0.2300853687 | 0.2663275394 | 0.7114404187 |
| Popularity baseline | Test | 0.3736350659 | 0.2286240097 | 0.2625267030 | 0.6795100759 |
| Residual retriever | Validation | N/A | N/A | 0.045040 | 0.248491 |
| Residual retriever | Test | N/A | N/A | 0.032480 | 0.206234 |
| Neural ranker | Validation | N/A | N/A | 0.181151 | 0.532570 |
| Neural ranker | Test | N/A | N/A | 0.187112 | 0.575556 |
| Selected production scorer | Validation | 0.4353637661 | 0.2732455409 | 0.3115227229 | 0.7296580790 |
| Selected production scorer | Test | 0.4477478201 | 0.2774359612 | 0.3175914351 | 0.7123735486 |

Selected production scorer policy:

- `ranker_topk_popularity_backfill`
- `alpha=1.0`, `beta=0.1`, `gamma=0.0`, `top_k_focus=20`

### C) Serving Benchmarks

| Validation Area | Result |
|---|---|
| Local API contract validation | `18/18` passed |
| Docker smoke test | `6/6` passed |
| Serving latency p50 | `29.70 ms` |
| Serving latency p95 | `35.05 ms` |
| Serving latency max | `36.64 ms` |
| Ollama explanation validation | `9/9` passed |
| Ollama explanation latency | about `70 to 73 seconds` |

Note: explanation generation is optional and intentionally post-processing; it does not change ranking order.

## Prerequisites

Operating system:

- Windows 11 PowerShell tested
- WSL/Linux likely compatible (commands may need path/shell adjustments)

Tools:

- Python 3.12
- uv
- Git
- Docker Desktop
- Ollama
- Optional CUDA-capable GPU for training
- CPU is sufficient for local API serving when artifacts already exist

Python/project stack:

- PyTorch
- FastAPI
- FAISS
- MLflow
- Polars / PyArrow
- scikit-learn and supporting libraries from `pyproject.toml`

Required local Ollama models:

- `qwen3:4b`
- `qwen3-embedding:0.6b`

```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

## Installation

```bash
git clone https://github.com/sundar139/Two-Tower-Recommendation-System.git
cd Two-Tower-Recommendation-System
```

Windows PowerShell:

```powershell
Copy-Item env.example .env -Force
uv sync --extra dev
uv run python verify.py
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Linux/macOS equivalent env copy:

```bash
cp env.example .env
```

## Dataset

Dataset: GroupLens MovieLens 25M

- URL: https://grouplens.org/datasets/movielens/25m/
- Broad contents: ratings, movies metadata, tags, links, genome scores/tags
- Raw dataset is not committed to this repository
- Download directly from GroupLens and follow their licensing/usage terms
- Place raw files under `data/raw/`
- Generated processed data is ignored by Git

Expected raw layout:

```text
data/raw/ml-25m/
  ratings.csv
  movies.csv
  tags.csv
  links.csv
  genome-scores.csv
  genome-tags.csv
```

## Usage

### A) Verify Environment

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
```

### B) Run Data Pipeline

```powershell
uv run python scripts/download_movielens.py --config configs/data.yaml
uv run python scripts/prepare_data.py --config configs/data.yaml --force
```

### C) Train and Evaluate Retrieval

```powershell
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --model-type baseline
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model baseline --split val

uv run python scripts/train_retriever.py --config configs/transformer_retrieval_residual.yaml --model-type residual_transformer --init-from-baseline artifacts/models/best_baseline_retriever.pt
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_residual.yaml --model residual_transformer --split val
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_residual.yaml --model residual_transformer --split test

uv run python scripts/export_faiss_index.py --config configs/transformer_retrieval_residual.yaml --model-type residual_transformer
```

### D) Train and Evaluate Ranker

```powershell
uv run python scripts/generate_ranker_candidates.py
uv run python scripts/train_ranker.py
uv run python scripts/evaluate_ranker.py --split val
uv run python scripts/evaluate_ranker.py --split test
uv run python scripts/select_production_scorer.py --config configs/ranker.yaml
uv run python scripts/check_production_scorer_acceptance.py --selection artifacts/reports/production_scorer_selection.json
```

### E) Run Local API

```powershell
uv run python scripts/run_api.py --config configs/serving.yaml --host 127.0.0.1 --port 8000
```

### F) Validate API

```powershell
uv run python scripts/validate_serving_api.py --base-url http://127.0.0.1:8000 --timeout-seconds 120 --max-explanation-items 3
```

### G) Run Docker

```powershell
uv run python scripts/check_docker_artifacts.py --config configs/serving.yaml
docker compose build
docker compose up recommender-api
uv run python scripts/docker_smoke_test.py
```

### H) Run Ollama Explanations

```powershell
ollama serve
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
uv run python scripts/validate_ollama_explanations.py --base-url http://127.0.0.1:8000 --ollama-url http://127.0.0.1:11434 --timeout-seconds 180 --max-explanation-items 3
```

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service identity and links |
| `/health` | GET | Liveness probe |
| `/healthz` | GET | Liveness alias |
| `/ready` | GET | Readiness probe |
| `/readyz` | GET | Readiness alias |
| `/metadata` | GET | Runtime metadata and scorer policy |
| `/recommendations` | POST | Primary recommendation endpoint |
| `/v1/recommend` | POST | Backward-compatible recommendation endpoint |
| `/recommendations/{user_id}` | GET | Convenience recommendation endpoint |
| `/users/{user_id}/history` | GET | Recent user history |
| `/v1/explain` | POST | Explanation endpoint |
| `/explanations/recommendations` | POST | Backward-compatible explanation endpoint |

Health:

```bash
curl http://127.0.0.1:8000/health
```

Metadata:

```bash
curl http://127.0.0.1:8000/metadata
```

Recommendations:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d "{\"user_idx\":0,\"k\":10,\"exclude_seen\":true,\"include_debug\":false,\"allow_cold_start\":true}"
```

Recommendations with explanations:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d "{\"user_idx\":0,\"k\":5,\"exclude_seen\":true,\"include_explanations\":true,\"max_explanation_items\":3}"
```

Cold start:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d "{\"user_idx\":999999999,\"k\":10,\"allow_cold_start\":true}"
```

Behavior notes:

- Explanations are post-processing only and do not change ranking order.
- Cold-start requests use popularity fallback when allowed.
- Invalid `k` returns structured validation errors.

## Experiment Tracking

MLflow is configured with a local SQLite backend and local artifact store:

- Metadata backend: `sqlite:///mlflow.db`
- Artifact directory: `mlruns/`

Start MLflow UI:

```powershell
uv run python scripts/start_mlflow_ui.py --run
```

Typical tracked outputs include:

- Hyperparameters and config values
- Training and validation losses
- NDCG@10, MRR@10, HR@10, Recall@50
- Selection and acceptance artifacts under `artifacts/reports/`

## Project Structure

```text
configs/                        # retrieval, ranker, serving, and data configs
scripts/                        # reproducible CLI workflows for train/eval/serve/validate
src/movie_recsys/               # core package
src/movie_recsys/modeling/      # retrieval models and training utilities
src/movie_recsys/ranking/       # ranker features, model, evaluation, acceptance
src/movie_recsys/serving/       # FastAPI app, schemas, artifact registry, explainers
tests/                          # unit and integration tests
data/                           # raw/interim/processed (only .gitkeep tracked)
artifacts/                      # generated models/indexes/reports (ignored)
mlruns/                         # MLflow artifacts (ignored)
```

## Design Decisions

### A) Why Two-Stage Retrieval and Ranking?

Retrieval is optimized for scalable candidate generation, while ranking is optimized for precision on a small candidate pool. This split reflects common production recommender system design.

### B) Why Residual Transformer Retriever?

Pure transformer retrieval underperformed during development. The residual/gated approach preserved baseline retrieval signal and added sequence modeling safely, leading to promotion of residual transformer as the production retrieval backbone.

### C) Why CL Stayed Experimental?

The CL retriever path was implemented and tested, but sample acceptance failed. It was intentionally not promoted because production quality criteria took precedence over architectural novelty.

### D) Why Compare Against Popularity?

Popularity is a strong MovieLens baseline. Recommender improvements are not credible unless they are compared against it and evaluated under stable guardrails.

### E) Why Recall-Constrained Production Scorer?

An earlier hybrid scorer improved NDCG but dropped Recall@50 too much. The final top-k ranker plus popularity backfill policy improved NDCG while preserving and improving Recall@50.

### F) Why FastAPI and Docker?

FastAPI provides a clear serving contract and easy validation. Docker local packaging makes serving reproducible and separates training artifacts from runtime API infrastructure.

### G) Why Ollama Explanations Are Optional?

Explanations are post-processing metadata. They do not affect ranking order, and fail-open behavior keeps recommendation availability when LLM calls are unavailable.

### H) Why Artifacts Are Mounted in Docker?

Model checkpoints and index artifacts are large and evolve independently from API code. Mounting keeps images smaller and supports reproducible local validation with approved artifacts.

## Limitations

- Offline MovieLens evaluation does not prove online business impact.
- No real-time feedback loop is implemented.
- No user authentication is included.
- No cloud deployment pipeline is included.
- No frontend experience is included.
- Docker image is local-serving oriented.
- Ollama explanations can be slow.
- CL retriever remains experimental.
- Serving requires local model/index artifacts.
- Popularity remains a strong signal and is intentionally used in final scoring policy.

## Acknowledgements

- GroupLens MovieLens 25M dataset
- PyTorch
- FAISS
- FastAPI
- MLflow
- Ollama

Contrastive-learning note:

The contrastive component was inspired by CL-EPIDTN-style contrastive recommendation ideas, but this repository does not claim a full paper reproduction.

## Author

**Rohith Sundar Jonnalagadda**  
[LinkedIn](https://www.linkedin.com/in/rohithsundarj/) · MS Computer Science, Kennesaw State University

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.