# FastAPI Serving Workflow (Step 6)

## Overview

This serving layer exposes local recommendation inference with:

- residual transformer retrieval + FAISS candidate lookup
- neural ranker rescoring
- production scorer policy: `ranker_topk_popularity_backfill`

The API runs fully local and uses artifacts from the project workspace.

## Configuration

Serving config file:

- `configs/serving.yaml`

Key fields:

- `paths.retrieval_config`: retrieval architecture and data-path config
- `paths.ranker_config`: ranker training/evaluation config
- `paths.faiss_dir`: FAISS bundle directory (`index.faiss`, mapping, metadata)
- `paths.residual_checkpoint`: residual retriever checkpoint
- `paths.ranker_checkpoint`: neural ranker checkpoint
- `paths.ranker_feature_manifest`: ranker feature-column contract
- `runtime.sample_data`: when `true`, retrieval config uses `data/processed/sample`

The current repository artifacts are aligned to `runtime.sample_data: true`.

## Run API Locally

```powershell
uv run python scripts/run_api.py --config configs/serving.yaml --host 127.0.0.1 --port 8000
```

## Smoke Test

```powershell
uv run python scripts/smoke_test_api.py --base-url http://127.0.0.1:8000 --user-idx 0 --top-k 20 --require-ready
```

The smoke script checks:

- `GET /healthz`
- `GET /readyz`
- `POST /v1/recommend`

## Endpoints

- `GET /healthz`
  - liveness probe
  - expected payload: `{ "status": "ok" }`

- `GET /readyz`
  - readiness probe
  - payload includes:
    - `status`
    - `ready`
    - `model_loaded`
    - `startup_error`

- `POST /v1/recommend`
  - request:
    - `user_idx` (int)
    - `top_k` (int, bounded by serving config)
  - response:
    - `user_idx`
    - `requested_top_k`
    - `returned_top_k`
    - `policy_name`
    - `total_candidates`
    - `recommendations[]` with `rank`, `item_idx`, and scores

## Error Contract

Errors use a structured payload:

```json
{
  "detail": {
    "error": "error_code",
    "message": "human readable message"
  }
}
```

Common error codes:

- `artifacts_not_ready`
- `service_not_ready`
- `user_not_found`
- `invalid_top_k`
- `no_candidates`
- `feature_mismatch`
