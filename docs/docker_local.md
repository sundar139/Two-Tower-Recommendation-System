# Step 7A Docker Local Workflow

## 1. Scope

Step 7A packages the approved local FastAPI recommendation service into a reproducible Docker workflow.

In scope:

- local image build for API runtime
- local Docker Compose service for the existing Step 6B contract
- mounted local artifacts and processed data
- local smoke validation against running container

Out of scope:

- cloud deployment
- authentication
- Ollama explanations

## 2. Why Artifacts Are Mounted (Not Baked)

Serving depends on large local model artifacts and FAISS files that are produced by training/evaluation workflows.

Mounting artifacts instead of baking them into the image keeps:

- image rebuilds fast for code-only changes
- artifact provenance explicit on the host
- serving behavior aligned with approved local checkpoints and FAISS bundles

## 3. Required Local Artifacts

The preflight checker validates the following paths:

- configs/serving.yaml
- configs/ranker.yaml
- configs/transformer_retrieval_residual.yaml
- artifacts/models/best_residual_transformer_retriever.pt
- artifacts/models/best_neural_ranker.pt
- artifacts/faiss/index.faiss
- artifacts/faiss/index_metadata.json
- artifacts/faiss/item_idx_mapping.parquet
- data/processed/items.parquet
- data/processed/users.parquet
- data/processed/interactions_train.parquet
- data/processed/user_id_map.parquet
- data/processed/item_id_map.parquet

## 4. Preflight Command

```powershell
uv run python scripts/check_docker_artifacts.py --config configs/serving.yaml
```

Expected result:

- FOUND/MISSING table
- final ok: true

If final ok is false, fix missing files before building or running Docker.

## 5. Build Command

```powershell
docker compose build
```

## 6. Run Command

```powershell
docker compose up recommender-api
```

Service mapping:

- host port: 8000
- container port: 8000
- mounted volumes:
  - ./artifacts -> /app/artifacts (read-only)
  - ./data/processed -> /app/data/processed (read-only)
  - ./configs -> /app/configs (read-only)

## 7. Smoke Test Command

Run in a second terminal after the service is up:

```powershell
uv run python scripts/docker_smoke_test.py --base-url http://127.0.0.1:8000 --known-user-idx 0 --k 10
```

Outputs:

- pass/fail summary by endpoint
- endpoint latency values
- artifacts/reports/docker_smoke_test.json
- final ok true/false

## 8. Troubleshooting

Missing artifacts:

- rerun preflight and create/copy the listed missing files exactly as reported.

Wrong FAISS path:

- verify paths.faiss_dir in configs/serving.yaml points to the directory that contains index.faiss, index_metadata.json, and item_idx_mapping.parquet.

Container cannot see mounted volume:

- verify docker-compose.yml volume paths exist on host and are mounted as /app/artifacts, /app/data/processed, and /app/configs.

Port already in use:

- stop conflicting process or update compose port mapping and smoke-test base URL consistently.

## 9. Known Limitations

- local Docker only
- no cloud deployment yet
- no auth
- no Ollama explanations yet
