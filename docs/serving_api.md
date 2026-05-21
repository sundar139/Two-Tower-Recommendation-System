# Step 7B Serving API Contract

## Scope

Step 6B/7B finalizes local FastAPI serving for the approved recommendation stack:

- residual transformer retriever backbone
- FAISS candidate retrieval
- neural ranker rescoring
- production scorer policy: `ranker_topk_popularity_backfill`

Step 6B adds contract-compatible aliases, cold-start fallback behavior, and local validation/benchmark tooling.

Step 7B adds optional local Ollama-powered explanation generation with fail-open behavior and explanation-focused endpoints.

## Approved Production Policy

- `policy`: `ranker_topk_popularity_backfill`
- `alpha`: `1.0`
- `beta`: `0.1`
- `gamma`: `0.0`
- `top_k_focus`: `20`

## Endpoint Table

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service identity and top-level links |
| `/health` | GET | Liveness probe |
| `/healthz` | GET | Backward-compatible liveness alias |
| `/ready` | GET | Readiness probe |
| `/readyz` | GET | Backward-compatible readiness alias |
| `/metadata` | GET | App metadata, scorer policy, artifact names, runtime limits |
| `/recommendations` | POST | Primary recommendation endpoint |
| `/v1/recommend` | POST | Backward-compatible recommendation endpoint |
| `/v1/explain` | POST | Generate explanations for recommended items |
| `/explanations/recommendations` | POST | Backward-compatible explanation endpoint alias |
| `/recommendations/{user_id}` | GET | Convenience wrapper around recommendation request |
| `/users/{user_id}/history` | GET | Recent user history capped to 100 rows |

## Backward-Compatible Aliases

- Health aliases: `/health` and `/healthz` are equivalent.
- Readiness aliases: `/ready` and `/readyz` are equivalent.
- Recommendation aliases: `/recommendations` and `/v1/recommend` use the same request normalization and response schema.
- Explanation aliases: `/v1/explain` and `/explanations/recommendations` use the same request normalization and response schema.

## Request Schema Compatibility

The recommendation request accepts both snake_case and camelCase aliases:

- `user_id` or `userId`
- `user_idx` or `userIdx`
- `k` or `top_k`
- `exclude_seen` or `excludeSeen`
- `include_debug` or `includeDebug`
- `allow_cold_start` or `allowColdStart`
- `candidate_top_k` or `candidateTopK`
- `include_explanations` or `includeExplanations`
- `explanation_style` or `explanationStyle` (`concise` or `detailed`)
- `max_explanation_items` or `maxExplanationItems`

Validation and identity rules:

- `k` must be within serving runtime bounds (`1..max_k`).
- If provided, `candidate_top_k` must satisfy `candidate_top_k >= k` and `candidate_top_k <= 500`.
- Supplying both `user_id` and `user_idx` returns a `400` invalid request.
- If both are missing and `allow_cold_start=false`, request returns `400`.

## Recommendation Response Example

```json
{
  "user_id": 709,
  "user_idx": 0,
  "k": 10,
  "cold_start": false,
  "scorer_policy": "ranker_topk_popularity_backfill",
  "explanation_status": "generated",
  "overall_explanation": "Top picks combine your strong Drama/Crime affinity with high ranker confidence.",
  "recommendations": [
    {
      "movieId": 318,
      "item_idx": 263,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "release_year": 1994,
      "final_score": 0.91,
      "residual_score": 0.29,
      "ranker_score": 0.84,
      "popularity_score": 0.71,
      "rank_position": 1,
      "scorer_policy": "ranker_topk_popularity_backfill",
      "explanation": "Strong Drama/Crime overlap with your recent history and high combined scoring support this top rank."
    }
  ],
  "debug": null
}
```

`explanation_status` values:

- `disabled`: explanation generation not requested or globally disabled
- `generated`: explanations successfully generated
- `unavailable`: Ollama unavailable while fail-open kept recommendations intact
- `failed`: unexpected explanation error while fail-open kept recommendations intact

## Explanation Endpoint Request Example

```json
{
  "user_id": 709,
  "top_k": 10,
  "style": "concise",
  "max_explanation_items": 5,
  "include_debug": false
}
```

`/v1/explain` and `/explanations/recommendations` can also accept `recommendation_items` to explain a precomputed list without rerunning retrieval/ranking.

## Cold-Start Behavior

Unknown user handling:

- `allow_cold_start=true`:
  - API returns popularity-backed recommendations.
  - Response includes `cold_start=true`.
  - `scorer_policy` is `popularity_fallback`.
- `allow_cold_start=false`:
  - API returns `404` with structured error payload.

Cold-start response rows include:

- `movieId`
- `item_idx`
- `title`
- `genres`
- `release_year`
- `final_score`
- `popularity_score`
- `rank_position`

## Local Validation Script

Run full contract validation (including explanation checks) against a live server:

```powershell
uv run python scripts/validate_serving_api.py --base-url http://127.0.0.1:8000 --known-user-idx 0 --k 10
```

Outputs:

- pass/fail table per check
- per-endpoint latencies
- JSON report at `artifacts/reports/serving_api_validation.json`

Run targeted Ollama explanation validation:

```powershell
uv run python scripts/validate_ollama_explanations.py --base-url http://127.0.0.1:8000 --ollama-url http://127.0.0.1:11434 --known-user-idx 0 --k 10
```

Outputs:

- pass/fail table focused on explanation flow
- Ollama health and configured model presence checks
- JSON report at `artifacts/reports/ollama_explanation_validation.json`

## Local Latency Benchmark Script

Run sequential latency benchmark:

```powershell
uv run python scripts/benchmark_serving_api.py --base-url http://127.0.0.1:8000 --num-users 50 --k 10
```

Outputs:

- `p50`, `p95`, `max` latency
- `success_count`, `failure_count`
- JSON report at `artifacts/reports/serving_api_latency.json`

## Step 7A Docker Local Packaging

Step 7A packages this API for reproducible local Docker runs without changing retrieval or scorer behavior.

Why artifacts are mounted instead of baked into the image:

- model checkpoints and FAISS bundles are large and updated independently of application code
- mounting keeps the image generic and reproducible while using approved local artifacts
- local validation can fail fast when required files are missing

Local Docker sequence:

```powershell
uv run python scripts/check_docker_artifacts.py --config configs/serving.yaml
docker compose build
docker compose up recommender-api
uv run python scripts/docker_smoke_test.py --base-url http://127.0.0.1:8000 --known-user-idx 0 --k 10
```

Detailed Docker runbook:

- `docs/docker_local.md`

## Known Limitations

- Ollama explanations require a running local Ollama daemon and installed models
- local Docker only
- no cloud deployment yet
- no authentication
- local model/data artifacts are required
