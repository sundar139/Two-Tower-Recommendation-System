# Step 6B Serving API Contract

## Scope

Step 6B finalizes local FastAPI serving for the approved recommendation stack:

- residual transformer retriever backbone
- FAISS candidate retrieval
- neural ranker rescoring
- production scorer policy: `ranker_topk_popularity_backfill`

Step 6B also adds contract-compatible aliases, cold-start fallback behavior, and local validation/benchmark tooling.

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
| `/recommendations/{user_id}` | GET | Convenience wrapper around recommendation request |
| `/users/{user_id}/history` | GET | Recent user history capped to 100 rows |

## Backward-Compatible Aliases

- Health aliases: `/health` and `/healthz` are equivalent.
- Readiness aliases: `/ready` and `/readyz` are equivalent.
- Recommendation aliases: `/recommendations` and `/v1/recommend` use the same request normalization and response schema.

## Request Schema Compatibility

The recommendation request accepts both snake_case and camelCase aliases:

- `user_id` or `userId`
- `user_idx` or `userIdx`
- `k` or `top_k`
- `exclude_seen` or `excludeSeen`
- `include_debug` or `includeDebug`
- `allow_cold_start` or `allowColdStart`
- `candidate_top_k` or `candidateTopK`

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
      "scorer_policy": "ranker_topk_popularity_backfill"
    }
  ],
  "debug": null
}
```

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

Run full contract validation against a live server:

```powershell
uv run python scripts/validate_serving_api.py --base-url http://127.0.0.1:8000 --known-user-idx 0 --k 10
```

Outputs:

- pass/fail table per check
- per-endpoint latencies
- JSON report at `artifacts/reports/serving_api_validation.json`

## Local Latency Benchmark Script

Run sequential latency benchmark:

```powershell
uv run python scripts/benchmark_serving_api.py --base-url http://127.0.0.1:8000 --num-users 50 --k 10
```

Outputs:

- `p50`, `p95`, `max` latency
- `success_count`, `failure_count`
- JSON report at `artifacts/reports/serving_api_latency.json`

## Known Limitations

- no Ollama explanations yet
- no Docker packaging yet
- no cloud deployment yet
- no authentication
- local model/data artifacts are required
