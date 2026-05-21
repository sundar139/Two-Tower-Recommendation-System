# Step 7B Ollama Recommendation Explanations

## 1. Scope

Step 7B adds an optional local explanation layer for recommendation responses.

In scope:

- post-hoc explanation generation for already-ranked recommendations
- explanation support on `POST /recommendations`, `POST /v1/recommend`, `POST /v1/explain`, and `POST /explanations/recommendations`
- fail-open handling when Ollama is unavailable

Out of scope:

- changing recommendation retrieval, ranking, scoring, or ordering
- cloud deployment
- frontend/UI
- Dockerized Ollama service

## 2. Installed Models

Required local Ollama models:

- `qwen3:4b`
- `qwen3-embedding:0.6b`

## 3. Environment Variables

Set in `env.example` (copy to `.env` for local runs):

- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `OLLAMA_CHAT_MODEL=qwen3:4b`
- `OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b`
- `OLLAMA_EXPLANATION_TIMEOUT_SECONDS=30`
- `OLLAMA_EXPLANATION_TEMPERATURE=0.2`
- `OLLAMA_EXPLANATION_MAX_ITEMS=10`
- `OLLAMA_EXPLANATION_ENABLED=true`

## 4. Start Ollama

```powershell
ollama serve
ollama list
```

`ollama list` should include both required models.

## 5. Start API

```powershell
uv run python scripts/run_api.py --config configs/serving.yaml --host 127.0.0.1 --port 8000
```

## 6. Request Explanations

Recommendation endpoint with explanations:

```json
{
  "user_id": 709,
  "k": 10,
  "include_explanations": true,
  "explanation_style": "concise",
  "max_explanation_items": 5
}
```

Dedicated explanation endpoint (recommend first, then explain):

```json
{
  "user_id": 709,
  "top_k": 10,
  "style": "concise",
  "max_explanation_items": 5
}
```

## 7. Example Response

```json
{
  "user_id": 709,
  "user_idx": 0,
  "k": 10,
  "cold_start": false,
  "scorer_policy": "ranker_topk_popularity_backfill",
  "explanation_status": "generated",
  "overall_explanation": "Top picks align with your recent Drama/Crime preference and strong ranker support.",
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
      "explanation": "Strong genre alignment with your recent history plus high combined ranking signals keeps this item near the top."
    }
  ]
}
```

## 8. Failure Behavior

`explanation_status` values:

- `disabled`: explanations were not requested or are globally disabled
- `generated`: explanations generated successfully
- `unavailable`: Ollama unavailable while fail-open preserved recommendations
- `failed`: explanation generation failed while fail-open preserved recommendations

If `fail_open=false`, explanation failures can return structured `503` responses.

## 9. Ranking Invariance Guarantee

Explanations are strictly post-processing metadata.

- no reordering of recommendation rows
- no filtering/removal of rows
- no score mutation
- no scorer policy changes

## 10. Validation

General serving validation:

```powershell
uv run python scripts/validate_serving_api.py --base-url http://127.0.0.1:8000 --known-user-idx 0 --k 10
```

Ollama explanation validation:

```powershell
uv run python scripts/validate_ollama_explanations.py --base-url http://127.0.0.1:8000 --ollama-url http://127.0.0.1:11434 --known-user-idx 0 --k 10
```

Generated reports:

- `artifacts/reports/serving_api_validation.json`
- `artifacts/reports/ollama_explanation_validation.json`

## 11. Known Limitations

- explanations run locally against host Ollama only
- explanations are approximate natural-language summaries, not guarantees
- no streaming explanation responses yet
- no frontend/UI yet
- no Dockerized Ollama sidecar/service in this step
