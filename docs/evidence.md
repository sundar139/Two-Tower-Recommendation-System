# Evidence and Validation

## Quality Gate Summary

| Gate | Result |
|---|---|
| `verify.py` | pass=10, warn=0, fail=0 |
| `ruff check .` | passed |
| `mypy src` | passed |
| `pytest -q` | passed |

## Retrieval Metrics (Full Data)

Source reports:

- `artifacts/reports/two_tower_full_summary.json`
- `artifacts/reports/residual_transformer_full_summary.json`

| Model | Split | HR@10 | MRR@10 | NDCG@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| Popularity | Validation | 0.050358 | 0.017927 | 0.025420 | 0.152007 |
| Two-Tower | Validation | 0.085761 | 0.031154 | 0.043782 | 0.221523 |
| Residual Transformer | Validation | 0.106902 | 0.040598 | 0.055954 | 0.266412 |
| Popularity | Test | 0.046261 | 0.016258 | 0.023188 | 0.139580 |
| Two-Tower | Test | 0.067099 | 0.023971 | 0.033933 | 0.190093 |
| Residual Transformer | Test | 0.081003 | 0.028951 | 0.040985 | 0.222907 |

## Ranker Metrics (Full Data)

Source reports:

- `artifacts/reports/ranker_eval_val.json`
- `artifacts/reports/ranker_eval_test.json`

| Model | Split | HR@10 | MRR@10 | NDCG@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| Popularity | Validation | 0.385364 | 0.230085 | 0.266328 | 0.711440 |
| Residual Retriever | Validation | 0.090569 | 0.031410 | 0.045040 | 0.248491 |
| Neural Ranker | Validation | 0.281478 | 0.150470 | 0.181151 | 0.532570 |
| Popularity | Test | 0.373635 | 0.228624 | 0.262527 | 0.679510 |
| Residual Retriever | Test | 0.067519 | 0.022010 | 0.032480 | 0.206234 |
| Neural Ranker | Test | 0.294949 | 0.154198 | 0.187112 | 0.575556 |

## Production Scorer Metrics (Approved Step 5D)

Source reports:

- `artifacts/reports/production_scorer_selection.json`
- `artifacts/reports/production_scorer_acceptance.json`

Selected policy:

- `ranker_topk_popularity_backfill`
- `alpha=1.0`, `beta=0.1`, `gamma=0.0`, `top_k_focus=20`

| Policy | Split | HR@10 | MRR@10 | NDCG@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| Popularity | Validation | 0.385364 | 0.230085 | 0.266328 | 0.711440 |
| Selected scorer | Validation | 0.435364 | 0.273246 | 0.311523 | 0.729658 |
| Popularity | Test | 0.373635 | 0.228624 | 0.262527 | 0.679510 |
| Selected scorer | Test | 0.447748 | 0.277436 | 0.317591 | 0.712374 |

Acceptance decision:

- `acceptance_passed=true`
- recall guard passed (`recall50_relative_drop_vs_popularity_le_5pct=true`)

## API Validation Evidence

Source reports:

- `artifacts/reports/serving_api_validation_step7b.json`
- `artifacts/reports/serving_api_validation.json`
- `artifacts/reports/serving_api_latency.json`

Summary:

- Core API contract validation: `18/18` passed.
- Extended validation run (including explanation aliases and parity checks): `26/26` passed.
- Latency benchmark (`50` successful requests):
  - `p50=29.70 ms`
  - `p95=35.05 ms`
  - `max=36.64 ms`

## Docker Validation Evidence

Source report:

- `artifacts/reports/docker_smoke_test.json`

Summary:

- Docker smoke checks passed: `6/6`
- No failed checks

## Ollama Explanation Validation Evidence

Source report:

- `artifacts/reports/ollama_explanation_validation.json`

Summary:

- Explanation validation checks passed: `9/9`
- `explanation_status=generated`
- `explanations_generated=3`
- `POST /recommendations include_explanations=true` latency: about `70.9s`
- `POST /v1/explain` latency: about `73.0s`
- Ranking invariance check passed

## MLflow Tracking Notes

- Tracking backend: SQLite (`mlflow.db`)
- Artifact root: local `mlruns/`
- All train/eval/export steps log run IDs and report links for reproducibility.
- Selection and acceptance artifacts are saved under `artifacts/reports/` and are linked in docs.

## Artifact Hygiene Notes

- Generated folders are ignored by Git: `artifacts/`, `mlruns/`, model checkpoints, local DB/cache outputs.
- `data/` keeps only `.gitkeep` sentinels under version control.
- `.env` and `mlflow.db` are ignored and validated through `git check-ignore`.
