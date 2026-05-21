# Project Log

1. Built the data pipeline for MovieLens-25M ingestion, preprocessing, and leakage-safe chronological splitting.
2. Implemented the plain two-tower retriever baseline and validated it against popularity.
3. First transformer retriever attempts underperformed and were held back after diagnostics.
4. Stabilized the retrieval path with a residual/gated transformer and passed acceptance checks.
5. Implemented CL contrastive residual retrieval experiments; kept the branch experimental after acceptance did not pass.
6. Added the neural ranker on top of residual top-200 candidates and validated ranker improvements versus residual retrieval.
7. Ran initial production scorer selection; first winner failed the recall guard against popularity.
8. Added recall-constrained scorer selection and approved `ranker_topk_popularity_backfill` with guard-compliant recall.
9. Implemented FastAPI local serving and validated API behavior, schema parity, and latency.
10. Added Docker local packaging with artifact preflight and smoke validation.
11. Added optional local Ollama explanations and validated fail-open behavior and ranking invariance.
12. Completed final documentation polish for architecture, evidence, reproducibility, and portfolio presentation.
