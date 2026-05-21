# GitHub Profile Summary

## 1) Short Resume Bullet Version

- Built an end-to-end MovieLens-25M recommender system with residual transformer retrieval, neural ranking, and recall-constrained production scoring.
- Improved production-quality offline ranking over popularity (`NDCG@10`: `0.311523` val, `0.317591` test) while preserving/improving `Recall@50` (`0.729658` val, `0.712374` test).
- Productized the stack with FastAPI serving, Docker local packaging, automated validation, and optional fail-open local Ollama explanations.

## 2) Long Project Description

This project implements a production-style two-stage recommendation system on MovieLens-25M. Candidate retrieval is handled by an approved residual transformer retriever exported to FAISS for fast top-200 lookup. A neural ranker then reorders candidates, and a recall-constrained scorer policy (`ranker_topk_popularity_backfill`) combines ranker and popularity signals while protecting recall behavior against a strong popularity baseline.

The project includes reproducible training/evaluation workflows, MLflow tracking with a local SQLite backend, and robust validation reports. The serving layer is implemented with FastAPI and includes deterministic response behavior, cold-start fallback, and strict schema checks. Docker packaging supports local deployment by mounting approved model/index artifacts. Optional local Ollama explanations are integrated as post-processing metadata with fail-open behavior so recommendation ranking remains unchanged if explanation generation is unavailable.

## 3) STAR/CAR Format Description

### Situation

A realistic recommendation system needed to move beyond model experimentation and demonstrate production-minded decision criteria against a strong popularity baseline.

### Task

Design and validate a retrieval-ranking-serving pipeline that could improve ranking quality while preserving recall safety and operational reproducibility.

### Action

- Built and validated a residual transformer retrieval backbone.
- Implemented full-data neural ranking with deterministic candidate generation and leakage checks.
- Added recall-constrained scorer selection and rejected earlier candidates that failed recall guardrails.
- Deployed local FastAPI and Docker workflows with automated validation scripts.
- Integrated optional local Ollama explanations with ranking-invariance guarantees.

### Result

- Selected scorer outperformed popularity on NDCG@10 (`0.311523` val, `0.317591` test).
- Recall guard preserved/improved Recall@50 (`0.729658` val, `0.712374` test vs popularity `0.711440` val, `0.679510` test).
- API validation passed (`18/18`), Docker smoke passed (`6/6`), and Ollama explanation validation passed (`9/9`).

## 4) LinkedIn Post Draft

I just wrapped an end-to-end recommender systems project on MovieLens-25M focused on production decision quality, not just model novelty.

Highlights:

- Built a two-stage architecture: residual transformer retrieval -> neural ranking -> recall-constrained production scorer.
- Benchmarked against popularity (strong baseline) and enforced a Recall@50 guard before promotion.
- Final selected scorer (`ranker_topk_popularity_backfill`) achieved `NDCG@10=0.311523` (val) and `0.317591` (test), with Recall@50 preserved/improved versus popularity.
- Productionized locally with FastAPI, Docker packaging, validation scripts, and optional fail-open Ollama explanations.

What I liked most: the project forced practical tradeoff decisions between ranking lift and retrieval safety, which is where recommendation systems become engineering work rather than leaderboard work.

## 5) Interview Explanation

### Problem

Recommend relevant movies at scale while keeping retrieval quality stable and serving behavior reproducible.

### Architecture

Two-stage pipeline:

1. Residual transformer retriever generates FAISS top-200 candidates.
2. Neural ranker scores candidates for top-k quality.
3. Recall-constrained scorer combines ranker and popularity.
4. FastAPI serves recommendations, with optional explanation layer.

### Modeling Choices

- Kept popularity as a first-class baseline.
- Promoted residual transformer over transformer-only and CL branches based on acceptance outcomes.
- Kept CL retriever experimental because acceptance did not pass.
- Chose scorer policy only after enforcing Recall@50 guard against popularity.

### Evaluation

- Offline metrics: HR@10, MRR@10, NDCG@10, Recall@50.
- Split-aware reporting for validation and test.
- Automated acceptance checks for leakage, determinism, finite scores, and recall constraints.

### Productionization

- FastAPI endpoints with schema and behavior validation.
- Local Docker packaging with artifact preflight.
- MLflow tracking with reproducible local backend.
- Optional Ollama explanations with fail-open behavior and ranking invariance.

### Tradeoffs

- Highest NDCG candidate from an earlier scorer sweep was rejected due to recall guard failure.
- Explanations are optional and slower due to local LLM generation.
- Docker is currently local artifact-mounted rather than cloud-native model distribution.

## 6) ATS Keywords

- recommender systems
- two-tower retrieval
- residual transformer
- candidate generation
- neural ranker
- learning-to-rank
- FAISS
- FastAPI
- Docker
- MLflow
- model evaluation
- offline metrics
- NDCG
- Recall@50
- production scorer
- ranking guardrails
- experiment tracking
- Python
- PyTorch
- feature engineering
