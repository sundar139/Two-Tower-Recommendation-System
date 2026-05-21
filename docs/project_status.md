# Project Status

## Step-by-Step Status

| Step | Scope | Status |
|---|---|---|
| Step 1 | Data pipeline and splits | Approved |
| Step 2 | Plain two-tower retriever baseline | Approved |
| Step 3 | Residual/gated transformer retriever | Approved (production retrieval backbone) |
| Step 4 | CL contrastive retriever | Implemented, experimental |
| Step 5D | Recall-constrained production scorer | Approved |
| Step 6 | FastAPI local serving | Approved |
| Step 7A | Docker local packaging | Approved |
| Step 7B | Ollama explanation endpoint | Approved |
| Step 8 | Final documentation and portfolio polish | In progress |

## Approved Components

- Residual transformer retriever for candidate retrieval.
- FAISS top-k candidate retrieval in serving path.
- Neural ranker second-stage scoring.
- Production scorer policy: `ranker_topk_popularity_backfill` (`alpha=1.0`, `beta=0.1`, `gamma=0.0`, `top_k_focus=20`).
- FastAPI serving contract and local benchmarking.
- Docker local packaging with mounted artifacts.
- Optional local Ollama explanation endpoints with fail-open behavior.

## Experimental Components

- CL residual retriever path (`cl_residual_transformer`) remains experimental and is not promoted.
- Additional scorer-policy experimentation beyond approved Step 5D selection is not required for current release.

## Future Work

- Online experimentation beyond offline MovieLens metrics.
- Cloud deployment and environment hardening.
- Authn/authz and API governance controls.
- UI product surface for recommendation experience.
- Monitoring for drift, latency regression, and explanation quality in long-running environments.
