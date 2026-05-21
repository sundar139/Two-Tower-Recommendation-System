# System Architecture

## End-to-End Overview

```mermaid
flowchart LR
    A[MovieLens-25M Raw Data] --> B[Data Pipeline]
    B --> C[Residual Transformer Retriever]
    C --> D[FAISS Index and Top-200 Candidates]
    D --> E[Neural Ranker]
    E --> F[Recall-Constrained Production Scorer]
    F --> G[FastAPI Service]
    G --> H[Optional Ollama Explanations]
```

The approved production path is residual retrieval plus neural ranking plus recall-constrained scorer selection.
The CL retriever branch remains experimental and is not in the production serving path.

## Training Architecture

```mermaid
flowchart TB
    A[prepare_data.py] --> B[Train Plain Two-Tower]
    B --> C[Train Residual Transformer Retriever]
    C --> D[Evaluate Retriever and Export FAISS]
    D --> E[Generate Top-200 Ranker Candidates]
    E --> F[Build Ranker Features]
    F --> G[Train Neural Ranker]
    G --> H[Evaluate Ranker vs Residual and Popularity]
    H --> I[Select Production Scorer under Recall Guard]
```

Key training outputs:

- Retriever checkpoint: `artifacts/models/best_residual_transformer_retriever.pt`
- FAISS bundle: `artifacts/faiss/index.faiss`, metadata, mapping parquet
- Ranker checkpoint: `artifacts/models/best_neural_ranker.pt`
- Scorer selection report: `artifacts/reports/production_scorer_selection.json`

## Serving Architecture

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant R as Residual Retriever
    participant F as FAISS
    participant N as Neural Ranker
    participant S as Production Scorer
    participant O as Ollama (optional)

    Client->>API: POST /recommendations
    API->>R: Encode user
    R->>F: Retrieve top-200
    F-->>API: Candidate ids and residual scores
    API->>N: Score candidates
    N-->>API: Ranker scores
    API->>S: Apply ranker_topk_popularity_backfill
    S-->>API: Final top-k with policy metadata
    opt include_explanations=true
        API->>O: Generate item-level explanations
        O-->>API: explanation text
    end
    API-->>Client: Recommendations response
```

## Artifact Flow

```mermaid
flowchart LR
    A[Training Runs] --> B[artifacts/models]
    A --> C[artifacts/faiss]
    A --> D[artifacts/reports]
    B --> E[Serving Artifact Registry]
    C --> E
    E --> F[API Runtime]
    D --> G[Portfolio Evidence Docs]
```

## Production Scorer Logic

Selected policy:

- `ranker_topk_popularity_backfill`
- `alpha=1.0`
- `beta=0.1`
- `gamma=0.0`
- `top_k_focus=20`

Decision rule:

1. Compute hybrid score for primary ranking window with ranker and popularity weights.
2. Use popularity backfill beyond the focus window to protect recall behavior.
3. Select using validation NDCG@10 only, while enforcing `Recall@50 >= 0.95 * popularity Recall@50`.

## Explanation Layer Logic

Explanations are post-processing metadata only:

- Retrieval, ranking, and scorer ordering are completed first.
- Explanation generation can be requested per call.
- If Ollama is unavailable and fail-open is enabled, recommendations are still returned with `explanation_status=unavailable`.
- Explanations do not mutate scores or reorder the ranked list.

## Mermaid Validation Note

The Mermaid blocks in this document use standard GitHub Markdown-supported syntax (`flowchart`, `sequenceDiagram`) and are intended to render directly in repository viewers that support Mermaid.
