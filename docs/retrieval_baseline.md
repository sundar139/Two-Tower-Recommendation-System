# Plain Two-Tower Retrieval Baseline

## Why Start With A Plain Baseline

The first modeling baseline is intentionally simple and deterministic:

- popularity baseline for a non-neural reference point
- plain two-tower retrieval model
- exact inner-product retrieval with FAISS `IndexFlatIP`

This gives a stable quality floor before introducing transformer sequence encoding or contrastive learning.

## Popularity Baseline

The popularity baseline ranks items by `positive_count` (or `popularity_score` when needed), then filters items already seen by the user in training history.

It is evaluated with:

- HR@10
- MRR@10
- NDCG@10
- Recall@50

## Two-Tower Architecture

### User Tower

Inputs:

- learned `user_id` embedding
- mean pooled embedding of last-50 positive history items
- dense/static user feature vector through an MLP

The concatenated representation is projected to a shared embedding space and L2-normalized.

### Item Tower

Inputs:

- learned `item_id` embedding
- structured item feature vector through an MLP

The concatenated representation is projected to the same shared embedding space and L2-normalized.

### Retrieval Score

User and item vectors use dot-product similarity in the shared space:

$score(u, i) = u^\top i$

## In-Batch Negatives

Training uses an InfoNCE-style in-batch cross-entropy objective:

- batch user embeddings $U \in \mathbb{R}^{B \times d}$
- batch item embeddings $V \in \mathbb{R}^{B \times d}$
- logits $L = \frac{U V^\top}{\tau}$
- labels are diagonal targets $\{0, 1, ..., B-1\}$

Each non-matching pair in the batch acts as a negative sample.

## Why FAISS Flat IP First

`IndexFlatIP` is exact (not approximate), so baseline retrieval quality can be validated without ANN approximation effects.

Later steps can move to IVF/HNSW/PQ variants once baseline behavior is validated.

## Evaluation Metrics

- HR@10: whether at least one ground-truth item is in top-10.
- MRR@10: reciprocal rank of the first relevant item in top-10.
- NDCG@10: discounted gain normalized by ideal ranking.
- Recall@50: fraction of relevant items recovered in top-50.

## Current Limitations

Not included in this baseline:

- transformer sequence encoder
- CL-EPIDTN contrastive learning objective
- neural reranker
- online serving APIs

## Overfit Diagnostic Note

The overfit smoke test checks whether loss clearly trends down over the first ~100 updates.
It does not require strictly monotonic per-step decrease, because mini-batch optimization is noisy
and short-term increases are normal. Diagnostic fields include both raw and smoothed trend signals.

## Step 2 Validation Commands

### Sample Validation Commands

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split val --sample
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split test --sample
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split test --sample
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml --sample
git status --short
git ls-files data artifacts mlruns models .venv
```

### Full-Data Validation Commands

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split val
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model popularity --split test
uv run python scripts/train_retriever.py --config configs/retrieval.yaml
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split val
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model two_tower --split test
uv run python scripts/export_faiss_index.py --config configs/retrieval.yaml
git status --short
git ls-files data artifacts mlruns models .venv
```
