# Transformer Retriever Workflow

## Architecture

The project now supports two user-sequence encoders while keeping the same item tower and in-batch retrieval loss.

## Baseline Versus Transformer

- Baseline retriever user tower: mean pooling over history item embeddings.
- Transformer retriever user tower: custom pre-LN transformer over history item embeddings.

Shared properties:

- user ID embedding
- static user feature MLP
- item ID embedding + item feature MLP
- final projection to a shared embedding space
- L2-normalized user/item vectors

## Transformer Sequence Encoder Details

- Causal attention mask prevents attending to future positions.
- Padding mask prevents attention to padded history positions.
- Attention backend: `torch.nn.functional.scaled_dot_product_attention`.
- Transformer block style: Pre-LN with residual connections.
- Feed-forward block: GELU + dropout.
- Sequence pooling options:
  - `last`: last valid hidden state
  - `mean`: masked mean pooling over valid positions

## Commands

## Environment Verification

```powershell
Copy-Item env.example .env -Force
uv run python verify.py
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Start MLflow UI

```powershell
uvx mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

or

```powershell
uv run python scripts/start_mlflow_ui.py
```

## Sample Baseline

```powershell
uv run python scripts/train_retriever.py --config configs/retrieval.yaml --sample --model-type baseline
uv run python scripts/evaluate_retriever.py --config configs/retrieval.yaml --model baseline --split val --sample
```

## Sample Transformer

```powershell
uv run python scripts/train_retriever.py --config configs/transformer_retrieval.yaml --sample --model-type transformer
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval.yaml --model transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval.yaml --model transformer --split test --sample
```

## Sample Ablation

```powershell
uv run python scripts/compare_retrievers.py --sample
```

## Full Transformer Commands

```powershell
uv run python scripts/train_retriever.py --config configs/transformer_retrieval.yaml --model-type transformer
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval.yaml --model transformer --split val
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval.yaml --model transformer --split test
uv run python scripts/export_faiss_index.py --config configs/transformer_retrieval.yaml --model-type transformer
```

## Metrics Table (Fill After Runs)

| Split | Model | HR@10 | MRR@10 | NDCG@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| val | popularity | - | - | - | - |
| val | baseline | - | - | - | - |
| val | transformer | - | - | - | - |
| test | popularity | - | - | - | - |
| test | baseline | - | - | - | - |
| test | transformer | - | - | - | - |

## Current Limitations

- no contrastive learning objective yet
- no neural ranker yet
- no FastAPI API layer yet
- no Ollama explanation endpoints yet
