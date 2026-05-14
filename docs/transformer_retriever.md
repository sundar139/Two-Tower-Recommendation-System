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
uv run python scripts/diagnose_transformer_retriever.py --config configs/transformer_retrieval.yaml --sample
uv run python scripts/run_transformer_ablation.py --sample
uv run python scripts/train_retriever.py --config configs/transformer_retrieval_stable.yaml --sample --model-type transformer
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split test --sample
```

## Sample Ablation

```powershell
uv run python scripts/run_transformer_ablation.py --sample
uv run python scripts/compare_retrievers.py --sample
```

## Full Transformer Commands

```powershell
uv run python scripts/train_retriever.py --config configs/transformer_retrieval_stable.yaml --model-type transformer
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split val
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_stable.yaml --model transformer --split test
uv run python scripts/export_faiss_index.py --config configs/transformer_retrieval_stable.yaml --model-type transformer
```

Only run the full commands if sample validation beats the baseline and recall remains stable.

## Failure Analysis (Current)

Previous sample transformer run underperformed both popularity and baseline. Stabilization diagnostics show:

- no NaN/inf in logits or embeddings
- loss decreases over short-run steps (10 -> 100)
- healthy gradient flow into user/item/position/attention/FFN parameters

This indicates optimization and representation quality issues rather than a masking numerical failure.

## Stable Config Selection

Selected file: `configs/transformer_retrieval_stable.yaml`

Key settings:

- learning_rate: `2e-4`
- dropout: `0.05`
- pooling: `mean`
- layers: `1`
- heads: `2`
- scheduler: `warmup_cosine`
- weight_decay: `1e-6`
- epochs: `10`

## Latest Sample Metrics

| Split | Model | HR@10 | MRR@10 | NDCG@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| val | popularity | 0.047000 | 0.017380 | 0.024347 | 0.146000 |
| val | baseline | 0.055000 | 0.016221 | 0.025047 | 0.155000 |
| val | transformer (stable) | 0.037000 | 0.012628 | 0.018203 | 0.124000 |
| test | popularity | 0.039000 | 0.012310 | 0.018442 | 0.122000 |
| test | baseline | 0.035000 | 0.008891 | 0.014924 | 0.132000 |
| test | transformer (stable) | 0.021000 | 0.009515 | 0.012216 | 0.112000 |

## MLflow Run Links

- baseline sample train: `http://127.0.0.1:5000/#/experiments/1/runs/e9bf170a6e324b9ca7917d994cd939f5`
- stable transformer sample train: `http://127.0.0.1:5000/#/experiments/1/runs/4940a9671a354a8fa27c8c9f4cf809fa`
- stable transformer sample val eval: `http://127.0.0.1:5000/#/experiments/1/runs/6b0778d71388441fb1ebd40bf6cead3b`
- stable transformer sample test eval: `http://127.0.0.1:5000/#/experiments/1/runs/8cfcb807deda4aacbedab923e4910775`
- best ablation trial: `http://127.0.0.1:5000/#/experiments/1/runs/745306f30a1145ef97e7f4bd716a0b68`

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
