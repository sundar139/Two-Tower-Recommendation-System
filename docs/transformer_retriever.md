# Transformer Retriever Workflow

## Architecture

The project now supports baseline, transformer, residual transformer, and CL residual transformer
retrievers while keeping the same item tower and in-batch retrieval objective.

It now also supports a residual transformer variant that starts from baseline behavior and adds gated transformer refinement.
It also supports CL residual training with user/item contrastive losses on top of the residual model.

## Baseline Versus Transformer

- Baseline retriever user tower: mean pooling over history item embeddings.
- Transformer retriever user tower: custom pre-LN transformer over history item embeddings.

Shared properties:

- user ID embedding
- static user feature MLP
- item ID embedding + item feature MLP
- final projection to a shared embedding space
- L2-normalized user/item vectors

Residual transformer specific properties:

- user sequence is blended as `baseline_context + sigmoid(gate) * (transformer_context - baseline_context)`
- gate is initialized near baseline (`initial_transformer_gate: -2.944`, alpha ~0.05)
- model can be initialized from a baseline checkpoint (`--init-from-baseline`)

CL residual specific properties:

- model type: `cl_residual_transformer`
- initialization from residual checkpoint (`--init-from-residual`)
- user two-view contrastive loss via sequence augmentations
- item contrastive loss between item-id and item-feature views
- optional user/item alignment loss
- total loss: retrieval + weighted auxiliary contrastive terms

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

## Sample Residual Transformer

```powershell
uv run python scripts/train_retriever.py --config configs/transformer_retrieval_residual.yaml --sample --model-type residual_transformer --init-from-baseline artifacts/models/best_baseline_retriever.pt
uv run python scripts/run_residual_transformer_ablation.py --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_residual.yaml --model residual_transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/transformer_retrieval_residual.yaml --model residual_transformer --split test --sample
uv run python scripts/compare_retrievers.py --sample
```

## Sample CL Residual Transformer

```powershell
uv run python scripts/run_contrastive_ablation.py --sample
uv run python scripts/train_retriever.py --config configs/cl_retrieval.yaml --sample --model-type cl_residual_transformer --init-from-residual artifacts/models/best_residual_transformer_retriever.pt
uv run python scripts/evaluate_retriever.py --config configs/cl_retrieval.yaml --model cl_residual_transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/cl_retrieval.yaml --model cl_residual_transformer --split test --sample
uv run python scripts/export_faiss_index.py --config configs/cl_retrieval.yaml --sample --model-type cl_residual_transformer
uv run python scripts/check_contrastive_acceptance.py --summary artifacts/reports/contrastive_ablation_sample.json --sample
```

Only run full-data CL after sample acceptance passes (`full_data_cl_allowed: true`).

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

## Full Residual Commands

```powershell
uv run python scripts/run_full_residual_training.py --max-runtime-hours 4
```

Resume full residual training:

```powershell
uv run python scripts/run_full_residual_training.py --resume-from artifacts/models/checkpoints/residual_transformer_epoch_3.pt --max-runtime-hours 4
```

Evaluation-only full residual workflow:

```powershell
uv run python scripts/run_full_residual_training.py --evaluate-only
```

Acceptance checker:

```powershell
uv run python scripts/check_residual_acceptance.py --summary artifacts/reports/full_residual_transformer_summary.json
```

CL-EPIDTN remains blocked until `scripts/check_contrastive_acceptance.py` reports `full_data_cl_allowed: true`.

## Failure Analysis (Current)

Previous sample transformer run underperformed both popularity and baseline. Stabilization diagnostics show:

- `raw_history_length_before_truncation`: `79.131`
- `model_history_length_after_truncation`: `34.057`
- `valid_tokens_seen_by_transformer`: `34.057`

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
| val | baseline | 0.050000 | 0.014749 | 0.022808 | 0.150000 |
| val | transformer (stable) | 0.036000 | 0.010375 | 0.016133 | 0.130000 |
| val | residual transformer | 0.055000 | 0.018227 | 0.026668 | 0.156000 |
| test | popularity | 0.039000 | 0.012310 | 0.018442 | 0.122000 |
| test | baseline | 0.028000 | 0.008383 | 0.012850 | 0.117000 |
| test | transformer (stable) | 0.022000 | 0.008679 | 0.011733 | 0.100000 |
| test | residual transformer | 0.035000 | 0.007512 | 0.013695 | 0.130000 |

## MLflow Run Links

- baseline sample train: `http://127.0.0.1:5000/#/experiments/1/runs/300602b1ac22462d957c9ab180df903c`
- stable transformer sample train: `http://127.0.0.1:5000/#/experiments/1/runs/181a2151033041d4aa4ec54d1d1c15bc`
- residual sample train: `http://127.0.0.1:5000/#/experiments/1/runs/d753cc7d2cfd477786284ae464cde99c`
- best residual ablation trial: `http://127.0.0.1:5000/#/experiments/1/runs/a23e7bfa2e4b4273abad6acbd7925109`
- residual sample val eval: `http://127.0.0.1:5000/#/experiments/1/runs/2015307a14b14e9a922269be082091aa`
- residual sample test eval: `http://127.0.0.1:5000/#/experiments/1/runs/f77c96a712f04439b5637f72f95dda32`

## Metrics Table (Fill After Runs)

| Split | Model | HR@10 | MRR@10 | NDCG@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| val | popularity | - | - | - | - |
| val | baseline | - | - | - | - |
| val | transformer | - | - | - | - |
| val | residual_transformer | - | - | - | - |
| test | popularity | - | - | - | - |
| test | baseline | - | - | - | - |
| test | transformer | - | - | - | - |
| test | residual_transformer | - | - | - | - |

## Contrastive Decision Snapshot

- first CL attempt was not promoted
- second focused CL sweep best trial: `focused_proj_warm_anchor_u050_i020_t007_a001`
- sample acceptance result: `acceptance_passed: false`
- sample acceptance result: `full_data_cl_allowed: false`
- CL remains experimental
- residual transformer remains the production retrieval backbone
- ranker work should continue with residual-transformer artifacts

## Current Limitations

- no neural ranker yet
- no FastAPI API layer yet
- no Ollama explanation endpoints yet
