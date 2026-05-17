# Contrastive Residual Retriever

## Overview

The contrastive retriever extends the residual transformer model with CL-EPIDTN-style auxiliary
losses while preserving baseline retrieval behavior.

Model type:

- `cl_residual_transformer`

Config:

- `configs/cl_retrieval.yaml`

## Objectives

Total loss combines retrieval and auxiliary contrastive losses:

$$
\mathcal{L}_{total} = \mathcal{L}_{retrieval}
+ \lambda_u \mathcal{L}_{user}
+ \lambda_i \mathcal{L}_{item}
+ \lambda_a \mathcal{L}_{align}
$$

Components:

- retrieval loss: in-batch cross-entropy over user/item logits
- user CL loss: symmetric InfoNCE over two augmented history views
- item CL loss: symmetric InfoNCE between item-id and item-feature views
- alignment CL loss: optional symmetric InfoNCE between user and item embeddings

## Sequence Augmentations

User history augmentations are applied per sample to generate two views:

- token masking
- token dropout
- random crop with minimum ratio
- local reorder in a small window

Augmentations preserve `[B, L]` shapes and masks and avoid collapsing non-empty histories to
all-empty views.

## Initialization Rule

CL training is expected to initialize from a residual checkpoint:

```powershell
uv run python scripts/train_retriever.py --config configs/cl_retrieval.yaml --sample --model-type cl_residual_transformer --init-from-residual artifacts/models/best_residual_transformer_retriever.pt
```

Random initialization is blocked by default unless `--allow-random-init` is explicitly set.

## Sample Workflow

```powershell
uv run python scripts/run_contrastive_ablation.py --sample
uv run python scripts/evaluate_retriever.py --config configs/cl_retrieval.yaml --model cl_residual_transformer --split val --sample
uv run python scripts/evaluate_retriever.py --config configs/cl_retrieval.yaml --model cl_residual_transformer --split test --sample
uv run python scripts/export_faiss_index.py --config configs/cl_retrieval.yaml --sample --model-type cl_residual_transformer
uv run python scripts/check_contrastive_acceptance.py --summary artifacts/reports/contrastive_ablation_sample.json --sample
```

## Acceptance Gate

`scripts/check_contrastive_acceptance.py` reports `full_data_cl_allowed`.

Full-data CL should run only when this value is `true`.

## Reports

Generated artifacts:

- `artifacts/reports/contrastive_ablation_sample.json`
- `artifacts/reports/contrastive_ablation_sample.md`
- `artifacts/reports/contrastive_acceptance_sample.json`
- `artifacts/reports/contrastive_acceptance_sample.md`

## Stabilization Decision (Latest)

First attempt:

- The initial broad CL sweep was not accepted as a stable promotion candidate.

Second focused sweep (6 trials):

- best trial: `focused_proj_warm_anchor_u050_i020_t007_a001`
- best val metrics:
	- `hr@10`: `0.044000`
	- `mrr@10`: `0.017020`
	- `ndcg@10`: `0.023174`
	- `recall@50`: `0.158000`

Acceptance outcome:

- `acceptance_passed: false`
- `full_data_cl_allowed: false`
- failed reason: no primary acceptance rule passed
- secondary safety checks passed (FAISS parity, recall collapse guard, finite loss checks)

Current policy:

- CL remains experimental.
- Residual transformer remains the production retrieval backbone.
- Full-data CL is blocked until a future sample run passes acceptance.
- Downstream ranker experiments should consume residual-transformer retrieval artifacts.
