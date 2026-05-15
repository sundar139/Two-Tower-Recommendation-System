"""Diagnostics for transformer retriever behavior on sample/full data."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer

from movie_recsys.modeling.datasets import (
    RetrievalDataset,
    load_feature_tables,
    make_retrieval_dataloader,
)
from movie_recsys.modeling.evaluator import evaluate_two_tower
from movie_recsys.modeling.losses import InBatchCrossEntropyLoss
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import load_retrieval_config

app = typer.Typer(add_completion=False)


def _history_stats(users_df: pl.DataFrame, *, history_length: int) -> dict[str, float]:
    histories = users_df.get_column("train_history_item_idx").to_list()
    lengths = np.asarray([len(v or []) for v in histories], dtype=np.float32)
    truncated_lengths = np.clip(lengths, 0.0, float(history_length))
    return {
        "raw_history_length_before_truncation": float(lengths.mean()) if lengths.size else 0.0,
        "model_history_length_after_truncation": (
            float(truncated_lengths.mean()) if truncated_lengths.size else 0.0
        ),
        "valid_tokens_seen_by_transformer": (
            float(truncated_lengths.mean()) if truncated_lengths.size else 0.0
        ),
        "pct_users_empty_history": float((lengths == 0).mean() * 100.0) if lengths.size else 0.0,
        "pct_users_history_len_1": float((lengths == 1).mean() * 100.0) if lengths.size else 0.0,
    }


def _attention_entropy_per_layer_head(
    attention_weights: list[torch.Tensor],
    history_mask: torch.Tensor,
) -> dict[str, float]:
    entropies: dict[str, float] = {}
    valid_queries = (
        history_mask.unsqueeze(1)
        .unsqueeze(-1)
        .expand(-1, attention_weights[0].shape[1], -1, 1)
    )

    for layer_idx, attn in enumerate(attention_weights):
        # attn: [B, H, L, L]
        probs = torch.clamp(attn, min=1e-12)
        entropy = -(probs * torch.log(probs)).sum(dim=-1, keepdim=True)
        masked_entropy = entropy * valid_queries.float()
        denom = valid_queries.float().sum().clamp_min(1.0)
        avg_entropy = float(masked_entropy.sum().item() / denom.item())
        entropies[f"layer_{layer_idx}_avg_attention_entropy"] = avg_entropy

        head_entropy = masked_entropy.sum(dim=(0, 2, 3)) / valid_queries.float().sum(
            dim=(0, 2, 3)
        ).clamp_min(1.0)
        for head_idx, value in enumerate(head_entropy.tolist()):
            entropies[f"layer_{layer_idx}_head_{head_idx}_entropy"] = float(value)

    return entropies


def _named_grad_norms(model: TransformerRetriever) -> dict[str, float]:
    groups = {
        "item_embeddings": ["item_tower.item_embedding"],
        "user_embeddings": ["user_tower.user_embedding"],
        "positional_embeddings": ["user_tower.sequence_encoder.position_embeddings"],
        "attention_projections": [
            "user_tower.sequence_encoder.blocks",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ],
        "ffn_parameters": ["user_tower.sequence_encoder.blocks", "ffn"],
    }

    norms: dict[str, float] = {}
    for label, patterns in groups.items():
        sq_sum = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if label in {"item_embeddings", "user_embeddings", "positional_embeddings"}:
                if any(token in name for token in patterns):
                    sq_sum += float((param.grad.detach().float().norm() ** 2).item())
            else:
                if all(token in name for token in patterns[:1]) and any(
                    token in name for token in patterns[1:]
                ):
                    sq_sum += float((param.grad.detach().float().norm() ** 2).item())
        norms[f"grad_norm_{label}"] = float(np.sqrt(max(sq_sum, 0.0)))

    return norms


@app.command()
def main(
    config: Path = typer.Option(Path("configs/transformer_retrieval.yaml"), "--config"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_retrieval_config(config, sample=sample)
    cfg.model.model_type = "transformer"

    feature_tables = load_feature_tables(cfg)
    train_ds = RetrievalDataset(
        str(cfg.train_path),
        feature_tables,
        history_length=cfg.train.history_length,
    )
    train_loader = make_retrieval_dataloader(
        train_ds,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        seed=cfg.train.random_seed,
    )

    model = TransformerRetriever(
        config=cfg,
        num_users=feature_tables.user_features.shape[0],
        num_items_with_padding=feature_tables.item_features.shape[0] + 1,
        user_feature_dim=feature_tables.user_features.shape[1],
        item_feature_dim=feature_tables.item_features.shape[1],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = InBatchCrossEntropyLoss()

    users_df = pl.read_parquet(cfg.users_path)
    train_df = pl.read_parquet(cfg.train_path)
    val_df = pl.read_parquet(cfg.val_path)
    history_stats = _history_stats(users_df, history_length=cfg.train.history_length)

    loss_at_steps: dict[int, float] = {}
    nan_or_inf_detected = False
    first_batch = None

    model.train()
    for step, batch in zip(range(1, 101), itertools.cycle(train_loader), strict=False):
        if first_batch is None:
            first_batch = batch

        device_batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(device_batch)

        for key in ["logits", "user_emb", "item_emb"]:
            if not torch.isfinite(output[key]).all():
                nan_or_inf_detected = True

        loss = criterion(output["logits"])
        if not torch.isfinite(loss):
            nan_or_inf_detected = True
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()

        if step in {10, 50, 100}:
            loss_at_steps[step] = float(loss.item())

    if first_batch is None:
        raise RuntimeError("No batches available for diagnostics")

    device_first_batch = {k: v.to(device) for k, v in first_batch.items()}
    model.eval()
    with torch.no_grad():
        user_emb, debug = model.encode_user_with_debug(device_first_batch)

    attention_weights = debug.get("attention_weights", [])
    if not isinstance(attention_weights, list):
        attention_weights = []

    attention_entropy = (
        _attention_entropy_per_layer_head(attention_weights, device_first_batch["history_mask"])
        if attention_weights
        else {}
    )

    # Fresh grad pass for named gradient norms.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    grad_out = model(device_first_batch)
    grad_loss = criterion(grad_out["logits"])
    grad_loss.backward()
    grad_norms = _named_grad_norms(model)

    model.eval()
    eval_result, _emb, _latency = evaluate_two_tower(
        model,
        train_df,
        val_df,
        users_df,
        feature_tables,
        history_length=cfg.train.history_length,
    )

    loss_decreasing = False
    if {10, 50, 100}.issubset(set(loss_at_steps.keys())):
        loss_decreasing = loss_at_steps[100] < loss_at_steps[10]

    report = {
        "sample": sample,
        "history_stats": history_stats,
        "attention_entropy": attention_entropy,
        "gradient_norms": grad_norms,
        "train_loss_at_steps": loss_at_steps,
        "short_run_val_ndcg@10": float(eval_result.metrics["ndcg@10"]),
        "loss_decreasing": loss_decreasing,
        "nan_or_inf_detected": nan_or_inf_detected,
        "user_embedding_norm_mean": float(torch.linalg.norm(user_emb, dim=1).mean().item()),
    }

    typer.echo(report)


if __name__ == "__main__":
    app()
