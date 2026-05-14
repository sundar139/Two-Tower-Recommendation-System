"""Overfit smoke test for plain two-tower retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader, Subset

from movie_recsys.modeling.datasets import (
    RetrievalDataset,
    collate_retrieval_batch,
    load_feature_tables,
)
from movie_recsys.modeling.losses import InBatchCrossEntropyLoss
from movie_recsys.modeling.retrieval import TwoTowerRetriever
from movie_recsys.training.config import load_retrieval_config
from movie_recsys.utils.reproducibility import set_global_seed

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config"),
    sample_size: int = typer.Option(1000, "--sample-size"),
    steps: int = typer.Option(100, "--steps"),
) -> None:
    cfg = load_retrieval_config(config, sample=True)
    set_global_seed(cfg.train.random_seed)

    tables = load_feature_tables(cfg)
    dataset = RetrievalDataset(str(cfg.train_path), tables, history_length=cfg.train.history_length)

    if sample_size < len(dataset):
        indices = list(range(sample_size))
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    loader = DataLoader(
        subset,
        batch_size=min(cfg.train.train_batch_size, 128),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda rows: collate_retrieval_batch(
            rows, history_length=cfg.train.history_length
        ),
        generator=torch.Generator().manual_seed(cfg.train.random_seed),
    )

    model = TwoTowerRetriever(
        config=cfg,
        num_users=tables.user_features.shape[0],
        num_items_with_padding=tables.item_features.shape[0] + 1,
        user_feature_dim=tables.user_features.shape[1],
        item_feature_dim=tables.item_features.shape[1],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    criterion = InBatchCrossEntropyLoss()

    losses: list[float] = []
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = criterion(output["logits"])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            step += 1

    initial_loss = losses[0] if losses else float("nan")
    final_loss = losses[-1] if losses else float("nan")
    best_loss = min(losses) if losses else float("nan")
    non_decreasing_steps = sum(1 for i in range(1, len(losses)) if losses[i] >= losses[i - 1])
    percent_loss_reduction = (
        ((initial_loss - final_loss) / initial_loss) * 100.0
        if losses and np.isfinite(initial_loss) and initial_loss > 0
        else float("nan")
    )

    rolling_window = min(10, len(losses))
    if rolling_window > 1:
        rolled = np.convolve(
            np.array(losses, dtype=np.float64),
            np.ones(rolling_window),
            mode="valid",
        )
        rolled = rolled / rolling_window
        smoothed_loss_decreased = bool(rolled[-1] < rolled[0])
    else:
        smoothed_loss_decreased = bool(final_loss < initial_loss)

    payload = {
        "steps": len(losses),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "loss_decreased": bool(final_loss < initial_loss),
        "percent_loss_reduction": percent_loss_reduction,
        "number_of_non_decreasing_steps": non_decreasing_steps,
        "smoothed_loss_decreased": smoothed_loss_decreased,
    }
    typer.echo(payload)


if __name__ == "__main__":
    app()
