"""Export FAISS flat IP index from trained two-tower checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import mlflow
import numpy as np
import polars as pl
import torch
import typer

from movie_recsys.modeling.artifacts import load_checkpoint, save_json
from movie_recsys.modeling.datasets import load_feature_tables
from movie_recsys.modeling.faiss_index import (
    build_flat_ip_index,
    load_faiss_bundle,
    save_faiss_bundle,
    search_index,
)
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import load_retrieval_config
from movie_recsys.training.mlflow_utils import (
    configure_mlflow,
    log_artifacts,
    log_metrics,
    log_training_params,
    print_mlflow_run_summary,
    set_retrieval_tags,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
    sample: bool = typer.Option(False, "--sample"),
    model_type: Literal["baseline", "transformer"] | None = typer.Option(None, "--model-type"),
) -> None:
    cfg = load_retrieval_config(config, sample=sample)
    normalized_model_type = model_type or cfg.model.model_type
    if normalized_model_type == "two_tower":
        normalized_model_type = "baseline"
    cfg.model.model_type = cast(Literal["baseline", "transformer"], normalized_model_type)
    experiment = configure_mlflow(cfg)
    tables = load_feature_tables(cfg)

    common_kwargs = {
        "config": cfg,
        "num_users": tables.user_features.shape[0],
        "num_items_with_padding": tables.item_features.shape[0] + 1,
        "user_feature_dim": tables.user_features.shape[1],
        "item_feature_dim": tables.item_features.shape[1],
    }
    if normalized_model_type == "transformer":
        model = TransformerRetriever(**common_kwargs)
    else:
        model = BaselineRetriever(**common_kwargs)

    checkpoint_path = checkpoint or (
        cfg.paths.model_output_dir / f"best_{normalized_model_type}_retriever.pt"
    )
    ckpt = load_checkpoint(checkpoint_path)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        item_features = torch.tensor(tables.item_features, dtype=torch.float32, device=device)
        item_idx_shifted = torch.arange(
            1, tables.item_features.shape[0] + 1, dtype=torch.long, device=device
        )
        item_emb = (
            model.item_tower(item_idx_shifted, item_features).cpu().numpy().astype(np.float32)
        )

    index = build_flat_ip_index(item_emb)
    item_indices = np.arange(tables.item_features.shape[0], dtype=np.int64)
    paths = save_faiss_bundle(
        index, item_indices, cfg.paths.index_output_dir, embedding_dim=item_emb.shape[1]
    )

    train_df = pl.read_parquet(cfg.train_path)
    first_user = int(train_df.get_column("user_idx")[0])
    users_df = pl.read_parquet(cfg.users_path)
    row = users_df.filter(pl.col("user_idx") == first_user)
    history = row.get_column("train_history_item_idx").to_list()[0] or []
    history = [int(v) + 1 for v in history[-cfg.train.history_length :]]

    history_tensor = torch.zeros((1, cfg.train.history_length), dtype=torch.long, device=device)
    history_mask = torch.zeros((1, cfg.train.history_length), dtype=torch.bool, device=device)
    if history:
        history_tensor[0, -len(history) :] = torch.tensor(history, dtype=torch.long, device=device)
        history_mask[0, -len(history) :] = True

    with torch.no_grad():
        user_features = np.asarray(tables.user_features[first_user], dtype=np.float32)
        user_emb = (
            model.user_tower(
                torch.tensor([first_user], dtype=torch.long, device=device),
                history_tensor,
                history_mask,
                torch.from_numpy(user_features).unsqueeze(0).to(device),
            )
            .cpu()
            .numpy()
        )

    reloaded_index, reloaded_mapping = load_faiss_bundle(cfg.paths.index_output_dir)
    top_items, _scores, latency = search_index(reloaded_index, user_emb, reloaded_mapping, 200)
    brute = (item_emb @ user_emb[0]).argsort()[::-1][:10]
    faiss_top10 = top_items[0][:10]

    payload = {
        "model_type": normalized_model_type,
        "faiss_top10": [int(x) for x in faiss_top10],
        "brute_top10": [int(x) for x in brute],
        "top10_match": [int(x) for x in faiss_top10] == [int(x) for x in brute],
        "top200_latency_ms": latency,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    report_path = cfg.paths.report_output_dir / f"faiss_export_report_{normalized_model_type}.json"
    save_json(report_path, payload)

    with mlflow.start_run(run_name=f"faiss_export_{normalized_model_type}") as run:
        set_retrieval_tags(model_type=normalized_model_type, split="val", sample=sample)
        log_training_params(cfg)
        log_metrics(
            {
                "faiss_top200_latency_ms": float(latency),
                "faiss_top10_match": 1.0 if payload["top10_match"] else 0.0,
            }
        )
        mlflow.log_param("faiss_artifact_path", str(cfg.paths.index_output_dir))
        mlflow.log_param("faiss_model_type", normalized_model_type)
        log_artifacts(
            [
                Path(paths["index"]),
                Path(paths["mapping"]),
                Path(paths["metadata"]),
                report_path,
            ]
        )
        print_mlflow_run_summary(
            config=cfg,
            run=run,
            experiment_id=experiment.experiment_id,
        )

    typer.echo(payload)


if __name__ == "__main__":
    app()
