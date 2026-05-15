"""Compare popularity, baseline, transformer, and residual retrievers on the same split(s)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
import torch
import typer

from movie_recsys.modeling.artifacts import load_checkpoint, save_json, save_markdown
from movie_recsys.modeling.datasets import load_feature_tables
from movie_recsys.modeling.evaluator import evaluate_popularity_baseline, evaluate_two_tower
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.modeling.retrieval import BaselineRetriever
from movie_recsys.modeling.transformer_retrieval import TransformerRetriever
from movie_recsys.training.config import RetrievalConfig, load_retrieval_config

app = typer.Typer(add_completion=False)


def _split_df(cfg: RetrievalConfig, split: str) -> pl.DataFrame:
    if split == "val":
        return pl.read_parquet(cfg.val_path)
    return pl.read_parquet(cfg.test_path)


def _build_model(
    cfg: RetrievalConfig,
    tables,
    model_type: Literal["baseline", "transformer", "residual_transformer"],
):
    common_kwargs = {
        "config": cfg,
        "num_users": tables.user_features.shape[0],
        "num_items_with_padding": tables.item_features.shape[0] + 1,
        "user_feature_dim": tables.user_features.shape[1],
        "item_feature_dim": tables.item_features.shape[1],
    }
    if model_type == "transformer":
        return TransformerRetriever(**common_kwargs)
    if model_type == "residual_transformer":
        return ResidualTransformerRetriever(**common_kwargs)
    return BaselineRetriever(**common_kwargs)


def _evaluate_model(
    cfg: RetrievalConfig,
    model_type: Literal["baseline", "transformer", "residual_transformer"],
    split: str,
    train_df: pl.DataFrame,
    split_df: pl.DataFrame,
    users_df: pl.DataFrame,
    tables,
) -> dict[str, float]:
    checkpoint = cfg.paths.model_output_dir / f"best_{model_type}_retriever.pt"
    model = _build_model(cfg, tables, model_type)
    model.load_state_dict(load_checkpoint(checkpoint)["model_state_dict"])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    result, _item_emb, _latency = evaluate_two_tower(
        model,
        train_df,
        split_df,
        users_df,
        tables,
        history_length=cfg.train.history_length,
    )
    return result.metrics


def _to_markdown(payload: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = []
    for split, values in payload.items():
        lines.append(f"## {split}")
        lines.append("| Model | HR@10 | MRR@10 | NDCG@10 | Recall@50 |")
        lines.append("|---|---:|---:|---:|---:|")
        for model_name in ["popularity", "baseline", "transformer", "residual_transformer"]:
            metrics = values[model_name]
            lines.append(
                f"| {model_name} | {metrics['hr@10']:.6f} | {metrics['mrr@10']:.6f} | "
                f"{metrics['ndcg@10']:.6f} | {metrics['recall@50']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


@app.command()
def main(
    baseline_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--baseline-config"),
    transformer_config: Path = typer.Option(
        Path("configs/transformer_retrieval_stable.yaml"),
        "--transformer-config",
    ),
    residual_config: Path = typer.Option(
        Path("configs/transformer_retrieval_residual.yaml"),
        "--residual-config",
    ),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    baseline_cfg = load_retrieval_config(baseline_config, sample=sample)
    transformer_cfg = load_retrieval_config(transformer_config, sample=sample)
    residual_cfg = load_retrieval_config(residual_config, sample=sample)

    train_df = pl.read_parquet(baseline_cfg.train_path)
    users_df = pl.read_parquet(baseline_cfg.users_path)
    items_df = pl.read_parquet(baseline_cfg.items_path)
    baseline_tables = load_feature_tables(baseline_cfg)
    transformer_tables = load_feature_tables(transformer_cfg)
    residual_tables = load_feature_tables(residual_cfg)

    payload: dict[str, dict[str, dict[str, float]]] = {}
    for split in ["val", "test"]:
        split_df = _split_df(baseline_cfg, split)
        pop_eval = evaluate_popularity_baseline(train_df, split_df, items_df)
        base_eval = _evaluate_model(
            baseline_cfg,
            "baseline",
            split,
            train_df,
            split_df,
            users_df,
            baseline_tables,
        )
        tf_eval = _evaluate_model(
            transformer_cfg,
            "transformer",
            split,
            train_df,
            split_df,
            users_df,
            transformer_tables,
        )
        residual_eval = _evaluate_model(
            residual_cfg,
            "residual_transformer",
            split,
            train_df,
            split_df,
            users_df,
            residual_tables,
        )
        payload[split] = {
            "popularity": pop_eval.metrics,
            "baseline": base_eval,
            "transformer": tf_eval,
            "residual_transformer": residual_eval,
        }

    report_name = "retriever_ablation_sample" if sample else "retriever_ablation_full"
    report_json = baseline_cfg.paths.report_output_dir / f"{report_name}.json"
    report_md = baseline_cfg.paths.report_output_dir / f"{report_name}.md"
    save_json(report_json, payload)
    save_markdown(report_md, _to_markdown(payload))

    typer.echo({"report_json": str(report_json), "report_md": str(report_md), "payload": payload})


if __name__ == "__main__":
    app()
