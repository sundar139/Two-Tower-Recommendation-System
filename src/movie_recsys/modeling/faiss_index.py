"""FAISS index utilities for exact inner-product retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl


def build_flat_ip_index(item_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(item_embeddings.shape[1])
    index.add(item_embeddings.astype(np.float32))
    return index


def save_faiss_bundle(
    index: faiss.IndexFlatIP,
    item_indices: np.ndarray,
    output_dir: Path,
    *,
    embedding_dim: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    mapping_path = output_dir / "item_idx_mapping.parquet"
    meta_path = output_dir / "index_metadata.json"

    faiss.write_index(index, str(index_path))
    pl.DataFrame(
        {"faiss_pos": np.arange(len(item_indices)), "item_idx": item_indices}
    ).write_parquet(mapping_path)
    meta_path.write_text(
        json.dumps({"embedding_dim": embedding_dim, "num_items": int(len(item_indices))}, indent=2),
        encoding="utf-8",
    )
    return {"index": index_path, "mapping": mapping_path, "metadata": meta_path}


def load_faiss_bundle(output_dir: Path) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    index = faiss.read_index(str(output_dir / "index.faiss"))
    mapping = pl.read_parquet(output_dir / "item_idx_mapping.parquet")
    item_idx = mapping.sort("faiss_pos").get_column("item_idx").to_numpy().astype(np.int64)
    return index, item_idx


def search_index(
    index: faiss.IndexFlatIP,
    queries: np.ndarray,
    item_mapping: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    scores, indices = index.search(queries.astype(np.float32), top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    mapped = item_mapping[indices]
    return mapped, scores, elapsed_ms
