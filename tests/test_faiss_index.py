from __future__ import annotations

from pathlib import Path

import numpy as np

from movie_recsys.modeling.faiss_index import (
    build_flat_ip_index,
    load_faiss_bundle,
    save_faiss_bundle,
    search_index,
)


def test_faiss_matches_bruteforce(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    items = rng.normal(size=(100, 16)).astype(np.float32)
    query = rng.normal(size=(1, 16)).astype(np.float32)

    index = build_flat_ip_index(items)
    mapping = np.arange(items.shape[0], dtype=np.int64)
    paths = save_faiss_bundle(index, mapping, tmp_path, embedding_dim=16)

    loaded_index, loaded_mapping = load_faiss_bundle(tmp_path)
    top_items, _scores, _lat = search_index(loaded_index, query, loaded_mapping, 10)

    brute = (items @ query[0]).argsort()[::-1][:10]
    assert top_items.shape == (1, 10)
    assert top_items[0].tolist() == brute.tolist()
    assert paths["index"].exists()
