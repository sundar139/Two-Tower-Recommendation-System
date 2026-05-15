from __future__ import annotations

import numpy as np
import polars as pl
import torch

from movie_recsys.modeling.datasets import FeatureTables, RetrievalDataset
from movie_recsys.modeling.losses import InBatchCrossEntropyLoss
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.training.config import load_retrieval_config


def _config():
    cfg = load_retrieval_config("configs/transformer_retrieval_residual.yaml", sample=True)
    cfg.model.dropout = 0.0
    cfg.train.history_length = 5
    return cfg


def _batch(batch_size: int, history_length: int) -> dict[str, torch.Tensor]:
    return {
        "user_idx": torch.randint(0, 32, (batch_size,)),
        "item_idx": torch.randint(1, 64, (batch_size,)),
        "history_item_idx": torch.randint(0, 64, (batch_size, history_length)),
        "history_mask": torch.randint(0, 2, (batch_size, history_length)).bool(),
        "user_features": torch.randn(batch_size, 6),
        "item_features": torch.randn(batch_size, 7),
    }


def _build_model(cfg):
    return ResidualTransformerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )


def test_residual_user_embedding_norm_and_logits_shape() -> None:
    cfg = _config()
    model = _build_model(cfg)
    out = model(_batch(8, cfg.train.history_length))

    assert out["user_emb"].shape == (8, cfg.model.embedding_dim)
    assert out["item_emb"].shape == (8, cfg.model.embedding_dim)
    assert out["logits"].shape == (8, 8)
    norms = torch.linalg.norm(out["user_emb"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_gate_alpha_matches_sigmoid_of_initial_value() -> None:
    cfg = _config()
    model = _build_model(cfg)
    expected = torch.sigmoid(torch.tensor(cfg.model.initial_transformer_gate)).item()
    got = model.user_tower.transformer_gate_alpha().item()
    assert abs(got - expected) < 1e-6


def test_gate_zero_equivalent_to_baseline_context_in_debug_path() -> None:
    cfg = _config()
    model = _build_model(cfg)
    batch = _batch(4, cfg.train.history_length)

    with torch.no_grad():
        model.user_tower.transformer_gate.fill_(-100.0)
        _emb, debug = model.encode_user_with_debug(batch)

    user_seq = debug["user_seq"]
    baseline_context = debug["baseline_context"]
    assert isinstance(user_seq, torch.Tensor)
    assert isinstance(baseline_context, torch.Tensor)
    assert torch.allclose(user_seq, baseline_context, atol=1e-4)


def test_gate_one_equivalent_to_transformer_context_in_debug_path() -> None:
    cfg = _config()
    model = _build_model(cfg)
    batch = _batch(4, cfg.train.history_length)

    with torch.no_grad():
        model.user_tower.transformer_gate.fill_(100.0)
        _emb, debug = model.encode_user_with_debug(batch)

    user_seq = debug["user_seq"]
    transformer_context = debug["transformer_context"]
    assert isinstance(user_seq, torch.Tensor)
    assert isinstance(transformer_context, torch.Tensor)
    assert torch.allclose(user_seq, transformer_context, atol=1e-4)


def test_residual_loss_finite_and_gradients_flow() -> None:
    cfg = _config()
    model = _build_model(cfg)
    criterion = InBatchCrossEntropyLoss()

    out = model(_batch(8, cfg.train.history_length))
    loss = criterion(out["logits"])
    assert torch.isfinite(loss)
    loss.backward()

    gate_grad = model.user_tower.transformer_gate.grad
    assert gate_grad is not None
    assert float(gate_grad.abs().sum().item()) > 0.0

    q_grad = model.user_tower.sequence_encoder.blocks[0].q_proj.weight.grad
    assert q_grad is not None
    assert float(q_grad.abs().sum().item()) > 0.0


def test_empty_history_safe_no_nan() -> None:
    cfg = _config()
    model = _build_model(cfg)
    batch = _batch(4, cfg.train.history_length)
    batch["history_item_idx"] = torch.zeros_like(batch["history_item_idx"])
    batch["history_mask"] = torch.zeros_like(batch["history_mask"])

    out = model(batch)
    assert torch.isfinite(out["user_emb"]).all()
    assert torch.isfinite(out["logits"]).all()


def test_eval_mode_deterministic_with_same_input() -> None:
    torch.manual_seed(123)
    cfg = _config()
    model = _build_model(cfg)
    model.eval()
    batch = _batch(6, cfg.train.history_length)

    out1 = model(batch)
    out2 = model(batch)
    assert torch.allclose(out1["user_emb"], out2["user_emb"], atol=1e-6)
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-6)


def test_encode_user_with_debug_contains_attention_and_gate() -> None:
    cfg = _config()
    model = _build_model(cfg)
    batch = _batch(4, cfg.train.history_length)

    emb, debug = model.encode_user_with_debug(batch)
    assert emb.shape == (4, cfg.model.embedding_dim)
    attention = debug.get("attention_weights")
    assert isinstance(attention, list)
    assert len(attention) == cfg.model.transformer_layers
    assert "transformer_gate_alpha" in debug


def test_history_is_truncated_to_configured_length_and_excludes_target(tmp_path) -> None:
    interactions = pl.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 0, 0, 0],
            "item_idx": [1, 2, 3, 4, 5, 6, 7],
            "timestamp": [1, 2, 3, 4, 5, 6, 7],
            "explicit_rating": [1.0] * 7,
        }
    )
    path = tmp_path / "interactions.parquet"
    interactions.write_parquet(path)

    features = FeatureTables(
        user_features=np.zeros((1, 2), dtype=np.float32),
        item_features=np.zeros((8, 3), dtype=np.float32),
        user_feature_columns=["a", "b"],
        item_feature_columns=["x", "y", "z"],
    )

    dataset = RetrievalDataset(str(path), features, history_length=3)
    row = dataset[6]
    history = row["history_item_idx"].tolist()

    assert history == [5, 6, 7]
    assert row["item_idx"] == 8
    assert row["item_idx"] not in history


def test_padding_side_matches_pooling_expectation() -> None:
    cfg = _config()
    model = _build_model(cfg)
    model.eval()

    batch = _batch(2, cfg.train.history_length)
    batch["history_mask"] = torch.tensor(
        [[False, False, True, True, True], [False, True, True, True, True]],
        dtype=torch.bool,
    )

    out = model(batch)
    assert torch.isfinite(out["user_emb"]).all()
