from __future__ import annotations

import torch

from movie_recsys.modeling.cl_retrieval import CLResidualTransformerRetriever
from movie_recsys.training.config import load_retrieval_config


def _config():
    cfg = load_retrieval_config("configs/cl_retrieval.yaml", sample=True)
    cfg.model.dropout = 0.0
    cfg.train.history_length = 8
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
    return CLResidualTransformerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )


def test_cl_retriever_forward_outputs_losses_and_total_matches_weighted_sum() -> None:
    torch.manual_seed(123)
    cfg = _config()
    model = _build_model(cfg)

    out = model(_batch(8, cfg.train.history_length))

    assert out["logits"].shape == (8, 8)
    assert torch.isfinite(out["retrieval_loss"])
    assert torch.isfinite(out["user_contrastive_loss"])
    assert torch.isfinite(out["item_contrastive_loss"])
    assert torch.isfinite(out["alignment_contrastive_loss"])
    assert torch.isfinite(out["total_loss"])

    expected_total = out["retrieval_loss"]
    expected_total = expected_total + (cfg.model.lambda_user_cl * out["user_contrastive_loss"])
    expected_total = expected_total + (cfg.model.lambda_item_cl * out["item_contrastive_loss"])
    expected_total = expected_total + (
        cfg.model.lambda_alignment_cl * out["alignment_contrastive_loss"]
    )
    assert torch.allclose(out["total_loss"], expected_total, atol=1e-6)


def test_cl_retriever_total_loss_backpropagates_to_contrastive_heads() -> None:
    torch.manual_seed(9)
    cfg = _config()
    model = _build_model(cfg)

    out = model(_batch(8, cfg.train.history_length))
    out["total_loss"].backward()

    id_proj_grad = model.item_id_view_proj.weight.grad
    feat_proj_grad = model.item_feature_view_proj.weight.grad
    gate_grad = model.user_tower.transformer_gate.grad

    assert id_proj_grad is not None
    assert feat_proj_grad is not None
    assert gate_grad is not None
    assert float(id_proj_grad.abs().sum().item()) > 0.0
    assert float(feat_proj_grad.abs().sum().item()) > 0.0
    assert float(gate_grad.abs().sum().item()) > 0.0


def test_cl_retriever_zero_lambdas_disable_auxiliary_losses() -> None:
    torch.manual_seed(17)
    cfg = _config()
    cfg.model.lambda_user_cl = 0.0
    cfg.model.lambda_item_cl = 0.0
    cfg.model.lambda_alignment_cl = 0.0
    model = _build_model(cfg)

    out = model(_batch(6, cfg.train.history_length))

    assert float(out["user_contrastive_loss"].item()) == 0.0
    assert float(out["item_contrastive_loss"].item()) == 0.0
    assert float(out["alignment_contrastive_loss"].item()) == 0.0
    assert torch.allclose(out["total_loss"], out["retrieval_loss"], atol=1e-8)
