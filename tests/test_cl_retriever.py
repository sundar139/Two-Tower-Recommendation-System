from __future__ import annotations

import torch

from movie_recsys.modeling.cl_retrieval import CLResidualTransformerRetriever
from movie_recsys.modeling.residual_transformer_retrieval import ResidualTransformerRetriever
from movie_recsys.modeling.trainer import _compute_contrastive_weight_scale, _residual_anchor_loss
from movie_recsys.training.config import RetrievalConfig, load_retrieval_config


def _config() -> RetrievalConfig:
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


def _build_model(cfg: RetrievalConfig) -> CLResidualTransformerRetriever:
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
    expected_total = expected_total + (
        model.effective_lambda_user_cl * out["user_contrastive_loss"]
    )
    expected_total = expected_total + (
        model.effective_lambda_item_cl * out["item_contrastive_loss"]
    )
    expected_total = expected_total + (
        model.effective_lambda_alignment_cl * out["alignment_contrastive_loss"]
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
    user_head_grad: torch.Tensor | None = None
    item_head_grad: torch.Tensor | None = None
    if model.user_contrastive_projection is not None:
        user_head = model.user_contrastive_projection[0]
        assert isinstance(user_head, torch.nn.Linear)
        user_head_grad = user_head.weight.grad
    if model.item_contrastive_projection is not None:
        item_head = model.item_contrastive_projection[0]
        assert isinstance(item_head, torch.nn.Linear)
        item_head_grad = item_head.weight.grad
    gate_grad = model.user_tower.transformer_gate.grad

    assert id_proj_grad is not None
    assert feat_proj_grad is not None
    assert user_head_grad is not None
    assert item_head_grad is not None
    assert gate_grad is not None
    assert float(torch.abs(id_proj_grad).sum().item()) > 0.0
    assert float(torch.abs(feat_proj_grad).sum().item()) > 0.0
    assert float(torch.abs(user_head_grad).sum().item()) > 0.0
    assert float(torch.abs(item_head_grad).sum().item()) > 0.0
    assert float(torch.abs(gate_grad).sum().item()) > 0.0


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


def test_cl_projection_heads_change_contrastive_dimension_only() -> None:
    torch.manual_seed(21)
    cfg = _config()
    cfg.model.contrastive_projection_dim = 96
    model = _build_model(cfg)
    batch = _batch(4, cfg.train.history_length)

    user_emb = model.encode_user(batch)
    item_emb = model.encode_item(batch)

    projected_user = model._project_user_for_contrastive(user_emb)
    projected_item = model._project_item_for_contrastive(item_emb)

    assert user_emb.shape[1] == cfg.model.embedding_dim
    assert item_emb.shape[1] == cfg.model.embedding_dim
    assert projected_user.shape[1] == cfg.model.contrastive_projection_dim
    assert projected_item.shape[1] == cfg.model.contrastive_projection_dim


def test_cl_effective_lambdas_override_configured_weights() -> None:
    torch.manual_seed(23)
    cfg = _config()
    model = _build_model(cfg)
    model.set_effective_contrastive_weights(
        lambda_user_cl=0.0,
        lambda_item_cl=0.0,
        lambda_alignment_cl=0.0,
    )

    out = model(_batch(6, cfg.train.history_length))

    assert float(out["user_contrastive_loss"].item()) == 0.0
    assert float(out["item_contrastive_loss"].item()) == 0.0
    assert float(out["alignment_contrastive_loss"].item()) == 0.0
    assert torch.allclose(out["total_loss"], out["retrieval_loss"], atol=1e-8)


def test_contrastive_weight_scale_warmup_and_decay_schedule() -> None:
    cfg = _config()
    cfg.train.epochs = 6
    cfg.model.contrastive_warmup_epochs = 2
    cfg.model.contrastive_decay_start_epoch = 4
    cfg.model.contrastive_min_weight_scale = 0.5

    scales = [_compute_contrastive_weight_scale(cfg, epoch=e) for e in range(cfg.train.epochs)]

    assert scales[0] == 0.0
    assert scales[1] == 0.5
    assert scales[2] == 1.0
    assert scales[3] == 1.0
    assert scales[4] == 1.0
    assert scales[5] == 0.5


def test_residual_anchor_loss_zero_for_identical_and_higher_for_perturbed() -> None:
    torch.manual_seed(42)
    student_user = torch.randn(8, 16)
    student_user = torch.nn.functional.normalize(student_user, p=2, dim=1)
    student_item = torch.randn(8, 16)
    student_item = torch.nn.functional.normalize(student_item, p=2, dim=1)

    same_anchor = _residual_anchor_loss(
        student_user_emb=student_user,
        student_item_emb=student_item,
        teacher_user_emb=student_user.clone(),
        teacher_item_emb=student_item.clone(),
    )

    perturbed_anchor = _residual_anchor_loss(
        student_user_emb=student_user,
        student_item_emb=student_item,
        teacher_user_emb=student_user + 0.1,
        teacher_item_emb=student_item - 0.1,
    )

    assert torch.isfinite(same_anchor)
    assert torch.isfinite(perturbed_anchor)
    assert float(same_anchor.item()) < 1e-6
    assert float(perturbed_anchor.item()) > float(same_anchor.item())


def test_residual_anchor_teacher_parameters_are_frozen() -> None:
    cfg = _config()
    teacher = ResidualTransformerRetriever(
        config=cfg,
        num_users=32,
        num_items_with_padding=64,
        user_feature_dim=6,
        item_feature_dim=7,
    )
    teacher.requires_grad_(False)

    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_residual_anchor_term_increases_total_loss_when_teacher_differs() -> None:
    torch.manual_seed(101)
    cfg = _config()
    model = _build_model(cfg)
    out = model(_batch(8, cfg.train.history_length))

    anchor = _residual_anchor_loss(
        student_user_emb=out["user_emb"],
        student_item_emb=out["item_emb"],
        teacher_user_emb=out["user_emb"].detach() + 0.1,
        teacher_item_emb=out["item_emb"].detach() - 0.1,
    )
    lambda_anchor = 0.01
    total_with_anchor = out["total_loss"] + (lambda_anchor * anchor)

    assert torch.isfinite(anchor)
    assert float(anchor.item()) > 0.0
    assert float(total_with_anchor.item()) > float(out["total_loss"].item())
