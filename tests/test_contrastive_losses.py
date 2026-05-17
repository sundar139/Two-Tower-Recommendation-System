from __future__ import annotations

import pytest
import torch

from movie_recsys.modeling.contrastive import (
    info_nce_loss,
    pairwise_contrastive_logits,
    symmetric_info_nce_loss,
)


def test_pairwise_contrastive_logits_shape_and_temperature_validation() -> None:
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)

    logits = pairwise_contrastive_logits(a, b, temperature=0.1)
    assert logits.shape == (4, 4)

    with pytest.raises(ValueError):
        pairwise_contrastive_logits(a, b, temperature=0.0)


def test_info_nce_loss_low_for_perfectly_aligned_orthogonal_views() -> None:
    eye = torch.eye(3)
    loss, logits = info_nce_loss(eye, eye, temperature=0.1)

    assert logits.shape == (3, 3)
    assert float(loss.item()) < 1e-3


def test_symmetric_info_nce_matches_transposed_logits_for_same_views() -> None:
    torch.manual_seed(123)
    views = torch.randn(6, 10)

    loss, logits_ab, logits_ba = symmetric_info_nce_loss(
        views,
        views,
        temperature=0.2,
    )

    assert torch.isfinite(loss)
    assert logits_ab.shape == (6, 6)
    assert logits_ba.shape == (6, 6)
    assert torch.allclose(logits_ab, logits_ba.T, atol=1e-6)
