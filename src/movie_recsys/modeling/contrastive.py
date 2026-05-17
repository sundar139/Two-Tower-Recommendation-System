"""Contrastive loss helpers for CL-enhanced retrieval models."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def pairwise_contrastive_logits(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return pairwise logits for InfoNCE-style losses."""
    if temperature <= 0.0:
        msg = "temperature must be positive"
        raise ValueError(msg)
    norm_a = functional.normalize(view_a, p=2, dim=1)
    norm_b = functional.normalize(view_b, p=2, dim=1)
    return (norm_a @ norm_b.T) / temperature


def info_nce_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-direction InfoNCE loss where diagonal pairs are positives."""
    logits = pairwise_contrastive_logits(query, key, temperature=temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = functional.cross_entropy(logits, labels)
    return loss, logits


def symmetric_info_nce_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE used for two-view contrastive learning."""
    loss_ab, logits_ab = info_nce_loss(view_a, view_b, temperature=temperature)
    loss_ba, logits_ba = info_nce_loss(view_b, view_a, temperature=temperature)
    return 0.5 * (loss_ab + loss_ba), logits_ab, logits_ba
