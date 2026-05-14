"""Loss functions for retrieval."""

from __future__ import annotations

import torch
from torch import nn


class InBatchCrossEntropyLoss(nn.Module):
    """In-batch negatives with cross-entropy labels [0..B-1]."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        labels = torch.arange(logits.shape[0], device=logits.device)
        return nn.functional.cross_entropy(logits, labels)
