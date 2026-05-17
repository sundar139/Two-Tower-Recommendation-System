"""Loss functions for pointwise and pairwise ranking objectives."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, cast

import torch
from torch import nn


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
	"""Pointwise BCE with logits loss."""

	criterion = nn.BCEWithLogitsLoss()
	return cast(torch.Tensor, criterion(logits, labels))


def bpr_loss(
	logits: torch.Tensor,
	labels: torch.Tensor,
	query_ids: list[str],
	*,
	margin: float,
) -> torch.Tensor:
	"""Pairwise BPR objective computed within each query group."""

	groups: dict[str, list[int]] = defaultdict(list)
	for idx, query_id in enumerate(query_ids):
		groups[query_id].append(idx)

	losses: list[torch.Tensor] = []
	for indices in groups.values():
		idx_tensor = torch.tensor(indices, device=logits.device, dtype=torch.long)
		group_logits = logits[idx_tensor]
		group_labels = labels[idx_tensor]

		pos_mask = group_labels > 0.5
		neg_mask = group_labels <= 0.5
		if int(pos_mask.sum().item()) == 0 or int(neg_mask.sum().item()) == 0:
			continue

		pos_scores = group_logits[pos_mask].view(-1, 1)
		neg_scores = group_logits[neg_mask].view(1, -1)
		pairwise_diff = pos_scores - neg_scores - margin
		losses.append(-torch.nn.functional.logsigmoid(pairwise_diff).mean())

	if not losses:
		return logits.sum() * 0.0
	return torch.stack(losses).mean()


def hybrid_loss(
	logits: torch.Tensor,
	labels: torch.Tensor,
	query_ids: list[str],
	*,
	margin: float,
	bce_weight: float = 0.5,
) -> torch.Tensor:
	"""Hybrid pointwise+pairwise objective."""

	pointwise = bce_loss(logits, labels)
	pairwise = bpr_loss(logits, labels, query_ids, margin=margin)
	return bce_weight * pointwise + (1.0 - bce_weight) * pairwise


def compute_loss(
	*,
	loss_type: Literal["bce", "bpr", "hybrid"],
	logits: torch.Tensor,
	labels: torch.Tensor,
	query_ids: list[str],
	pairwise_margin: float,
) -> torch.Tensor:
	"""Dispatch loss computation by configured objective type."""

	if loss_type == "bce":
		return bce_loss(logits, labels)
	if loss_type == "bpr":
		return bpr_loss(logits, labels, query_ids, margin=pairwise_margin)
	return hybrid_loss(logits, labels, query_ids, margin=pairwise_margin)
