"""Neural ranker model definitions."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class NeuralRanker(nn.Module):
	"""MLP ranker that predicts one logit score per candidate."""

	def __init__(
		self,
		*,
		input_dim: int,
		hidden_dims: list[int],
		dropout: float,
		use_layer_norm: bool = True,
	) -> None:
		super().__init__()
		if input_dim <= 0:
			msg = "input_dim must be > 0"
			raise ValueError(msg)
		if not hidden_dims:
			msg = "hidden_dims must contain at least one layer width"
			raise ValueError(msg)

		layers: list[nn.Module] = []
		prev_dim = input_dim
		for width in hidden_dims:
			layers.append(nn.Linear(prev_dim, width))
			if use_layer_norm:
				layers.append(nn.LayerNorm(width))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			prev_dim = width

		self.backbone = nn.Sequential(*layers)
		self.output = nn.Linear(prev_dim, 1)

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		hidden = self.backbone(features)
		logits = self.output(hidden)
		return cast(torch.Tensor, logits.squeeze(-1))
