"""Tests for neural ranker model forward behavior."""

from __future__ import annotations

import torch

from movie_recsys.ranking.losses import bce_loss, bpr_loss
from movie_recsys.ranking.model import NeuralRanker


def test_ranker_forward_output_shape() -> None:
	model = NeuralRanker(input_dim=6, hidden_dims=[16, 8], dropout=0.1)
	features = torch.randn(4, 6)
	logits = model(features)
	assert tuple(logits.shape) == (4,)


def test_bce_and_bpr_losses_are_finite() -> None:
	logits = torch.tensor([0.2, -0.1, 0.6, 0.0], dtype=torch.float32)
	labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
	query_ids = ["q1", "q1", "q2", "q2"]

	pointwise = bce_loss(logits, labels)
	pairwise = bpr_loss(logits, labels, query_ids, margin=0.2)
	assert torch.isfinite(pointwise)
	assert torch.isfinite(pairwise)


def test_small_overfit_reduces_loss() -> None:
	torch.manual_seed(42)
	n_rows = 1000
	dim = 8

	features = torch.randn(n_rows, dim)
	true_weights = torch.randn(dim)
	labels = ((features @ true_weights) > 0).float()

	model = NeuralRanker(input_dim=dim, hidden_dims=[32, 16], dropout=0.0)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

	with torch.no_grad():
		initial_loss = float(bce_loss(model(features), labels).item())

	for _ in range(80):
		optimizer.zero_grad(set_to_none=True)
		loss = bce_loss(model(features), labels)
		loss.backward()
		optimizer.step()

	with torch.no_grad():
		final_loss = float(bce_loss(model(features), labels).item())

	assert final_loss < initial_loss
