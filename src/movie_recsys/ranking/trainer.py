"""Training loop for the neural ranker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, cast

import mlflow
import torch

from movie_recsys.modeling.artifacts import save_checkpoint
from movie_recsys.modeling.metrics import aggregate_ranking_metrics
from movie_recsys.ranking.config import RankerConfig
from movie_recsys.ranking.dataset import RankerDataset, make_ranker_dataloader
from movie_recsys.ranking.features import build_ranker_features
from movie_recsys.ranking.losses import compute_loss
from movie_recsys.ranking.model import NeuralRanker


@dataclass(slots=True)
class RankerTrainingResult:
	best_checkpoint: Path
	last_checkpoint: Path
	best_val_metrics: dict[str, float]
	final_train_loss: float
	mlflow_run_id: str
	mlflow_run_url: str
	elapsed_seconds: float


def _select_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_scheduler(
	cfg: RankerConfig,
	optimizer: torch.optim.Optimizer,
	*,
	steps_per_epoch: int,
) -> tuple[
	torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
	bool,
]:
	if cfg.scheduler == "none":
		return None, False
	if cfg.scheduler == "cosine":
		return (
			torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer,
				T_max=max(cfg.epochs, 1),
			),
			False,
		)
	if cfg.scheduler == "plateau":
		return (
			torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode="max",
				factor=0.5,
				patience=2,
				min_lr=1e-6,
			),
			False,
		)

	total_steps = max(steps_per_epoch * max(cfg.epochs, 1), 1)
	warmup_steps = min(max(cfg.warmup_steps, 0), total_steps)

	def _lr_lambda(step: int) -> float:
		if warmup_steps > 0 and step < warmup_steps:
			return max(float(step + 1) / float(warmup_steps), 1e-8)
		decay_steps = max(total_steps - warmup_steps, 1)
		progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
		return 0.5 * (1.0 + math.cos(math.pi * progress))

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda), True


def _evaluate_loader(
	model: NeuralRanker,
	loader: torch.utils.data.DataLoader[dict[str, Any]],
	*,
	device: torch.device,
	amp_enabled: bool,
) -> dict[str, float]:
	model.eval()
	grouped_scores: dict[str, list[tuple[float, int]]] = {}
	targets_by_query: dict[str, set[int]] = {}

	with torch.no_grad():
		for batch in loader:
			features = cast(torch.Tensor, batch["features"]).to(device)
			query_ids = cast(list[str], batch["query_ids"])
			item_idx = cast(torch.Tensor, batch["item_idx"]).cpu().tolist()
			labels = cast(torch.Tensor, batch["labels"]).cpu().tolist()

			with torch.autocast(
				device_type=device.type,
				enabled=amp_enabled and device.type == "cuda",
			):
				logits = model(features)
			scores = logits.detach().cpu().tolist()

			for idx, query_id in enumerate(query_ids):
				grouped_scores.setdefault(query_id, []).append(
					(float(scores[idx]), int(item_idx[idx]))
				)
				if float(labels[idx]) > 0.5:
					targets_by_query.setdefault(query_id, set()).add(int(item_idx[idx]))

	predictions: dict[int, list[int]] = {}
	targets: dict[int, set[int]] = {}
	for query_idx, query_id in enumerate(sorted(grouped_scores.keys())):
		ranked = sorted(grouped_scores[query_id], key=lambda value: value[0], reverse=True)
		predictions[query_idx] = [item for _score, item in ranked]
		targets[query_idx] = targets_by_query.get(query_id, set())

	return aggregate_ranking_metrics(predictions, targets)


def _mlflow_run_url(cfg: RankerConfig, *, run_id: str, experiment_id: str) -> str:
	return f"{cfg.mlflow_ui_url.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


def train_ranker(
	cfg: RankerConfig,
	*,
	sample: bool,
	run_name: str | None = None,
) -> RankerTrainingResult:
	torch.manual_seed(cfg.random_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(cfg.random_seed)

	build_ranker_features(cfg, sample=sample)

	train_features = cfg.features_path(split="train", sample=sample)
	val_features = cfg.features_path(split="val", sample=sample)

	train_dataset = RankerDataset(str(train_features), sort_by_query=True)
	val_dataset = RankerDataset(
		str(val_features),
		feature_columns=train_dataset.feature_columns,
		sort_by_query=True,
	)

	train_loader = make_ranker_dataloader(
		train_dataset,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=0,
		seed=cfg.random_seed,
	)
	val_loader = make_ranker_dataloader(
		val_dataset,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=0,
		seed=cfg.random_seed,
	)

	device = _select_device()
	model = NeuralRanker(
		input_dim=train_dataset.feature_dim,
		hidden_dims=cfg.hidden_dims,
		dropout=cfg.dropout,
		use_layer_norm=True,
	).to(device)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=cfg.learning_rate,
		weight_decay=cfg.weight_decay,
	)
	scheduler, step_per_batch = _build_scheduler(
		cfg,
		optimizer,
		steps_per_epoch=max(len(train_loader), 1),
	)

	scaler = torch.amp.GradScaler(enabled=cfg.amp_enabled and device.type == "cuda")

	mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
	mlflow.set_experiment(cfg.mlflow_experiment_name)

	best_metric = float("-inf")
	best_metrics = {"hr@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}
	best_checkpoint = cfg.best_checkpoint
	last_checkpoint = cfg.last_checkpoint
	train_loss_value = 0.0
	start = monotonic()

	run_label = run_name or f"train_neural_ranker_{'sample' if sample else 'full'}"
	with mlflow.start_run(run_name=run_label) as run:
		mlflow.log_params(
			{
				"model_type": cfg.model_type,
				"retrieval_backbone": cfg.retrieval_backbone,
				"candidate_top_k": cfg.candidate_top_k,
				"ranker_top_k": cfg.ranker_top_k,
				"batch_size": cfg.batch_size,
				"epochs": cfg.epochs,
				"learning_rate": cfg.learning_rate,
				"weight_decay": cfg.weight_decay,
				"dropout": cfg.dropout,
				"hidden_dims": ",".join(str(v) for v in cfg.hidden_dims),
				"loss_type": cfg.loss_type,
				"pairwise_margin": cfg.pairwise_margin,
				"scheduler": cfg.scheduler,
				"warmup_steps": cfg.warmup_steps,
				"amp_enabled": cfg.amp_enabled,
				"sample": sample,
			}
		)

		for epoch in range(cfg.epochs):
			model.train()
			epoch_losses: list[float] = []

			for batch in train_loader:
				features = cast(torch.Tensor, batch["features"]).to(device)
				labels = cast(torch.Tensor, batch["labels"]).to(device)
				query_ids = cast(list[str], batch["query_ids"])

				optimizer.zero_grad(set_to_none=True)
				with torch.autocast(
					device_type=device.type,
					enabled=cfg.amp_enabled and device.type == "cuda",
				):
					logits = model(features)
					loss = compute_loss(
						loss_type=cfg.loss_type,
						logits=logits,
						labels=labels,
						query_ids=query_ids,
						pairwise_margin=cfg.pairwise_margin,
					)

				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
				scaler.step(optimizer)
				scaler.update()

				if scheduler is not None and step_per_batch:
					scheduler.step()

				epoch_losses.append(float(loss.detach().cpu().item()))

			train_loss_value = float(sum(epoch_losses) / max(len(epoch_losses), 1))
			val_metrics = _evaluate_loader(
				model,
				val_loader,
				device=device,
				amp_enabled=cfg.amp_enabled,
			)

			if scheduler is not None and not step_per_batch:
				if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
					scheduler.step(val_metrics["ndcg@10"])
				else:
					scheduler.step()

			mlflow.log_metric("train_loss", train_loss_value, step=epoch)
			for name, value in val_metrics.items():
				mlflow.log_metric(name.replace("@", "_at_"), float(value), step=epoch)

			if val_metrics["ndcg@10"] > best_metric:
				best_metric = val_metrics["ndcg@10"]
				best_metrics = {k: float(v) for k, v in val_metrics.items()}
				save_checkpoint(
					best_checkpoint,
					{
						"model_state_dict": model.state_dict(),
						"feature_columns": train_dataset.feature_columns,
						"hidden_dims": cfg.hidden_dims,
						"dropout": cfg.dropout,
						"use_layer_norm": True,
						"epoch": epoch,
						"best_val_metrics": best_metrics,
						"train_loss": train_loss_value,
						"config": cfg.model_dump(mode="json"),
						"mlflow_run_id": run.info.run_id,
					},
				)

		save_checkpoint(
			last_checkpoint,
			{
				"model_state_dict": model.state_dict(),
				"feature_columns": train_dataset.feature_columns,
				"hidden_dims": cfg.hidden_dims,
				"dropout": cfg.dropout,
				"use_layer_norm": True,
				"epoch": cfg.epochs - 1,
				"best_val_metrics": best_metrics,
				"train_loss": train_loss_value,
				"config": cfg.model_dump(mode="json"),
				"mlflow_run_id": run.info.run_id,
			},
		)

		mlflow.log_artifact(str(best_checkpoint))
		mlflow.log_artifact(str(last_checkpoint))

		elapsed = monotonic() - start
		return RankerTrainingResult(
			best_checkpoint=best_checkpoint,
			last_checkpoint=last_checkpoint,
			best_val_metrics=best_metrics,
			final_train_loss=train_loss_value,
			mlflow_run_id=cast(str, run.info.run_id),
			mlflow_run_url=_mlflow_run_url(
				cfg,
				run_id=cast(str, run.info.run_id),
				experiment_id=cast(str, run.info.experiment_id),
			),
			elapsed_seconds=elapsed,
		)
