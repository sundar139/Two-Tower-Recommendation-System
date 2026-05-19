"""Training loop for the neural ranker."""

from __future__ import annotations

import gc
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
	best_epoch: int
	start_epoch: int
	completed_epochs: int
	final_train_loss: float
	mlflow_run_id: str
	mlflow_run_url: str
	elapsed_seconds: float
	stopped_due_to_runtime: bool
	resumed_from: Path | None
	last_epoch_checkpoint: Path | None
	feature_count: int
	train_query_count: int
	val_query_count: int


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


def _epoch_checkpoint_path(cfg: RankerConfig, *, epoch: int) -> Path:
	checkpoint_dir = cfg.paths.ranker_model_dir / "checkpoints"
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	return checkpoint_dir / f"neural_ranker_epoch_{epoch}.pt"


def _checkpoint_payload(
	*,
	model: NeuralRanker,
	optimizer: torch.optim.Optimizer,
	scheduler: (
		torch.optim.lr_scheduler.LRScheduler
		| torch.optim.lr_scheduler.ReduceLROnPlateau
		| None
	),
	scaler: torch.amp.GradScaler,
	epoch: int,
	best_metric: float,
	best_val_metrics: dict[str, float],
	train_loss: float,
	feature_columns: list[str],
	cfg: RankerConfig,
	mlflow_run_id: str,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"epoch": epoch,
		"best_metric": best_metric,
		"best_val_metrics": best_val_metrics,
		"train_loss": train_loss,
		"config": cfg.model_dump(mode="json"),
		"random_seed": cfg.random_seed,
		"feature_columns": feature_columns,
		"mlflow_run_id": mlflow_run_id,
		"hidden_dims": cfg.hidden_dims,
		"dropout": cfg.dropout,
		"use_layer_norm": True,
		"scaler_state_dict": scaler.state_dict(),
	}
	if scheduler is not None:
		payload["scheduler_state_dict"] = scheduler.state_dict()
	return payload


def train_ranker(
	cfg: RankerConfig,
	*,
	sample: bool,
	run_name: str | None = None,
	resume_from: Path | None = None,
	checkpoint_every_epoch: bool = False,
	max_runtime_hours: float | None = None,
	eval_every_epoch: bool = False,
	save_last: bool = True,
	rebuild_features: bool = False,
) -> RankerTrainingResult:
	torch.manual_seed(cfg.random_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(cfg.random_seed)

	train_features = cfg.features_path(split="train", sample=sample)
	val_features = cfg.features_path(split="val", sample=sample)
	test_features = cfg.features_path(split="test", sample=sample)
	if rebuild_features or any(
		not feature_path.exists()
		for feature_path in [train_features, val_features, test_features]
	):
		build_ranker_features(cfg, sample=sample)

	train_dataset = RankerDataset(str(train_features), sort_by_query=True)
	feature_columns = train_dataset.feature_columns
	feature_count = train_dataset.feature_dim
	val_dataset: RankerDataset | None = None
	val_loader: torch.utils.data.DataLoader[dict[str, Any]] | None = None
	val_candidate_count = -1
	val_query_count = -1
	if eval_every_epoch:
		val_dataset = RankerDataset(
			str(val_features),
			feature_columns=feature_columns,
			sort_by_query=True,
		)
		val_loader = make_ranker_dataloader(
			val_dataset,
			batch_size=cfg.batch_size,
			shuffle=False,
			num_workers=0,
			seed=cfg.random_seed,
		)
		val_candidate_count = len(val_dataset)
		val_query_count = len(set(val_dataset.query_ids))

	train_query_count = len(set(train_dataset.query_ids))

	train_loader = make_ranker_dataloader(
		train_dataset,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=0,
		seed=cfg.random_seed,
	)

	device = _select_device()
	model = NeuralRanker(
		input_dim=feature_count,
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
	best_epoch = -1
	best_checkpoint = cfg.best_checkpoint
	last_checkpoint = cfg.last_checkpoint
	train_loss_value = 0.0
	start_epoch = 0
	completed_epochs = 0
	stopped_due_to_runtime = False
	last_epoch_checkpoint: Path | None = None
	start = monotonic()
	deadline = (
		start + (float(max_runtime_hours) * 3600.0)
		if max_runtime_hours is not None and max_runtime_hours > 0.0
		else None
	)

	if resume_from is not None:
		resume_payload = cast(dict[str, Any], torch.load(resume_from, map_location="cpu"))
		model.load_state_dict(cast(dict[str, Any], resume_payload["model_state_dict"]))
		optimizer_state = resume_payload.get("optimizer_state_dict")
		if isinstance(optimizer_state, dict):
			optimizer.load_state_dict(optimizer_state)
		scheduler_state = resume_payload.get("scheduler_state_dict")
		if scheduler is not None and isinstance(scheduler_state, dict):
			scheduler.load_state_dict(scheduler_state)
		scaler_state = resume_payload.get("scaler_state_dict")
		if isinstance(scaler_state, dict):
			scaler.load_state_dict(scaler_state)
		start_epoch = int(resume_payload.get("epoch", -1)) + 1
		best_metric = float(resume_payload.get("best_metric", best_metric))
		stored_best = resume_payload.get("best_val_metrics")
		if isinstance(stored_best, dict):
			best_metrics = {
				k: float(v)
				for k, v in stored_best.items()
				if isinstance(v, (int, float))
			}
		best_epoch = int(resume_payload.get("best_epoch", best_epoch))

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
				"feature_count": feature_count,
				"train_candidate_count": len(train_dataset),
				"val_candidate_count": val_candidate_count,
				"train_query_count": train_query_count,
				"val_query_count": val_query_count,
				"resumed_from": str(resume_from) if resume_from is not None else "",
				"rebuild_features": rebuild_features,
				"sample": sample,
			}
		)

		for epoch in range(start_epoch, cfg.epochs):
			if deadline is not None and monotonic() >= deadline:
				stopped_due_to_runtime = True
				break

			model.train()
			epoch_losses: list[float] = []

			for batch in train_loader:
				if deadline is not None and monotonic() >= deadline:
					stopped_due_to_runtime = True
					break

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

				optimizer_stepped = False
				if scaler.is_enabled():
					prev_scale = scaler.get_scale()
					scaler.scale(loss).backward()
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
					scaler.step(optimizer)
					scaler.update()
					optimizer_stepped = scaler.get_scale() >= prev_scale
				else:
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
					optimizer.step()
					optimizer_stepped = True

				if scheduler is not None and step_per_batch and optimizer_stepped:
					scheduler.step()

				epoch_losses.append(float(loss.detach().cpu().item()))

			if not epoch_losses:
				break

			train_loss_value = float(sum(epoch_losses) / max(len(epoch_losses), 1))
			val_metrics = best_metrics
			if eval_every_epoch:
				if val_loader is None:
					msg = "Validation loader is unavailable while eval_every_epoch is enabled"
					raise RuntimeError(msg)
				val_metrics = _evaluate_loader(
					model,
					val_loader,
					device=device,
					amp_enabled=cfg.amp_enabled,
				)

			if scheduler is not None and not step_per_batch and eval_every_epoch:
				if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
					scheduler.step(val_metrics["ndcg@10"])
				else:
					scheduler.step()

			mlflow.log_metric("train_loss", train_loss_value, step=epoch)
			if eval_every_epoch:
				for name, value in val_metrics.items():
					mlflow.log_metric(name.replace("@", "_at_"), float(value), step=epoch)

			if eval_every_epoch and val_metrics["ndcg@10"] > best_metric:
				best_metric = val_metrics["ndcg@10"]
				best_metrics = {k: float(v) for k, v in val_metrics.items()}
				best_epoch = epoch
				save_checkpoint(
					best_checkpoint,
					_checkpoint_payload(
						model=model,
						optimizer=optimizer,
						scheduler=scheduler,
						scaler=scaler,
						epoch=epoch,
						best_metric=best_metric,
						best_val_metrics=best_metrics,
						train_loss=train_loss_value,
						feature_columns=feature_columns,
						cfg=cfg,
						mlflow_run_id=cast(str, run.info.run_id),
					),
				)

			if checkpoint_every_epoch:
				last_epoch_checkpoint = _epoch_checkpoint_path(cfg, epoch=epoch)
				save_checkpoint(
					last_epoch_checkpoint,
					_checkpoint_payload(
						model=model,
						optimizer=optimizer,
						scheduler=scheduler,
						scaler=scaler,
						epoch=epoch,
						best_metric=best_metric,
						best_val_metrics=best_metrics,
						train_loss=train_loss_value,
						feature_columns=feature_columns,
						cfg=cfg,
						mlflow_run_id=cast(str, run.info.run_id),
					),
				)

			completed_epochs = epoch + 1
			if stopped_due_to_runtime:
				break

		if not eval_every_epoch:
			del train_loader
			del train_dataset
			gc.collect()
			if device.type == "cuda":
				torch.cuda.empty_cache()

			val_dataset = RankerDataset(
				str(val_features),
				feature_columns=feature_columns,
				sort_by_query=True,
			)
			val_loader = make_ranker_dataloader(
				val_dataset,
				batch_size=cfg.batch_size,
				shuffle=False,
				num_workers=0,
				seed=cfg.random_seed,
			)
			val_candidate_count = len(val_dataset)
			val_query_count = len(set(val_dataset.query_ids))
			best_metrics = _evaluate_loader(
				model,
				val_loader,
				device=device,
				amp_enabled=cfg.amp_enabled,
			)
			best_metric = best_metrics["ndcg@10"]
			best_epoch = completed_epochs - 1
			for name, value in best_metrics.items():
				mlflow.log_metric(name.replace("@", "_at_"), float(value), step=completed_epochs)
			save_checkpoint(
				best_checkpoint,
				_checkpoint_payload(
					model=model,
					optimizer=optimizer,
					scheduler=scheduler,
					scaler=scaler,
					epoch=max(completed_epochs - 1, 0),
					best_metric=best_metric,
					best_val_metrics=best_metrics,
					train_loss=train_loss_value,
					feature_columns=feature_columns,
					cfg=cfg,
					mlflow_run_id=cast(str, run.info.run_id),
				),
			)

		if save_last:
			save_checkpoint(
				last_checkpoint,
				_checkpoint_payload(
					model=model,
					optimizer=optimizer,
					scheduler=scheduler,
					scaler=scaler,
					epoch=max(completed_epochs - 1, 0),
					best_metric=best_metric,
					best_val_metrics=best_metrics,
					train_loss=train_loss_value,
					feature_columns=feature_columns,
					cfg=cfg,
					mlflow_run_id=cast(str, run.info.run_id),
				),
			)

		if best_checkpoint.exists():
			mlflow.log_artifact(str(best_checkpoint))
		if save_last and last_checkpoint.exists():
			mlflow.log_artifact(str(last_checkpoint))
		if last_epoch_checkpoint is not None and last_epoch_checkpoint.exists():
			mlflow.log_artifact(str(last_epoch_checkpoint))

		elapsed = monotonic() - start
		return RankerTrainingResult(
			best_checkpoint=best_checkpoint,
			last_checkpoint=last_checkpoint,
			best_val_metrics=best_metrics,
			best_epoch=best_epoch,
			start_epoch=start_epoch,
			completed_epochs=completed_epochs,
			final_train_loss=train_loss_value,
			mlflow_run_id=cast(str, run.info.run_id),
			mlflow_run_url=_mlflow_run_url(
				cfg,
				run_id=cast(str, run.info.run_id),
				experiment_id=cast(str, run.info.experiment_id),
			),
			elapsed_seconds=elapsed,
			stopped_due_to_runtime=stopped_due_to_runtime,
			resumed_from=resume_from,
			last_epoch_checkpoint=last_epoch_checkpoint,
			feature_count=feature_count,
			train_query_count=train_query_count,
			val_query_count=val_query_count,
		)
