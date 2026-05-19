"""Training loop for the neural ranker."""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, cast

import mlflow
import numpy as np
import polars as pl
import torch

from movie_recsys.modeling.artifacts import save_checkpoint
from movie_recsys.modeling.metrics import (
	hit_rate_at_k,
	mrr_at_k,
	ndcg_at_k,
	recall_at_k,
)
from movie_recsys.ranking.config import RankerConfig
from movie_recsys.ranking.dataset import (
	iter_ranker_feature_batches,
	iter_ranker_query_groups,
)
from movie_recsys.ranking.features import (
	build_ranker_features,
	load_feature_manifest,
	resolve_feature_columns_from_artifacts,
	resolve_feature_shard_paths,
)
from movie_recsys.ranking.losses import compute_loss
from movie_recsys.ranking.model import NeuralRanker
from movie_recsys.utils.system import (
	MemoryStatus,
	get_memory_status,
	log_memory_status,
	should_stop_for_memory,
)

LOGGER = logging.getLogger(__name__)


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
	stopped_due_to_memory: bool
	resumed_from: Path | None
	last_epoch_checkpoint: Path | None
	feature_count: int
	train_query_count: int
	val_query_count: int
	train_row_count: int
	val_row_count: int
	memory_status: dict[str, float | None] | None
	full_smoke: bool


@dataclass(slots=True)
class _MetricAccumulator:
	hr_sum: float = 0.0
	mrr_sum: float = 0.0
	ndcg_sum: float = 0.0
	recall_sum: float = 0.0
	query_count: int = 0

	def update(self, ranked_items: list[int], targets: set[int]) -> None:
		self.hr_sum += hit_rate_at_k(ranked_items, targets, 10)
		self.mrr_sum += mrr_at_k(ranked_items, targets, 10)
		self.ndcg_sum += ndcg_at_k(ranked_items, targets, 10)
		self.recall_sum += recall_at_k(ranked_items, targets, 50)
		self.query_count += 1

	def as_dict(self) -> dict[str, float]:
		if self.query_count <= 0:
			return {"hr@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}
		n = float(self.query_count)
		return {
			"hr@10": float(self.hr_sum / n),
			"mrr@10": float(self.mrr_sum / n),
			"ndcg@10": float(self.ndcg_sum / n),
			"recall@50": float(self.recall_sum / n),
		}


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
	best_epoch: int,
	full_smoke: bool,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"epoch": epoch,
		"best_epoch": best_epoch,
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
		"full_smoke": full_smoke,
	}
	if scheduler is not None:
		payload["scheduler_state_dict"] = scheduler.state_dict()
	return payload


def _target_items(group: pl.DataFrame) -> set[int]:
	positives = {
		int(value)
		for value in group.filter(pl.col("label") == 1).get_column("item_idx").to_list()
	}
	if positives:
		return positives
	return {int(group.get_column("target_item_idx")[0])}


def _rank_items(group: pl.DataFrame, *, score_column: str) -> list[int]:
	ranked = group.sort(
		[score_column, "residual_rank", "item_idx"],
		descending=[True, False, False],
	)
	return [int(value) for value in ranked.get_column("item_idx").to_list()]


def _evaluate_streaming(
	model: NeuralRanker,
	*,
	feature_paths: list[Path],
	feature_columns: list[str],
	batch_size: int,
	max_queries: int | None,
	device: torch.device,
	log_prefix: str,
	log_every_queries: int,
) -> tuple[dict[str, float], int]:
	columns = [
		"query_id",
		"item_idx",
		"target_item_idx",
		"label",
		"residual_score",
		"residual_rank",
		*feature_columns,
	]

	accumulator = _MetricAccumulator()
	model.eval()
	query_counter = 0

	with torch.no_grad():
		for group in iter_ranker_query_groups(
			[str(path) for path in feature_paths],
			columns=columns,
			batch_size_rows=max(batch_size * 8, 4096),
			max_queries=max_queries,
		):
			features = group.select(feature_columns).to_numpy().astype("float32", copy=False)
			score_parts: list[Any] = []
			for start in range(0, features.shape[0], batch_size):
				end = min(start + batch_size, features.shape[0])
				batch = torch.from_numpy(features[start:end]).to(device)
				logits = model(batch)
				score_parts.append(logits.detach().cpu().numpy())
			ranker_values = (
				np.concatenate(score_parts).astype(np.float32, copy=False)
				if score_parts
				else np.zeros((0,), dtype=np.float32)
			)

			scored = group.with_columns(pl.Series(name="ranker_score", values=ranker_values))
			finite = scored.select(
				[
					pl.col("ranker_score").is_finite().all().alias("ranker_ok"),
					pl.col("residual_score").is_finite().all().alias("residual_ok"),
				]
			).row(0)
			if not bool(finite[0]) or not bool(finite[1]):
				msg = "Detected non-finite scores while evaluating ranker"
				raise ValueError(msg)

			targets = _target_items(scored)
			accumulator.update(_rank_items(scored, score_column="ranker_score"), targets)
			query_counter += 1

			if log_every_queries > 0 and query_counter % log_every_queries == 0:
				log_memory_status(
					f"{log_prefix} queries={query_counter}",
					logger=LOGGER,
					disk_path=feature_paths[0].parent,
				)

	return accumulator.as_dict(), query_counter


def _split_stats_from_manifest(
	cfg: RankerConfig,
	*,
	sample: bool,
	split: str,
) -> tuple[int, int]:
	manifest = load_feature_manifest(cfg, sample=sample)
	splits_payload = manifest.get("splits", {})
	if not isinstance(splits_payload, dict):
		return -1, -1
	split_payload = splits_payload.get(split, {})
	if not isinstance(split_payload, dict):
		return -1, -1
	row_count = int(split_payload.get("row_count", -1))
	query_count = int(split_payload.get("query_count", -1))
	return row_count, query_count


def _memory_status_payload(status: MemoryStatus) -> dict[str, float | None]:
	return {
		"ram_percent": status.ram_percent,
		"ram_used_gb": status.ram_used_gb,
		"ram_total_gb": status.ram_total_gb,
		"pagefile_percent": status.pagefile_percent,
		"pagefile_used_gb": status.pagefile_used_gb,
		"pagefile_total_gb": status.pagefile_total_gb,
		"disk_free_gb": status.disk_free_gb,
	}


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
	full_smoke: bool = False,
	max_train_rows: int | None = None,
	max_val_queries: int | None = None,
	log_file_path: Path | None = None,
) -> RankerTrainingResult:
	torch.manual_seed(cfg.random_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(cfg.random_seed)

	if full_smoke and sample:
		msg = "full_smoke requires full artifacts; do not combine with sample=True"
		raise ValueError(msg)

	required_splits = ("train", "val", "test")
	if rebuild_features or any(
		not resolve_feature_shard_paths(cfg, split=split, sample=sample)
		for split in required_splits
	):
		build_ranker_features(
			cfg,
			sample=sample,
			feature_shard_size=cfg.feature_shard_size,
			resume=not rebuild_features,
			overwrite=rebuild_features,
			splits=required_splits,
		)

	train_feature_paths = resolve_feature_shard_paths(cfg, split="train", sample=sample)
	val_feature_paths = resolve_feature_shard_paths(cfg, split="val", sample=sample)
	if not train_feature_paths:
		msg = "Missing train ranker feature artifacts"
		raise FileNotFoundError(msg)
	if not val_feature_paths:
		msg = "Missing val ranker feature artifacts"
		raise FileNotFoundError(msg)

	feature_columns = resolve_feature_columns_from_artifacts(
		cfg,
		sample=sample,
		fallback_split="train",
	)
	if not feature_columns:
		msg = "Unable to resolve ranker feature columns from artifacts"
		raise ValueError(msg)
	feature_count = len(feature_columns)

	train_row_count, train_query_count = _split_stats_from_manifest(
		cfg,
		sample=sample,
		split="train",
	)
	val_row_count, val_query_count = _split_stats_from_manifest(
		cfg,
		sample=sample,
		split="val",
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

	train_batch_size = cfg.batch_size if sample else cfg.full_train_batch_size
	eval_batch_size = cfg.batch_size if sample else cfg.full_eval_batch_size
	effective_epochs = 1 if full_smoke else cfg.epochs
	effective_train_rows = int(max_train_rows) if max_train_rows is not None else None
	effective_val_queries = int(max_val_queries) if max_val_queries is not None else None

	if train_row_count > 0 and train_batch_size > 0:
		estimated_steps = int(math.ceil(float(train_row_count) / float(train_batch_size)))
	else:
		estimated_steps = 100
	scheduler, step_per_batch = _build_scheduler(
		cfg,
		optimizer,
		steps_per_epoch=max(estimated_steps, 1),
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
	stopped_due_to_memory = False
	last_epoch_checkpoint: Path | None = None
	start = monotonic()
	deadline = (
		start + (float(max_runtime_hours) * 3600.0)
		if max_runtime_hours is not None and max_runtime_hours > 0.0
		else None
	)
	memory_status: dict[str, float | None] | None = None

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
		run_id = cast(str, run.info.run_id)
		experiment_id = cast(str, run.info.experiment_id)
		run_url = _mlflow_run_url(cfg, run_id=run_id, experiment_id=experiment_id)

		mlflow.log_params(
			{
				"model_type": cfg.model_type,
				"retrieval_backbone": cfg.retrieval_backbone,
				"candidate_top_k": cfg.candidate_top_k,
				"ranker_top_k": cfg.ranker_top_k,
				"batch_size": cfg.batch_size,
				"full_train_batch_size": cfg.full_train_batch_size,
				"full_eval_batch_size": cfg.full_eval_batch_size,
				"epochs": effective_epochs,
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
				"train_candidate_count": train_row_count,
				"val_candidate_count": val_row_count,
				"train_query_count": train_query_count,
				"val_query_count": val_query_count,
				"resumed_from": str(resume_from) if resume_from is not None else "",
				"rebuild_features": rebuild_features,
				"sample": sample,
				"full_smoke": full_smoke,
				"max_train_rows": (
					effective_train_rows if effective_train_rows is not None else -1
				),
				"max_val_queries": (
					effective_val_queries if effective_val_queries is not None else -1
				),
			}
		)
		mlflow.set_tag("mlflow_run_url", run_url)

		for epoch in range(start_epoch, effective_epochs):
			if deadline is not None and monotonic() >= deadline:
				stopped_due_to_runtime = True
				break

			model.train()
			epoch_losses: list[float] = []
			rows_processed_epoch = 0
			epoch_start = monotonic()

			for batch_index, batch in enumerate(
				iter_ranker_feature_batches(
				[str(path) for path in train_feature_paths],
				feature_columns=feature_columns,
				batch_size=max(train_batch_size, 1),
				shuffle=True,
				seed=cfg.random_seed + (epoch * 9973),
				max_rows=effective_train_rows,
				),
				start=1,
			):
				if deadline is not None and monotonic() >= deadline:
					stopped_due_to_runtime = True
					break

				features = cast(torch.Tensor, batch["features"]).to(device)
				labels = cast(torch.Tensor, batch["labels"]).to(device)
				query_ids = cast(list[str], batch["query_ids"])
				rows_processed_epoch += int(features.shape[0])

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

				if batch_index % max(cfg.log_every_batches, 1) == 0:
					status = log_memory_status(
						"[ranker-train]"
						f" split=train"
						f" epoch={epoch + 1}/{effective_epochs}"
						f" batch={batch_index}"
						f" rows={rows_processed_epoch}"
						f" loss={epoch_losses[-1]:.6f}"
						f" elapsed_s={monotonic() - epoch_start:.1f}"
						" checkpoint="
						f"{last_epoch_checkpoint if last_epoch_checkpoint is not None else '-'}"
						f" mlflow={run_url}"
						f" log_file={log_file_path if log_file_path is not None else '-'}",
						logger=LOGGER,
						disk_path=cfg.paths.ranker_model_dir,
					)
					if not sample and should_stop_for_memory(
						cfg.max_ram_percent,
						cfg.max_pagefile_percent,
						disk_path=cfg.paths.ranker_model_dir,
					):
						stopped_due_to_memory = True
						memory_status = _memory_status_payload(status)
						break

			if not epoch_losses:
				break

			train_loss_value = float(sum(epoch_losses) / max(len(epoch_losses), 1))
			val_metrics = best_metrics
			if eval_every_epoch or full_smoke:
				val_metrics, evaluated_queries = _evaluate_streaming(
					model,
					feature_paths=val_feature_paths,
					feature_columns=feature_columns,
					batch_size=max(eval_batch_size, 1),
					max_queries=effective_val_queries,
					device=device,
					log_prefix=(
						"[ranker-train-eval]"
						f" split=val"
						f" epoch={epoch + 1}/{effective_epochs}"
					),
					log_every_queries=max(cfg.log_every_batches, 1),
				)
				if evaluated_queries > 0:
					val_query_count = evaluated_queries

			if scheduler is not None and not step_per_batch and (eval_every_epoch or full_smoke):
				if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
					scheduler.step(val_metrics["ndcg@10"])
				else:
					scheduler.step()

			mlflow.log_metric("train_loss", train_loss_value, step=epoch)
			if eval_every_epoch or full_smoke:
				for name, value in val_metrics.items():
					mlflow.log_metric(name.replace("@", "_at_"), float(value), step=epoch)

			if (eval_every_epoch or full_smoke) and val_metrics["ndcg@10"] > best_metric:
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
						mlflow_run_id=run_id,
						best_epoch=best_epoch,
						full_smoke=full_smoke,
					),
				)

			if checkpoint_every_epoch or stopped_due_to_runtime or stopped_due_to_memory:
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
						mlflow_run_id=run_id,
						best_epoch=best_epoch,
						full_smoke=full_smoke,
					),
				)

			completed_epochs = epoch + 1
			if stopped_due_to_runtime or stopped_due_to_memory:
				break

		if not (eval_every_epoch or full_smoke):
			best_metrics, evaluated_queries = _evaluate_streaming(
				model,
				feature_paths=val_feature_paths,
				feature_columns=feature_columns,
				batch_size=max(eval_batch_size, 1),
				max_queries=effective_val_queries,
				device=device,
				log_prefix="[ranker-train-eval] split=val final",
				log_every_queries=max(cfg.log_every_batches, 1),
			)
			if evaluated_queries > 0:
				val_query_count = evaluated_queries
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
					mlflow_run_id=run_id,
					best_epoch=best_epoch,
					full_smoke=full_smoke,
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
					mlflow_run_id=run_id,
					best_epoch=best_epoch,
					full_smoke=full_smoke,
				),
			)

		if best_checkpoint.exists():
			mlflow.log_artifact(str(best_checkpoint))
		if save_last and last_checkpoint.exists():
			mlflow.log_artifact(str(last_checkpoint))
		if last_epoch_checkpoint is not None and last_epoch_checkpoint.exists():
			mlflow.log_artifact(str(last_epoch_checkpoint))

		if memory_status is None:
			memory_status = _memory_status_payload(
				get_memory_status(disk_path=cfg.paths.ranker_model_dir)
			)

		elapsed = monotonic() - start
		gc.collect()
		if device.type == "cuda":
			torch.cuda.empty_cache()

		return RankerTrainingResult(
			best_checkpoint=best_checkpoint,
			last_checkpoint=last_checkpoint,
			best_val_metrics=best_metrics,
			best_epoch=best_epoch,
			start_epoch=start_epoch,
			completed_epochs=completed_epochs,
			final_train_loss=train_loss_value,
			mlflow_run_id=run_id,
			mlflow_run_url=run_url,
			elapsed_seconds=elapsed,
			stopped_due_to_runtime=stopped_due_to_runtime,
			stopped_due_to_memory=stopped_due_to_memory,
			resumed_from=resume_from,
			last_epoch_checkpoint=last_epoch_checkpoint,
			feature_count=feature_count,
			train_query_count=train_query_count,
			val_query_count=val_query_count,
			train_row_count=train_row_count,
			val_row_count=val_row_count,
			memory_status=memory_status,
			full_smoke=full_smoke,
		)
