"""Coverage for memory-safe ranker streaming workflow pieces."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest
import torch

import movie_recsys.ranking.trainer as trainer_module
from movie_recsys.ranking.config import RankerConfig, RankerPathsConfig
from movie_recsys.ranking.evaluator import evaluate_ranker_split
from movie_recsys.ranking.model import NeuralRanker


class _DummyRunInfo:
	run_id = "dummy-run"
	experiment_id = "0"


class _DummyRun:
	info = _DummyRunInfo()

	def __enter__(self) -> _DummyRun:
		return self

	def __exit__(self, _exc_type, _exc, _tb) -> bool:
		return False


class _DummyMlflow:
	def set_tracking_uri(self, _uri: str) -> None:
		return None

	def set_experiment(self, _name: str) -> None:
		return None

	def start_run(self, run_name: str | None = None) -> _DummyRun:
		_ = run_name
		return _DummyRun()

	def log_params(self, _params: dict[str, object]) -> None:
		return None

	def log_metric(self, _name: str, _value: float, step: int) -> None:
		_ = step
		return None

	def set_tag(self, _name: str, _value: str) -> None:
		return None

	def log_artifact(self, _path: str) -> None:
		return None


def _ranker_cfg(tmp_path: Path) -> RankerConfig:
	paths = RankerPathsConfig(
		ranker_candidate_dir=tmp_path / "candidates",
		ranker_feature_dir=tmp_path / "features",
		ranker_model_dir=tmp_path / "models",
		ranker_report_dir=tmp_path / "reports",
	)
	paths.ranker_candidate_dir.mkdir(parents=True, exist_ok=True)
	paths.ranker_feature_dir.mkdir(parents=True, exist_ok=True)
	paths.ranker_model_dir.mkdir(parents=True, exist_ok=True)
	paths.ranker_report_dir.mkdir(parents=True, exist_ok=True)

	return RankerConfig(
		retriever_config=tmp_path / "retriever.yaml",
		retriever_checkpoint=tmp_path / "retriever.pt",
		mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
		mlflow_artifact_root=str(tmp_path / "mlruns"),
		mlflow_experiment_name="ranker-stream-tests",
		mlflow_ui_host="127.0.0.1",
		mlflow_ui_port=5000,
		mlflow_ui_url="http://127.0.0.1:5000",
		paths=paths,
		epochs=2,
		batch_size=2,
		full_train_batch_size=2,
		full_eval_batch_size=2,
		amp_enabled=False,
		log_every_batches=1,
		hidden_dims=[8, 4],
		max_ram_percent=85.0,
	)


def _write_feature_split(path: Path, *, split: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	pl.DataFrame(
		{
			"query_id": [f"{split}_q1", f"{split}_q1", f"{split}_q2", f"{split}_q2"],
			"item_idx": [10, 20, 30, 40],
			"target_item_idx": [10, 10, 30, 30],
			"label": [1, 0, 1, 0],
			"residual_score": [0.9, 0.5, 0.8, 0.4],
			"residual_rank": [1, 2, 1, 2],
		}
	).write_parquet(path)


def _patch_trainer_artifacts(
	monkeypatch: pytest.MonkeyPatch,
	*,
	train_path: Path,
	val_path: Path,
	test_path: Path,
) -> None:
	path_by_split = {
		"train": train_path,
		"val": val_path,
		"test": test_path,
	}

	monkeypatch.setattr(trainer_module, "mlflow", _DummyMlflow())
	monkeypatch.setattr(trainer_module, "build_ranker_features", lambda *_args, **_kwargs: {})
	monkeypatch.setattr(
		trainer_module,
		"resolve_feature_shard_paths",
		lambda _cfg, *, split, sample: [path_by_split[split]],
	)
	monkeypatch.setattr(
		trainer_module,
		"resolve_feature_columns_from_artifacts",
		lambda _cfg, *, sample, fallback_split: ["residual_score", "residual_rank"],
	)
	monkeypatch.setattr(
		trainer_module,
		"load_feature_manifest",
		lambda _cfg, *, sample: {
			"splits": {
				"train": {"row_count": 4, "query_count": 2},
				"val": {"row_count": 4, "query_count": 2},
				"test": {"row_count": 4, "query_count": 2},
			}
		},
	)


def test_memory_guard_stops_with_checkpoint_and_flushes_logs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	cfg = _ranker_cfg(tmp_path)
	train_path = tmp_path / "features" / "full" / "train" / "part_000000.parquet"
	val_path = tmp_path / "features" / "full" / "val" / "part_000000.parquet"
	test_path = tmp_path / "features" / "full" / "test" / "part_000000.parquet"
	_write_feature_split(train_path, split="train")
	_write_feature_split(val_path, split="val")
	_write_feature_split(test_path, split="test")

	_patch_trainer_artifacts(
		monkeypatch,
		train_path=train_path,
		val_path=val_path,
		test_path=test_path,
	)
	monkeypatch.setattr(trainer_module, "should_stop_for_memory", lambda *_a, **_k: True)

	log_file = tmp_path / "train_guard.log"
	logging.basicConfig(
		level=logging.INFO,
		handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8")],
		force=True,
	)

	result = trainer_module.train_ranker(
		cfg,
		sample=False,
		eval_every_epoch=True,
		checkpoint_every_epoch=True,
		save_last=True,
		log_file_path=log_file,
	)

	assert result.stopped_due_to_memory is True
	assert result.last_epoch_checkpoint is not None
	assert result.last_epoch_checkpoint.exists()
	assert log_file.exists()
	assert log_file.read_text(encoding="utf-8")


def test_full_smoke_uses_limited_train_rows(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	cfg = _ranker_cfg(tmp_path)
	train_path = tmp_path / "features" / "full" / "train" / "part_000000.parquet"
	val_path = tmp_path / "features" / "full" / "val" / "part_000000.parquet"
	test_path = tmp_path / "features" / "full" / "test" / "part_000000.parquet"
	_write_feature_split(train_path, split="train")
	_write_feature_split(val_path, split="val")
	_write_feature_split(test_path, split="test")

	_patch_trainer_artifacts(
		monkeypatch,
		train_path=train_path,
		val_path=val_path,
		test_path=test_path,
	)
	monkeypatch.setattr(trainer_module, "should_stop_for_memory", lambda *_a, **_k: False)

	captured_max_rows: dict[str, int | None] = {"value": None}
	original_iter = trainer_module.iter_ranker_feature_batches

	def _wrapped_iter(*args, **kwargs):
		captured_max_rows["value"] = kwargs.get("max_rows")
		yield from original_iter(*args, **kwargs)

	monkeypatch.setattr(trainer_module, "iter_ranker_feature_batches", _wrapped_iter)

	result = trainer_module.train_ranker(
		cfg,
		sample=False,
		full_smoke=True,
		max_train_rows=3,
		max_val_queries=1,
		eval_every_epoch=True,
		save_last=False,
		log_file_path=tmp_path / "train_smoke.log",
	)

	assert captured_max_rows["value"] == 3
	assert result.full_smoke is True
	assert result.completed_epochs == 1
	assert result.val_query_count <= 1


def test_evaluator_aggregates_query_chunks_and_writes_logs(tmp_path: Path) -> None:
	feature_a = tmp_path / "shards" / "part_000000.parquet"
	feature_b = tmp_path / "shards" / "part_000001.parquet"
	feature_a.parent.mkdir(parents=True, exist_ok=True)

	pl.DataFrame(
		{
			"query_id": ["q1", "q1", "q2"],
			"item_idx": [10, 20, 30],
			"target_item_idx": [10, 10, 30],
			"label": [1, 0, 1],
			"residual_score": [0.9, 0.4, 0.8],
			"residual_rank": [1, 2, 1],
		}
	).write_parquet(feature_a)
	pl.DataFrame(
		{
			"query_id": ["q3", "q3"],
			"item_idx": [50, 60],
			"target_item_idx": [50, 50],
			"label": [1, 0],
			"residual_score": [0.7, 0.2],
			"residual_rank": [1, 2],
		}
	).write_parquet(feature_b)

	checkpoint_path = tmp_path / "checkpoint.pt"
	model = NeuralRanker(input_dim=2, hidden_dims=[8, 4], dropout=0.0, use_layer_norm=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"hidden_dims": [8, 4],
			"dropout": 0.0,
			"use_layer_norm": True,
		},
		checkpoint_path,
	)

	log_file = tmp_path / "evaluate.log"
	logger = logging.getLogger("ranker-evaluator-test")
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	logger.propagate = False
	handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
	logger.addHandler(handler)

	result = evaluate_ranker_split(
		feature_paths=[feature_a, feature_b],
		feature_columns=["residual_score", "residual_rank"],
		checkpoint_path=checkpoint_path,
		split="val",
		report_dir=tmp_path,
		batch_size=2,
		log_every_queries=1,
		log_file_path=log_file,
		logger=logger,
	)
	handler.flush()
	handler.close()

	assert result.query_count == 3
	assert result.row_count == 5
	assert set(result.ranker_metrics.keys()) == {"hr@10", "mrr@10", "ndcg@10", "recall@50"}
	assert result.report_path.exists()
	assert result.scored_candidates_path.exists()
	assert log_file.exists()
	assert log_file.read_text(encoding="utf-8")
