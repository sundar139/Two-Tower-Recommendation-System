from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from movie_recsys.modeling.artifacts import load_checkpoint, save_checkpoint
from movie_recsys.modeling.trainer import (
    _build_training_checkpoint_payload,
    _load_resume_training_state,
    train_retriever,
)
from movie_recsys.training.config import load_retrieval_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load module spec for {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acceptance_module = _load_script_module(
    "check_residual_acceptance",
    PROJECT_ROOT / "scripts" / "check_residual_acceptance.py",
)
runner_module = _load_script_module(
    "run_full_residual_training",
    PROJECT_ROOT / "scripts" / "run_full_residual_training.py",
)

evaluate_acceptance = acceptance_module.evaluate_acceptance
build_train_command = runner_module.build_train_command
build_eval_command = runner_module.build_eval_command
build_export_command = runner_module.build_export_command


def _minimal_config(tmp_path: Path):
    cfg = load_retrieval_config("configs/retrieval.yaml", sample=True)
    cfg.paths.model_output_dir = tmp_path / "models"
    cfg.paths.report_output_dir = tmp_path / "reports"
    cfg.paths.index_output_dir = tmp_path / "faiss"
    cfg.train.num_workers = 0
    return cfg


def test_checkpoint_payload_roundtrip_contains_resume_fields(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    payload = _build_training_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        best_metric=0.123,
        best_checkpoint_path=tmp_path / "best.pt",
        config=cfg,
        model_type="residual_transformer",
        mlflow_run_id="run-123",
        best_metrics={"ndcg@10": 0.123},
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, payload)

    loaded = load_checkpoint(checkpoint_path)
    assert loaded["epoch"] == 3
    assert loaded["best_metric"] == 0.123
    assert loaded["model_type"] == "residual_transformer"
    assert loaded["mlflow_run_id"] == "run-123"
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert "scheduler_state_dict" in loaded
    assert "best_checkpoint_path" in loaded
    assert "config" in loaded
    assert "random_seed" in loaded


def test_resume_state_starts_from_next_epoch(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)

    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    sample = torch.randn(2, 3)
    loss = model(sample).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    resume_payload = _build_training_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        best_metric=0.42,
        best_checkpoint_path=tmp_path / "best.pt",
        config=cfg,
        model_type="baseline",
        mlflow_run_id="run-xyz",
        best_metrics={"ndcg@10": 0.42},
    )
    resume_path = tmp_path / "resume.pt"
    save_checkpoint(resume_path, resume_payload)

    model_resumed = torch.nn.Linear(3, 3)
    optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
    scheduler_resumed = torch.optim.lr_scheduler.StepLR(optimizer_resumed, step_size=1)

    state = _load_resume_training_state(
        resume_from=resume_path,
        model=model_resumed,
        optimizer=optimizer_resumed,
        scheduler=scheduler_resumed,
    )

    assert state["start_epoch"] == 6
    assert float(state["best_metric"]) == 0.42
    assert state["mlflow_run_id"] == "run-xyz"
    assert optimizer_resumed.state_dict()["state"]

    for current, restored in zip(model.parameters(), model_resumed.parameters(), strict=True):
        assert torch.allclose(current, restored)


def test_max_runtime_stop_writes_last_checkpoint(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    cfg.train.epochs = 3
    cfg.train.train_batch_size = 1024

    result = train_retriever(
        cfg,
        sample=True,
        model_type="baseline",
        max_runtime_hours=1e-8,
        save_last=True,
        checkpoint_every_epoch=False,
        eval_every_epoch=False,
        run_name="runtime_stop_test",
    )

    assert result.stopped_due_to_runtime
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.exists()

    checkpoint = load_checkpoint(result.last_checkpoint)
    assert "optimizer_state_dict" in checkpoint
    assert "model_state_dict" in checkpoint
    assert "epoch" in checkpoint


def test_acceptance_checker_rules_pass_and_fail() -> None:
    baseline_val = {"hr@10": 0.10, "mrr@10": 0.04, "ndcg@10": 0.05, "recall@50": 0.20}
    baseline_test = {"hr@10": 0.09, "mrr@10": 0.03, "ndcg@10": 0.04, "recall@50": 0.18}
    popularity_val = {"hr@10": 0.08, "mrr@10": 0.03, "ndcg@10": 0.04, "recall@50": 0.17}
    popularity_test = {"hr@10": 0.07, "mrr@10": 0.02, "ndcg@10": 0.03, "recall@50": 0.16}

    pass_result = evaluate_acceptance(
        residual_val={"hr@10": 0.11, "mrr@10": 0.05, "ndcg@10": 0.051, "recall@50": 0.21},
        residual_test={"hr@10": 0.10, "mrr@10": 0.04, "ndcg@10": 0.041, "recall@50": 0.19},
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        popularity_val=popularity_val,
        popularity_test=popularity_test,
    )
    assert pass_result["acceptance_passed"]
    assert pass_result["cl_epidtn_unblocked"]

    fail_result = evaluate_acceptance(
        residual_val={"hr@10": 0.09, "mrr@10": 0.03, "ndcg@10": 0.045, "recall@50": 0.19},
        residual_test={"hr@10": 0.08, "mrr@10": 0.02, "ndcg@10": 0.035, "recall@50": 0.17},
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        popularity_val=popularity_val,
        popularity_test=popularity_test,
    )
    assert not fail_result["acceptance_passed"]
    assert not fail_result["cl_epidtn_unblocked"]
    assert fail_result["residual_should_remain_experimental"]


def test_full_runner_builds_expected_commands() -> None:
    train_command = build_train_command(
        config=Path("configs/transformer_retrieval_residual.yaml"),
        baseline_checkpoint=Path("artifacts/models/best_baseline_retriever.pt"),
        resume_from=Path("artifacts/models/checkpoints/residual_transformer_epoch_3.pt"),
        max_runtime_hours=4.0,
        run_name="full_residual_test",
    )
    joined_train = " ".join(train_command).replace("\\", "/")
    assert "scripts/train_retriever.py" in joined_train
    assert "--model-type residual_transformer" in joined_train
    assert "--init-from-baseline artifacts/models/best_baseline_retriever.pt" in joined_train
    assert "--checkpoint-every-epoch" in joined_train
    assert "--eval-every-epoch" in joined_train
    assert (
        "--resume-from artifacts/models/checkpoints/residual_transformer_epoch_3.pt"
        in joined_train
    )
    assert "--max-runtime-hours 4.0" in joined_train

    eval_command = build_eval_command(
        config=Path("configs/transformer_retrieval_residual.yaml"),
        model="residual_transformer",
        split="val",
        checkpoint=Path("artifacts/models/best_residual_transformer_retriever.pt"),
    )
    joined_eval = " ".join(eval_command).replace("\\", "/")
    assert "scripts/evaluate_retriever.py" in joined_eval
    assert "--model residual_transformer" in joined_eval
    assert "--split val" in joined_eval

    export_command = build_export_command(
        config=Path("configs/transformer_retrieval_residual.yaml"),
        checkpoint=Path("artifacts/models/best_residual_transformer_retriever.pt"),
    )
    joined_export = " ".join(export_command).replace("\\", "/")
    assert "scripts/export_faiss_index.py" in joined_export
    assert "--model-type residual_transformer" in joined_export
