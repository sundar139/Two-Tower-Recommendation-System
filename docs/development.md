# Development Guide

## Toolchain

- OS target: Windows 11 (PowerShell)
- Python: 3.12
- Package manager: uv

Setup:

```powershell
uv sync --extra dev
```

## PyTorch CUDA Note

This repository is configured to resolve PyTorch from the `cu128` index through uv:

- index URL: `https://download.pytorch.org/whl/cu128`
- source alias: `pytorch-cu128`

If wheel resolution fails in a specific environment, update the uv index configuration to the latest supported official PyTorch CUDA index and record the reason in this document before proceeding.

## Windows Troubleshooting

- Paths with spaces can break some uv trampoline entry points in certain setups.
- If a command like `uv run pytest` fails due trampoline path canonicalization, use module execution form:
	- `uv run python -m pytest -q`
	- `uv run python -m ruff check .`
	- `uv run python -m mypy src`

## Ollama Verification

- `scripts/verify_environment.py` performs an optional health check at `http://localhost:11434/api/tags`.
- Ollama offline status is reported as `WARN`, not a hard failure.

## Commit Workflow

- Commit after each successful implementation part.
- Keep commit scope narrow and message explicit.
- Never commit raw datasets, processed datasets, MLflow runs, artifacts, model checkpoints, or cache folders.
