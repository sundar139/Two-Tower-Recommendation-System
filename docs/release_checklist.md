# Release Checklist

## Quality Gates

- [ ] `Copy-Item env.example .env -Force` executed
- [ ] `uv run python verify.py` passes (`10 pass, 0 warn, 0 fail`)
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest -q` passes

## Serving Validation

- [ ] Core serving validation report confirms `18/18` passed
- [ ] Latency benchmark report is present and reviewed (`p50/p95/max`)
- [ ] Response schema parity checks pass for endpoint aliases

## Docker Validation

- [ ] Artifact preflight check passes
- [ ] `docker compose build` succeeds
- [ ] Docker smoke test report confirms `6/6` passed

## Ollama Validation

- [ ] Local Ollama daemon reachable
- [ ] Required models available (`qwen3:4b`, `qwen3-embedding:0.6b`)
- [ ] Explanation validation report confirms `9/9` passed
- [ ] Ranking invariance checks pass with explanations enabled

## Artifact Hygiene

- [ ] `.env` is ignored by Git
- [ ] `mlflow.db` is ignored by Git
- [ ] Generated artifacts remain untracked (`artifacts`, `mlruns`, checkpoints, caches)
- [ ] Only data directory `.gitkeep` files are tracked

## Documentation

- [ ] README reflects final approved architecture and metrics
- [ ] `docs/architecture.md` diagrams render as valid Mermaid
- [ ] `docs/evidence.md` tables align with artifact reports
- [ ] `docs/project_status.md` reflects approved vs experimental scope
- [ ] `docs/project_log.md` captures major milestones accurately

## Final Repository Check

- [ ] README and docs links reviewed
- [ ] `git status --short` is clean
- [ ] No uncommitted generated files
