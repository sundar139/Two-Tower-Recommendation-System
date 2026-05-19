"""Generate neural ranker candidates from residual retrieval outputs."""

from __future__ import annotations

from pathlib import Path

import typer

from movie_recsys.ranking.candidates import generate_ranker_candidates
from movie_recsys.ranking.config import load_ranker_config

app = typer.Typer(add_completion=False)


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	sample: bool = typer.Option(False, "--sample"),
	chunk_size: int = typer.Option(5000, "--chunk-size"),
	resume: bool = typer.Option(False, "--resume"),
	max_users: int | None = typer.Option(None, "--max-users"),
	splits: str = typer.Option("train,val,test", "--splits"),
	overwrite: bool = typer.Option(False, "--overwrite"),
	progress_every: int = typer.Option(1000, "--progress-every"),
) -> None:
	ranker_cfg = load_ranker_config(config)
	selected_splits = tuple(
		part.strip().lower()
		for part in splits.split(",")
		if part.strip()
	)
	outputs = generate_ranker_candidates(
		ranker_cfg,
		sample=sample,
		chunk_size=chunk_size,
		resume=resume,
		max_users=max_users,
		splits=selected_splits,
		overwrite=overwrite,
		progress_every=progress_every,
	)
	for key, path in outputs.items():
		typer.echo(f"{key}: {path}")


if __name__ == "__main__":
	app()
