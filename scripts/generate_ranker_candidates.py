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
) -> None:
	ranker_cfg = load_ranker_config(config)
	outputs = generate_ranker_candidates(ranker_cfg, sample=sample)
	for key, path in outputs.items():
		typer.echo(f"{key}: {path}")


if __name__ == "__main__":
	app()
