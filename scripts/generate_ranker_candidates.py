"""Generate neural ranker candidates from residual retrieval outputs."""

from __future__ import annotations

from pathlib import Path

import typer

from movie_recsys.ranking.candidates import generate_ranker_candidates
from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.features import build_ranker_features

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
	feature_shard_size: int = typer.Option(500000, "--feature-shard-size"),
	resume_features: bool = typer.Option(False, "--resume-features"),
	overwrite_features: bool = typer.Option(False, "--overwrite-features"),
	generate_features: bool = typer.Option(False, "--generate-features"),
) -> None:
	ranker_cfg = load_ranker_config(config)
	selected_splits = tuple(
		part.strip().lower()
		for part in splits.split(",")
		if part.strip()
	)
	outputs: dict[str, Path] = {}

	candidate_paths = {
		split: ranker_cfg.candidate_path(split=split, sample=sample)
		for split in selected_splits
	}
	needs_candidate_generation = (
		overwrite
		or resume
		or max_users is not None
		or any(not path.exists() for path in candidate_paths.values())
	)
	if needs_candidate_generation:
		outputs.update(
			generate_ranker_candidates(
				ranker_cfg,
				sample=sample,
				chunk_size=chunk_size,
				resume=resume,
				max_users=max_users,
				splits=selected_splits,
				overwrite=overwrite,
				progress_every=progress_every,
			)
		)
	else:
		for split, path in candidate_paths.items():
			outputs[f"{split}_candidates"] = path

	features_requested = (
		generate_features
		or resume_features
		or overwrite_features
		or feature_shard_size != 500000
	)
	if features_requested:
		feature_outputs = build_ranker_features(
			ranker_cfg,
			sample=sample,
			feature_shard_size=feature_shard_size,
			resume=resume_features,
			overwrite=overwrite_features,
			splits=selected_splits,
		)
		outputs.update(feature_outputs)

	for key, path in outputs.items():
		typer.echo(f"{key}: {path}")


if __name__ == "__main__":
	app()
