"""Compare residual retrieval ordering and neural ranker ordering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from movie_recsys.ranking.config import load_ranker_config
from movie_recsys.ranking.evaluator import write_comparison_markdown

app = typer.Typer(add_completion=False)


def _load_json(path: Path) -> dict[str, Any]:
	if not path.exists():
		msg = f"Required report not found: {path}"
		raise FileNotFoundError(msg)
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if not isinstance(payload, dict):
		msg = f"Expected object JSON in {path}"
		raise ValueError(msg)
	return payload


@app.command()
def main(
	config: Path = typer.Option(Path("configs/ranker.yaml"), "--config"),
	sample: bool = typer.Option(False, "--sample"),
) -> None:
	_ = sample
	ranker_cfg = load_ranker_config(config)
	report_dir = ranker_cfg.paths.ranker_report_dir
	val_path = report_dir / "ranker_eval_val.json"
	test_path = report_dir / "ranker_eval_test.json"
	output_md = report_dir / "ranker_comparison.md"

	val_report = _load_json(val_path)
	test_report = _load_json(test_path)
	write_comparison_markdown(
		val_report=val_report,
		test_report=test_report,
		output_path=output_md,
	)

	typer.echo(f"comparison_markdown: {output_md}")


if __name__ == "__main__":
	app()
