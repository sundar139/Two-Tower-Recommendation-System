"""CLI for downloading and extracting MovieLens-25M."""

from __future__ import annotations

from pathlib import Path

import typer

from movie_recsys.config import load_data_config
from movie_recsys.data.download import DownloadError, download_movielens

app = typer.Typer(add_completion=False)


@app.command()
def main(
	config: Path = typer.Option(Path("configs/data.yaml"), "--config", help="Data config path."),
	force: bool = typer.Option(False, "--force", help="Re-download and re-extract dataset."),
	skip_checksum: bool = typer.Option(
		False,
		"--skip-checksum",
		help="Skip checksum validation even when expected checksum is configured.",
	),
) -> None:
	"""Download MovieLens zip and extract required files into data/raw."""

	data_config = load_data_config(config)
	try:
		summary = download_movielens(data_config, force=force, skip_checksum=skip_checksum)
	except DownloadError as exc:
		typer.secho(f"ERROR: {exc}", fg=typer.colors.RED)
		raise typer.Exit(code=1) from exc

	typer.echo("MovieLens download summary")
	typer.echo(f"  downloaded: {'yes' if summary.downloaded else 'no (already present)'}")
	typer.echo(f"  extracted: {'yes' if summary.extracted else 'no (already present)'}")
	if summary.checksum_status == "verified":
		typer.echo("  checksum: verified")
	elif summary.checksum_status == "skipped_by_flag":
		typer.echo("  checksum: skipped (--skip-checksum)")
	else:
		typer.echo("  checksum: skipped (expected checksum not configured)")
	typer.echo(f"  zip_path: {summary.zip_path}")
	typer.echo(f"  extract_dir: {summary.extract_dir}")
	typer.echo(f"  files_found ({len(summary.files_found)}): {', '.join(summary.files_found)}")
	if summary.missing_files:
		typer.secho(
			f"  missing_files ({len(summary.missing_files)}): {', '.join(summary.missing_files)}",
			fg=typer.colors.YELLOW,
		)
		raise typer.Exit(code=2)
	typer.echo("  missing_files (0): none")


if __name__ == "__main__":
	app()
