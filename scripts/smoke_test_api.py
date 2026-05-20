"""Basic API smoke test script for local development."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    """Placeholder smoke test command."""

    typer.echo("Smoke test scaffold is in place.")


if __name__ == "__main__":
    app()
