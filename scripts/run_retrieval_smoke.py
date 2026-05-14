"""Quick smoke command for retrieval baseline workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(config: Path = typer.Option(Path("configs/retrieval.yaml"), "--config")) -> None:
    commands = [
        [
            sys.executable,
            "scripts/evaluate_retriever.py",
            "--config",
            str(config),
            "--model",
            "popularity",
            "--split",
            "val",
            "--sample",
        ],
        [sys.executable, "scripts/train_retriever.py", "--config", str(config), "--sample"],
        [
            sys.executable,
            "scripts/evaluate_retriever.py",
            "--config",
            str(config),
            "--model",
            "two_tower",
            "--split",
            "val",
            "--sample",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    app()
