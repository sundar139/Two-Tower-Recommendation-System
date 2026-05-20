"""Run local FastAPI serving app for recommendation inference."""

from __future__ import annotations

import typer
import uvicorn

from movie_recsys.serving.config import load_serving_config

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    """Start FastAPI app using serving config host and port."""

    config = load_serving_config()
    uvicorn.run(
        "movie_recsys.serving.app:create_app",
        factory=True,
        host=config.api.host,
        port=config.api.port,
        log_level=config.api.log_level,
    )


if __name__ == "__main__":
    app()
