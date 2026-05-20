"""Lightweight telemetry helpers for API runtime logging."""

from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str = "movie_recsys.serving") -> logging.Logger:
    """Return a process-wide logger for serving modules."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured log line as compact JSON."""

    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, sort_keys=True, default=str))
