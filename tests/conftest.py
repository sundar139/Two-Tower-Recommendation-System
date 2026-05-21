"""Pytest bootstrap configuration shared across test modules."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root_on_path()
