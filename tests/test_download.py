from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from movie_recsys.config import DataConfig
from movie_recsys.data.download import DownloadError, _safe_extract_zip, download_movielens


def _write_zip(path: Path, entries: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def test_safe_extract_blocks_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    _write_zip(zip_path, {"../escape.txt": "nope"})

    with pytest.raises(DownloadError):
        _safe_extract_zip(zip_path, tmp_path / "extract")


def test_download_skips_network_when_zip_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_root = raw_dir / "ml-25m"
    extract_root.mkdir(parents=True, exist_ok=True)
    for file_name in [
        "ratings.csv",
        "movies.csv",
        "tags.csv",
        "genome-scores.csv",
        "genome-tags.csv",
        "links.csv",
    ]:
        (extract_root / file_name).write_text("x\n", encoding="utf-8")

    zip_path = raw_dir / "ml-25m.zip"
    _write_zip(zip_path, {"ml-25m/ratings.csv": "userId,movieId,rating,timestamp\n"})

    def _raise_stream(*args: object, **kwargs: object) -> io.BytesIO:
        raise AssertionError("network should not be used when zip already exists")

    monkeypatch.setattr("movie_recsys.data.download.httpx.stream", _raise_stream)

    cfg = DataConfig(raw_data_dir=raw_dir).resolve_paths(tmp_path)
    summary = download_movielens(cfg)
    assert summary.downloaded is False
    assert summary.extracted is False
    assert summary.missing_files == []