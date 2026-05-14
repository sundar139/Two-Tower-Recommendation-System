"""MovieLens download and extraction utilities."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

from movie_recsys.config import DataConfig
from movie_recsys.constants import REQUIRED_MOVIELENS_FILES


@dataclass(slots=True)
class DownloadSummary:
	downloaded: bool
	extracted: bool
	files_found: list[str]
	missing_files: list[str]
	zip_path: Path
	extract_dir: Path


class DownloadError(RuntimeError):
	"""Raised when download or extraction fails."""


def download_movielens(
	config: DataConfig,
	*,
	force: bool = False,
	skip_checksum: bool = False,
) -> DownloadSummary:
	"""Download and extract MovieLens-25M dataset idempotently."""

	raw_dir = config.raw_data_dir
	raw_dir.mkdir(parents=True, exist_ok=True)
	zip_path = raw_dir / config.expected_zip_name
	extract_dir = raw_dir / "ml-25m"

	downloaded = False
	extracted = False

	if force and zip_path.exists():
		zip_path.unlink()
	if force and extract_dir.exists():
		for path in sorted(extract_dir.rglob("*"), reverse=True):
			if path.is_file():
				path.unlink()
			else:
				path.rmdir()
		extract_dir.rmdir()

	if not zip_path.exists():
		_stream_download(config.movielens_url, zip_path)
		downloaded = True

	if not skip_checksum and config.expected_checksum:
		actual = _sha256(zip_path)
		expected = config.expected_checksum.lower().strip()
		if actual != expected:
			raise DownloadError(
				"Checksum mismatch for downloaded archive. "
				f"Expected {expected}, got {actual}."
			)

	if not extract_dir.exists() or force:
		_safe_extract_zip(zip_path, raw_dir)
		extracted = True

	files_found, missing_files = _check_required_files(extract_dir)
	return DownloadSummary(
		downloaded=downloaded,
		extracted=extracted,
		files_found=files_found,
		missing_files=missing_files,
		zip_path=zip_path,
		extract_dir=extract_dir,
	)


def _stream_download(url: str, target_path: Path) -> None:
	target_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
			response.raise_for_status()
			total = int(response.headers.get("Content-Length", "0") or 0)
			with target_path.open("wb") as file_obj, tqdm(
				desc=f"Downloading {target_path.name}",
				total=total,
				unit="B",
				unit_scale=True,
				unit_divisor=1024,
			) as progress:
				for chunk in response.iter_bytes(chunk_size=1024 * 64):
					if not chunk:
						continue
					file_obj.write(chunk)
					progress.update(len(chunk))
	except httpx.HTTPError as exc:
		if target_path.exists():
			target_path.unlink(missing_ok=True)
		raise DownloadError(f"Failed to download MovieLens archive: {exc}") from exc


def _sha256(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open("rb") as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b""):
			hasher.update(block)
	return hasher.hexdigest()


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
	try:
		with zipfile.ZipFile(zip_path) as archive:
			for member in archive.infolist():
				member_path = output_dir / member.filename
				resolved = member_path.resolve()
				if not str(resolved).startswith(str(output_dir.resolve())):
					raise DownloadError(f"Blocked unsafe zip path: {member.filename}")
			archive.extractall(output_dir)
	except zipfile.BadZipFile as exc:
		raise DownloadError("Corrupted zip archive. Re-download with --force.") from exc


def _check_required_files(extract_dir: Path) -> tuple[list[str], list[str]]:
	found: list[str] = []
	missing: list[str] = []
	for name in REQUIRED_MOVIELENS_FILES:
		if (extract_dir / name).exists():
			found.append(name)
		else:
			missing.append(name)
	return found, missing
