"""System runtime helpers for memory-aware long-running jobs."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
	import psutil
except ImportError:  # pragma: no cover - dependency may be optional in some environments.
	psutil = None


@dataclass(slots=True)
class MemoryStatus:
	ram_percent: float | None
	ram_used_gb: float | None
	ram_total_gb: float | None
	pagefile_percent: float | None
	pagefile_used_gb: float | None
	pagefile_total_gb: float | None
	disk_free_gb: float | None


def _bytes_to_gb(value: float | int) -> float:
	return float(value) / (1024.0**3)


def get_memory_status(*, disk_path: Path | None = None) -> MemoryStatus:
	"""Return current RAM/pagefile usage and free disk space for runtime guardrails."""

	disk_root = disk_path if disk_path is not None else Path.cwd()
	try:
		disk_usage = shutil.disk_usage(disk_root)
		disk_free_gb = _bytes_to_gb(disk_usage.free)
	except (FileNotFoundError, OSError):
		disk_free_gb = None

	if psutil is None:
		return MemoryStatus(
			ram_percent=None,
			ram_used_gb=None,
			ram_total_gb=None,
			pagefile_percent=None,
			pagefile_used_gb=None,
			pagefile_total_gb=None,
			disk_free_gb=disk_free_gb,
		)

	virtual = psutil.virtual_memory()
	swap = psutil.swap_memory()
	return MemoryStatus(
		ram_percent=float(virtual.percent),
		ram_used_gb=_bytes_to_gb(virtual.used),
		ram_total_gb=_bytes_to_gb(virtual.total),
		pagefile_percent=float(swap.percent) if swap.total > 0 else None,
		pagefile_used_gb=_bytes_to_gb(swap.used) if swap.total > 0 else None,
		pagefile_total_gb=_bytes_to_gb(swap.total) if swap.total > 0 else None,
		disk_free_gb=disk_free_gb,
	)


def log_memory_status(
	prefix: str,
	*,
	logger: logging.Logger | None = None,
	disk_path: Path | None = None,
) -> MemoryStatus:
	"""Emit memory/disk telemetry and return the sampled values."""

	status = get_memory_status(disk_path=disk_path)
	parts = [prefix]
	if status.ram_percent is not None:
		parts.append(
			"ram="
			f"{status.ram_percent:.1f}%"
			f" ({(status.ram_used_gb or 0.0):.2f}GiB/{(status.ram_total_gb or 0.0):.2f}GiB)"
		)
	if status.pagefile_percent is not None:
		page_used = status.pagefile_used_gb or 0.0
		page_total = status.pagefile_total_gb or 0.0
		parts.append(
			"pagefile="
			f"{status.pagefile_percent:.1f}%"
			f" ({page_used:.2f}GiB/{page_total:.2f}GiB)"
		)
	if status.disk_free_gb is not None:
		parts.append(f"disk_free={status.disk_free_gb:.2f}GiB")
	message = " | ".join(parts)
	if logger is not None:
		logger.info(message)
	else:
		print(message, flush=True)
	return status


def should_stop_for_memory(
	max_ram_percent: float,
	max_pagefile_percent: float | None = None,
	*,
	disk_path: Path | None = None,
) -> bool:
	"""Return True when memory thresholds indicate the run should stop safely."""

	status = get_memory_status(disk_path=disk_path)
	if status.ram_percent is not None and status.ram_percent >= max_ram_percent:
		return True
	return bool(
		max_pagefile_percent is not None
		and status.pagefile_percent is not None
		and status.pagefile_percent >= max_pagefile_percent
	)
