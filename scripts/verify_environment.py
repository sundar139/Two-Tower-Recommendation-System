"""Environment readiness checks for local development."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import asdict, dataclass

import httpx
import orjson


@dataclass(slots=True)
class CheckResult:
	name: str
	status: str
	details: str
	required: bool


def _check_python_version() -> CheckResult:
	required = (3, 12)
	current = sys.version_info[:3]
	if current[0:2] != required:
		return CheckResult(
			"python_version",
			"FAIL",
			f"Expected Python {required[0]}.{required[1]}, found {platform.python_version()}",
			True,
		)
	return CheckResult("python_version", "PASS", platform.python_version(), True)


def _check_import(module_name: str, required: bool = True) -> CheckResult:
	try:
		importlib.import_module(module_name)
	except Exception as exc:  # noqa: BLE001
		return CheckResult(module_name, "FAIL" if required else "WARN", str(exc), required)
	return CheckResult(module_name, "PASS", "imported", required)


def _check_torch_cuda() -> list[CheckResult]:
	try:
		torch = importlib.import_module("torch")
	except Exception as exc:  # noqa: BLE001
		return [CheckResult("torch", "FAIL", str(exc), True)]

	cuda_available = bool(torch.cuda.is_available())
	cuda_ver = torch.version.cuda
	details = f"cuda_available={cuda_available}, torch_cuda_version={cuda_ver}"
	status = "PASS" if cuda_available else "WARN"
	return [
		CheckResult("torch", "PASS", "imported", True),
		CheckResult("torch_cuda", status, details, False),
	]


def _check_ollama() -> CheckResult:
	try:
		resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
		if resp.status_code == 200:
			return CheckResult("ollama", "PASS", "reachable at localhost:11434", False)
		return CheckResult("ollama", "WARN", f"unexpected status: {resp.status_code}", False)
	except Exception as exc:  # noqa: BLE001
		return CheckResult("ollama", "WARN", f"offline or unreachable: {exc}", False)


def collect_results(include_ollama: bool = True) -> list[CheckResult]:
	results: list[CheckResult] = []
	results.append(_check_python_version())
	results.extend(_check_torch_cuda())
	results.extend(
		[
			_check_import("faiss"),
			_check_import("polars"),
			_check_import("pyarrow"),
			_check_import("mlflow"),
			_check_import("optuna"),
			_check_import("httpx"),
		]
	)
	if include_ollama:
		results.append(_check_ollama())
	return results


def main() -> int:
	results = collect_results(include_ollama=True)

	for result in results:
		print(f"[{result.status}] {result.name}: {result.details}")

	summary = {
		"pass": sum(1 for r in results if r.status == "PASS"),
		"warn": sum(1 for r in results if r.status == "WARN"),
		"fail": sum(1 for r in results if r.status == "FAIL"),
	}
	print("\nSummary:", summary)
	print("Structured output:")
	print(orjson.dumps([asdict(r) for r in results], option=orjson.OPT_INDENT_2).decode("utf-8"))

	required_failures = [r for r in results if r.required and r.status == "FAIL"]
	return 1 if required_failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
