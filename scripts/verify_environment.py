"""Environment readiness checks for local development."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import asdict, dataclass

import httpx
import orjson

REQUIRED_OLLAMA_MODELS = ["qwen3:4b", "qwen3-embedding:0.6b"]


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


def _ollama_base_urls() -> list[str]:
	configured = [
		os.getenv("OLLAMA_BASE_URL"),
		os.getenv("OLLAMA_HOST"),
		"http://localhost:11434",
		"http://127.0.0.1:11434",
	]
	urls: list[str] = []
	for raw_url in configured:
		if raw_url is None:
			continue
		candidate = raw_url.strip().rstrip("/")
		if candidate and candidate not in urls:
			urls.append(candidate)
	return urls


def _ollama_available_models(base_url: str) -> tuple[set[str] | None, str | None]:
	try:
		resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
	except Exception as exc:  # noqa: BLE001
		return None, f"{base_url} unreachable: {exc}"

	if resp.status_code != 200:
		return None, f"{base_url} unexpected status: {resp.status_code}"

	payload = resp.json()
	models = payload.get("models", []) if isinstance(payload, dict) else []
	available = {
		str(model.get("name", "")).strip() for model in models if isinstance(model, dict)
	}
	return available, None


def _check_ollama() -> CheckResult:
	endpoint_errors: list[str] = []
	best_url: str | None = None
	best_available: set[str] = set()

	for base_url in _ollama_base_urls():
		available, error = _ollama_available_models(base_url)
		if error is not None:
			endpoint_errors.append(error)
			continue

		if available is None:
			continue

		missing = [name for name in REQUIRED_OLLAMA_MODELS if name not in available]
		if not missing:
			return CheckResult(
				"ollama",
				"PASS",
				f"reachable at {base_url} and required models installed",
				False,
			)

		if len(available) > len(best_available):
			best_available = available
			best_url = base_url

	pull_lines = "\n".join([f"ollama pull {name}" for name in REQUIRED_OLLAMA_MODELS])
	if best_url is not None:
		missing = [name for name in REQUIRED_OLLAMA_MODELS if name not in best_available]
		return CheckResult(
			"ollama",
			"WARN",
			"reachable at "
			f"{best_url} but missing required models: {', '.join(missing)}"
			f"\nInstall with:\n{pull_lines}",
			False,
		)

	error_detail = "; ".join(endpoint_errors) if endpoint_errors else "no endpoints checked"
	return CheckResult(
		"ollama",
		"WARN",
		"offline or unreachable: "
		f"{error_detail}\nWhen online, ensure models are installed:\n{pull_lines}",
		False,
	)


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
