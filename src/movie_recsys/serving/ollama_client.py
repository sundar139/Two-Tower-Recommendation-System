"""Lightweight local Ollama client for recommendation explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class OllamaClientConfig:
    """Connection settings for local Ollama API calls."""

    base_url: str
    chat_model: str
    embedding_model: str = "qwen3-embedding:0.6b"
    timeout_seconds: float = 30.0
    temperature: float = 0.2


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama cannot be reached or times out."""


class OllamaClient:
    """Minimal JSON API client for Ollama health, generation, and embeddings."""

    def __init__(
        self,
        config: OllamaClientConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = http_client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
        )
        self._owns_client = http_client is None

    @property
    def config(self) -> OllamaClientConfig:
        return self._config

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def check_health(self) -> bool:
        response = self._request("GET", "/api/tags")
        return response.status_code == 200

    def generate_explanation(self, prompt: str) -> str:
        payload = {
            "model": self._config.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._config.temperature},
        }
        response = self._request("POST", "/api/generate", json_payload=payload)
        body = self._parse_json_body(response)
        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            msg = "Ollama response did not include a non-empty 'response' field"
            raise ValueError(msg)
        return text.strip()

    def embed_text(self, text: str) -> list[float]:
        payload = {
            "model": self._config.embedding_model,
            "input": text,
        }
        response = self._request("POST", "/api/embed", json_payload=payload)
        body = self._parse_json_body(response)

        if isinstance(body.get("embeddings"), list) and body["embeddings"]:
            first = body["embeddings"][0]
            if isinstance(first, list):
                return [float(value) for value in first]

        if isinstance(body.get("embedding"), list):
            return [float(value) for value in body["embedding"]]

        msg = "Ollama embed response did not include embedding values"
        raise ValueError(msg)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(method, path, json=json_payload)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

        if response.status_code >= 500:
            raise OllamaUnavailableError(
                f"Ollama returned server error {response.status_code}: {response.text}"
            )
        if response.status_code >= 400:
            msg = f"Ollama returned error {response.status_code}: {response.text}"
            raise ValueError(msg)
        return response

    @staticmethod
    def _parse_json_body(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            msg = "Ollama response JSON must be an object"
            raise ValueError(msg)
        return payload
