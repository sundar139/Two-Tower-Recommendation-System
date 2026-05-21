"""Tests for local Ollama client behavior."""

from __future__ import annotations

import httpx
import pytest

from movie_recsys.serving.ollama_client import (
    OllamaClient,
    OllamaClientConfig,
    OllamaUnavailableError,
)


def _client(handler: httpx.MockTransport) -> OllamaClient:
    config = OllamaClientConfig(
        base_url="http://127.0.0.1:11434",
        chat_model="qwen3:4b",
        embedding_model="qwen3-embedding:0.6b",
        timeout_seconds=2.0,
        temperature=0.2,
    )
    http_client = httpx.Client(
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        transport=handler,
    )
    return OllamaClient(config=config, http_client=http_client)


def test_check_health_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(status_code=200, json={"models": []})
        return httpx.Response(status_code=404, json={"error": "unexpected"})

    client = _client(httpx.MockTransport(handler))
    assert client.check_health() is True


def test_generate_explanation_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/generate":
            return httpx.Response(status_code=200, json={"response": "Explanation text"})
        return httpx.Response(status_code=404, json={"error": "unexpected"})

    client = _client(httpx.MockTransport(handler))
    text = client.generate_explanation("Explain this recommendation")
    assert text == "Explanation text"


def test_generate_explanation_timeout_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(OllamaUnavailableError):
        client.generate_explanation("Explain this recommendation")


def test_generate_explanation_unavailable_server_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(OllamaUnavailableError):
        client.generate_explanation("Explain this recommendation")


def test_embed_text_supports_embeddings_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/embed":
            return httpx.Response(status_code=200, json={"embeddings": [[0.1, 0.2, 0.3]]})
        return httpx.Response(status_code=404, json={"error": "unexpected"})

    client = _client(httpx.MockTransport(handler))
    vector = client.embed_text("A test sentence")
    assert vector == [0.1, 0.2, 0.3]


def test_generate_explanation_rejects_empty_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/generate":
            return httpx.Response(status_code=200, json={"response": ""})
        return httpx.Response(status_code=404, json={"error": "unexpected"})

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        client.generate_explanation("Explain this recommendation")
