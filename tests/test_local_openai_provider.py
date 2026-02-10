"""Tests for the LocalOpenAIProvider (vLLM-mlx compatible)."""

import json

import pytest

from rde.providers.local_openai_provider import LocalOpenAIProvider


def _mock_chat_response(content="Hello!", model="default", prompt_tokens=10, completion_tokens=5):
    """Build a mock OpenAI-compatible chat/completions response."""
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class _FakeHTTPResponse:
    """Minimal mock of httpx.Response."""

    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def json(self):
        return self._data


@pytest.mark.asyncio
async def test_complete_basic(monkeypatch):
    """Basic chat completion returns proper LLMResponse."""
    provider = LocalOpenAIProvider(base_url="http://localhost:9999/v1", model="test-model")

    mock_response = _FakeHTTPResponse(_mock_chat_response("Test output", "test-model"))

    async def mock_post(self, url, headers=None, json=None):
        return mock_response

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await provider.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="test-model",
        temperature=0.5,
        max_tokens=100,
    )

    assert result.content == "Test output"
    assert result.model == "test-model"
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 5
    assert result.estimated_cost == 0.0
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_complete_local_model_alias(monkeypatch):
    """When model='local', uses default_model instead."""
    provider = LocalOpenAIProvider(base_url="http://localhost:9999/v1", model="qwen3-8b")

    captured_payload = {}

    async def mock_post(self, url, headers=None, json=None):
        captured_payload.update(json)
        return _FakeHTTPResponse(_mock_chat_response("ok", "qwen3-8b"))

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await provider.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="local",
    )

    assert captured_payload["model"] == "qwen3-8b"


@pytest.mark.asyncio
async def test_complete_http_error(monkeypatch):
    """HTTP errors raise RuntimeError."""
    provider = LocalOpenAIProvider(base_url="http://localhost:9999/v1")

    async def mock_post(self, url, headers=None, json=None):
        return _FakeHTTPResponse({"error": "not found"}, status_code=404)

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(RuntimeError, match="Local server error 404"):
        await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="test",
        )


def test_supports_model():
    provider = LocalOpenAIProvider(model="qwen3-8b")
    assert provider.supports_model("local")
    assert provider.supports_model("qwen3-8b")
    assert provider.supports_model("QWEN3-8B")  # Case insensitive
    assert provider.supports_model("vllm-local")
    assert provider.supports_model("mlx-model")
    assert provider.supports_model("~/Models/something")
    assert provider.supports_model("/absolute/path")
    assert not provider.supports_model("gpt-5")
    assert not provider.supports_model("claude-sonnet")


def test_supports_model_default():
    provider = LocalOpenAIProvider(model="default")
    assert provider.supports_model("local")
    assert provider.supports_model("default")


@pytest.mark.asyncio
async def test_complete_with_response_format(monkeypatch):
    """response_format is forwarded in payload."""
    provider = LocalOpenAIProvider(base_url="http://localhost:9999/v1")

    captured_payload = {}

    async def mock_post(self, url, headers=None, json=None):
        captured_payload.update(json)
        return _FakeHTTPResponse(_mock_chat_response('{"answer": 42}'))

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await provider.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="default",
        response_format={"type": "json_object"},
    )

    assert captured_payload["response_format"] == {"type": "json_object"}
