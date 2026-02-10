"""Tests for providers and model router."""

from __future__ import annotations

import pytest

from rde.models import ModelConfig, TraceConfig
from rde.providers.base import BaseProvider, LLMResponse
from rde.providers.router import ModelRouter


class MockProvider(BaseProvider):
    """A test provider that returns canned responses."""

    def __init__(self, family: str, response_text: str = "mock response"):
        self.family = family
        self.response_text = response_text
        self.call_count = 0

    async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
        self.call_count += 1
        return LLMResponse(
            content=self.response_text,
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=50.0,
        )

    def supports_model(self, model: str) -> bool:
        return self.family in model.lower()


def test_router_get_provider(monkeypatch):
    """Router finds the correct provider for a model string."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    router = ModelRouter(ModelConfig())
    # Inject mock providers
    mock_a = MockProvider("claude")
    mock_b = MockProvider("gpt")
    router._providers = {"anthropic": mock_a, "openai": mock_b}

    assert router.get_provider("claude-opus-4-6") is mock_a
    assert router.get_provider("gpt-5") is mock_b


def test_router_no_provider_raises(monkeypatch):
    """Router raises ValueError for unknown model."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    router = ModelRouter(ModelConfig())
    router._providers = {"anthropic": MockProvider("claude")}

    with pytest.raises(ValueError, match="No provider found"):
        router.get_provider("totally-unknown-model")


@pytest.mark.asyncio
async def test_router_complete(monkeypatch):
    """Router routes complete() to the correct provider."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    router = ModelRouter(ModelConfig())
    mock = MockProvider("claude", "\\boxed{42}")
    router._providers = {"anthropic": mock}

    response = await router.complete(
        messages=[{"role": "user", "content": "test"}],
        model="claude-sonnet-4-5-20250929",
    )
    assert response.content == "\\boxed{42}"
    assert mock.call_count == 1


def test_assign_trace_models_round_robin(monkeypatch):
    """Round-robin assignment cycles through available models."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    config = ModelConfig(
        trace_models=["claude-sonnet-4-5-20250929", "gpt-5", "gemini-2.5-pro"],
        trace_assignment="round_robin",
    )
    router = ModelRouter(config)
    router._providers = {
        "anthropic": MockProvider("claude"),
        "openai": MockProvider("gpt"),
        "google": MockProvider("gemini"),
    }

    traces = [
        TraceConfig(role="A", perspective="a", system_prompt="a"),
        TraceConfig(role="B", perspective="b", system_prompt="b"),
        TraceConfig(role="C", perspective="c", system_prompt="c"),
    ]
    assigned = router.assign_trace_models(traces)
    assert assigned == ["claude-sonnet-4-5-20250929", "gpt-5", "gemini-2.5-pro"]


def test_assign_trace_models_with_preference(monkeypatch):
    """Model preference on a trace overrides round-robin."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    config = ModelConfig(
        trace_models=["claude-sonnet-4-5-20250929", "gpt-5"],
        trace_assignment="round_robin",
    )
    router = ModelRouter(config)
    router._providers = {
        "anthropic": MockProvider("claude"),
        "openai": MockProvider("gpt"),
    }

    traces = [
        TraceConfig(role="A", perspective="a", system_prompt="a", model_preference="gpt-5"),
        TraceConfig(role="B", perspective="b", system_prompt="b"),
    ]
    assigned = router.assign_trace_models(traces)
    assert assigned[0] == "gpt-5"  # preference honored
    # round-robin: index 1 % 2 models = index 1 → gpt-5
    assert assigned[1] == "gpt-5"


def test_assign_no_available_models_raises(monkeypatch):
    """Raises ValueError when no configured trace models have providers."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    config = ModelConfig(trace_models=["nonexistent-model"])
    router = ModelRouter(config)
    router._providers = {"anthropic": MockProvider("claude")}

    traces = [TraceConfig(role="A", perspective="a", system_prompt="a")]
    with pytest.raises(ValueError, match="No trace models available"):
        router.assign_trace_models(traces)
