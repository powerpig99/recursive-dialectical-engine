"""Tests for the TraceExecutor — trace execution and context strategies."""

from __future__ import annotations

import pytest

from rde.environment import ContextEnvironment
from rde.models import RecursionBudget, TraceConfig, TraceResult
from rde.providers.base import LLMResponse
from rde.trace import TraceExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRouter:
    """Minimal mock of ModelRouter for TraceExecutor tests."""

    def __init__(self, content: str = "Default response", config=None):
        self._content = content
        self.config = config or _fake_model_config()

    async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
        return LLMResponse(content=self._content, model=model, latency_ms=10.0)


class _FakeConfig:
    """Minimal stand-in for ModelConfig attributes accessed by TraceExecutor."""

    def __init__(self):
        self.orchestrator_model = "test-model"
        self.arbiter_model = "test-model"
        self.trace_models = ["test-a", "test-b"]
        self.sub_lm_models = ["test-sub"]
        self.trace_assignment = "round_robin"


def _fake_model_config():
    return _FakeConfig()


def _default_trace_config(**overrides) -> TraceConfig:
    """Build a default TraceConfig for tests."""
    defaults = {
        "role": "Logician",
        "perspective": "Formal logic",
        "system_prompt": "You are a logician.",
        "context_strategy": "full",
        "temperature": 0.7,
        "model_preference": "any",
    }
    defaults.update(overrides)
    return TraceConfig(**defaults)


# ---------------------------------------------------------------------------
# TraceExecutor.execute tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_trace_result():
    """Basic execution produces a TraceResult."""
    router = FakeRouter("Analysis... the answer is clear.")
    env = ContextEnvironment("What is 2+2?")
    executor = TraceExecutor(router, env)

    result = await executor.execute(_default_trace_config(), "test-model")

    assert isinstance(result, TraceResult)
    assert result.role == "Logician"
    assert result.model_used == "test-model"
    assert result.error is None
    assert result.raw_output == "Analysis... the answer is clear."


@pytest.mark.asyncio
async def test_execute_extracts_boxed_answer():
    """Boxed answers are parsed from raw output."""
    router = FakeRouter("The result is \\boxed{42}.")
    env = ContextEnvironment("What is the answer?")
    executor = TraceExecutor(router, env)

    result = await executor.execute(_default_trace_config(), "test-model")

    assert result.extracted_answer == "42"


@pytest.mark.asyncio
async def test_execute_handles_error():
    """Provider error is captured gracefully in TraceResult."""

    class ErrorRouter:
        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            raise RuntimeError("API connection failed")

    env = ContextEnvironment("test prompt")
    executor = TraceExecutor(ErrorRouter(), env)

    result = await executor.execute(_default_trace_config(), "test-model")

    assert result.error is not None
    assert "API connection failed" in result.error
    assert result.raw_output == ""


# ---------------------------------------------------------------------------
# Context strategy tests (_build_user_prompt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_prompt_full_strategy():
    """'full' context strategy passes entire prompt."""
    router = FakeRouter()
    env = ContextEnvironment("The full problem text here.")
    executor = TraceExecutor(router, env)

    config = _default_trace_config(context_strategy="full")
    prompt = executor._build_user_prompt(config)

    assert prompt == "The full problem text here."


@pytest.mark.asyncio
async def test_build_prompt_partition_strategy():
    """'partition:structural' splits and returns all parts joined."""
    router = FakeRouter()
    env = ContextEnvironment("Part one.\n\nPart two.\n\nPart three.")
    executor = TraceExecutor(router, env)

    config = _default_trace_config(context_strategy="partition:structural")
    prompt = executor._build_user_prompt(config)

    # Should contain the full text (all partitions joined)
    assert "Part one" in prompt
    assert "Part two" in prompt
    assert "Part three" in prompt


@pytest.mark.asyncio
async def test_build_prompt_partition_index():
    """'partition:structural:0' returns only the first partition."""
    router = FakeRouter()
    env = ContextEnvironment("Part one.\n\nPart two.\n\nPart three.")
    executor = TraceExecutor(router, env)

    config = _default_trace_config(context_strategy="partition:structural:0")
    prompt = executor._build_user_prompt(config)

    assert "Section 1/" in prompt
    assert "Part one" in prompt


@pytest.mark.asyncio
async def test_build_prompt_search_strategy():
    """'search:pattern' filters context by regex."""
    router = FakeRouter()
    env = ContextEnvironment("The cat sat on the mat. The dog ran in the park.")
    executor = TraceExecutor(router, env)

    config = _default_trace_config(context_strategy="search:cat")
    prompt = executor._build_user_prompt(config)

    assert "cat" in prompt
    assert "Relevant excerpts" in prompt


@pytest.mark.asyncio
async def test_build_prompt_search_no_match_fallback():
    """'search:nonexistent' falls back to full prompt when no matches."""
    router = FakeRouter()
    env = ContextEnvironment("The cat sat on the mat.")
    executor = TraceExecutor(router, env)

    config = _default_trace_config(context_strategy="search:zzz_no_match_zzz")
    prompt = executor._build_user_prompt(config)

    assert prompt == "The cat sat on the mat."


# ---------------------------------------------------------------------------
# Budget tracking tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_recorded_after_call():
    """budget.total_calls is incremented after execution."""
    router = FakeRouter("Result \\boxed{42}")
    env = ContextEnvironment("test")
    budget = RecursionBudget()
    executor = TraceExecutor(router, env, budget=budget)

    assert budget.total_calls == 0
    await executor.execute(_default_trace_config(), "test-model")
    assert budget.total_calls == 1


@pytest.mark.asyncio
async def test_no_budget_no_error():
    """Execution works fine without a budget."""
    router = FakeRouter("Result")
    env = ContextEnvironment("test")
    executor = TraceExecutor(router, env, budget=None)

    result = await executor.execute(_default_trace_config(), "test-model")
    assert result.error is None


# ---------------------------------------------------------------------------
# Trace ID generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_id_includes_role():
    """Trace ID starts with the lowercase role name."""
    router = FakeRouter("output")
    env = ContextEnvironment("test")
    executor = TraceExecutor(router, env)

    result = await executor.execute(_default_trace_config(role="Contrarian"), "test-model")

    assert result.trace_id.startswith("contrarian_")
