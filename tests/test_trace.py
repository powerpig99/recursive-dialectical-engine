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

    def get_fallback_model(self, failed_model):
        return None


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
async def test_execute_records_cost_in_budget():
    """Estimated cost is recorded against the recursion budget."""

    class CostRouter(FakeRouter):
        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            return LLMResponse(
                content="ok",
                model=model,
                latency_ms=1.0,
                estimated_cost=1.25,
            )

    env = ContextEnvironment("What is 2+2?")
    budget = RecursionBudget(max_total_calls=10)
    executor = TraceExecutor(CostRouter(), env, budget=budget)

    await executor.execute(_default_trace_config(), "test-model")

    assert budget.total_calls == 1
    assert budget.total_cost_usd == pytest.approx(1.25)


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

        def get_fallback_model(self, failed_model):
            return None

    env = ContextEnvironment("test prompt")
    executor = TraceExecutor(ErrorRouter(), env)

    result = await executor.execute(_default_trace_config(), "test-model")

    assert result.error is not None
    assert "API connection failed" in result.error
    assert result.raw_output == ""


@pytest.mark.asyncio
async def test_execute_fallback_on_error():
    """When primary model fails and fallback is available, use it."""

    class FallbackRouter:
        def __init__(self):
            self.call_count = 0

        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            self.call_count += 1
            if model == "primary-model":
                raise RuntimeError("Primary down")
            return LLMResponse(content="fallback answer", model="fallback-model", latency_ms=5.0)

        def get_fallback_model(self, failed_model):
            if failed_model == "primary-model":
                return "fallback-model"
            return None

    env = ContextEnvironment("test prompt")
    router = FallbackRouter()
    executor = TraceExecutor(router, env)

    result = await executor.execute(_default_trace_config(), "primary-model")

    assert result.error is None
    assert result.fallback_used is True
    assert result.original_model == "primary-model"
    assert result.model_used == "fallback-model"
    assert result.raw_output == "fallback answer"
    assert router.call_count == 2  # primary + fallback


@pytest.mark.asyncio
async def test_execute_fallback_also_fails():
    """When both primary and fallback fail, capture combined error."""

    class AllFailRouter:
        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            raise RuntimeError(f"{model} failed")

        def get_fallback_model(self, failed_model):
            if failed_model == "model-a":
                return "model-b"
            return None

    env = ContextEnvironment("test prompt")
    executor = TraceExecutor(AllFailRouter(), env)

    result = await executor.execute(_default_trace_config(), "model-a")

    assert result.error is not None
    assert "Primary" in result.error
    assert "Fallback" in result.error


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
