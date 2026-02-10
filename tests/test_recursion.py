"""Tests for Phase 3: recursion, budget management, sub-dialectics, and call tree."""

from __future__ import annotations

import json

import pytest

from rde.arbiter import Arbiter
from rde.engine import DialecticalEngine
from rde.environment import ContextEnvironment
from rde.models import (
    CallTreeNode,
    ConfidenceLevel,
    ModelConfig,
    RecursionBudget,
)
from rde.providers.base import BaseProvider, LLMResponse
from rde.trace import TraceExecutor
from rde.utils.visualizer import call_tree_to_ascii, call_tree_to_json


# ---------------------------------------------------------------------------
# RecursionBudget tests
# ---------------------------------------------------------------------------


def test_budget_can_recurse():
    budget = RecursionBudget(max_depth=3, max_total_calls=10)
    assert budget.can_recurse() is True


def test_budget_depth_limit():
    budget = RecursionBudget(max_depth=2, current_depth=2)
    assert budget.can_recurse() is False
    assert "depth" in budget.exhausted_reason.lower()


def test_budget_call_limit():
    budget = RecursionBudget(max_total_calls=5, total_calls=5)
    assert budget.can_recurse() is False
    assert "calls" in budget.exhausted_reason.lower()


def test_budget_cost_limit():
    budget = RecursionBudget(max_cost_usd=1.0, total_cost_usd=1.5)
    assert budget.can_recurse() is False
    assert "cost" in budget.exhausted_reason.lower()


def test_budget_child_increments_depth():
    parent = RecursionBudget(max_depth=3, current_depth=1, total_calls=5)
    child = parent.child_budget()
    assert child.current_depth == 2
    assert child.total_calls == 5  # inherited
    assert child.max_depth == 3  # same limit


def test_budget_record_call():
    budget = RecursionBudget()
    assert budget.total_calls == 0
    budget.record_call(cost_usd=0.01)
    assert budget.total_calls == 1
    assert budget.total_cost_usd == pytest.approx(0.01)


def test_budget_exhausted_reason_available():
    budget = RecursionBudget()
    assert "available" in budget.exhausted_reason.lower()


# ---------------------------------------------------------------------------
# spawn_sub_lm tests
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    """Provider that returns a fixed response."""

    def __init__(self, response: str = "sub-lm answer"):
        self.response = response
        self.call_count = 0

    async def complete(self, messages, model, **kwargs):
        self.call_count += 1
        return LLMResponse(content=self.response, model=model, latency_ms=5.0)

    def supports_model(self, model: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_spawn_sub_lm_wired(monkeypatch):
    """spawn_sub_lm works when router is wired."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig()
    router = ModelRouter(config)
    provider = FakeProvider("factored result: 42")
    router._providers = {"test": provider}

    budget = RecursionBudget(max_depth=3)
    env = ContextEnvironment(
        "test prompt",
        router=router,
        budget=budget,
        sub_lm_models=["test-model"],
    )

    result = await env.spawn_sub_lm("What is 6*7?")
    assert result == "factored result: 42"
    assert provider.call_count == 1
    assert len(env.recursion_log) == 1
    assert env.recursion_log[0]["type"] == "sub_lm"
    assert budget.total_calls == 1


@pytest.mark.asyncio
async def test_spawn_sub_lm_budget_exhausted(monkeypatch):
    """spawn_sub_lm returns budget message when exhausted."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig()
    router = ModelRouter(config)
    router._providers = {"test": FakeProvider()}

    budget = RecursionBudget(max_total_calls=0)  # Already exhausted
    env = ContextEnvironment(
        "test",
        router=router,
        budget=budget,
        sub_lm_models=["test-model"],
    )

    result = await env.spawn_sub_lm("test")
    assert "BUDGET EXHAUSTED" in result


@pytest.mark.asyncio
async def test_spawn_sub_lm_no_models_configured(monkeypatch):
    """spawn_sub_lm returns message when no sub-LM models configured."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig()
    router = ModelRouter(config)
    router._providers = {"test": FakeProvider()}

    env = ContextEnvironment("test", router=router, sub_lm_models=[])

    result = await env.spawn_sub_lm("test")
    assert "NO SUB-LM" in result


# ---------------------------------------------------------------------------
# spawn_sub_dialectic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_sub_dialectic(monkeypatch):
    """Sub-dialectic creates a mini-RDE and returns an EngineResult."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig(trace_models=["test-a", "test-b", "test-c"])
    router = ModelRouter(config)
    provider = FakeProvider("\\boxed{42}")
    router._providers = {"test": provider}

    budget = RecursionBudget(max_depth=3)
    env = ContextEnvironment("parent prompt", router=router, budget=budget)
    executor = TraceExecutor(router, env, budget=budget)

    result = await executor.spawn_sub_dialectic("What is 6*7?")
    assert "42" in result.resolution
    assert result.confidence == ConfidenceLevel.NECESSARY  # All traces agree
    assert budget.total_calls >= 1  # Budget was consumed


@pytest.mark.asyncio
async def test_spawn_sub_dialectic_budget_exhausted(monkeypatch):
    """Sub-dialectic returns UNRESOLVED when budget is exhausted."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig(trace_models=["test-a"])
    router = ModelRouter(config)
    router._providers = {"test": FakeProvider()}

    budget = RecursionBudget(max_depth=1, current_depth=1)  # Already at max
    env = ContextEnvironment("test", router=router, budget=budget)
    executor = TraceExecutor(router, env, budget=budget)

    result = await executor.spawn_sub_dialectic("sub-problem")
    assert result.confidence == ConfidenceLevel.UNRESOLVED
    assert "BUDGET EXHAUSTED" in result.resolution


@pytest.mark.asyncio
async def test_spawn_sub_dialectic_depth_tracking(monkeypatch):
    """Sub-dialectic increments depth correctly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig(trace_models=["test-a", "test-b", "test-c"])
    router = ModelRouter(config)
    router._providers = {"test": FakeProvider("\\boxed{ok}")}

    budget = RecursionBudget(max_depth=5, current_depth=2)
    env = ContextEnvironment("test", router=router, budget=budget)
    executor = TraceExecutor(router, env, budget=budget)

    result = await executor.spawn_sub_dialectic("sub")
    assert result.confidence != ConfidenceLevel.UNRESOLVED
    # Budget calls should have been consumed
    assert budget.total_calls > 0


# ---------------------------------------------------------------------------
# Recursive arbiter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arbiter_sub_arbitration(monkeypatch):
    """Arbiter spawns sub-arbitration on unresolved interference dimensions."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    call_counter = {"n": 0}

    class SubArbiterProvider(BaseProvider):
        async def complete(self, messages, model, **kwargs):
            call_counter["n"] += 1
            system = messages[0]["content"] if messages else ""

            # Arbiter call — return unresolved with interference
            if "Recursive Arbiter" in system:
                return LLMResponse(
                    content=json.dumps({
                        "resolution": "unclear",
                        "causal_chain": "conflicting evidence",
                        "confidence": "unresolved",
                        "interference_detected": ["time ordering", "observer bias"],
                        "traces_adopted": [],
                        "traces_rejected": [],
                        "shadows": ["fundamental ambiguity"],
                    }),
                    model=model,
                    latency_ms=5.0,
                )

            # Trace calls (for sub-dialectics) — all agree
            return LLMResponse(content="\\boxed{resolved}", model=model, latency_ms=5.0)

        def supports_model(self, model: str) -> bool:
            return True

    from rde.providers.router import ModelRouter
    from rde.models import NormalizedTrace

    config = ModelConfig(trace_models=["test-a", "test-b", "test-c"])
    router = ModelRouter(config)
    router._providers = {"test": SubArbiterProvider()}

    budget = RecursionBudget(max_depth=3, max_total_calls=50)
    env = ContextEnvironment("test problem", router=router, budget=budget)

    traces = [
        NormalizedTrace(trace_id="t1", role="A", model_family="test", conclusion="yes"),
        NormalizedTrace(trace_id="t2", role="B", model_family="test", conclusion="no"),
    ]

    arbiter = Arbiter(router, "test-arbiter")
    result = await arbiter.arbitrate(traces, env, budget=budget)

    # Should have been upgraded from unresolved to contingent via sub-arbitration
    assert result.confidence == ConfidenceLevel.CONTINGENT
    assert "Sub-arbitration" in result.resolution or "sub-arbitration" in str(result.shadows)
    assert budget.total_calls > 1


# ---------------------------------------------------------------------------
# Call tree visualization tests
# ---------------------------------------------------------------------------


def test_call_tree_to_json():
    tree = CallTreeNode(
        node_type="engine",
        depth=0,
        latency_ms=1000,
        children=[
            CallTreeNode(
                node_type="trace",
                role="Believer",
                model="claude-sonnet",
                depth=1,
                output_summary="\\boxed{42}",
                latency_ms=500,
            ),
            CallTreeNode(
                node_type="sub_lm",
                model="haiku",
                depth=1,
                input_summary="What is X?",
                output_summary="X is 7",
                latency_ms=100,
            ),
        ],
    )
    json_str = call_tree_to_json(tree)
    parsed = json.loads(json_str)
    assert parsed["node_type"] == "engine"
    assert len(parsed["children"]) == 2
    assert parsed["children"][0]["role"] == "Believer"


def test_call_tree_to_ascii():
    tree = CallTreeNode(
        node_type="engine",
        depth=0,
        children=[
            CallTreeNode(
                node_type="trace",
                role="Logician",
                model="gpt-5",
                depth=1,
                latency_ms=200,
                output_summary="42",
            ),
            CallTreeNode(
                node_type="arbiter",
                model="claude-opus",
                depth=1,
                latency_ms=150,
                children=[
                    CallTreeNode(
                        node_type="sub_dialectic",
                        depth=2,
                        output_summary="resolved",
                        latency_ms=300,
                    ),
                ],
            ),
        ],
    )
    ascii_tree = call_tree_to_ascii(tree)
    assert "[engine]" in ascii_tree
    assert "Logician" in ascii_tree
    assert "[arbiter]" in ascii_tree
    assert "[sub_dialectic]" in ascii_tree


# ---------------------------------------------------------------------------
# Engine with budget integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_with_budget(monkeypatch):
    """Engine passes budget through to environment and traces."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    class SimpleProvider(BaseProvider):
        async def complete(self, messages, model, **kwargs):
            return LLMResponse(content="\\boxed{42}", model=model, latency_ms=5.0)

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig(trace_models=["test-a", "test-b", "test-c"])
    budget = RecursionBudget(max_depth=2, max_total_calls=100)

    async with DialecticalEngine(config) as engine:
        engine.router._providers = {"test": SimpleProvider()}
        result = await engine.run("What is 2+2?", use_orchestrator=False, budget=budget)

    assert result.consensus_reached is True
    assert result.call_tree is not None
    assert result.call_tree.node_type == "engine"


@pytest.mark.asyncio
async def test_engine_call_tree_populated(monkeypatch):
    """Engine result includes a call tree."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    class SimpleProvider(BaseProvider):
        async def complete(self, messages, model, **kwargs):
            return LLMResponse(content="\\boxed{yes}", model=model, latency_ms=5.0)

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig(trace_models=["test-a", "test-b", "test-c"])

    async with DialecticalEngine(config) as engine:
        engine.router._providers = {"test": SimpleProvider()}
        result = await engine.run("question", use_orchestrator=False)

    assert result.call_tree is not None
    json_str = result.call_tree.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["node_type"] == "engine"
