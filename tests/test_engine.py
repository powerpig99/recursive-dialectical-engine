"""Tests for the DialecticalEngine with mocked providers."""

from __future__ import annotations

import json

import pytest

from rde.engine import DialecticalEngine
from rde.models import ConfidenceLevel, ModelConfig
from rde.providers.base import BaseProvider, LLMResponse


class FakeProvider(BaseProvider):
    """Provider that returns configurable responses per model."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = ""):
        self._responses = responses or {}
        self._default = default

    async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
        content = self._responses.get(model, self._default)
        return LLMResponse(content=content, model=model, latency_ms=10.0)

    def supports_model(self, model: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_engine_consensus(monkeypatch):
    """All traces agree → consensus, no arbiter called."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    config = ModelConfig(
        trace_models=["test-a", "test-b", "test-c"],
    )

    async with DialecticalEngine(config) as engine:
        # Replace router with fake that returns same answer for all traces
        fake = FakeProvider(default="Analysis...\n\\boxed{42}")
        engine.router._providers = {"test": fake}

        result = await engine.run("What is the answer?", use_orchestrator=False)

    assert result.consensus_reached is True
    assert result.confidence == ConfidenceLevel.NECESSARY
    assert "42" in result.resolution
    assert result.arbitration is None


@pytest.mark.asyncio
async def test_engine_disagreement_triggers_arbiter(monkeypatch):
    """Traces disagree → arbiter is called."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    arbiter_json = json.dumps({
        "resolution": "1/2",
        "causal_chain": "The accidental condition changes the probability",
        "confidence": "necessary",
        "interference_detected": [],
        "traces_adopted": ["Logician"],
        "traces_rejected": ["Believer"],
        "shadows": ["This assumes the host cannot cheat"],
    })

    call_counter = {"trace": 0}

    class SequentialProvider(BaseProvider):
        """Returns different answers for sequential calls."""

        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            call_counter["trace"] += 1
            # First 3 calls are traces, 4th is arbiter
            if call_counter["trace"] <= 3:
                answers = ["\\boxed{2/3}", "\\boxed{1/2}", "\\boxed{1/2}"]
                content = answers[(call_counter["trace"] - 1) % 3]
                return LLMResponse(content=content, model=model, latency_ms=10.0)
            else:
                return LLMResponse(content=arbiter_json, model=model, latency_ms=10.0)

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig(
        trace_models=["test-a", "test-b", "test-c"],
        arbiter_model="test-arbiter",
    )

    async with DialecticalEngine(config) as engine:
        engine.router._providers = {"test": SequentialProvider()}

        result = await engine.run("Monty Hall problem", use_orchestrator=False)

    assert result.consensus_reached is False
    assert result.arbitration is not None
    assert result.resolution == "1/2"
    assert result.confidence == ConfidenceLevel.NECESSARY
    assert len(result.shadows) > 0


@pytest.mark.asyncio
async def test_engine_all_traces_fail(monkeypatch):
    """All traces error → unresolved result."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    class FailingProvider(BaseProvider):
        async def complete(self, messages, model, **kwargs):
            raise RuntimeError("API down")

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig(trace_models=["test-a", "test-b", "test-c"])

    async with DialecticalEngine(config) as engine:
        engine.router._providers = {"test": FailingProvider()}

        result = await engine.run("test", use_orchestrator=False)

    assert result.confidence == ConfidenceLevel.UNRESOLVED
    assert result.resolution == "All traces failed"
    assert all(tr.error is not None for tr in result.trace_results)


@pytest.mark.asyncio
async def test_engine_with_orchestrator(monkeypatch):
    """Orchestrator designs traces (mocked), then engine runs them."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    orchestrator_response = json.dumps({
        "problem_type": "arithmetic",
        "decomposition_rationale": "Simple problem, classic triad",
        "constraint_level": "structured",
        "traces": [
            {
                "role": "Believer",
                "perspective": "intuition",
                "system_prompt": "Answer with \\boxed{}",
                "temperature": 0.6,
            },
            {
                "role": "Logician",
                "perspective": "deduction",
                "system_prompt": "Derive answer with \\boxed{}",
                "temperature": 0.3,
            },
        ],
    })

    call_counter = {"n": 0}

    class OrchestratedProvider(BaseProvider):
        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                # Orchestrator call
                return LLMResponse(content=orchestrator_response, model=model, latency_ms=10.0)
            else:
                # Trace calls — all agree
                return LLMResponse(content="\\boxed{4}", model=model, latency_ms=10.0)

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig(
        orchestrator_model="test-orch",
        trace_models=["test-a", "test-b"],
    )

    async with DialecticalEngine(config) as engine:
        engine.router._providers = {"test": OrchestratedProvider()}
        result = await engine.run("What is 2+2?", use_orchestrator=True)

    assert result.consensus_reached is True
    assert "4" in result.resolution
    assert len(result.trace_results) == 2


@pytest.mark.asyncio
async def test_engine_json_output(monkeypatch):
    """EngineResult serializes to valid JSON."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    config = ModelConfig(trace_models=["test-a", "test-b", "test-c"])

    async with DialecticalEngine(config) as engine:
        fake = FakeProvider(default="\\boxed{42}")
        engine.router._providers = {"test": fake}

        result = await engine.run("test", use_orchestrator=False)

    json_str = result.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["resolution"] == "42"
    assert parsed["consensus_reached"] is True
