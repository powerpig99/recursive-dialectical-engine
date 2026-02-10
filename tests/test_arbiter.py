"""Tests for the Arbiter — trace disagreement resolution."""

from __future__ import annotations

import json

import pytest

from rde.arbiter import Arbiter
from rde.environment import ContextEnvironment
from rde.models import (
    ArbitrationResult,
    ConfidenceLevel,
    NormalizedTrace,
)
from rde.providers.base import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRouter:
    """Minimal mock of ModelRouter for Arbiter tests."""

    def __init__(self, response_content: str = ""):
        self._response_content = response_content

    async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
        return LLMResponse(content=self._response_content, model=model, latency_ms=5.0)


def _make_traces(n: int = 2, same_conclusion: bool = False) -> list[NormalizedTrace]:
    """Build synthetic normalized traces."""
    conclusions = ["42", "43", "44"]
    traces = []
    for i in range(n):
        traces.append(NormalizedTrace(
            trace_id=f"trace-{i}",
            role=["Believer", "Logician", "Contrarian"][i % 3],
            model_family=["anthropic", "google", "xai"][i % 3],
            conclusion="42" if same_conclusion else conclusions[i % len(conclusions)],
            reasoning_chain=[f"Step 1 by trace {i}", f"Step 2 by trace {i}"],
            confidence=0.8,
            raw_output=f"Analysis by trace {i}",
        ))
    return traces


def _arbiter_json(**overrides) -> str:
    """Build a valid arbiter JSON response."""
    data = {
        "resolution": "The answer is 42",
        "causal_chain": "By logical necessity, the answer follows.",
        "confidence": "necessary",
        "interference_detected": [],
        "traces_adopted": ["Believer"],
        "traces_rejected": ["Logician"],
        "shadows": ["Assumes standard interpretation"],
    }
    data.update(overrides)
    return f"```json\n{json.dumps(data)}\n```"


# ---------------------------------------------------------------------------
# Arbiter.arbitrate tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arbiter_produces_resolution():
    """Normal arbitration returns an ArbitrationResult."""
    router = FakeRouter(_arbiter_json())
    arbiter = Arbiter(router, "test-model")
    env = ContextEnvironment("test prompt")

    result = await arbiter.arbitrate(_make_traces(2), env)

    assert isinstance(result, ArbitrationResult)
    assert result.resolution == "The answer is 42"
    assert result.confidence == ConfidenceLevel.NECESSARY


@pytest.mark.asyncio
async def test_arbiter_fallback_on_json_failure():
    """Malformed LLM output triggers fallback to longest chain."""
    router = FakeRouter("This is not valid JSON at all.")
    arbiter = Arbiter(router, "test-model")
    env = ContextEnvironment("test prompt")

    traces = _make_traces(2)
    # Make second trace have longer reasoning
    traces[1].reasoning_chain = ["Step 1", "Step 2", "Step 3", "Step 4"]

    result = await arbiter.arbitrate(traces, env)

    assert result.confidence == ConfidenceLevel.CONTINGENT
    assert "Fallback" in result.causal_chain
    assert traces[1].role in result.traces_adopted


@pytest.mark.asyncio
async def test_arbiter_causal_chain_populated():
    """causal_chain field is non-empty on successful arbitration."""
    router = FakeRouter(_arbiter_json(causal_chain="Chain of causal reasoning here"))
    arbiter = Arbiter(router, "test-model")
    env = ContextEnvironment("test prompt")

    result = await arbiter.arbitrate(_make_traces(2), env)

    assert result.causal_chain != ""
    assert "causal" in result.causal_chain.lower()


@pytest.mark.asyncio
async def test_arbiter_shadows_populated():
    """shadows list is present and non-empty."""
    router = FakeRouter(_arbiter_json(shadows=["Shadow A", "Shadow B"]))
    arbiter = Arbiter(router, "test-model")
    env = ContextEnvironment("test prompt")

    result = await arbiter.arbitrate(_make_traces(2), env)

    assert len(result.shadows) == 2
    assert "Shadow A" in result.shadows


@pytest.mark.asyncio
async def test_arbiter_traces_adopted_rejected():
    """trace attribution fields populated correctly."""
    router = FakeRouter(_arbiter_json(
        traces_adopted=["Believer", "Contrarian"],
        traces_rejected=["Logician"],
    ))
    arbiter = Arbiter(router, "test-model")
    env = ContextEnvironment("test prompt")

    result = await arbiter.arbitrate(_make_traces(3), env)

    assert "Believer" in result.traces_adopted
    assert "Logician" in result.traces_rejected


@pytest.mark.asyncio
async def test_arbiter_confidence_levels():
    """Test all three confidence levels parse correctly."""
    env = ContextEnvironment("test prompt")

    for level in ["necessary", "contingent", "unresolved"]:
        router = FakeRouter(_arbiter_json(confidence=level))
        arbiter = Arbiter(router, "test-model")
        result = await arbiter.arbitrate(_make_traces(2), env)
        assert result.confidence == ConfidenceLevel(level)


@pytest.mark.asyncio
async def test_arbiter_uses_configured_temperature():
    """Arbiter passes its temperature to the router."""
    captured = {}

    class CapturingRouter:
        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            captured["temperature"] = temperature
            return LLMResponse(content=_arbiter_json(), model=model, latency_ms=5.0)

    arbiter = Arbiter(CapturingRouter(), "test-model", temperature=0.05)
    env = ContextEnvironment("test prompt")
    await arbiter.arbitrate(_make_traces(2), env)

    assert captured["temperature"] == 0.05


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


def test_convergence_necessary_stops():
    """High confidence (necessary) triggers convergence."""
    arbiter = Arbiter(FakeRouter(), "test-model")
    arb_result = ArbitrationResult(
        resolution="42",
        causal_chain="by necessity",
        confidence=ConfidenceLevel.NECESSARY,
    )

    conv = arbiter.check_convergence(arb_result, iteration=1, max_iterations=5)

    assert conv.should_stop is True
    assert "necessary" in conv.reason.lower()


def test_convergence_budget_stops():
    """Budget exhaustion triggers convergence."""
    arbiter = Arbiter(FakeRouter(), "test-model")
    arb_result = ArbitrationResult(
        resolution="maybe 42",
        causal_chain="unclear",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["Something remains"],
    )

    conv = arbiter.check_convergence(arb_result, iteration=5, max_iterations=5)

    assert conv.should_stop is True
    assert "budget" in conv.reason.lower() or "exhausted" in conv.reason.lower()


def test_convergence_no_shadows_stops():
    """No shadows → clean resolution, stop."""
    arbiter = Arbiter(FakeRouter(), "test-model")
    arb_result = ArbitrationResult(
        resolution="42",
        causal_chain="clearly",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=[],
    )

    conv = arbiter.check_convergence(arb_result, iteration=1, max_iterations=5)

    assert conv.should_stop is True
    assert "clean" in conv.reason.lower() or "no shadow" in conv.reason.lower()


def test_convergence_novel_shadows_continue():
    """Novel shadows with budget remaining → continue."""
    arbiter = Arbiter(FakeRouter(), "test-model")
    arb_result = ArbitrationResult(
        resolution="maybe",
        causal_chain="partial",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["New dimension to explore"],
    )

    conv = arbiter.check_convergence(arb_result, iteration=1, max_iterations=5)

    assert conv.should_stop is False
    assert conv.next_framings == ["New dimension to explore"]


def test_convergence_repeating_shadows_stop():
    """Repeating shadows → diminishing returns, stop."""
    arbiter = Arbiter(FakeRouter(), "test-model")
    arb_result = ArbitrationResult(
        resolution="maybe",
        causal_chain="partial",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["Same old shadow"],
    )
    prior = [["Same old shadow"]]

    conv = arbiter.check_convergence(
        arb_result, iteration=2, max_iterations=5, prior_shadows=prior
    )

    assert conv.should_stop is True
    assert "repeat" in conv.reason.lower()


# ---------------------------------------------------------------------------
# Shadow report tests
# ---------------------------------------------------------------------------


def test_shadow_report_generation():
    """Human-readable shadow report includes all iterations."""
    arbiter = Arbiter(FakeRouter(), "test-model")
    arb_result = ArbitrationResult(
        resolution="42",
        causal_chain="by necessity",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["Final shadow"],
    )
    prior_shadows = [["Shadow from iter 1"], ["Shadow from iter 2"]]

    report = arbiter._build_shadow_report(arb_result, prior_shadows)

    assert "Shadow Report" in report
    assert "Shadow from iter 1" in report
    assert "Shadow from iter 2" in report
    assert "Final shadow" in report
    assert "Iteration 1" in report
    assert "Iteration 2" in report


# ---------------------------------------------------------------------------
# Fallback tests
# ---------------------------------------------------------------------------


def test_fallback_selects_longest_chain():
    """Fallback picks the trace with most reasoning steps."""
    arbiter = Arbiter(FakeRouter(), "test-model")
    traces = _make_traces(3)
    traces[2].reasoning_chain = ["A", "B", "C", "D", "E"]  # longest

    result = arbiter._fallback_arbitration(traces)

    assert result.confidence == ConfidenceLevel.CONTINGENT
    assert traces[2].role in result.traces_adopted
    assert "Fallback" in result.causal_chain
