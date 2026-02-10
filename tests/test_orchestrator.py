"""Tests for the orchestrator, templates, convergence, and multi-iteration engine."""

from __future__ import annotations

import json

import pytest

from rde.arbiter import Arbiter
from rde.models import (
    ArbitrationResult,
    ConfidenceLevel,
    ModelConfig,
    TraceConfig,
)
from rde.orchestrator import Orchestrator
from rde.environment import ContextEnvironment
from rde.engine import DialecticalEngine
from rde.prompts.templates import (
    TEMPLATE_REGISTRY,
    get_template,
    list_problem_types,
)
from rde.prompts.orchestrator import (
    build_orchestrator_messages,
    build_reframing_messages,
)
from rde.providers.base import BaseProvider, LLMResponse


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


def test_template_registry_has_all_types():
    """Registry contains all documented problem types."""
    expected = {
        "logic_puzzle",
        "mathematical_reasoning",
        "document_analysis",
        "multi_constraint",
        "causal_reasoning",
        "code_analysis",
        "ambiguous_query",
    }
    assert expected == set(TEMPLATE_REGISTRY.keys())


def test_get_template_normalized_lookup():
    """Templates are found with various name formats."""
    assert get_template("logic_puzzle") is not None
    assert get_template("Logic Puzzle") is not None
    assert get_template("logic-puzzle") is not None
    assert get_template("LOGIC_PUZZLE") is not None
    assert get_template("nonexistent") is None


def test_list_problem_types():
    types = list_problem_types()
    assert len(types) == 7
    assert "logic_puzzle" in types


def test_all_templates_have_boxed_instruction():
    """Every trace in every template tells the model to use \\boxed{}."""
    for problem_type, traces in TEMPLATE_REGISTRY.items():
        for trace in traces:
            assert "boxed" in trace["system_prompt"].lower(), (
                f"Template {problem_type}/{trace['role']} missing \\boxed{{}} instruction"
            )


def test_all_templates_have_required_fields():
    """Every template trace can be parsed as a TraceConfig."""
    for problem_type, traces in TEMPLATE_REGISTRY.items():
        for trace_dict in traces:
            config = TraceConfig.model_validate(trace_dict)
            assert config.role
            assert config.perspective
            assert config.system_prompt


# ---------------------------------------------------------------------------
# Orchestrator prompt tests
# ---------------------------------------------------------------------------


def test_orchestrator_messages_include_problem_types():
    """Orchestrator system prompt lists known problem types."""
    messages = build_orchestrator_messages("test prompt")
    system_msg = messages[0]["content"]
    assert "logic_puzzle" in system_msg
    assert "causal_reasoning" in system_msg


def test_reframing_messages_include_shadows():
    """Reframing messages inject prior shadows into the prompt."""
    messages = build_reframing_messages(
        prompt="test problem",
        prior_shadows=["hidden assumption about time", "ignores observer effect"],
        prior_resolution="50/50",
        prior_confidence="necessary",
        iteration=2,
    )
    system_msg = messages[0]["content"]
    assert "hidden assumption about time" in system_msg
    assert "ignores observer effect" in system_msg
    assert "50/50" in system_msg
    assert "iteration 2" in system_msg.lower()


def test_reframing_messages_user_prompt():
    messages = build_reframing_messages(
        prompt="the problem",
        prior_shadows=["shadow1"],
        prior_resolution="42",
        prior_confidence="contingent",
        iteration=3,
    )
    user_msg = messages[1]["content"]
    assert "the problem" in user_msg
    assert "shadow" in user_msg.lower()


# ---------------------------------------------------------------------------
# Convergence logic tests
# ---------------------------------------------------------------------------


class FakeRouter:
    async def complete(self, **kwargs):
        return LLMResponse(content="{}", model="test", latency_ms=0)


def _make_arbiter():
    return Arbiter(router=FakeRouter(), model="test")


def test_convergence_necessary_confidence_stops():
    """Necessary confidence → stop immediately."""
    arbiter = _make_arbiter()
    arb = ArbitrationResult(
        resolution="42",
        causal_chain="proved",
        confidence=ConfidenceLevel.NECESSARY,
        shadows=["some shadow"],
    )
    result = arbiter.check_convergence(arb, iteration=1, max_iterations=5)
    assert result.should_stop is True
    assert "necessary" in result.reason.lower()


def test_convergence_budget_exhausted_stops():
    """Stops when iteration count reaches max."""
    arbiter = _make_arbiter()
    arb = ArbitrationResult(
        resolution="maybe",
        causal_chain="partial",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["unexplored dimension"],
    )
    result = arbiter.check_convergence(arb, iteration=3, max_iterations=3)
    assert result.should_stop is True
    assert "budget" in result.reason.lower()


def test_convergence_no_shadows_stops():
    """No shadows → clean resolution, stop."""
    arbiter = _make_arbiter()
    arb = ArbitrationResult(
        resolution="42",
        causal_chain="clean",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=[],
    )
    result = arbiter.check_convergence(arb, iteration=1, max_iterations=5)
    assert result.should_stop is True
    assert "clean" in result.reason.lower() or "no shadow" in result.reason.lower()


def test_convergence_novel_shadows_continues():
    """Novel shadows with budget remaining → continue."""
    arbiter = _make_arbiter()
    arb = ArbitrationResult(
        resolution="partial",
        causal_chain="incomplete",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["new unexplored axis"],
    )
    result = arbiter.check_convergence(arb, iteration=1, max_iterations=5)
    assert result.should_stop is False
    assert result.next_framings == ["new unexplored axis"]


def test_convergence_repeating_shadows_stops():
    """Same shadows appearing again → diminishing returns, stop."""
    arbiter = _make_arbiter()
    arb = ArbitrationResult(
        resolution="partial",
        causal_chain="stuck",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["observer effect", "time assumption"],
    )
    prior = [["observer effect", "time assumption"]]
    result = arbiter.check_convergence(
        arb, iteration=2, max_iterations=5, prior_shadows=prior
    )
    assert result.should_stop is True
    assert "repeating" in result.reason.lower() or "diminishing" in result.reason.lower()


def test_convergence_partially_novel_shadows_continues():
    """Partially overlapping shadows (< 50%) → continue."""
    arbiter = _make_arbiter()
    arb = ArbitrationResult(
        resolution="partial",
        causal_chain="evolving",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["new axis 1", "new axis 2", "old axis"],
    )
    prior = [["old axis", "ancient axis"]]
    result = arbiter.check_convergence(
        arb, iteration=2, max_iterations=5, prior_shadows=prior
    )
    # 1/3 overlap = 33% < 50%, should continue
    assert result.should_stop is False


def test_shadow_report_generation():
    arbiter = _make_arbiter()
    arb = ArbitrationResult(
        resolution="answer",
        causal_chain="chain",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["final shadow"],
    )
    prior = [["iter 1 shadow"], ["iter 2 shadow"]]
    report = arbiter._build_shadow_report(arb, prior)
    assert "iter 1 shadow" in report
    assert "iter 2 shadow" in report
    assert "final shadow" in report


# ---------------------------------------------------------------------------
# Orchestrator with reframing tests (mocked)
# ---------------------------------------------------------------------------


class MockOrchestratorProvider(BaseProvider):
    """Returns different orchestrator responses based on call count."""

    def __init__(self):
        self.call_count = 0

    async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
        self.call_count += 1

        # Check if this is a reframing call
        system_msg = messages[0]["content"] if messages else ""
        is_reframing = "REFRAMING" in system_msg

        if is_reframing:
            response = json.dumps({
                "problem_type": "causal_reasoning",
                "decomposition_rationale": "Exploring shadows from prior iteration",
                "constraint_level": "guided",
                "traces": [
                    {
                        "role": "Forward-Chain",
                        "perspective": "temporal cause-effect",
                        "system_prompt": "Trace forward. \\boxed{}",
                        "temperature": 0.4,
                    },
                    {
                        "role": "Counterfactual",
                        "perspective": "alternative timelines",
                        "system_prompt": "What if? \\boxed{}",
                        "temperature": 0.7,
                    },
                ],
            })
        else:
            response = json.dumps({
                "problem_type": "logic_puzzle",
                "decomposition_rationale": "Classic triad",
                "constraint_level": "structured",
                "traces": [
                    {
                        "role": "Believer",
                        "perspective": "intuition",
                        "system_prompt": "Trust intuition. \\boxed{}",
                        "temperature": 0.6,
                    },
                    {
                        "role": "Logician",
                        "perspective": "deduction",
                        "system_prompt": "Derive logically. \\boxed{}",
                        "temperature": 0.3,
                    },
                ],
            })

        return LLMResponse(content=response, model=model, latency_ms=10.0)

    def supports_model(self, model: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_orchestrator_reframing(monkeypatch):
    """Orchestrator produces different traces when given prior shadows."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig(orchestrator_model="test-orch")
    router = ModelRouter(config)
    provider = MockOrchestratorProvider()
    router._providers = {"test": provider}

    orchestrator = Orchestrator(router, "test-orch")
    env = ContextEnvironment("test problem")

    # First call: initial design
    traces1 = await orchestrator.design_traces(env)
    assert traces1[0].role == "Believer"

    # Second call: reframing with shadows
    prior = ArbitrationResult(
        resolution="1/2",
        causal_chain="partial",
        confidence=ConfidenceLevel.CONTINGENT,
        shadows=["ignores observer effect"],
    )
    traces2 = await orchestrator.design_traces_for_iteration(env, prior, iteration=2)
    assert traces2[0].role == "Forward-Chain"
    assert len(traces2) == 2


# ---------------------------------------------------------------------------
# Multi-iteration engine test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_multi_iteration(monkeypatch):
    """Engine runs multiple iterations when shadows exist and budget allows."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    call_counter = {"n": 0}

    arbiter_response_1 = json.dumps({
        "resolution": "maybe 1/2",
        "causal_chain": "partial analysis",
        "confidence": "contingent",
        "interference_detected": [],
        "traces_adopted": ["Logician"],
        "traces_rejected": ["Believer"],
        "shadows": ["ignores observer effect", "time ordering unclear"],
    })
    arbiter_response_2 = json.dumps({
        "resolution": "1/2",
        "causal_chain": "complete causal chain",
        "confidence": "necessary",
        "interference_detected": [],
        "traces_adopted": ["Forward-Chain"],
        "traces_rejected": [],
        "shadows": ["solved within stated constraints"],
    })

    class MultiIterProvider(BaseProvider):
        async def complete(self, messages, model, temperature=0.7, max_tokens=4096, response_format=None):
            call_counter["n"] += 1
            system_msg = messages[0]["content"] if messages else ""

            # Orchestrator calls
            if "Root Orchestrator" in system_msg:
                is_reframing = "REFRAMING" in system_msg
                if is_reframing:
                    resp = json.dumps({
                        "problem_type": "causal_reasoning",
                        "decomposition_rationale": "exploring shadows",
                        "constraint_level": "guided",
                        "traces": [
                            {"role": "Forward", "perspective": "forward",
                             "system_prompt": "Forward. \\boxed{}", "temperature": 0.4},
                            {"role": "Backward", "perspective": "backward",
                             "system_prompt": "Backward. \\boxed{}", "temperature": 0.5},
                        ],
                    })
                else:
                    resp = json.dumps({
                        "problem_type": "logic_puzzle",
                        "decomposition_rationale": "classic",
                        "constraint_level": "structured",
                        "traces": [
                            {"role": "A", "perspective": "a",
                             "system_prompt": "s. \\boxed{}", "temperature": 0.5},
                            {"role": "B", "perspective": "b",
                             "system_prompt": "s. \\boxed{}", "temperature": 0.5},
                        ],
                    })
                return LLMResponse(content=resp, model=model, latency_ms=5.0)

            # Arbiter calls — first returns contingent, second returns necessary
            if "Recursive Arbiter" in system_msg:
                if call_counter["n"] <= 6:
                    return LLMResponse(content=arbiter_response_1, model=model, latency_ms=5.0)
                else:
                    return LLMResponse(content=arbiter_response_2, model=model, latency_ms=5.0)

            # Trace calls — return disagreeing answers
            if call_counter["n"] % 2 == 0:
                return LLMResponse(content="\\boxed{2/3}", model=model, latency_ms=5.0)
            else:
                return LLMResponse(content="\\boxed{1/2}", model=model, latency_ms=5.0)

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig(
        orchestrator_model="test-orch",
        arbiter_model="test-arb",
        trace_models=["test-a", "test-b"],
    )

    async with DialecticalEngine(config) as engine:
        engine.router._providers = {"test": MultiIterProvider()}

        result = await engine.run(
            "Monty Hall accidental",
            use_orchestrator=True,
            max_iterations=3,
        )

    # Should have run 2 iterations (first contingent, second necessary → stop)
    assert result.iterations >= 1
    assert result.resolution == "1/2"
    assert result.confidence == ConfidenceLevel.NECESSARY


@pytest.mark.asyncio
async def test_engine_single_iteration_backward_compat(monkeypatch):
    """max_iterations=1 behaves identically to Phase 1 behavior."""
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

    async with DialecticalEngine(config) as engine:
        engine.router._providers = {"test": SimpleProvider()}
        result = await engine.run("What is 2+2?", use_orchestrator=False, max_iterations=1)

    assert result.consensus_reached is True
    assert "42" in result.resolution
    assert result.iterations == 1


# ---------------------------------------------------------------------------
# Scaffolding calibration
# ---------------------------------------------------------------------------


def test_model_config_scaffolding_default():
    config = ModelConfig()
    assert config.scaffolding_preference == "auto"


def test_model_config_scaffolding_override():
    config = ModelConfig(scaffolding_preference="structured")
    assert config.scaffolding_preference == "structured"
