"""Tests for data models."""


from rde.models import (
    ArbitrationResult,
    ConfidenceLevel,
    ConstraintLevel,
    EngineResult,
    ModelConfig,
    OrchestratorOutput,
    TraceConfig,
    TraceResult,
)


def test_trace_config_defaults():
    tc = TraceConfig(role="Believer", perspective="intuition", system_prompt="test")
    assert tc.temperature == 0.7
    assert tc.model_preference == "any"
    assert tc.context_strategy == "full"
    assert tc.can_recurse is False
    assert tc.recursion_budget == 0


def test_trace_config_json_round_trip():
    tc = TraceConfig(
        role="Logician",
        perspective="deduction",
        system_prompt="test prompt",
        temperature=0.3,
    )
    json_str = tc.model_dump_json()
    restored = TraceConfig.model_validate_json(json_str)
    assert restored == tc


def test_orchestrator_output_validation():
    raw = {
        "problem_type": "logic_puzzle",
        "decomposition_rationale": "testing",
        "constraint_level": "structured",
        "traces": [
            {
                "role": "Believer",
                "perspective": "intuition",
                "system_prompt": "test",
            }
        ],
    }
    output = OrchestratorOutput.model_validate(raw)
    assert output.constraint_level == ConstraintLevel.STRUCTURED
    assert len(output.traces) == 1


def test_trace_result_with_error():
    tr = TraceResult(
        trace_id="t1",
        role="Believer",
        model_used="test",
        raw_output="",
        error="Connection timeout",
    )
    assert tr.error == "Connection timeout"
    assert tr.extracted_answer is None


def test_arbitration_result_confidence_enum():
    ar = ArbitrationResult(
        resolution="50/50",
        causal_chain="test chain",
        confidence=ConfidenceLevel.NECESSARY,
        shadows=["test shadow"],
    )
    assert ar.confidence == ConfidenceLevel.NECESSARY
    assert ar.confidence.value == "necessary"


def test_engine_result_consensus():
    result = EngineResult(
        resolution="42",
        confidence=ConfidenceLevel.NECESSARY,
        consensus_reached=True,
    )
    assert result.consensus_reached
    assert result.iterations == 1
    assert result.arbitration is None


def test_model_config_defaults():
    config = ModelConfig()
    assert "claude-opus" in config.orchestrator_model
    assert len(config.trace_models) == 3
    assert config.trace_assignment == "round_robin"


def test_model_config_local_path():
    config = ModelConfig()
    assert "Qwen3" in config.local_model_path
