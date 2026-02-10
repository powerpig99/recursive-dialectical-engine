"""Tests for the Phase 6 training data pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rde.training.collector import TRAINING_PROBLEMS, load_results, save_results
from rde.training.evaluator import compute_deltas, format_evaluation_table
from rde.training.formatter import (
    format_all,
    format_full_dialectic,
    format_resolution,
    format_role_specific,
    to_anthropic_format,
    to_gemini_format,
    to_generic_jsonl,
)
from rde.training.models import (
    CollectedResult,
    CollectionConfig,
    DistillationStrategy,
    EvaluationResult,
    FormattingConfig,
    ProviderFormat,
    TrainingExample,
)
from rde.utils.metrics import IndependenceReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collected_result(
    problem_name: str = "test_problem",
    num_traces: int = 3,
    consensus: bool = False,
    failed: bool = False,
) -> CollectedResult:
    """Build a synthetic CollectedResult for tests."""
    trace_results = []
    normalized_traces = []
    for i in range(num_traces):
        role = ["Believer", "Logician", "Contrarian"][i % 3]
        family = ["anthropic", "google", "xai"][i % 3]
        trace_results.append({
            "trace_id": f"trace-{i}",
            "role": role,
            "model_used": f"model-{family}",
            "raw_output": f"Analysis from {role}: The answer is derived by {role.lower()} reasoning.\n\\boxed{{42}}",
            "extracted_answer": "42",
            "error": None,
            "latency_ms": 100.0,
            "token_usage": {},
        })
        normalized_traces.append({
            "trace_id": f"trace-{i}",
            "role": role,
            "model_family": family,
            "conclusion": "42",
            "reasoning_chain": [f"Step 1 by {role}", f"Step 2 by {role}"],
            "confidence": 0.8,
            "evidence_cited": [],
            "raw_output": f"Analysis from {role}",
        })

    engine_result = {
        "resolution": "All traces failed" if failed else "The answer is 42",
        "confidence": "unresolved" if failed else "necessary",
        "shadows": [] if failed else ["Assumes standard interpretation"],
        "causal_chain": "" if failed else "By logical necessity, the answer follows from the premises.",
        "iterations": 1,
        "trace_results": trace_results if not failed else [],
        "normalized_traces": normalized_traces if not failed else [],
        "arbitration": None if consensus else {
            "resolution": "The answer is 42",
            "causal_chain": "By logical necessity",
            "confidence": "necessary",
            "shadows": ["Assumes standard interpretation"],
        },
        "total_latency_ms": 500.0,
        "consensus_reached": consensus,
        "call_tree": None,
    }

    return CollectedResult(
        problem_name=problem_name,
        problem_prompt="What is the answer to life, the universe, and everything?",
        run_index=0,
        engine_result=engine_result,
        independence_report={"conclusion_agreement": 1.0, "model_diversity_score": 1.0} if not failed else None,
        collection_timestamp="2026-02-10T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_training_example_defaults():
    ex = TrainingExample(
        input_text="test",
        output_text="output",
        strategy=DistillationStrategy.RESOLUTION,
    )
    assert ex.source_problem == ""
    assert ex.metadata == {}


def test_training_example_json_round_trip():
    ex = TrainingExample(
        input_text="What is 2+2?",
        output_text="RESOLUTION: 4",
        strategy=DistillationStrategy.RESOLUTION,
        source_problem="math_test",
    )
    dumped = ex.model_dump_json()
    restored = TrainingExample.model_validate_json(dumped)
    assert restored == ex


def test_collection_config_defaults():
    config = CollectionConfig()
    assert config.output_path == "training_data/engine_results.jsonl"
    assert config.runs_per_problem == 1
    assert config.include_metrics is True


def test_collected_result_with_metrics():
    result = _make_collected_result()
    assert result.independence_report is not None
    assert result.independence_report["conclusion_agreement"] == 1.0


def test_distillation_strategy_enum():
    assert DistillationStrategy.RESOLUTION.value == "resolution"
    assert DistillationStrategy.FULL_DIALECTIC.value == "full_dialectic"
    assert DistillationStrategy.ROLE_SPECIFIC.value == "role_specific"


def test_provider_format_enum():
    assert ProviderFormat.GEMINI.value == "gemini"
    assert ProviderFormat.ANTHROPIC.value == "anthropic"
    assert ProviderFormat.JSONL_GENERIC.value == "jsonl_generic"


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------


def test_save_load_round_trip():
    results = [_make_collected_result("problem_a"), _make_collected_result("problem_b")]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)

    try:
        save_results(results, path)
        loaded = load_results(path)
        assert len(loaded) == 2
        assert loaded[0].problem_name == "problem_a"
        assert loaded[1].problem_name == "problem_b"
        assert loaded[0].engine_result["resolution"] == "The answer is 42"
    finally:
        path.unlink(missing_ok=True)


def test_collected_result_structure():
    result = _make_collected_result(num_traces=3)
    assert len(result.engine_result["trace_results"]) == 3
    assert len(result.engine_result["normalized_traces"]) == 3
    assert result.collection_timestamp != ""


def test_training_problems_not_empty():
    assert len(TRAINING_PROBLEMS) >= 20
    for problem in TRAINING_PROBLEMS:
        assert "name" in problem
        assert "prompt" in problem
        assert len(problem["prompt"]) > 10


# ---------------------------------------------------------------------------
# Formatter tests — Strategy A
# ---------------------------------------------------------------------------


def test_format_resolution_produces_example():
    result = _make_collected_result()
    ex = format_resolution(result)
    assert ex is not None
    assert "RESOLUTION:" in ex.output_text
    assert "42" in ex.output_text
    assert "CAUSAL CHAIN:" in ex.output_text
    assert "SHADOWS:" in ex.output_text
    assert ex.strategy == DistillationStrategy.RESOLUTION


def test_format_resolution_skips_failed():
    result = _make_collected_result(failed=True)
    ex = format_resolution(result)
    assert ex is None


# ---------------------------------------------------------------------------
# Formatter tests — Strategy B
# ---------------------------------------------------------------------------


def test_format_full_dialectic_includes_all_traces():
    result = _make_collected_result(num_traces=3)
    ex = format_full_dialectic(result)
    assert ex is not None
    assert "PERSPECTIVE 1" in ex.output_text
    assert "PERSPECTIVE 2" in ex.output_text
    assert "PERSPECTIVE 3" in ex.output_text
    assert "Believer" in ex.output_text
    assert "Logician" in ex.output_text
    assert "Contrarian" in ex.output_text


def test_format_full_dialectic_includes_synthesis():
    result = _make_collected_result()
    ex = format_full_dialectic(result)
    assert ex is not None
    assert "SYNTHESIS:" in ex.output_text
    assert ex.strategy == DistillationStrategy.FULL_DIALECTIC


def test_format_full_dialectic_skips_single_trace():
    result = _make_collected_result(num_traces=1)
    ex = format_full_dialectic(result)
    assert ex is None


# ---------------------------------------------------------------------------
# Formatter tests — Strategy C
# ---------------------------------------------------------------------------


def test_format_role_specific_produces_per_trace():
    result = _make_collected_result(num_traces=3)
    examples = format_role_specific(result)
    assert len(examples) == 3
    for ex in examples:
        assert ex.strategy == DistillationStrategy.ROLE_SPECIFIC


def test_format_role_specific_includes_role():
    result = _make_collected_result(num_traces=2)
    examples = format_role_specific(result)
    assert any("ROLE: Believer" in ex.input_text for ex in examples)
    assert any("ROLE: Logician" in ex.input_text for ex in examples)
    for ex in examples:
        assert "PROBLEM:" in ex.input_text


# ---------------------------------------------------------------------------
# Provider format tests
# ---------------------------------------------------------------------------


def test_to_gemini_format():
    examples = [
        TrainingExample(
            input_text="question",
            output_text="answer",
            strategy=DistillationStrategy.RESOLUTION,
        )
    ]
    formatted = to_gemini_format(examples)
    assert len(formatted) == 1
    assert formatted[0]["text_input"] == "question"
    assert formatted[0]["output"] == "answer"


def test_to_anthropic_format():
    examples = [
        TrainingExample(
            input_text="question",
            output_text="answer",
            strategy=DistillationStrategy.RESOLUTION,
        )
    ]
    formatted = to_anthropic_format(examples)
    assert len(formatted) == 1
    msgs = formatted[0]["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "question"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "answer"


def test_to_generic_jsonl():
    examples = [
        TrainingExample(
            input_text="q",
            output_text="a",
            strategy=DistillationStrategy.RESOLUTION,
        )
    ]
    formatted = to_generic_jsonl(examples)
    assert formatted[0]["messages"][0]["role"] == "user"
    assert formatted[0]["messages"][1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# format_all filtering tests
# ---------------------------------------------------------------------------


def test_format_all_filters_by_trace_count():
    results = [
        _make_collected_result(num_traces=1),  # Should be filtered
        _make_collected_result(num_traces=3),  # Should pass
    ]
    config = FormattingConfig(
        strategies=[DistillationStrategy.RESOLUTION],
        min_trace_count=2,
    )
    examples = format_all(results, config)
    # Only the 3-trace result should produce an example
    assert len(examples) == 1


def test_format_all_filters_consensus_when_required():
    results = [
        _make_collected_result(consensus=True),   # Should be filtered
        _make_collected_result(consensus=False),   # Should pass
    ]
    config = FormattingConfig(
        strategies=[DistillationStrategy.RESOLUTION],
        require_arbitration=True,
    )
    examples = format_all(results, config)
    assert len(examples) == 1


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


def test_compute_deltas_positive():
    ft = IndependenceReport(avg_reasoning_divergence=0.8, avg_jaccard_distance=0.7)
    bl = IndependenceReport(avg_reasoning_divergence=0.5, avg_jaccard_distance=0.4)
    deltas = compute_deltas(ft, bl)
    assert deltas["divergence_delta"] == pytest.approx(0.3)
    assert deltas["jaccard_delta"] == pytest.approx(0.3)


def test_compute_deltas_negative():
    ft = IndependenceReport(avg_reasoning_divergence=0.2, avg_jaccard_distance=0.1)
    bl = IndependenceReport(avg_reasoning_divergence=0.6, avg_jaccard_distance=0.5)
    deltas = compute_deltas(ft, bl)
    assert deltas["divergence_delta"] == pytest.approx(-0.4)
    assert deltas["jaccard_delta"] == pytest.approx(-0.4)


def test_compute_deltas_handles_none():
    deltas = compute_deltas(None, None)
    assert deltas["divergence_delta"] == 0.0
    assert deltas["jaccard_delta"] == 0.0

    ft = IndependenceReport(avg_reasoning_divergence=0.5)
    deltas = compute_deltas(ft, None)
    assert deltas["divergence_delta"] == 0.0


def test_format_evaluation_table_empty():
    assert format_evaluation_table([]) == "No evaluation results."


def test_format_evaluation_table():
    results = [
        EvaluationResult(
            strategy=DistillationStrategy.RESOLUTION,
            problem_name="monty_hall",
            finetuned_confidence="necessary",
            baseline_confidence="contingent",
            finetuned_avg_divergence=0.6,
            baseline_avg_divergence=0.8,
            divergence_delta=-0.2,
            finetuned_avg_jaccard=0.5,
            baseline_avg_jaccard=0.7,
            jaccard_delta=-0.2,
        )
    ]
    table = format_evaluation_table(results)
    assert "monty_hall" in table
    assert "resolution" in table
    assert "necessary" in table
