"""Tests for the trace normalizer."""

from rde.models import TraceResult
from rde.normalizer import TraceNormalizer


def test_normalize_boxed_answer():
    norm = TraceNormalizer()
    tr = TraceResult(
        trace_id="t1",
        role="Logician",
        model_used="claude-sonnet",
        raw_output="Analysis...\n\\boxed{1/2}",
    )
    nt = norm.normalize(tr)
    assert nt.conclusion == "1/2"
    assert nt.model_family == "anthropic"
    assert nt.role == "Logician"


def test_normalize_marker_line():
    norm = TraceNormalizer()
    tr = TraceResult(
        trace_id="t1",
        role="Believer",
        model_used="gpt-5",
        raw_output="Some reasoning.\nTherefore, the answer is 42.",
    )
    nt = norm.normalize(tr)
    assert "therefore" in nt.conclusion.lower() or "42" in nt.conclusion


def test_normalize_fallback_last_line():
    norm = TraceNormalizer()
    tr = TraceResult(
        trace_id="t1",
        role="Contrarian",
        model_used="gemini-2.5-pro",
        raw_output="Line 1\nLine 2\nFinal line",
    )
    nt = norm.normalize(tr)
    assert nt.conclusion == "Final line"
    assert nt.model_family == "google"


def test_extract_reasoning_chain():
    norm = TraceNormalizer()
    tr = TraceResult(
        trace_id="t1",
        role="Logician",
        model_used="local",
        raw_output="1. First step\n2. Second step\n- Bullet point\nConclusion\n\\boxed{42}",
    )
    nt = norm.normalize(tr)
    assert len(nt.reasoning_chain) == 3  # 2 numbered + 1 bullet


def test_consensus_all_agree():
    norm = TraceNormalizer()
    from rde.models import NormalizedTrace

    traces = [
        NormalizedTrace(trace_id="t1", role="A", model_family="a", conclusion="42"),
        NormalizedTrace(trace_id="t2", role="B", model_family="b", conclusion="42"),
        NormalizedTrace(trace_id="t3", role="C", model_family="c", conclusion="42"),
    ]
    assert norm.check_consensus(traces) is True


def test_consensus_disagree():
    norm = TraceNormalizer()
    from rde.models import NormalizedTrace

    traces = [
        NormalizedTrace(trace_id="t1", role="A", model_family="a", conclusion="2/3"),
        NormalizedTrace(trace_id="t2", role="B", model_family="b", conclusion="1/2"),
        NormalizedTrace(trace_id="t3", role="C", model_family="c", conclusion="1/2"),
    ]
    assert norm.check_consensus(traces) is False


def test_consensus_whitespace_insensitive():
    norm = TraceNormalizer()
    from rde.models import NormalizedTrace

    traces = [
        NormalizedTrace(trace_id="t1", role="A", model_family="a", conclusion="1/2"),
        NormalizedTrace(trace_id="t2", role="B", model_family="b", conclusion=" 1/2 "),
    ]
    assert norm.check_consensus(traces) is True


def test_detect_model_families():
    norm = TraceNormalizer()
    assert norm._detect_model_family("claude-opus-4-6") == "anthropic"
    assert norm._detect_model_family("gpt-5") == "openai"
    assert norm._detect_model_family("gemini-2.5-pro") == "google"
    assert norm._detect_model_family("grok-3") == "xai"
    assert norm._detect_model_family("kimi-for-coding") == "kimi"
    assert norm._detect_model_family("~/Models/Qwen3-8B-4bit") == "local"
    assert norm._detect_model_family("something-else") == "unknown"
