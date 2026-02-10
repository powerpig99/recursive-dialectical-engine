"""Tests for trace independence metrics."""

import pytest

from rde.models import NormalizedTrace
from rde.utils.metrics import (
    IndependenceReport,
    agreement_for_different_reasons,
    compute_all,
    conclusion_agreement,
    jaccard_distance,
    model_diversity_score,
    reasoning_chain_divergence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trace(
    trace_id: str,
    conclusion: str,
    reasoning: list[str] | None = None,
    raw_output: str = "",
    model_family: str = "anthropic",
) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=trace_id,
        role="test",
        model_family=model_family,
        conclusion=conclusion,
        reasoning_chain=reasoning or [],
        raw_output=raw_output or conclusion,
    )


# ---------------------------------------------------------------------------
# conclusion_agreement
# ---------------------------------------------------------------------------


def test_conclusion_agreement_all_same():
    traces = [_trace("a", "42"), _trace("b", "42"), _trace("c", "42")]
    assert conclusion_agreement(traces) == pytest.approx(1.0)


def test_conclusion_agreement_all_different():
    traces = [_trace("a", "yes"), _trace("b", "no"), _trace("c", "maybe")]
    assert conclusion_agreement(traces) == pytest.approx(0.0)


def test_conclusion_agreement_partial():
    """2 of 3 agree -> 1 matching pair out of 3 pairs."""
    traces = [_trace("a", "42"), _trace("b", "42"), _trace("c", "99")]
    assert conclusion_agreement(traces) == pytest.approx(1 / 3)


def test_conclusion_agreement_single_trace():
    assert conclusion_agreement([_trace("a", "42")]) == 0.0


def test_conclusion_agreement_empty():
    assert conclusion_agreement([]) == 0.0


def test_conclusion_agreement_case_insensitive():
    traces = [_trace("a", "Yes"), _trace("b", "yes")]
    assert conclusion_agreement(traces) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# reasoning_chain_divergence
# ---------------------------------------------------------------------------


def test_reasoning_divergence_identical():
    traces = [
        _trace("a", "42", reasoning=["step 1", "step 2"]),
        _trace("b", "42", reasoning=["step 1", "step 2"]),
    ]
    metrics = reasoning_chain_divergence(traces)
    assert len(metrics) == 1
    assert metrics[0].value == pytest.approx(0.0)


def test_reasoning_divergence_completely_different():
    traces = [
        _trace("a", "42", reasoning=["alpha beta gamma delta epsilon"]),
        _trace("b", "42", reasoning=["one two three four five six seven"]),
    ]
    metrics = reasoning_chain_divergence(traces)
    assert len(metrics) == 1
    assert metrics[0].value > 0.5  # Highly divergent


def test_reasoning_divergence_pairwise_count():
    """N=4 traces -> 6 pairs."""
    traces = [_trace(f"t{i}", f"c{i}") for i in range(4)]
    metrics = reasoning_chain_divergence(traces)
    assert len(metrics) == 6


# ---------------------------------------------------------------------------
# jaccard_distance
# ---------------------------------------------------------------------------


def test_jaccard_identical():
    traces = [
        _trace("a", "42", raw_output="the answer is clearly forty two"),
        _trace("b", "42", raw_output="the answer is clearly forty two"),
    ]
    metrics = jaccard_distance(traces)
    assert metrics[0].value == pytest.approx(0.0)


def test_jaccard_disjoint():
    traces = [
        _trace("a", "42", raw_output="alpha beta gamma"),
        _trace("b", "42", raw_output="one two three"),
    ]
    metrics = jaccard_distance(traces)
    assert metrics[0].value == pytest.approx(1.0)


def test_jaccard_partial_overlap():
    traces = [
        _trace("a", "42", raw_output="the quick brown fox"),
        _trace("b", "42", raw_output="the slow brown dog"),
    ]
    metrics = jaccard_distance(traces)
    # Shared: {the, brown} = 2, Union: {the,quick,brown,fox,slow,dog} = 6
    assert metrics[0].value == pytest.approx(1.0 - 2 / 6)


# ---------------------------------------------------------------------------
# agreement_for_different_reasons
# ---------------------------------------------------------------------------


def test_afdr_detected():
    """Same conclusion but very different reasoning -> AFDR."""
    traces = [
        _trace("a", "42", reasoning=["by mathematical induction we derive the result"]),
        _trace("b", "42", reasoning=["empirical observation of patterns suggests this"]),
    ]
    metrics = agreement_for_different_reasons(traces)
    assert metrics[0].value == 1.0


def test_afdr_not_detected_different_conclusions():
    """Different conclusions -> not AFDR regardless of reasoning."""
    traces = [
        _trace("a", "yes", reasoning=["reason A"]),
        _trace("b", "no", reasoning=["reason B"]),
    ]
    metrics = agreement_for_different_reasons(traces)
    assert metrics[0].value == 0.0


def test_afdr_not_detected_same_reasoning():
    """Same conclusion AND same reasoning -> not AFDR."""
    traces = [
        _trace("a", "42", reasoning=["step 1", "step 2"]),
        _trace("b", "42", reasoning=["step 1", "step 2"]),
    ]
    metrics = agreement_for_different_reasons(traces)
    assert metrics[0].value == 0.0


# ---------------------------------------------------------------------------
# model_diversity_score
# ---------------------------------------------------------------------------


def test_model_diversity_all_different():
    traces = [
        _trace("a", "42", model_family="anthropic"),
        _trace("b", "42", model_family="openai"),
        _trace("c", "42", model_family="google"),
    ]
    assert model_diversity_score(traces) == pytest.approx(1.0)


def test_model_diversity_all_same():
    traces = [
        _trace("a", "42", model_family="anthropic"),
        _trace("b", "42", model_family="anthropic"),
        _trace("c", "42", model_family="anthropic"),
    ]
    assert model_diversity_score(traces) == pytest.approx(1 / 3)


def test_model_diversity_empty():
    assert model_diversity_score([]) == 0.0


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_all_returns_report():
    traces = [
        _trace("a", "42", reasoning=["step A"], raw_output="analysis A", model_family="anthropic"),
        _trace("b", "42", reasoning=["step B"], raw_output="analysis B", model_family="openai"),
        _trace("c", "99", reasoning=["step C"], raw_output="analysis C", model_family="google"),
    ]
    report = await compute_all(traces, include_embeddings=False)
    assert isinstance(report, IndependenceReport)
    assert report.model_diversity_score == pytest.approx(1.0)
    assert report.summary != ""
    assert len(report.pairwise_metrics) > 0


@pytest.mark.asyncio
async def test_compute_all_single_trace():
    traces = [_trace("a", "42")]
    report = await compute_all(traces, include_embeddings=False)
    assert "Insufficient" in report.summary


@pytest.mark.asyncio
async def test_compute_all_no_embeddings():
    traces = [
        _trace("a", "yes", model_family="anthropic"),
        _trace("b", "no", model_family="openai"),
    ]
    report = await compute_all(traces, include_embeddings=False)
    assert report.embedding_distances is None


@pytest.mark.asyncio
async def test_compute_all_pairwise_count():
    """3 traces -> 3 pairs per metric type, 3 metric types = 9 total."""
    traces = [
        _trace("a", "x", reasoning=["r1"], raw_output="out1"),
        _trace("b", "y", reasoning=["r2"], raw_output="out2"),
        _trace("c", "z", reasoning=["r3"], raw_output="out3"),
    ]
    report = await compute_all(traces, include_embeddings=False)
    # 3 pairs * 3 metric types (divergence + jaccard + afdr)
    assert len(report.pairwise_metrics) == 9
