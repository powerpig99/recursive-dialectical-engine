"""Trace independence metrics for the Recursive Dialectical Engine.

Measures how independent trace projections are from each other — the core
scientific claim of the RDE architecture. All text-based metrics use stdlib
only. Optional embedding-based metrics use the Google Generative AI API.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..models import NormalizedTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PairwiseMetric(BaseModel):
    """A metric computed between two specific traces."""

    trace_a: str
    trace_b: str
    metric_name: str
    value: float
    detail: str = ""


class IndependenceReport(BaseModel):
    """Complete independence analysis across a set of traces."""

    conclusion_agreement: float = 0.0
    model_diversity_score: float = 0.0
    avg_reasoning_divergence: float = 0.0
    avg_jaccard_distance: float = 0.0
    agreement_for_different_reasons_count: int = 0
    pairwise_metrics: list[PairwiseMetric] = Field(default_factory=list)
    embedding_distances: list[PairwiseMetric] | None = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Text-based metrics (stdlib only)
# ---------------------------------------------------------------------------


def _normalize_conclusion(text: str) -> str:
    """Normalize a conclusion for comparison."""
    return text.strip().lower()


def _pairs(traces: list[NormalizedTrace]) -> list[tuple[NormalizedTrace, NormalizedTrace]]:
    """All unique pairs from a list of traces."""
    return list(itertools.combinations(traces, 2))


def conclusion_agreement(traces: list[NormalizedTrace]) -> float:
    """Fraction of trace pairs with matching conclusions.

    Returns 0.0 (all different) to 1.0 (all agree).
    Returns 0.0 for fewer than 2 traces.
    """
    pairs = _pairs(traces)
    if not pairs:
        return 0.0
    matches = sum(
        1
        for a, b in pairs
        if _normalize_conclusion(a.conclusion) == _normalize_conclusion(b.conclusion)
    )
    return matches / len(pairs)


def reasoning_chain_divergence(traces: list[NormalizedTrace]) -> list[PairwiseMetric]:
    """Measure divergence between reasoning chains using SequenceMatcher.

    Returns 0.0 (identical chains) to 1.0 (completely different) per pair.
    """
    metrics = []
    for a, b in _pairs(traces):
        chain_a = " ".join(a.reasoning_chain)
        chain_b = " ".join(b.reasoning_chain)
        if not chain_a and not chain_b:
            ratio = 1.0  # Both empty = no shared reasoning
        elif not chain_a or not chain_b:
            ratio = 0.0  # One empty = maximum divergence
        else:
            ratio = SequenceMatcher(None, chain_a, chain_b).ratio()
        metrics.append(
            PairwiseMetric(
                trace_a=a.trace_id,
                trace_b=b.trace_id,
                metric_name="reasoning_chain_divergence",
                value=1.0 - ratio,
            )
        )
    return metrics


def jaccard_distance(traces: list[NormalizedTrace]) -> list[PairwiseMetric]:
    """Token-level Jaccard distance on raw outputs.

    Returns 0.0 (identical token sets) to 1.0 (no shared tokens).
    """
    metrics = []
    for a, b in _pairs(traces):
        tokens_a = set(a.raw_output.lower().split())
        tokens_b = set(b.raw_output.lower().split())
        union = tokens_a | tokens_b
        if not union:
            dist = 0.0
        else:
            intersection = tokens_a & tokens_b
            dist = 1.0 - len(intersection) / len(union)
        metrics.append(
            PairwiseMetric(
                trace_a=a.trace_id,
                trace_b=b.trace_id,
                metric_name="jaccard_distance",
                value=dist,
            )
        )
    return metrics


def agreement_for_different_reasons(
    traces: list[NormalizedTrace],
    divergence_threshold: float = 0.3,
) -> list[PairwiseMetric]:
    """Detect pairs that agree on conclusion but via different reasoning.

    This is epistemically significant: it indicates genuine structural
    independence rather than shared training bias.

    Value 1.0 = same conclusion, different reasoning (AFDR detected).
    Value 0.0 = either different conclusions or similar reasoning.
    """
    divergences = {
        (m.trace_a, m.trace_b): m.value for m in reasoning_chain_divergence(traces)
    }
    metrics = []
    for a, b in _pairs(traces):
        same_conclusion = (
            _normalize_conclusion(a.conclusion) == _normalize_conclusion(b.conclusion)
        )
        div = divergences.get((a.trace_id, b.trace_id), 0.0)
        is_afdr = same_conclusion and div > divergence_threshold
        metrics.append(
            PairwiseMetric(
                trace_a=a.trace_id,
                trace_b=b.trace_id,
                metric_name="agreement_for_different_reasons",
                value=1.0 if is_afdr else 0.0,
                detail=f"conclusion_match={same_conclusion}, divergence={div:.3f}",
            )
        )
    return metrics


def model_diversity_score(traces: list[NormalizedTrace]) -> float:
    """Fraction of unique model families among traces.

    Returns 1.0 when every trace is from a different family.
    Returns 1/N when all traces are from the same family.
    Returns 0.0 for empty input.
    """
    if not traces:
        return 0.0
    families = set(t.model_family for t in traces)
    return len(families) / len(traces)


# ---------------------------------------------------------------------------
# Embedding-based metrics (Google Generative AI API)
# ---------------------------------------------------------------------------


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance between two vectors. 0.0 = identical, 2.0 = opposite."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


async def embedding_distance(
    traces: list[NormalizedTrace],
    google_api_key: str | None = None,
) -> list[PairwiseMetric] | None:
    """Compute pairwise cosine distance on embeddings via Google API.

    Uses text-embedding-004 model. Returns None if API is unavailable.
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.info("No GOOGLE_API_KEY; skipping embedding metrics")
        return None

    if len(traces) < 2:
        return []

    try:
        from google import generativeai as genai

        genai.configure(api_key=api_key)

        # Batch embed all trace outputs
        texts = [t.raw_output for t in traces]
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
        )
        embeddings = result["embedding"]

        # Compute pairwise distances
        metrics = []
        for i, j in itertools.combinations(range(len(traces)), 2):
            dist = _cosine_distance(embeddings[i], embeddings[j])
            metrics.append(
                PairwiseMetric(
                    trace_a=traces[i].trace_id,
                    trace_b=traces[j].trace_id,
                    metric_name="embedding_cosine_distance",
                    value=dist,
                )
            )
        return metrics

    except Exception:
        logger.warning("Embedding distance failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


async def compute_all(
    traces: list[NormalizedTrace],
    include_embeddings: bool = True,
) -> IndependenceReport:
    """Compute all available independence metrics.

    Text-based metrics are always computed. Embedding metrics are computed
    only if include_embeddings=True and GOOGLE_API_KEY is available.
    """
    if len(traces) < 2:
        return IndependenceReport(
            summary=f"Insufficient traces for independence analysis ({len(traces)} trace(s))",
        )

    # Text-based metrics
    ca = conclusion_agreement(traces)
    mds = model_diversity_score(traces)
    rcd = reasoning_chain_divergence(traces)
    jd = jaccard_distance(traces)
    afdr = agreement_for_different_reasons(traces)

    avg_rcd = sum(m.value for m in rcd) / len(rcd) if rcd else 0.0
    avg_jd = sum(m.value for m in jd) / len(jd) if jd else 0.0
    afdr_count = sum(1 for m in afdr if m.value > 0.5)

    all_pairwise = rcd + jd + afdr

    # Optional embedding metrics
    emb = None
    if include_embeddings:
        emb = await embedding_distance(traces)

    # Summary
    lines = [
        f"Traces: {len(traces)} | Model diversity: {mds:.2f}",
        f"Conclusion agreement: {ca:.2f} | Avg Jaccard distance: {avg_jd:.2f}",
        f"Avg reasoning divergence: {avg_rcd:.2f}",
        f"Agreement-for-different-reasons pairs: {afdr_count}",
    ]
    if emb is not None:
        avg_emb = sum(m.value for m in emb) / len(emb) if emb else 0.0
        lines.append(f"Avg embedding distance: {avg_emb:.3f}")

    return IndependenceReport(
        conclusion_agreement=ca,
        model_diversity_score=mds,
        avg_reasoning_divergence=avg_rcd,
        avg_jaccard_distance=avg_jd,
        agreement_for_different_reasons_count=afdr_count,
        pairwise_metrics=all_pairwise,
        embedding_distances=emb,
        summary="\n".join(lines),
    )
