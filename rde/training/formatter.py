"""Format collected RDE results into fine-tuning training data.

Three distillation strategies:
  A) Resolution: problem -> resolution + causal_chain + shadows
  B) Full dialectic: problem -> multi-perspective analysis + synthesis
  C) Role-specific: (problem + role) -> single trace output
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import (
    CollectedResult,
    DistillationStrategy,
    FormattingConfig,
    ProviderFormat,
    TrainingExample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy A: Resolution distillation
# ---------------------------------------------------------------------------


def format_resolution(result: CollectedResult) -> TrainingExample | None:
    """Problem -> resolution + causal_chain + shadows.

    Teaches the model to produce the final arbiter output directly.
    Returns None if the result has no usable arbitration.
    """
    er = result.engine_result
    resolution = er.get("resolution", "")
    if not resolution or resolution == "All traces failed":
        return None

    # Build output text
    parts = [f"RESOLUTION: {resolution}"]

    causal_chain = er.get("causal_chain", "")
    if causal_chain:
        parts.append(f"\nCAUSAL CHAIN: {causal_chain}")

    confidence = er.get("confidence", "")
    if confidence:
        parts.append(f"\nCONFIDENCE: {confidence}")

    shadows = er.get("shadows", [])
    if shadows:
        parts.append("\nSHADOWS:")
        for s in shadows:
            parts.append(f"- {s}")

    return TrainingExample(
        input_text=result.problem_prompt,
        output_text="\n".join(parts),
        strategy=DistillationStrategy.RESOLUTION,
        source_problem=result.problem_name,
        source_run_index=result.run_index,
    )


# ---------------------------------------------------------------------------
# Strategy B: Full dialectic distillation
# ---------------------------------------------------------------------------


def format_full_dialectic(result: CollectedResult) -> TrainingExample | None:
    """Problem -> multi-perspective analysis + synthesis.

    Teaches the model to produce the entire dialectical process.
    Returns None if there are insufficient traces.
    """
    er = result.engine_result
    traces = er.get("normalized_traces", [])
    if len(traces) < 2:
        return None

    parts = []

    # Each trace as a perspective
    for i, trace in enumerate(traces, 1):
        role = trace.get("role", f"Trace {i}")
        family = trace.get("model_family", "unknown")
        conclusion = trace.get("conclusion", "")
        chain = trace.get("reasoning_chain", [])

        parts.append(f"PERSPECTIVE {i} ({role} - {family}):")
        if chain:
            for step in chain:
                parts.append(f"  {step}")
        parts.append(f"Conclusion: {conclusion}")
        parts.append("")

    # Synthesis from arbitration
    resolution = er.get("resolution", "")
    if resolution:
        parts.append("SYNTHESIS:")
        parts.append(resolution)

        causal_chain = er.get("causal_chain", "")
        if causal_chain:
            parts.append(f"\nReasoning: {causal_chain}")

    shadows = er.get("shadows", [])
    if shadows:
        parts.append("\nSHADOWS:")
        for s in shadows:
            parts.append(f"- {s}")

    return TrainingExample(
        input_text=result.problem_prompt,
        output_text="\n".join(parts),
        strategy=DistillationStrategy.FULL_DIALECTIC,
        source_problem=result.problem_name,
        source_run_index=result.run_index,
    )


# ---------------------------------------------------------------------------
# Strategy C: Role-specific distillation
# ---------------------------------------------------------------------------


def format_role_specific(result: CollectedResult) -> list[TrainingExample]:
    """(Problem + role description) -> role-specific trace output.

    Produces one training example per trace.
    """
    er = result.engine_result
    traces = er.get("trace_results", [])
    examples = []

    for trace in traces:
        if trace.get("error"):
            continue
        raw_output = trace.get("raw_output", "")
        if not raw_output:
            continue

        role = trace.get("role", "unknown")
        input_text = f"ROLE: {role}\n\nPROBLEM: {result.problem_prompt}"

        examples.append(
            TrainingExample(
                input_text=input_text,
                output_text=raw_output,
                strategy=DistillationStrategy.ROLE_SPECIFIC,
                source_problem=result.problem_name,
                source_run_index=result.run_index,
                metadata={"role": role, "model_used": trace.get("model_used", "")},
            )
        )

    return examples


# ---------------------------------------------------------------------------
# Provider formatters
# ---------------------------------------------------------------------------


def to_gemini_format(examples: list[TrainingExample]) -> list[dict]:
    """Convert to Gemini tuning format: {text_input, output}."""
    return [
        {"text_input": ex.input_text, "output": ex.output_text}
        for ex in examples
    ]


def to_anthropic_format(examples: list[TrainingExample]) -> list[dict]:
    """Convert to Anthropic fine-tuning format: messages list."""
    return [
        {
            "messages": [
                {"role": "user", "content": ex.input_text},
                {"role": "assistant", "content": ex.output_text},
            ]
        }
        for ex in examples
    ]


def to_generic_jsonl(examples: list[TrainingExample]) -> list[dict]:
    """Convert to generic JSONL (OpenAI-compatible messages format)."""
    return [
        {
            "messages": [
                {"role": "user", "content": ex.input_text},
                {"role": "assistant", "content": ex.output_text},
            ]
        }
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# Formatting pipeline
# ---------------------------------------------------------------------------

_STRATEGY_FORMATTERS = {
    DistillationStrategy.RESOLUTION: lambda r: [x for x in [format_resolution(r)] if x],
    DistillationStrategy.FULL_DIALECTIC: lambda r: [x for x in [format_full_dialectic(r)] if x],
    DistillationStrategy.ROLE_SPECIFIC: format_role_specific,
}

_PROVIDER_FORMATTERS = {
    ProviderFormat.GEMINI: to_gemini_format,
    ProviderFormat.ANTHROPIC: to_anthropic_format,
    ProviderFormat.JSONL_GENERIC: to_generic_jsonl,
}


def format_all(
    results: list[CollectedResult],
    config: FormattingConfig,
) -> list[TrainingExample]:
    """Apply configured strategies to all results with filtering."""
    examples: list[TrainingExample] = []

    for result in results:
        er = result.engine_result

        # Filter: minimum trace count
        trace_count = len(er.get("trace_results", []))
        if trace_count < config.min_trace_count:
            continue

        # Filter: require arbitration (skip consensus results)
        if config.require_arbitration and er.get("consensus_reached", False):
            continue

        # Apply each strategy
        for strategy in config.strategies:
            formatter = _STRATEGY_FORMATTERS[strategy]
            examples.extend(formatter(result))

    logger.info("Formatted %d training examples", len(examples))
    return examples


def save_formatted(
    examples: list[TrainingExample],
    path: str | Path,
    provider_format: ProviderFormat,
) -> None:
    """Write formatted training examples to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    formatter = _PROVIDER_FORMATTERS[provider_format]
    formatted = formatter(examples)

    with path.open("w") as f:
        for entry in formatted:
            f.write(json.dumps(entry) + "\n")

    logger.info("Saved %d formatted examples to %s", len(formatted), path)
