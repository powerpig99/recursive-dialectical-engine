"""Trace output normalizer for cross-model comparison."""

from __future__ import annotations

from .models import NormalizedTrace, TraceResult
from .utils.extraction import extract_boxed


class TraceNormalizer:
    """Normalizes trace outputs from heterogeneous models into a common format.

    This is NOT about making traces agree — it's about making them COMPARABLE.
    """

    def normalize(self, result: TraceResult) -> NormalizedTrace:
        """Extract structured information from a raw trace output."""
        # Use extracted_answer (from REPL FINAL) when available
        if result.extracted_answer:
            conclusion = result.extracted_answer
        else:
            conclusion = self._extract_conclusion(result.raw_output)
        reasoning_chain = self._extract_reasoning_chain(result.raw_output)
        model_family = self._detect_model_family(result.model_used)

        return NormalizedTrace(
            trace_id=result.trace_id,
            role=result.role,
            model_family=model_family,
            conclusion=conclusion,
            reasoning_chain=reasoning_chain,
            raw_output=result.raw_output,
        )

    def _extract_conclusion(self, text: str) -> str:
        """Try \\boxed{} first, then marker lines, then last non-empty line."""
        boxed = extract_boxed(text)
        if boxed:
            return boxed

        lines = text.strip().split("\n")
        for line in reversed(lines):
            line_lower = line.strip().lower()
            if any(
                marker in line_lower
                for marker in ["answer:", "conclusion:", "therefore,", "thus,", "result:"]
            ):
                return line.strip()

        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return text[:200]

    def _extract_reasoning_chain(self, text: str) -> list[str]:
        """Extract numbered steps or bullet points as reasoning chain."""
        steps = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and (
                (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)")
                or stripped.startswith("- ")
                or stripped.startswith("* ")
            ):
                steps.append(stripped)
        return steps

    def _detect_model_family(self, model: str) -> str:
        """Detect model family from model identifier string."""
        model_lower = model.lower()
        if "claude" in model_lower:
            return "anthropic"
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        if "gemini" in model_lower:
            return "google"
        if "grok" in model_lower or "xai" in model_lower:
            return "xai"
        if "kimi" in model_lower:
            return "kimi"
        if "qwen" in model_lower or "mlx" in model_lower or "models/" in model_lower:
            return "local"
        return "unknown"

    def check_consensus(self, normalized: list[NormalizedTrace]) -> bool:
        """Check if all traces agree (skip arbiter if so).

        Preserves the Dialectical-TTS consensus-skip pattern.
        """
        if len(normalized) < 2:
            return True
        first = self._normalize_answer_text(normalized[0].conclusion)
        return all(self._normalize_answer_text(t.conclusion) == first for t in normalized[1:])

    def _normalize_answer_text(self, text: str) -> str:
        """Normalize answer text for comparison."""
        return text.lower().strip().replace(" ", "")
