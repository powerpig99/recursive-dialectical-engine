"""Extraction utilities for parsing LLM outputs."""

from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """
    Extract content from \\boxed{...} with brace-balance parsing.
    Preserves the original Dialectical-TTS extraction logic.
    Returns None if no boxed content found.
    """
    start_marker = "\\boxed{"
    start_idx = text.rfind(start_marker)
    if start_idx == -1:
        return None
    content_start = start_idx + len(start_marker)
    balance = 1
    end_idx = content_start
    while balance > 0 and end_idx < len(text):
        if text[end_idx] == "{":
            balance += 1
        elif text[end_idx] == "}":
            balance -= 1
        end_idx += 1
    if balance != 0:
        return None
    return text[content_start : end_idx - 1].strip()


def extract_json_block(text: str) -> str | None:
    """
    Extract JSON from markdown code blocks or raw JSON in LLM output.
    Tries ```json ... ``` first, then raw { ... } or [ ... ] matching.
    """
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try raw JSON object (outermost braces)
    result = _extract_balanced(text, "{", "}")
    if result:
        return result
    # Try raw JSON array (outermost brackets)
    return _extract_balanced(text, "[", "]")


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Extract a balanced delimited block from text."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None
