"""Extraction utilities for parsing LLM outputs."""

from __future__ import annotations

import json
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
    Tries ```json ... ``` first, then raw { ... } or [ ... ] matching,
    then attempts JSON repair on near-valid JSON.
    """
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if _is_valid_json(candidate):
            return candidate
        # Try repairing the code block content
        repaired = _attempt_json_repair(candidate)
        if repaired is not None:
            return repaired
        return candidate  # Return as-is, let caller handle parse errors

    # Try raw JSON object (outermost braces)
    result = _extract_balanced(text, "{", "}")
    if result:
        if _is_valid_json(result):
            return result
        repaired = _attempt_json_repair(result)
        if repaired is not None:
            return repaired
        return result

    # Try raw JSON array (outermost brackets)
    result = _extract_balanced(text, "[", "]")
    if result:
        if _is_valid_json(result):
            return result
        repaired = _attempt_json_repair(result)
        if repaired is not None:
            return repaired
        return result

    return None


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


def _is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _attempt_json_repair(text: str) -> str | None:
    """Try common repairs on near-valid JSON.

    Handles: trailing commas, single quotes, JS-style comments,
    unquoted keys.  Returns repaired text if valid JSON, else None.
    """
    repaired = text

    # Remove JS-style line comments (// ...)
    repaired = re.sub(r"//[^\n]*", "", repaired)
    # Remove JS-style block comments (/* ... */)
    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)

    # Replace single quotes with double quotes (naive but handles common case)
    # Only replace when they look like JSON string delimiters
    repaired = re.sub(r"(?<=[\[{,:\s])'([^']*?)'(?=[\]},:\s])", r'"\1"', repaired)

    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # Quote unquoted keys: { key: "value" } → { "key": "value" }
    repaired = re.sub(
        r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r' "\1":', repaired
    )

    if _is_valid_json(repaired):
        return repaired
    return None


def validate_trace_config(data: dict) -> bool:
    """Validate that a dict has required TraceConfig fields."""
    required = {"role", "system_prompt"}
    return required.issubset(data.keys())
