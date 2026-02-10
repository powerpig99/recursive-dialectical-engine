"""Integration tests requiring real API keys.

Run with: uv run pytest tests/test_integration.py -m integration -v
"""

from __future__ import annotations

import os

import pytest

from rde.engine import DialecticalEngine
from rde.models import ModelConfig

# Skip all tests in this module if no API keys are available
pytestmark = pytest.mark.integration

HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANY_KEY = HAS_ANTHROPIC or HAS_OPENAI


MONTY_HALL_ACCIDENTAL = """\
You are on a game show with 3 doors. Behind one is a car; behind the others, goats.
1. You pick Door 1.
2. The host, Monty, walks toward the remaining doors.
3. He slips on a banana peel and accidentally crashes into Door 2, forcing it open.
4. Door 2 reveals a Goat.
5. Monty picks himself up and asks: "Do you want to switch to Door 3?"

Does switching to Door 3 increase your probability of winning, or does it remain 50/50?
Derive the probability mathematically based strictly on the "Accidental" condition."""


@pytest.mark.skipif(not HAS_ANY_KEY, reason="No API keys available")
@pytest.mark.asyncio
async def test_monty_hall_accidental():
    """The canonical test: should resolve to 50/50 with necessary confidence."""
    config = ModelConfig()

    # Use only available providers
    available_trace_models = []
    if HAS_ANTHROPIC:
        available_trace_models.append("claude-sonnet-4-5-20250929")
        config.arbiter_model = "claude-sonnet-4-5-20250929"
        config.orchestrator_model = "claude-sonnet-4-5-20250929"
    if HAS_OPENAI:
        available_trace_models.append("gpt-4o")

    if len(available_trace_models) < 2:
        # Duplicate to have at least 2 traces
        available_trace_models = available_trace_models * 3

    config.trace_models = available_trace_models[:3]

    async with DialecticalEngine(config) as engine:
        result = await engine.run(
            MONTY_HALL_ACCIDENTAL,
            use_orchestrator=False,
        )

    # The resolution should indicate 50/50 or 1/2
    resolution_lower = result.resolution.lower()
    assert any(
        indicator in resolution_lower
        for indicator in ["50/50", "1/2", "50%", "equal", "0.5"]
    ), f"Expected 50/50, got: {result.resolution}"


@pytest.mark.skipif(not HAS_ANY_KEY, reason="No API keys available")
@pytest.mark.asyncio
async def test_simple_arithmetic():
    """Simple test to verify basic engine flow with real APIs."""
    config = ModelConfig()

    if HAS_ANTHROPIC:
        config.trace_models = ["claude-sonnet-4-5-20250929"] * 3
        config.arbiter_model = "claude-sonnet-4-5-20250929"
    elif HAS_OPENAI:
        config.trace_models = ["gpt-4o"] * 3
        config.arbiter_model = "gpt-4o"

    async with DialecticalEngine(config) as engine:
        result = await engine.run("What is 17 * 23?", use_orchestrator=False)

    assert "391" in result.resolution
