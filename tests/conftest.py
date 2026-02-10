"""Shared test fixtures."""

from __future__ import annotations

import pytest

from rde.models import ModelConfig, TraceResult


MONTY_HALL_ACCIDENTAL = """\
You are on a game show with 3 doors. Behind one is a car; behind the others, goats.
1. You pick Door 1.
2. The host, Monty, walks toward the remaining doors.
3. He slips on a banana peel and accidentally crashes into Door 2, forcing it open.
4. Door 2 reveals a Goat.
5. Monty picks himself up and asks: "Do you want to switch to Door 3?"

Does switching to Door 3 increase your probability of winning, or does it remain 50/50?
Derive the probability mathematically based strictly on the "Accidental" condition."""


@pytest.fixture
def model_config():
    return ModelConfig(
        orchestrator_model="test-model",
        arbiter_model="test-model",
        trace_models=["test-model-a", "test-model-b", "test-model-c"],
    )


@pytest.fixture
def sample_trace_results():
    """Three trace results simulating Believer/Logician/Contrarian."""
    return [
        TraceResult(
            trace_id="believer_001",
            role="Believer",
            model_used="test-model-a",
            raw_output="My intuition says switching helps.\n\\boxed{2/3}",
            extracted_answer="2/3",
            latency_ms=100.0,
        ),
        TraceResult(
            trace_id="logician_001",
            role="Logician",
            model_used="test-model-b",
            raw_output=(
                "1. The host accidentally opened a door.\n"
                "2. The accidental condition means no information was conveyed.\n"
                "3. The probability is 1/2.\n"
                "\\boxed{1/2}"
            ),
            extracted_answer="1/2",
            latency_ms=150.0,
        ),
        TraceResult(
            trace_id="contrarian_001",
            role="Contrarian",
            model_used="test-model-c",
            raw_output=(
                "The intuitive answer of 2/3 is a TRAP.\n"
                "- In the standard Monty Hall, the host intentionally opens a goat door.\n"
                "- Here, the host ACCIDENTALLY opened door 2.\n"
                "- The probability is 50/50.\n"
                "\\boxed{1/2}"
            ),
            extracted_answer="1/2",
            latency_ms=200.0,
        ),
    ]
