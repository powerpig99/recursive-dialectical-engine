"""Tests for the ContextEnvironment."""

import pytest

from rde.environment import ContextEnvironment


SAMPLE_PROMPT = """\
You are on a game show with 3 doors.

Behind one is a car; behind the others, goats.

1. You pick Door 1.
2. The host opens Door 2.
3. Door 2 reveals a Goat."""


def test_peek():
    env = ContextEnvironment(SAMPLE_PROMPT)
    peeked = env.peek(0, 20)
    assert peeked == "You are on a game sh"


def test_peek_lines():
    env = ContextEnvironment(SAMPLE_PROMPT)
    # Lines 1-1 should be the first line
    first_line = env.peek_lines(1, 1)
    assert first_line == "You are on a game show with 3 doors."


def test_peek_lines_range():
    env = ContextEnvironment(SAMPLE_PROMPT)
    # Lines 5-7 should cover the numbered list
    result = env.peek_lines(5, 7)
    assert "1. You pick Door 1." in result
    assert "2. The host opens Door 2." in result
    assert "3. Door 2 reveals a Goat." in result


def test_peek_lines_clamps():
    env = ContextEnvironment("line1\nline2\nline3")
    # Out-of-range is clamped
    result = env.peek_lines(1, 100)
    assert "line1" in result
    assert "line3" in result


def test_search():
    env = ContextEnvironment(SAMPLE_PROMPT)
    matches = env.search(r"Door \d")
    assert len(matches) == 3
    assert "Door 1" in matches
    assert "Door 2" in matches


def test_search_lines():
    env = ContextEnvironment(SAMPLE_PROMPT)
    results = env.search_lines(r"Door \d")
    assert len(results) == 3
    # Each result is (line_number, line_content)
    line_numbers = [r[0] for r in results]
    assert all(isinstance(n, int) for n in line_numbers)
    assert all("Door" in r[1] for r in results)


def test_search_lines_no_match():
    env = ContextEnvironment(SAMPLE_PROMPT)
    results = env.search_lines(r"unicorn")
    assert results == []


def test_partition_structural():
    env = ContextEnvironment(SAMPLE_PROMPT)
    parts = env.partition("structural")
    assert len(parts) == 3  # 3 paragraphs separated by blank lines


def test_partition_single_paragraph():
    env = ContextEnvironment("No blank lines here at all")
    parts = env.partition("structural")
    assert len(parts) == 1


def test_spawn_sub_lm_no_router_raises():
    """spawn_sub_lm raises RuntimeError when no router is wired."""
    env = ContextEnvironment("test")
    with pytest.raises(RuntimeError, match="not wired"):
        import asyncio
        asyncio.run(env.spawn_sub_lm("sub-problem"))


def test_store_iteration():
    env = ContextEnvironment("test")
    assert env.current_iteration == 0
    env.store_iteration(results={"t1": "output"}, arbitration={"resolution": "42"})
    assert env.current_iteration == 1
    assert len(env.iteration_history) == 1
    assert env.iteration_history[0]["iteration"] == 0


# ---------------------------------------------------------------------------
# Phase 4: partition cache and new strategies
# ---------------------------------------------------------------------------

ENV_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "KIMI_API_KEY"]


def test_partition_unknown_strategy_falls_back():
    """Unknown strategies fall back to structural."""
    env = ContextEnvironment(SAMPLE_PROMPT)
    parts = env.partition("nonexistent")
    structural = env.partition("structural")
    assert parts == structural


def test_partition_semantic_falls_back_without_router():
    """Semantic partition falls back to structural when no router is wired."""
    env = ContextEnvironment(SAMPLE_PROMPT)
    parts = env.partition("semantic")
    structural = env.partition("structural")
    assert parts == structural


def test_partition_cache():
    """Partition results are cached."""
    env = ContextEnvironment(SAMPLE_PROMPT)
    parts1 = env.partition("structural")
    parts2 = env.partition("structural")
    assert parts1 is parts2  # Same object from cache


@pytest.mark.asyncio
async def test_prepare_partitions_semantic(monkeypatch):
    """Semantic partition via prepare_partitions() with mocked router."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    from rde.models import ModelConfig
    from rde.providers.base import BaseProvider, LLMResponse
    from rde.providers.router import ModelRouter

    class PartitionProvider(BaseProvider):
        async def complete(self, messages, model, **kwargs):
            return LLMResponse(
                content='["Section about doors", "Section about probability"]',
                model=model,
                latency_ms=5.0,
            )

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig()
    router = ModelRouter(config)
    router._providers = {"test": PartitionProvider()}

    env = ContextEnvironment(
        SAMPLE_PROMPT,
        router=router,
        sub_lm_models=["test-model"],
    )

    parts = await env.prepare_partitions("semantic")
    assert len(parts) == 2
    assert parts[0] == "Section about doors"

    # Verify cache — partition() returns same result
    cached = env.partition("semantic")
    assert cached is parts


@pytest.mark.asyncio
async def test_prepare_partitions_constraint_based(monkeypatch):
    """Constraint-based partition extracts constraints via LLM."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    from rde.models import ModelConfig
    from rde.providers.base import BaseProvider, LLMResponse
    from rde.providers.router import ModelRouter

    class ConstraintProvider(BaseProvider):
        async def complete(self, messages, model, **kwargs):
            return LLMResponse(
                content='["3 doors total", "1 car behind one door", "Host opens door accidentally"]',
                model=model,
                latency_ms=5.0,
            )

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig()
    router = ModelRouter(config)
    router._providers = {"test": ConstraintProvider()}

    env = ContextEnvironment(
        SAMPLE_PROMPT,
        router=router,
        sub_lm_models=["test-model"],
    )

    parts = await env.prepare_partitions("constraint-based")
    assert len(parts) == 3
    assert "3 doors" in parts[0]


@pytest.mark.asyncio
async def test_prepare_partitions_custom():
    """Custom partition uses user-supplied callable."""
    env = ContextEnvironment(SAMPLE_PROMPT)

    def split_by_sentences(text: str) -> list[str]:
        return [s.strip() for s in text.split(".") if s.strip()]

    parts = await env.prepare_partitions("custom", custom_fn=split_by_sentences)
    assert len(parts) > 1
    # Verify cache
    cached = env.partition("custom")
    assert cached is parts
