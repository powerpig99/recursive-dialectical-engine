"""Tests for long-context handling and context rot resistance."""

import pytest

from rde.environment import ContextEnvironment
from rde.models import ModelConfig, TraceConfig
from rde.providers.base import BaseProvider, LLMResponse
from rde.trace import TraceExecutor


# ---------------------------------------------------------------------------
# Long document with a key fact buried in the middle
# ---------------------------------------------------------------------------

LONG_DOCUMENT = (
    "Introduction: This document discusses urban planning.\n\n"
    + "\n\n".join(
        f"Section {i}: " + "Lorem ipsum dolor sit amet. " * 50 for i in range(1, 20)
    )
    + "\n\nSection 20: CRITICAL FINDING: The optimal park size is exactly 4.7 hectares "
    "based on the 2024 meta-analysis by Chen et al.\n\n"
    + "\n\n".join(
        f"Section {i}: " + "Consectetur adipiscing elit. " * 50 for i in range(21, 40)
    )
    + "\n\nConclusion: Urban planning requires careful consideration of multiple factors."
)


ENV_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "KIMI_API_KEY"]


class LongContextProvider(BaseProvider):
    """Provider that checks if the key fact is in the prompt."""

    async def complete(self, messages, model, **kwargs):
        user_msg = messages[-1]["content"] if messages else ""
        if "4.7 hectares" in user_msg:
            return LLMResponse(content="\\boxed{4.7 hectares}", model=model, latency_ms=5.0)
        return LLMResponse(content="\\boxed{unknown}", model=model, latency_ms=5.0)

    def supports_model(self, model: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# ContextEnvironment on long documents
# ---------------------------------------------------------------------------


def test_long_document_structural_partition():
    """Structural partition handles a long document."""
    env = ContextEnvironment(LONG_DOCUMENT)
    parts = env.partition("structural")
    assert len(parts) >= 40  # Many paragraphs


def test_search_finds_buried_fact():
    """Search finds a fact buried deep in a long document."""
    env = ContextEnvironment(LONG_DOCUMENT)
    matches = env.search(r"4\.7 hectares")
    assert len(matches) == 1


def test_peek_accesses_specific_region():
    """Peek can access the middle of a long document."""
    env = ContextEnvironment(LONG_DOCUMENT)
    idx = LONG_DOCUMENT.index("CRITICAL FINDING")
    peeked = env.peek(idx, idx + 100)
    assert "4.7 hectares" in peeked


# ---------------------------------------------------------------------------
# Trace execution with context strategies on long documents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_strategy_finds_buried_fact(monkeypatch):
    """Search context strategy routes relevant content to trace."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig(trace_models=["test-a"])
    router = ModelRouter(config)
    router._providers = {"test": LongContextProvider()}

    env = ContextEnvironment(LONG_DOCUMENT, router=router)
    executor = TraceExecutor(router, env)

    trace_config = TraceConfig(
        role="Searcher",
        perspective="Find specific facts",
        system_prompt="Find the answer. \\boxed{}",
        context_strategy="search:4\\.7 hectares",
    )

    result = await executor.execute(trace_config, "test-a")
    assert result.extracted_answer == "4.7 hectares"


@pytest.mark.asyncio
async def test_full_strategy_passes_everything(monkeypatch):
    """Full strategy passes the entire long document, including the buried fact."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    from rde.providers.router import ModelRouter

    config = ModelConfig(trace_models=["test-a"])
    router = ModelRouter(config)
    router._providers = {"test": LongContextProvider()}

    env = ContextEnvironment(LONG_DOCUMENT, router=router)
    executor = TraceExecutor(router, env)

    trace_config = TraceConfig(
        role="Reader",
        perspective="Read everything",
        system_prompt="Answer. \\boxed{}",
        context_strategy="full",
    )

    result = await executor.execute(trace_config, "test-a")
    assert result.extracted_answer == "4.7 hectares"


@pytest.mark.asyncio
async def test_partition_index_strategy(monkeypatch):
    """Partition with index routes only the selected section."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    from rde.providers.router import ModelRouter

    class IndexCheckProvider(BaseProvider):
        async def complete(self, messages, model, **kwargs):
            user_msg = messages[-1]["content"]
            if user_msg.startswith("[Section"):
                return LLMResponse(
                    content="\\boxed{got indexed section}", model=model, latency_ms=5.0
                )
            return LLMResponse(content="\\boxed{got full}", model=model, latency_ms=5.0)

        def supports_model(self, model: str) -> bool:
            return True

    config = ModelConfig(trace_models=["test-a"])
    router = ModelRouter(config)
    router._providers = {"test": IndexCheckProvider()}

    env = ContextEnvironment(LONG_DOCUMENT, router=router)
    executor = TraceExecutor(router, env)

    trace_config = TraceConfig(
        role="Indexer",
        perspective="Read one section",
        system_prompt="Answer. \\boxed{}",
        context_strategy="partition:structural:0",
    )

    result = await executor.execute(trace_config, "test-a")
    assert result.extracted_answer == "got indexed section"
