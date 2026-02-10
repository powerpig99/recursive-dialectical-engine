"""Abstract provider interface for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Unified response from any provider."""

    content: str
    model: str
    usage: dict = field(default_factory=dict)  # {"prompt_tokens": N, "completion_tokens": M}
    latency_ms: float = 0.0


class BaseProvider(ABC):
    """Abstract interface all providers implement."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request. Returns unified LLMResponse."""
        ...

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Whether this provider can serve the given model identifier."""
        ...

    async def close(self) -> None:
        """Cleanup async client resources. Override if needed."""
        pass
