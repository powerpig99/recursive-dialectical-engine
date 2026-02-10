"""Anthropic provider (Claude models)."""

from __future__ import annotations

import os
import time

from .base import BaseProvider, LLMResponse


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models.

    Handles the Anthropic-specific system message extraction
    (system is a top-level param, not a message role).
    """

    # Per-million-token pricing: (input, output)
    COSTS: dict[str, tuple[float, float]] = {
        "claude-opus-4-6": (15.0, 75.0),
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-haiku-4-5": (0.80, 4.0),
    }

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        # Extract system message — Anthropic uses a top-level `system` param
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                chat_messages.append(msg)

        start = time.perf_counter()
        response = await self._client.messages.create(
            model=model,
            system=system_text if system_text else "",
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = response.content[0].text if response.content else ""
        usage = {}
        estimated_cost = 0.0
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
            estimated_cost = self._estimate_cost(
                model, response.usage.input_tokens, response.usage.output_tokens
            )

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            latency_ms=elapsed_ms,
            estimated_cost=estimated_cost,
        )

    def supports_model(self, model: str) -> bool:
        return "claude" in model.lower()

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD from token counts."""
        model_lower = model.lower()
        for prefix, (inp, out) in self.COSTS.items():
            if prefix in model_lower:
                return (input_tokens * inp + output_tokens * out) / 1_000_000
        return 0.0

    async def close(self) -> None:
        await self._client.close()
