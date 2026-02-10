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
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            latency_ms=elapsed_ms,
        )

    def supports_model(self, model: str) -> bool:
        return "claude" in model.lower()

    async def close(self) -> None:
        await self._client.close()
