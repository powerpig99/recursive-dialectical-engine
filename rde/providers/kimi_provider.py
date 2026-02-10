"""Kimi Code provider.

Uses OpenAI-compatible API at https://api.kimi.com/coding/v1.
"""

from __future__ import annotations

import os
import time

from .base import BaseProvider, LLMResponse


class KimiProvider(BaseProvider):
    """Provider for Kimi Code via OpenAI-compatible API."""

    def __init__(self) -> None:
        import openai

        self._client = openai.AsyncOpenAI(
            api_key=os.environ.get("KIMI_API_KEY"),
            base_url="https://api.kimi.com/coding/v1",
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        start = time.perf_counter()
        response = await self._client.chat.completions.create(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=content or "",
            model=response.model or model,
            usage=usage,
            latency_ms=elapsed_ms,
        )

    def supports_model(self, model: str) -> bool:
        return "kimi" in model.lower()

    async def close(self) -> None:
        await self._client.close()
