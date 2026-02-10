"""OpenAI provider (GPT models)."""

from __future__ import annotations

import os
import time

from .base import BaseProvider, LLMResponse


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models. Direct OpenAI-compatible API."""

    # Per-million-token pricing: (input, output)
    COSTS: dict[str, tuple[float, float]] = {
        "gpt-5": (2.0, 8.0),
        "gpt-5-mini": (0.40, 1.60),
        "gpt-5.2": (2.0, 8.0),
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "o3": (2.0, 8.0),
    }

    def __init__(self) -> None:
        import openai

        self._client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
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
        estimated_cost = 0.0
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            estimated_cost = self._estimate_cost(
                model,
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0,
            )

        return LLMResponse(
            content=content or "",
            model=response.model or model,
            usage=usage,
            latency_ms=elapsed_ms,
            estimated_cost=estimated_cost,
        )

    def supports_model(self, model: str) -> bool:
        model_lower = model.lower()
        return any(
            prefix in model_lower for prefix in ("gpt", "o1", "o3", "o4", "chatgpt")
        )

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
