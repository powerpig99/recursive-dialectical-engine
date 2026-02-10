"""Local OpenAI-compatible provider (vLLM-mlx, LM Studio, Ollama, etc.).

Connects to any server that exposes an OpenAI-compatible /chat/completions
endpoint. The recommended backend for Apple Silicon is vLLM-mlx, which
provides continuous batching for parallel trace execution.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import BaseProvider, LLMResponse


class LocalOpenAIProvider(BaseProvider):
    """Provider for OpenAI-compatible local inference servers.

    Unlike the deprecated MLXProvider (which runs inference in-process and
    serializes all traces), this provider talks to an external server that
    can batch concurrent requests — critical for RDE's parallel trace
    execution.

    Environment variables:
        LOCAL_OPENAI_BASE_URL: Server base URL (default: http://localhost:8000/v1)
        LOCAL_OPENAI_MODEL: Default model name (default: "default")
        LOCAL_OPENAI_API_KEY: Optional API key for the server
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1")
        )
        self.default_model = (
            model
            or os.environ.get("LOCAL_OPENAI_MODEL", "default")
        )
        self._api_key = os.environ.get("LOCAL_OPENAI_API_KEY")

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        chosen_model = model if model != "local" else self.default_model
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        url = f"{self.base_url}/chat/completions"

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Local server error {resp.status_code}: {resp.text[:500]}"
                )
            data = resp.json()
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        usage_data = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
        }

        return LLMResponse(
            content=content or "",
            model=data.get("model", chosen_model),
            usage=usage,
            latency_ms=elapsed_ms,
            estimated_cost=0.0,  # Local inference is free
        )

    def supports_model(self, model: str) -> bool:
        model_lower = model.lower()
        return (
            model_lower == "local"
            or model_lower == self.default_model.lower()
            or "vllm" in model_lower
            or "mlx" in model_lower
            or model_lower.startswith("~/models")
            or model_lower.startswith("/")
        )

    async def close(self) -> None:
        pass  # httpx.AsyncClient is created per-request
