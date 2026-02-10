"""Google provider (Gemini models)."""

from __future__ import annotations

import os
import time

from .base import BaseProvider, LLMResponse


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models.

    Uses the google-generativeai SDK. Converts messages to Gemini's Content format.
    """

    # Per-million-token pricing: (input, output)
    COSTS: dict[str, tuple[float, float]] = {
        "gemini-2.5-pro": (1.25, 10.0),
        "gemini-2.5-flash": (0.15, 0.60),
        "gemini-2.0-flash": (0.10, 0.40),
    }

    def __init__(self) -> None:
        from google import generativeai as genai

        self._genai = genai
        self._genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        # Extract system instruction and convert to Gemini format
        system_instruction = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})

        gen_model = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction if system_instruction else None,
        )

        config = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if response_format and response_format.get("type") == "json_object":
            config.response_mime_type = "application/json"

        start = time.perf_counter()
        response = await gen_model.generate_content_async(
            contents,
            generation_config=config,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = response.text if response.text else ""
        usage = {}
        estimated_cost = 0.0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(
                response.usage_metadata, "candidates_token_count", 0
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            estimated_cost = self._estimate_cost(
                model, prompt_tokens or 0, completion_tokens or 0
            )

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            latency_ms=elapsed_ms,
            estimated_cost=estimated_cost,
        )

    def supports_model(self, model: str) -> bool:
        return "gemini" in model.lower()

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD from token counts."""
        model_lower = model.lower()
        for prefix, (inp, out) in self.COSTS.items():
            if prefix in model_lower:
                return (input_tokens * inp + output_tokens * out) / 1_000_000
        return 0.0
