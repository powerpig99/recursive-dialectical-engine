"""Model router — maps model identifiers to providers.

Only initializes providers for which API keys are available.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

from .base import BaseProvider, LLMResponse

if TYPE_CHECKING:
    from ..models import ModelConfig, TraceConfig


class ModelRouter:
    """Routes model strings to the correct provider and manages trace model assignment."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        from ..models import ModelConfig

        self.config = config or ModelConfig()
        self._providers: dict[str, BaseProvider] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize only providers with available API keys."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            from .anthropic_provider import AnthropicProvider

            self._providers["anthropic"] = AnthropicProvider()

        if os.environ.get("OPENAI_API_KEY"):
            from .openai_provider import OpenAIProvider

            self._providers["openai"] = OpenAIProvider()

        if os.environ.get("GOOGLE_API_KEY"):
            from .google_provider import GoogleProvider

            self._providers["google"] = GoogleProvider()

        if os.environ.get("XAI_API_KEY"):
            from .xai_provider import XAIProvider

            self._providers["xai"] = XAIProvider()

        if os.environ.get("KIMI_API_KEY"):
            from .kimi_provider import KimiProvider

            self._providers["kimi"] = KimiProvider()

        # Local OpenAI-compatible server (vLLM-mlx, LM Studio, Ollama)
        local_url = os.environ.get("LOCAL_OPENAI_BASE_URL")
        local_model = os.environ.get("LOCAL_OPENAI_MODEL")
        if local_url or local_model:
            from .local_openai_provider import LocalOpenAIProvider

            self._providers["local"] = LocalOpenAIProvider()
        else:
            # Legacy MLX fallback: available when mlx is installed and model path exists
            try:
                model_path = os.path.expanduser(self.config.local_model_path)
                if os.path.exists(model_path):
                    from .mlx_provider import MLXProvider

                    self._providers["local"] = MLXProvider(model_path)
            except ImportError:
                pass

    def get_provider(self, model: str) -> BaseProvider:
        """Find the provider that supports the given model identifier."""
        for provider in self._providers.values():
            if provider.supports_model(model):
                return provider
        available = list(self._providers.keys())
        raise ValueError(
            f"No provider found for model '{model}'. "
            f"Available providers: {available}"
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Route a completion request to the appropriate provider."""
        provider = self.get_provider(model)
        return await provider.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def assign_trace_models(self, traces: list[TraceConfig]) -> list[str]:
        """Assign models to traces based on the configured strategy.

        Returns a list of model identifiers, one per trace.
        Filters trace_models to only those with available providers.
        """
        available_models = [
            m for m in self.config.trace_models if self._model_available(m)
        ]
        if not available_models:
            raise ValueError(
                "No trace models available. Check API keys and model configuration. "
                f"Configured: {self.config.trace_models}"
            )

        assigned: list[str] = []
        for i, trace in enumerate(traces):
            if trace.model_preference != "any":
                # Support "family:model" syntax (e.g., "local:qwen3-8b", "openai:gpt-4.1")
                if ":" in trace.model_preference:
                    family, model_name = trace.model_preference.split(":", 1)
                    if family in self._providers:
                        assigned.append(model_name)
                        continue
                # Trace has a specific model preference
                if self._model_available(trace.model_preference):
                    assigned.append(trace.model_preference)
                    continue
                # Fall through to assignment strategy if preferred model unavailable

            if self.config.trace_assignment == "round_robin":
                assigned.append(available_models[i % len(available_models)])
            elif self.config.trace_assignment == "random":
                assigned.append(random.choice(available_models))
            else:
                # Default to round robin
                assigned.append(available_models[i % len(available_models)])

        return assigned

    def _model_available(self, model: str) -> bool:
        """Check if a model is servable by any initialized provider."""
        return any(p.supports_model(model) for p in self._providers.values())

    def get_fallback_model(self, failed_model: str) -> str | None:
        """Find a fallback model from a different provider family.

        Prefers models from a different provider than the one that failed.
        Returns None if no alternative is available.
        """
        failed_family = self._get_provider_name(failed_model)
        available = [
            m for m in self.config.trace_models if self._model_available(m)
        ]

        # Prefer different family first
        for m in available:
            if self._get_provider_name(m) != failed_family:
                return m

        # Same family but different model
        for m in available:
            if m != failed_model:
                return m

        return None

    def _get_provider_name(self, model: str) -> str:
        """Get the provider name for a model."""
        for name, provider in self._providers.items():
            if provider.supports_model(model):
                return name
        return "unknown"

    @property
    def available_providers(self) -> list[str]:
        """List of initialized provider names."""
        return list(self._providers.keys())

    async def close(self) -> None:
        """Close all provider clients."""
        for provider in self._providers.values():
            await provider.close()
