"""MLX provider for local Apple Silicon inference.

Uses mlx_lm for local model loading and generation.
Same pattern as the original Dialectical-TTS.
"""

from __future__ import annotations

import os
import time

from .base import BaseProvider, LLMResponse


class MLXProvider(BaseProvider):
    """Local inference via MLX on Apple Silicon.

    Loads a model from disk once at init. Generates synchronously
    (MLX can't parallelize on the same GPU) but wrapped in async interface.
    """

    def __init__(self, model_path: str | None = None):
        import warnings

        warnings.warn(
            "MLXProvider is deprecated. Use LocalOpenAIProvider with vLLM-mlx "
            "for better parallelism (continuous batching). See scripts/run_vllm_mlx.sh.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = model_path or os.environ.get(
            "RDE_LOCAL_MODEL_PATH", "~/Models/Qwen3-8B-4bit"
        )
        self._model_path = os.path.expanduser(path)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if self._model is not None:
            return
        from mlx_lm import load

        self._model, self._tokenizer = load(self._model_path)

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        self._ensure_loaded()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # Apply chat template
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(temp=temperature)
        start = time.perf_counter()
        output = generate(
            self._model,
            self._tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            verbose=False,
            sampler=sampler,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=output,
            model=self._model_path,
            usage={"prompt_tokens": 0, "completion_tokens": 0},  # MLX doesn't report this
            latency_ms=elapsed_ms,
        )

    def supports_model(self, model: str) -> bool:
        """True for local model paths or 'mlx' identifiers."""
        model_lower = model.lower()
        return (
            "mlx" in model_lower
            or model_lower.startswith("~/models")
            or model_lower.startswith("/")
            or model_lower == "local"
        )
