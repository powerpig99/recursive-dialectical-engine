"""Preset configurations for benchmark evaluation.

Configurations enable systematic comparison:
1. RDE_REPL_MULTI: Full RDE with REPL traces and multi-model — the main result
2. RDE_DIRECT_MULTI: Multi-model without REPL — shows dialectical value alone
3. SINGLE_MODEL_REPL: Single model with REPL — equivalent to RLM
4. VANILLA_BASE: Single model, single pass — the baseline

Model IDs (as of Feb 2026):
  Anthropic: claude-opus-4-6, claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001
  OpenAI:    gpt-5, gpt-5-mini, gpt-5.2
  Google:    gemini-2.5-pro, gemini-2.5-flash
  xAI:       grok-4-1-fast-reasoning
  Moonshot:  kimi-k2.5-preview
"""

from __future__ import annotations

from ..models import ModelConfig


def rde_repl_multi_config() -> tuple[str, ModelConfig, dict]:
    """RDE with REPL traces and multi-model composition (main result)."""
    config = ModelConfig(
        orchestrator_model="claude-sonnet-4-5-20250929",
        arbiter_model="claude-sonnet-4-5-20250929",
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gpt-5",
            "gemini-2.5-pro",
        ],
        sub_lm_models=["claude-haiku-4-5-20251001", "gpt-5-mini"],
    )
    run_opts = {
        "execution_mode": "repl",
        "use_orchestrator": True,
        "max_iterations": 1,
    }
    return "RDE (REPL, multi-model)", config, run_opts


def rde_direct_multi_config() -> tuple[str, ModelConfig, dict]:
    """RDE with direct traces and multi-model (dialectical value without REPL)."""
    config = ModelConfig(
        orchestrator_model="claude-sonnet-4-5-20250929",
        arbiter_model="claude-sonnet-4-5-20250929",
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gpt-5",
            "gemini-2.5-pro",
        ],
        sub_lm_models=["claude-haiku-4-5-20251001", "gpt-5-mini"],
    )
    run_opts = {
        "execution_mode": "direct",
        "use_orchestrator": True,
        "max_iterations": 1,
    }
    return "RDE (direct, multi-model)", config, run_opts


def single_model_repl_config(model: str = "gpt-5") -> tuple[str, ModelConfig, dict]:
    """Single model with REPL — equivalent to RLM for fair comparison."""
    config = ModelConfig(
        orchestrator_model=model,
        arbiter_model=model,
        trace_models=[model],
        sub_lm_models=[model],
    )
    run_opts = {
        "execution_mode": "repl",
        "use_orchestrator": False,
        "max_iterations": 1,
        "num_traces": 1,
    }
    return f"Single-model REPL ({model})", config, run_opts


def vanilla_base_config(model: str = "gpt-5") -> tuple[str, ModelConfig, dict]:
    """Single model, single pass — the baseline."""
    config = ModelConfig(
        orchestrator_model=model,
        arbiter_model=model,
        trace_models=[model],
        sub_lm_models=[model],
    )
    run_opts = {
        "execution_mode": "direct",
        "use_orchestrator": False,
        "max_iterations": 1,
        "num_traces": 1,
    }
    return f"Base model ({model})", config, run_opts


def single_model_repl_claude_config() -> tuple[str, ModelConfig, dict]:
    """Single Claude Sonnet with REPL — for long-context benchmarks."""
    return single_model_repl_config("claude-sonnet-4-5-20250929")


def vanilla_base_claude_config() -> tuple[str, ModelConfig, dict]:
    """Single Claude Sonnet, single pass — baseline for long-context."""
    return vanilla_base_config("claude-sonnet-4-5-20250929")


def single_model_repl_gemini_config() -> tuple[str, ModelConfig, dict]:
    """Single Gemini 2.5 Flash with REPL — cost-effective long-context (1M max)."""
    return single_model_repl_config("gemini-2.5-flash")


def vanilla_base_gemini_config() -> tuple[str, ModelConfig, dict]:
    """Single Gemini 2.5 Flash, single pass — cheap long-context baseline."""
    return vanilla_base_config("gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Local model configs (vLLM-mlx / LM Studio / Ollama)
# ---------------------------------------------------------------------------


def local_vanilla_config(model: str = "local") -> tuple[str, ModelConfig, dict]:
    """Single local model, single pass — baseline."""
    config = ModelConfig(
        orchestrator_model=model,
        arbiter_model=model,
        trace_models=[model],
        sub_lm_models=[model],
    )
    run_opts = {
        "execution_mode": "direct",
        "use_orchestrator": False,
        "max_iterations": 1,
        "num_traces": 1,
    }
    return f"Local vanilla ({model})", config, run_opts


def local_repl_config(model: str = "local") -> tuple[str, ModelConfig, dict]:
    """Single local model with REPL — RLM equivalent."""
    config = ModelConfig(
        orchestrator_model=model,
        arbiter_model=model,
        trace_models=[model],
        sub_lm_models=[model],
    )
    run_opts = {
        "execution_mode": "repl",
        "use_orchestrator": False,
        "max_iterations": 1,
        "num_traces": 1,
    }
    return f"Local REPL ({model})", config, run_opts


def local_dialectical_same_config(model: str = "local") -> tuple[str, ModelConfig, dict]:
    """3x same local model, dialectical — multi-trace without model diversity."""
    config = ModelConfig(
        orchestrator_model=model,
        arbiter_model=model,
        trace_models=[model, model, model],
        sub_lm_models=[model],
        orchestrator_temperature=0.3,
        arbiter_temperature=0.1,
    )
    run_opts = {
        "execution_mode": "direct",
        "use_orchestrator": True,
        "max_iterations": 1,
    }
    return f"Local dialectical 3x ({model})", config, run_opts


def local_dialectical_repl_same_config(model: str = "local") -> tuple[str, ModelConfig, dict]:
    """3x same local model, REPL + dialectical — full RDE, same model."""
    config = ModelConfig(
        orchestrator_model=model,
        arbiter_model=model,
        trace_models=[model, model, model],
        sub_lm_models=[model],
        orchestrator_temperature=0.3,
        arbiter_temperature=0.1,
    )
    run_opts = {
        "execution_mode": "repl",
        "use_orchestrator": True,
        "max_iterations": 1,
    }
    return f"Local REPL+dialectical 3x ({model})", config, run_opts


def local_dialectical_2iter_config(model: str = "local") -> tuple[str, ModelConfig, dict]:
    """3x same local model, dialectical with 2 iterations — shadow reframing."""
    config = ModelConfig(
        orchestrator_model=model,
        arbiter_model=model,
        trace_models=[model, model, model],
        sub_lm_models=[model],
        orchestrator_temperature=0.3,
        arbiter_temperature=0.1,
    )
    run_opts = {
        "execution_mode": "direct",
        "use_orchestrator": True,
        "max_iterations": 2,
    }
    return f"Local dialectical 3x 2-iter ({model})", config, run_opts


# Registry of all configs
ALL_CONFIGS = {
    # Frontier model configs
    "rde_repl_multi": rde_repl_multi_config,
    "rde_direct_multi": rde_direct_multi_config,
    "single_model_repl": single_model_repl_config,
    "single_model_repl_claude": single_model_repl_claude_config,
    "single_model_repl_gemini": single_model_repl_gemini_config,
    "vanilla_base": vanilla_base_config,
    "vanilla_base_claude": vanilla_base_claude_config,
    "vanilla_base_gemini": vanilla_base_gemini_config,
    # Local model configs
    "local_vanilla": local_vanilla_config,
    "local_repl": local_repl_config,
    "local_dialectical_same": local_dialectical_same_config,
    "local_dialectical_repl_same": local_dialectical_repl_same_config,
    "local_dialectical_2iter": local_dialectical_2iter_config,
}

# Subset for quick local-only runs
LOCAL_CONFIGS = {
    "local_vanilla": local_vanilla_config,
    "local_repl": local_repl_config,
    "local_dialectical_same": local_dialectical_same_config,
    "local_dialectical_repl_same": local_dialectical_repl_same_config,
    "local_dialectical_2iter": local_dialectical_2iter_config,
}
