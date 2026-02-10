"""Preset configurations for benchmark evaluation.

Four configurations enable systematic comparison:
1. RDE_REPL_MULTI: Full RDE with REPL traces and multi-model — the main result
2. RDE_DIRECT_MULTI: Multi-model without REPL — shows dialectical value alone
3. SINGLE_MODEL_REPL: Single model with REPL — equivalent to RLM
4. VANILLA_BASE: Single model, single pass — the baseline
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
            "gpt-4o",
            "gemini-2.5-pro",
        ],
        sub_lm_models=["claude-haiku-4-5-20251001", "gpt-4o-mini"],
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
            "gpt-4o",
            "gemini-2.5-pro",
        ],
        sub_lm_models=["claude-haiku-4-5-20251001", "gpt-4o-mini"],
    )
    run_opts = {
        "execution_mode": "direct",
        "use_orchestrator": True,
        "max_iterations": 1,
    }
    return "RDE (direct, multi-model)", config, run_opts


def single_model_repl_config(model: str = "gpt-4o") -> tuple[str, ModelConfig, dict]:
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


def vanilla_base_config(model: str = "gpt-4o") -> tuple[str, ModelConfig, dict]:
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


ALL_CONFIGS = {
    "rde_repl_multi": rde_repl_multi_config,
    "rde_direct_multi": rde_direct_multi_config,
    "single_model_repl": single_model_repl_config,
    "vanilla_base": vanilla_base_config,
}
