"""Ablation 5: Strong vs weak orchestrator model.

Tests whether orchestrator model strength affects trace design quality
and downstream arbitration results.

Compares:
  - strong_orch: claude-opus-4-6 as orchestrator
  - weak_orch: claude-haiku-4-5 as orchestrator
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Orchestrator Model Strength")
    print("=" * 60)

    config_strong = ModelConfig(
        orchestrator_model="claude-opus-4-6",
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-3",
        ],
    )

    config_weak = ModelConfig(
        orchestrator_model="claude-haiku-4-5-20251001",
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-3",
        ],
    )

    print("\n--- Running strong orchestrator (opus) ---")
    results_strong = await run_ablation("strong_orch", config_strong)

    print("\n--- Running weak orchestrator (haiku) ---")
    results_weak = await run_ablation("weak_orch", config_weak)

    print("\n" + format_results_table([results_strong, results_weak]))
    save_results_json(
        [results_strong, results_weak],
        "ablation_results/orchestrator_strength.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
