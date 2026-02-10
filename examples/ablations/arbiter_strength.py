"""Ablation 6: Strong vs weak arbiter model.

Tests whether arbiter model strength affects resolution quality.
The arbiter performs causal arbitration — a complex reasoning task
that may benefit from stronger models.

Compares:
  - strong_arb: claude-opus-4-6 as arbiter
  - weak_arb: claude-haiku-4-5 as arbiter
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Arbiter Model Strength")
    print("=" * 60)

    config_strong = ModelConfig(
        arbiter_model="claude-opus-4-6",
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-4-1-fast-reasoning",
        ],
    )

    config_weak = ModelConfig(
        arbiter_model="claude-haiku-4-5-20251001",
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-4-1-fast-reasoning",
        ],
    )

    print("\n--- Running strong arbiter (opus) ---")
    results_strong = await run_ablation("strong_arb", config_strong)

    print("\n--- Running weak arbiter (haiku) ---")
    results_weak = await run_ablation("weak_arb", config_weak)

    print("\n" + format_results_table([results_strong, results_weak]))
    save_results_json(
        [results_strong, results_weak],
        "ablation_results/arbiter_strength.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
