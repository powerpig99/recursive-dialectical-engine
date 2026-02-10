"""Ablation 7: Cheap vs expensive sub-LM models.

Tests whether sub-LM quality affects partition and decomposition quality.
Sub-LMs are used for semantic partitioning, constraint extraction, and
sub-dialectic spawning.

Compares:
  - cheap_sub: haiku-level sub-LMs
  - expensive_sub: sonnet-level sub-LMs
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Sub-LM Quality")
    print("=" * 60)

    config_cheap = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-4-1-fast-reasoning",
        ],
        sub_lm_models=[
            "claude-haiku-4-5-20251001",
        ],
    )

    config_expensive = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-4-1-fast-reasoning",
        ],
        sub_lm_models=[
            "claude-sonnet-4-5-20250929",
        ],
    )

    print("\n--- Running cheap sub-LMs (haiku) ---")
    results_cheap = await run_ablation("cheap_sub", config_cheap)

    print("\n--- Running expensive sub-LMs (sonnet) ---")
    results_expensive = await run_ablation("expensive_sub", config_expensive)

    print("\n" + format_results_table([results_cheap, results_expensive]))
    save_results_json(
        [results_cheap, results_expensive],
        "ablation_results/sub_lm_quality.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
