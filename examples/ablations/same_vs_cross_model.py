"""Ablation 1: Same-model vs cross-model traces.

The core RDE hypothesis: cross-model traces produce more independent
projections than same-model traces, leading to higher-quality arbitration.

Compares:
  - same_model: All traces use claude-sonnet-4-5
  - cross_model: Traces use claude-sonnet-4-5, gemini-2.5-pro, grok-3
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Same-model vs Cross-model Traces")
    print("=" * 60)

    # Condition A: All same model
    config_same = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-5-20250929",
        ],
    )

    # Condition B: Cross-model (default diverse set, excluding OpenAI)
    config_cross = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-3",
        ],
    )

    print("\n--- Running same-model condition ---")
    results_same = await run_ablation("same_model", config_same)

    print("\n--- Running cross-model condition ---")
    results_cross = await run_ablation("cross_model", config_cross)

    print("\n" + format_results_table([results_same, results_cross]))
    save_results_json([results_same, results_cross], "ablation_results/same_vs_cross_model.json")


if __name__ == "__main__":
    asyncio.run(main())
