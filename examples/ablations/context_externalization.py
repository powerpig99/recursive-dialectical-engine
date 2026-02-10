"""Ablation 2: With vs without ContextEnvironment.

Tests whether the externalized context REPL (ContextEnvironment) improves
reasoning quality compared to raw prompt injection.

Compares:
  - with_env: Normal engine run (context externalized via peek/search/partition)
  - without_env: All traces receive full context, no partitioning
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Context Externalization")
    print("=" * 60)

    config = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-4-1-fast-reasoning",
        ],
    )

    # Condition A: With orchestrator (designs context strategies)
    print("\n--- Running with orchestrator (adaptive context) ---")
    results_with = await run_ablation("with_orchestrator", config, use_orchestrator=True)

    # Condition B: Without orchestrator (all traces get full context)
    print("\n--- Running without orchestrator (full context only) ---")
    results_without = await run_ablation("without_orchestrator", config, use_orchestrator=False)

    print("\n" + format_results_table([results_with, results_without]))
    save_results_json(
        [results_with, results_without],
        "ablation_results/context_externalization.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
