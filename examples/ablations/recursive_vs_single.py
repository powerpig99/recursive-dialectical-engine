"""Ablation 3: Recursive vs single-pass arbitration.

Tests whether multi-iteration shadow-informed reframing improves
resolution quality over a single pass.

Compares:
  - single_pass: max_iterations=1 (no reframing)
  - multi_pass_2: max_iterations=2
  - multi_pass_3: max_iterations=3
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Recursive vs Single-pass Arbitration")
    print("=" * 60)

    config = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-3",
        ],
    )

    print("\n--- Running single-pass ---")
    results_1 = await run_ablation("single_pass", config, max_iterations=1)

    print("\n--- Running 2-iteration ---")
    results_2 = await run_ablation("multi_pass_2", config, max_iterations=2)

    print("\n--- Running 3-iteration ---")
    results_3 = await run_ablation("multi_pass_3", config, max_iterations=3)

    print("\n" + format_results_table([results_1, results_2, results_3]))
    save_results_json(
        [results_1, results_2, results_3],
        "ablation_results/recursive_vs_single.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
