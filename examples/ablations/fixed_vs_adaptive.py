"""Ablation 4: Fixed default traces vs orchestrator-designed traces.

Tests whether the orchestrator's adaptive trace design improves
quality over the fixed 3-trace default (Believer/Logician/Contrarian).

Compares:
  - fixed: Default 3 traces, no orchestrator
  - adaptive: Orchestrator designs problem-specific traces
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Fixed vs Adaptive Trace Design")
    print("=" * 60)

    config = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-3",
        ],
    )

    print("\n--- Running fixed traces (no orchestrator) ---")
    results_fixed = await run_ablation("fixed_traces", config, use_orchestrator=False)

    print("\n--- Running adaptive traces (orchestrator) ---")
    results_adaptive = await run_ablation("adaptive_traces", config, use_orchestrator=True)

    print("\n" + format_results_table([results_fixed, results_adaptive]))
    save_results_json(
        [results_fixed, results_adaptive],
        "ablation_results/fixed_vs_adaptive.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
