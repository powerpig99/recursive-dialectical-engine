"""Ablation 8: Open vs guided vs structured scaffolding.

Tests how the degree of structure imposed on traces affects output quality.
Open = free-form, guided = step hints, structured = strict format.

Compares:
  - open: Minimal trace structure
  - guided: Moderate structure with step suggestions
  - structured: Full structured output format
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Scaffolding Calibration")
    print("=" * 60)

    configs = {
        "open": ModelConfig(
            scaffolding_preference="open",
            trace_models=[
                "claude-sonnet-4-5-20250929",
                "gemini-2.5-pro",
                "grok-3",
            ],
        ),
        "guided": ModelConfig(
            scaffolding_preference="guided",
            trace_models=[
                "claude-sonnet-4-5-20250929",
                "gemini-2.5-pro",
                "grok-3",
            ],
        ),
        "structured": ModelConfig(
            scaffolding_preference="structured",
            trace_models=[
                "claude-sonnet-4-5-20250929",
                "gemini-2.5-pro",
                "grok-3",
            ],
        ),
    }

    all_results = []
    for name, config in configs.items():
        print(f"\n--- Running {name} scaffolding ---")
        results = await run_ablation(name, config)
        all_results.append(results)

    print("\n" + format_results_table(all_results))
    save_results_json(all_results, "ablation_results/scaffolding_calibration.json")


if __name__ == "__main__":
    asyncio.run(main())
