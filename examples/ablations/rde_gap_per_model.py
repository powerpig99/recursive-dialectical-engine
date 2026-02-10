"""Ablation 9: RDE gap per model family.

Measures how much each model family benefits from RDE vs standalone.
The "RDE gap" is the improvement in reasoning quality when a model
participates in the dialectical process vs reasoning alone.

Compares single-model standalone runs against the full RDE pipeline
for each available model family.
"""

from __future__ import annotations

import asyncio

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json

# Model families available (no OpenAI key)
MODEL_FAMILIES = {
    "claude": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-2.5-pro",
    "grok": "grok-3",
}


async def main() -> None:
    print("=" * 60)
    print("ABLATION: RDE Gap per Model Family")
    print("=" * 60)

    all_results = []

    # Standalone: each model solo (3 copies of same model, no orchestrator)
    for family, model in MODEL_FAMILIES.items():
        config = ModelConfig(
            trace_models=[model, model, model],
        )
        print(f"\n--- Running standalone {family} ---")
        results = await run_ablation(f"standalone_{family}", config, use_orchestrator=False)
        all_results.append(results)

    # Full RDE: cross-model with orchestrator
    config_rde = ModelConfig(
        trace_models=list(MODEL_FAMILIES.values()),
    )
    print("\n--- Running full RDE (cross-model) ---")
    results_rde = await run_ablation("full_rde", config_rde, use_orchestrator=True)
    all_results.append(results_rde)

    print("\n" + format_results_table(all_results))
    save_results_json(all_results, "ablation_results/rde_gap_per_model.json")


if __name__ == "__main__":
    asyncio.run(main())
