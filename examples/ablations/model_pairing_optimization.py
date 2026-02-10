"""Ablation 10: Optimal model pair combinations.

Tests all 2-model and 3-model combinations from available families
to find which pairings produce the highest trace independence.

Available models (no OpenAI key):
  - claude-sonnet-4-5
  - gemini-2.5-pro
  - grok-3
"""

from __future__ import annotations

import asyncio
import itertools

from rde.models import ModelConfig

from .harness import format_results_table, run_ablation, save_results_json

AVAILABLE_MODELS = {
    "claude": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-2.5-pro",
    "grok": "grok-3",
}


async def main() -> None:
    print("=" * 60)
    print("ABLATION: Model Pairing Optimization")
    print("=" * 60)

    all_results = []

    # All 2-model pairs
    for names in itertools.combinations(AVAILABLE_MODELS.keys(), 2):
        models = [AVAILABLE_MODELS[n] for n in names]
        condition = f"pair_{'_'.join(names)}"
        config = ModelConfig(trace_models=models)
        print(f"\n--- Running {condition} ---")
        results = await run_ablation(condition, config, use_orchestrator=False)
        all_results.append(results)

    # All 3-model combinations (just the one triple in this case)
    for names in itertools.combinations(AVAILABLE_MODELS.keys(), 3):
        models = [AVAILABLE_MODELS[n] for n in names]
        condition = f"triple_{'_'.join(names)}"
        config = ModelConfig(trace_models=models)
        print(f"\n--- Running {condition} ---")
        results = await run_ablation(condition, config, use_orchestrator=False)
        all_results.append(results)

    print("\n" + format_results_table(all_results))
    save_results_json(all_results, "ablation_results/model_pairing_optimization.json")


if __name__ == "__main__":
    asyncio.run(main())
