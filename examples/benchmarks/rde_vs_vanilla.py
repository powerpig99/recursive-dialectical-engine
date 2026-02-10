"""Benchmark: RDE vs. vanilla single-pass LLM.

Compares accuracy, latency, and cost across several test prompts.
Runs both RDE multi-trace dialectical reasoning and a vanilla single-pass
LLM call on the same prompts, then reports results.

Requirements:
    - At least one API key set (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
    - pip install recursive-dialectical-engine[cloud]

Usage:
    python -m examples.benchmarks.rde_vs_vanilla
"""

from __future__ import annotations

import asyncio
import time

from rde.engine import DialecticalEngine
from rde.models import ModelConfig, RecursionBudget


TEST_PROMPTS = [
    {
        "prompt": (
            "You are on a game show with 3 doors. Behind one is a car; behind the "
            "others, goats. You pick Door 1. The host ACCIDENTALLY opens Door 2 and "
            "reveals a Goat. Should you switch to Door 3 or stay with Door 1?"
        ),
        "expected_contains": "50",
        "label": "Monty Hall Accidental",
    },
    {
        "prompt": "What is the sum of the first 100 positive integers?",
        "expected_contains": "5050",
        "label": "Arithmetic (Gauss sum)",
    },
    {
        "prompt": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?"
        ),
        "expected_contains": "0.05",
        "label": "Bat and Ball",
    },
]


async def run_rde(prompt: str, config: ModelConfig) -> tuple[str, float]:
    """Run RDE and return (resolution, latency_ms)."""
    budget = RecursionBudget(max_depth=2, max_total_calls=20)
    async with DialecticalEngine(config) as engine:
        result = await engine.run(prompt, use_orchestrator=False, budget=budget)
    return result.resolution, result.total_latency_ms


async def run_vanilla(prompt: str, config: ModelConfig) -> tuple[str, float]:
    """Run a single-pass LLM call and return (response, latency_ms)."""
    from rde.providers.router import ModelRouter

    router = ModelRouter(config)
    try:
        models = config.trace_models
        if not models:
            models = list(router.available_models)
        model = models[0] if models else "claude-sonnet-4-5-20250929"

        start = time.perf_counter()
        response = await router.complete(
            messages=[
                {"role": "system", "content": "Answer precisely. Put your answer in \\boxed{}."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.3,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return response.content, elapsed
    finally:
        await router.close()


async def main() -> None:
    config = ModelConfig()
    print("=" * 70)
    print("RDE vs. Vanilla LLM Benchmark")
    print("=" * 70)

    for test in TEST_PROMPTS:
        print(f"\n--- {test['label']} ---")
        print(f"Prompt: {test['prompt'][:80]}...")

        try:
            rde_result, rde_ms = await run_rde(test["prompt"], config)
            rde_correct = test["expected_contains"] in rde_result
            print(f"  RDE:     {rde_result[:100]} ({rde_ms:.0f}ms) {'CORRECT' if rde_correct else 'WRONG'}")
        except Exception as e:
            print(f"  RDE:     ERROR: {e}")

        try:
            vanilla_result, vanilla_ms = await run_vanilla(test["prompt"], config)
            vanilla_correct = test["expected_contains"] in vanilla_result
            print(f"  Vanilla: {vanilla_result[:100]} ({vanilla_ms:.0f}ms) {'CORRECT' if vanilla_correct else 'WRONG'}")
        except Exception as e:
            print(f"  Vanilla: ERROR: {e}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
