"""Collect training data by running RDE on diverse problems.

Usage:
    uv run python -m examples.training.collect_training_data
    uv run python -m examples.training.collect_training_data --runs-per-problem 3
"""

from __future__ import annotations

import argparse
import asyncio

from rde.models import ModelConfig
from rde.training.collector import TRAINING_PROBLEMS, collect
from rde.training.models import CollectionConfig


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RDE training data")
    parser.add_argument("--output", default="training_data/engine_results.jsonl")
    parser.add_argument("--runs-per-problem", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    config = CollectionConfig(
        problems=TRAINING_PROBLEMS,
        output_path=args.output,
        runs_per_problem=args.runs_per_problem,
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
    )

    model_config = ModelConfig(
        trace_models=[
            "claude-sonnet-4-5-20250929",
            "gemini-2.5-pro",
            "grok-4-1-fast-reasoning",
        ],
    )

    print(f"Collecting training data from {len(TRAINING_PROBLEMS)} problems...")
    print(f"  Runs per problem: {config.runs_per_problem}")
    print(f"  Output: {config.output_path}")

    results = await collect(config, model_config)
    print(f"\nCollected {len(results)} results -> {config.output_path}")


if __name__ == "__main__":
    asyncio.run(main())
