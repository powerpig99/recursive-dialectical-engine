"""Evaluate a fine-tuned model against the multi-model baseline.

Usage:
    uv run python -m examples.training.evaluate_finetuned --model tunedModels/my-rde-model
    uv run python -m examples.training.evaluate_finetuned --model gemini-2.5-pro --strategy role_specific
"""

from __future__ import annotations

import argparse
import asyncio

from rde.models import ModelConfig
from rde.training.evaluator import (
    evaluate,
    format_evaluation_table,
    save_evaluation_results,
)
from rde.training.models import DistillationStrategy, EvaluationConfig

# Reuse canonical ablation problems for evaluation
EVAL_PROBLEMS: list[dict[str, str]] = [
    {
        "name": "monty_hall",
        "prompt": "Should you switch doors in the Monty Hall problem? Prove your answer.",
    },
    {
        "name": "trolley_problem",
        "prompt": (
            "A runaway trolley will kill 5 people. You can divert it to kill "
            "1 person instead. Should you? Analyze from multiple ethical frameworks."
        ),
    },
    {
        "name": "p_vs_np",
        "prompt": "Is P equal to NP? What is the strongest evidence for each side?",
    },
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned RDE model")
    parser.add_argument("--model", required=True, help="Fine-tuned model identifier")
    parser.add_argument(
        "--strategy",
        choices=["resolution", "full_dialectic", "role_specific"],
        default="resolution",
    )
    parser.add_argument("--output", default="evaluation_results/eval.json")
    args = parser.parse_args()

    config = EvaluationConfig(
        finetuned_model=args.model,
        baseline_config=ModelConfig(
            trace_models=[
                "claude-sonnet-4-5-20250929",
                "gemini-2.5-pro",
                "grok-4-1-fast-reasoning",
            ]
        ).model_dump(),
        problems=EVAL_PROBLEMS,
        strategies_to_evaluate=[DistillationStrategy(args.strategy)],
    )

    print(f"Evaluating fine-tuned model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Problems: {len(EVAL_PROBLEMS)}")
    print()

    results = await evaluate(config)

    print(format_evaluation_table(results))
    save_evaluation_results(results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
