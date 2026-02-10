"""Format collected training data for Gemini fine-tuning.

Usage:
    uv run python -m examples.training.format_for_gemini
    uv run python -m examples.training.format_for_gemini --strategy full_dialectic
"""

from __future__ import annotations

import argparse

from rde.training.collector import load_results
from rde.training.formatter import format_all, save_formatted
from rde.training.models import (
    DistillationStrategy,
    FormattingConfig,
    ProviderFormat,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Format training data for Gemini")
    parser.add_argument("--input", default="training_data/engine_results.jsonl")
    parser.add_argument("--output-dir", default="training_data/gemini")
    parser.add_argument(
        "--strategy",
        choices=["resolution", "full_dialectic", "role_specific"],
        default="resolution",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    strategy = DistillationStrategy(args.strategy)

    config = FormattingConfig(
        input_path=args.input,
        output_path=args.output_dir,
        strategies=[strategy],
        provider_format=ProviderFormat.GEMINI,
    )

    examples = format_all(results, config)
    output_file = f"{args.output_dir}/{strategy.value}.jsonl"
    save_formatted(examples, output_file, ProviderFormat.GEMINI)

    print(f"Formatted {len(examples)} examples -> {output_file}")
    print(f"Strategy: {strategy.value}")
    if examples:
        print(f"\nSample input (truncated): {examples[0].input_text[:100]}...")
        print(f"Sample output (truncated): {examples[0].output_text[:100]}...")


if __name__ == "__main__":
    main()
