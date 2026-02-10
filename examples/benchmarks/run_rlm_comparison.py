"""Run RDE benchmarks and compare against RLM paper results.

Usage:
    python -m examples.benchmarks.run_rlm_comparison
    python -m examples.benchmarks.run_rlm_comparison --benchmark oolong
    python -m examples.benchmarks.run_rlm_comparison --benchmark oolong_pairs --context-size 131000
    python -m examples.benchmarks.run_rlm_comparison --config rde_repl_multi --max-tasks 5
    python -m examples.benchmarks.run_rlm_comparison --config all --max-cost 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from rde.benchmarks.configs import ALL_CONFIGS
from rde.benchmarks.runner import BenchmarkResult, BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_single_config(
    config_name: str,
    config_fn,
    benchmark: str,
    context_size: int,
    max_tasks: int | None,
    max_cost: float,
) -> dict[str, BenchmarkResult]:
    """Run benchmarks for a single configuration."""
    name, model_config, run_opts = config_fn()
    execution_mode = run_opts.get("execution_mode", "direct")
    use_orchestrator = run_opts.get("use_orchestrator", True)
    max_iterations = run_opts.get("max_iterations", 1)
    num_traces = run_opts.get("num_traces")

    runner = BenchmarkRunner(
        engine_config=model_config,
        config_name=name,
        execution_mode=execution_mode,
        use_orchestrator=use_orchestrator,
        max_iterations=max_iterations,
        max_cost_usd=max_cost,
        num_traces=num_traces,
    )

    results: dict[str, BenchmarkResult] = {}

    if benchmark in ("oolong", "all"):
        logger.info("Running OOLONG benchmark for %s...", name)
        # Map approximate token counts to exact powers of 2 for oolong-synth
        context_len_map = {
            1000: 1024, 2000: 2048, 4000: 4096, 8000: 8192,
            16000: 16384, 32000: 32768, 64000: 65536,
            131000: 131072, 131072: 131072, 262000: 262144,
            524000: 524288, 1048000: 1048576,
        }
        oolong_ctx = context_len_map.get(context_size, context_size)
        results["oolong"] = await runner.run_oolong(
            context_len=oolong_ctx,
            max_tasks=max_tasks,
        )
        logger.info("OOLONG: %.1f%% accuracy", results["oolong"].aggregate_score)

    if benchmark in ("oolong_pairs", "all"):
        logger.info("Running OOLONG-Pairs benchmark for %s...", name)
        results["oolong_pairs"] = await runner.run_oolong_pairs(
            context_size=context_size,
            num_tasks=max_tasks,
        )
        logger.info("OOLONG-Pairs: %.1f%% F1", results["oolong_pairs"].aggregate_score)

    if benchmark in ("sniah", "all"):
        logger.info("Running S-NIAH benchmark for %s...", name)
        results["sniah"] = await runner.run_sniah(
            context_sizes=[context_size],
            tasks_per_size=max_tasks or 10,
        )
        logger.info("S-NIAH: %.1f%% accuracy", results["sniah"].aggregate_score)

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RDE benchmarks vs RLM")
    parser.add_argument(
        "--benchmark",
        choices=["oolong", "oolong_pairs", "sniah", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument(
        "--config",
        choices=[*ALL_CONFIGS.keys(), "all"],
        default="rde_repl_multi",
        help="Configuration preset (default: rde_repl_multi)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=131_000,
        help="Context size in tokens (default: 131000)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Max tasks per benchmark (default: all)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=50.0,
        help="Max cost in USD per run (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rlm_comparison.json",
        help="Output file for results JSON",
    )

    args = parser.parse_args()

    configs_to_run = (
        list(ALL_CONFIGS.items())
        if args.config == "all"
        else [(args.config, ALL_CONFIGS[args.config])]
    )

    all_results = {}
    for config_name, config_fn in configs_to_run:
        logger.info("=== Configuration: %s ===", config_name)
        results = await run_single_config(
            config_name=config_name,
            config_fn=config_fn,
            benchmark=args.benchmark,
            context_size=args.context_size,
            max_tasks=args.max_tasks,
            max_cost=args.max_cost,
        )
        all_results[config_name] = results

        # Print comparison table for this config
        print(f"\n--- {config_name} ---")
        print(BenchmarkRunner.comparison_table(results))

    # Save all results
    if all_results:
        # Flatten for saving
        flat_results = {}
        for config_name, results in all_results.items():
            for bench_name, result in results.items():
                flat_results[f"{config_name}/{bench_name}"] = result
        BenchmarkRunner.save_results(flat_results, args.output)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
