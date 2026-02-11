"""Run RDE benchmarks on local models via vLLM-mlx.

Designed for systematic local evaluation: vanilla vs REPL vs dialectical.

Usage:
    # Check server health first
    python -m examples.benchmarks.run_local_benchmarks --health

    # Round 1: Pipeline validation (1K, 10 tasks)
    python -m examples.benchmarks.run_local_benchmarks \
        --config local_vanilla --benchmark oolong --context-size 1024 --max-tasks 10

    # Run all local configs on one benchmark
    python -m examples.benchmarks.run_local_benchmarks \
        --config local_all --benchmark oolong --context-size 4096 --max-tasks 50

    # Full comparison at 32K
    python -m examples.benchmarks.run_local_benchmarks \
        --config local_all --benchmark all --context-size 32768 --max-tasks 50
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from rde.benchmarks.configs import ALL_CONFIGS, LOCAL_CONFIGS
from rde.benchmarks.runner import BenchmarkResult, BenchmarkRunner
from rde.providers.local_openai_provider import LocalOpenAIProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def check_health() -> str | None:
    """Check vLLM-mlx health, return model name or None."""
    provider = LocalOpenAIProvider()
    health = await provider.check_health()
    if health.get("ok"):
        models = health.get("models", [])
        model_name = models[0] if models else "unknown"
        print(f"Server healthy at {provider.base_url}")
        print(f"  Loaded model: {model_name}")
        print(f"  All models: {models}")
        return model_name
    else:
        print(f"Server NOT reachable at {provider.base_url}")
        print(f"  Error: {health.get('error')}")
        return None


async def run_config(
    config_name: str,
    config_fn,
    model_name: str,
    benchmark: str,
    context_size: int,
    max_tasks: int | None,
    max_cost: float,
) -> dict[str, BenchmarkResult]:
    """Run benchmarks for a single local config."""
    name, model_config, run_opts = config_fn(model_name)
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

    # Map approximate token counts to powers of 2 for oolong-synth
    context_len_map = {
        1000: 1024, 2000: 2048, 4000: 4096, 8000: 8192,
        16000: 16384, 32000: 32768, 64000: 65536,
        131000: 131072, 262000: 262144, 524000: 524288,
    }

    if benchmark in ("oolong", "all"):
        logger.info("Running OOLONG for %s...", name)
        oolong_ctx = context_len_map.get(context_size, context_size)
        results["oolong"] = await runner.run_oolong(
            context_len=oolong_ctx,
            max_tasks=max_tasks,
        )
        logger.info("OOLONG: %.1f%% accuracy", results["oolong"].aggregate_score)

    if benchmark in ("oolong_pairs", "all"):
        logger.info("Running OOLONG-Pairs for %s...", name)
        results["oolong_pairs"] = await runner.run_oolong_pairs(
            context_size=context_size,
            num_tasks=max_tasks,
        )
        logger.info("OOLONG-Pairs: %.1f%% F1", results["oolong_pairs"].aggregate_score)

    if benchmark in ("sniah", "all"):
        logger.info("Running S-NIAH for %s...", name)
        results["sniah"] = await runner.run_sniah(
            context_sizes=[context_size],
            tasks_per_size=max_tasks or 10,
        )
        logger.info("S-NIAH: %.1f%% accuracy", results["sniah"].aggregate_score)

    return results


def print_comparison(all_results: dict[str, dict[str, BenchmarkResult]]) -> None:
    """Print a comparison table across all configs."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Collect all benchmarks seen
    benchmarks = set()
    for results in all_results.values():
        benchmarks.update(results.keys())
    benchmarks = sorted(benchmarks)

    # Header
    header = "| Config | " + " | ".join(f"{b} (%)" for b in benchmarks) + " |"
    sep = "|" + "-" * (len("Config") + 2) + "|" + "|".join(
        "-" * (len(b) + 6) for b in benchmarks
    ) + "|"
    print(header)
    print(sep)

    # Rows
    for config_name, results in all_results.items():
        cells = []
        for b in benchmarks:
            r = results.get(b)
            cells.append(f"{r.aggregate_score:.1f}" if r else "-")
        print(f"| {config_name} | " + " | ".join(cells) + " |")

    # Independence metrics summary
    has_independence = False
    for results in all_results.values():
        for r in results.values():
            if r.independence_report:
                has_independence = True
                break

    if has_independence:
        print("\n--- Independence Metrics ---")
        for config_name, results in all_results.items():
            for bench_name, r in results.items():
                if r.independence_report:
                    rpt = r.independence_report
                    print(
                        f"  {config_name}/{bench_name}: "
                        f"agreement={rpt.conclusion_agreement:.2f} "
                        f"diversity={rpt.model_diversity_score:.2f} "
                        f"divergence={rpt.avg_reasoning_divergence:.2f} "
                        f"jaccard={rpt.avg_jaccard_distance:.2f} "
                        f"AFDR={rpt.agreement_for_different_reasons_count}"
                    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RDE benchmarks on local models")
    parser.add_argument(
        "--health", action="store_true",
        help="Check vLLM-mlx server health and exit",
    )
    parser.add_argument(
        "--benchmark",
        choices=["oolong", "oolong_pairs", "sniah", "all"],
        default="oolong",
        help="Which benchmark to run (default: oolong)",
    )
    parser.add_argument(
        "--config",
        choices=[*LOCAL_CONFIGS.keys(), "local_all", "all", *ALL_CONFIGS.keys()],
        default="local_vanilla",
        help="Configuration preset (default: local_vanilla)",
    )
    parser.add_argument(
        "--context-size", type=int, default=1024,
        help="Context size in tokens (default: 1024)",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Max tasks per benchmark (default: all)",
    )
    parser.add_argument(
        "--max-cost", type=float, default=0.0,
        help="Max cost in USD (default: 0 = unlimited for local)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/local",
        help="Output directory for results (default: results/local)",
    )

    args = parser.parse_args()

    # Health check
    model_name = await check_health()
    if args.health:
        sys.exit(0 if model_name else 1)

    if not model_name:
        print("\nERROR: No local server available. Start vLLM-mlx first:")
        print("  ./scripts/run_vllm_mlx.sh")
        sys.exit(1)

    # Resolve configs
    if args.config == "local_all":
        configs_to_run = list(LOCAL_CONFIGS.items())
    elif args.config == "all":
        configs_to_run = list(ALL_CONFIGS.items())
    elif args.config in LOCAL_CONFIGS:
        configs_to_run = [(args.config, LOCAL_CONFIGS[args.config])]
    else:
        configs_to_run = [(args.config, ALL_CONFIGS[args.config])]

    # Use unlimited cost for local models
    max_cost = args.max_cost if args.max_cost > 0 else 999999.0

    all_results: dict[str, dict[str, BenchmarkResult]] = {}
    flat_results: dict[str, BenchmarkResult] = {}

    for config_name, config_fn in configs_to_run:
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name} | MODEL: {model_name}")
        print(f"{'='*60}")

        results = await run_config(
            config_name=config_name,
            config_fn=config_fn,
            model_name=model_name,
            benchmark=args.benchmark,
            context_size=args.context_size,
            max_tasks=args.max_tasks,
            max_cost=max_cost,
        )

        all_results[config_name] = results
        for bench_name, r in results.items():
            flat_results[f"{config_name}/{bench_name}"] = r

    # Print comparison
    print_comparison(all_results)

    # Save results
    if flat_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize model name for filename
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{safe_model}_{args.benchmark}_{args.context_size}_{timestamp}.json"

        BenchmarkRunner.save_results(flat_results, output_path)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
