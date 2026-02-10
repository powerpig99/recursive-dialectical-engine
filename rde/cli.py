"""CLI entry point for the Recursive Dialectical Engine."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rde",
        description="Recursive Dialectical Engine — multi-model causal reasoning",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The problem to analyze. Use --stdin to read from stdin instead.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt from stdin",
    )
    parser.add_argument(
        "--no-orchestrator",
        action="store_true",
        help="Skip orchestrator LLM call; use default Believer/Logician/Contrarian traces",
    )
    parser.add_argument(
        "--orchestrator-model",
        default=None,
        help="Model for the orchestrator (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--arbiter-model",
        default=None,
        help="Model for the arbiter (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--trace-models",
        default=None,
        help="Comma-separated list of models for traces",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum iterations for shadow-informed reframing (default: 1)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum recursion depth for sub-dialectics (default: 3)",
    )
    parser.add_argument(
        "--trace-log",
        default=None,
        metavar="PATH",
        help="Save call tree JSON to this file path",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compute and display trace independence metrics",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON EngineResult",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


async def run_engine(args: argparse.Namespace) -> None:
    from .engine import DialecticalEngine
    from .models import ModelConfig, RecursionBudget

    config = ModelConfig()
    if args.orchestrator_model:
        config.orchestrator_model = args.orchestrator_model
    if args.arbiter_model:
        config.arbiter_model = args.arbiter_model
    if args.trace_models:
        config.trace_models = [m.strip() for m in args.trace_models.split(",")]

    # Get prompt
    if args.stdin:
        prompt = sys.stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: provide a prompt argument or use --stdin", file=sys.stderr)
        sys.exit(1)

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    budget = RecursionBudget(max_depth=args.max_depth)

    async with DialecticalEngine(config) as engine:
        result = await engine.run(
            prompt=prompt,
            use_orchestrator=not args.no_orchestrator,
            max_iterations=args.max_iterations,
            budget=budget,
        )

    if args.trace_log and result.call_tree:
        from .utils.visualizer import save_call_tree

        save_call_tree(result.call_tree, args.trace_log)
        print(f"Call tree saved to {args.trace_log}", file=sys.stderr)

    if args.metrics and result.normalized_traces:
        from .utils.metrics import compute_all

        report = await compute_all(result.normalized_traces, include_embeddings=True)
        print(f"\n{'='*60}")
        print("INDEPENDENCE METRICS:")
        print(report.summary)
        print(f"{'='*60}")

    if args.json_output:
        print(result.model_dump_json(indent=2))
    else:
        print_result(result)


def print_result(result) -> None:
    """Pretty-print an EngineResult."""
    print(f"\n{'='*60}")
    print(f"RESOLUTION: {result.resolution}")
    print(f"CONFIDENCE: {result.confidence.value}")
    print(f"CONSENSUS:  {result.consensus_reached}")
    print(f"ITERATIONS: {result.iterations}")
    print(f"LATENCY:    {result.total_latency_ms:.0f}ms")

    if result.shadows:
        print("\nSHADOWS:")
        for s in result.shadows:
            print(f"  - {s}")

    if result.causal_chain:
        print(f"\nCAUSAL CHAIN:\n  {result.causal_chain}")

    print(f"\nTRACES ({len(result.trace_results)}):")
    for tr in result.trace_results:
        status = "ERROR" if tr.error else "OK"
        answer = tr.extracted_answer or "(no boxed answer)"
        print(f"  [{status}] {tr.role} ({tr.model_used}): {answer}")
        if tr.error:
            print(f"         Error: {tr.error}")

    print(f"{'='*60}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")

    asyncio.run(run_engine(args))


if __name__ == "__main__":
    main()
