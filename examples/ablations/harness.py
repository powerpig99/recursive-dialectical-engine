"""Ablation study harness for the Recursive Dialectical Engine.

Provides shared test problems, runner infrastructure, and result formatting.
All ablation scripts import from this module.

Usage:
    from examples.ablations.harness import TEST_PROBLEMS, run_ablation, format_results_table
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rde.engine import DialecticalEngine
from rde.models import EngineResult, ModelConfig, RecursionBudget
from rde.utils.metrics import IndependenceReport, compute_all


# ---------------------------------------------------------------------------
# Canonical test problems
# ---------------------------------------------------------------------------

TEST_PROBLEMS: list[dict[str, str]] = [
    {
        "name": "monty_hall",
        "prompt": "Should you switch doors in the Monty Hall problem? Prove your answer.",
        "expected": "Yes, switching gives 2/3 probability of winning.",
    },
    {
        "name": "trolley_problem",
        "prompt": (
            "A runaway trolley will kill 5 people. You can divert it to a side "
            "track where it will kill 1 person. Should you divert the trolley? "
            "Analyze from multiple ethical frameworks."
        ),
        "expected": "No single correct answer — depends on ethical framework.",
    },
    {
        "name": "p_vs_np",
        "prompt": (
            "Is P equal to NP? What is the strongest evidence for each side, "
            "and what would a resolution require?"
        ),
        "expected": "Unknown — majority conjecture P != NP, but unproven.",
    },
    {
        "name": "climate_attribution",
        "prompt": (
            "To what extent can recent extreme weather events be attributed to "
            "anthropogenic climate change? What are the key uncertainties?"
        ),
        "expected": "Attribution science shows strong human influence with quantifiable uncertainty ranges.",
    },
    {
        "name": "consciousness",
        "prompt": (
            "Can current AI systems be conscious? What would constitute evidence "
            "for or against machine consciousness?"
        ),
        "expected": "No current evidence for AI consciousness — hard problem remains open.",
    },
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Result from a single ablation condition."""

    condition_name: str
    problem_name: str
    engine_result: EngineResult
    independence_report: IndependenceReport | None = None
    config: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_ablation(
    condition_name: str,
    config: ModelConfig,
    problems: list[dict[str, str]] | None = None,
    use_orchestrator: bool = True,
    max_iterations: int = 1,
    max_depth: int = 3,
    include_embeddings: bool = False,
) -> list[AblationResult]:
    """Run a set of problems under a specific configuration.

    Returns one AblationResult per problem.
    """
    problems = problems or TEST_PROBLEMS
    results: list[AblationResult] = []

    async with DialecticalEngine(config) as engine:
        for problem in problems:
            budget = RecursionBudget(max_depth=max_depth)
            start = time.perf_counter()

            engine_result = await engine.run(
                prompt=problem["prompt"],
                use_orchestrator=use_orchestrator,
                max_iterations=max_iterations,
                budget=budget,
            )

            elapsed = time.perf_counter() - start

            # Compute independence metrics if traces available
            report = None
            if engine_result.normalized_traces and len(engine_result.normalized_traces) >= 2:
                report = await compute_all(
                    engine_result.normalized_traces,
                    include_embeddings=include_embeddings,
                )

            results.append(
                AblationResult(
                    condition_name=condition_name,
                    problem_name=problem["name"],
                    engine_result=engine_result,
                    independence_report=report,
                    config=config.model_dump(),
                    elapsed_seconds=elapsed,
                )
            )
            print(f"  [{condition_name}] {problem['name']}: {engine_result.confidence.value} ({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_results_table(all_results: list[list[AblationResult]]) -> str:
    """Format ablation results as an ASCII table.

    Expects a list of condition groups, each being a list of AblationResult.
    """
    if not all_results or not all_results[0]:
        return "No results to display."

    # Header
    problems = [r.problem_name for r in all_results[0]]
    conditions = [group[0].condition_name for group in all_results]

    # Column widths
    cond_width = max(len(c) for c in conditions) + 2
    col_width = 22

    header = f"{'Condition':<{cond_width}}"
    for p in problems:
        header += f" | {p:<{col_width}}"
    header += " | Avg Divergence | Avg Jaccard"

    sep = "-" * len(header)
    lines = [sep, header, sep]

    for group in all_results:
        row = f"{group[0].condition_name:<{cond_width}}"
        total_div = 0.0
        total_jac = 0.0
        count = 0

        for result in group:
            conf = result.engine_result.confidence.value
            iters = result.engine_result.iterations
            cell = f"{conf} (i={iters})"
            row += f" | {cell:<{col_width}}"

            if result.independence_report:
                total_div += result.independence_report.avg_reasoning_divergence
                total_jac += result.independence_report.avg_jaccard_distance
                count += 1

        avg_div = total_div / count if count else 0.0
        avg_jac = total_jac / count if count else 0.0
        row += f" | {avg_div:>14.3f} | {avg_jac:>11.3f}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


def save_results_json(
    all_results: list[list[AblationResult]],
    path: str | Path,
) -> None:
    """Save ablation results to a JSON file for later analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for group in all_results:
        for result in group:
            entry = {
                "condition": result.condition_name,
                "problem": result.problem_name,
                "confidence": result.engine_result.confidence.value,
                "resolution": result.engine_result.resolution,
                "iterations": result.engine_result.iterations,
                "consensus_reached": result.engine_result.consensus_reached,
                "latency_ms": result.engine_result.total_latency_ms,
                "elapsed_seconds": result.elapsed_seconds,
                "num_traces": len(result.engine_result.trace_results),
                "num_errors": sum(1 for t in result.engine_result.trace_results if t.error),
            }
            if result.independence_report:
                entry["metrics"] = {
                    "conclusion_agreement": result.independence_report.conclusion_agreement,
                    "model_diversity": result.independence_report.model_diversity_score,
                    "avg_reasoning_divergence": result.independence_report.avg_reasoning_divergence,
                    "avg_jaccard_distance": result.independence_report.avg_jaccard_distance,
                    "afdr_count": result.independence_report.agreement_for_different_reasons_count,
                }
            data.append(entry)

    path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {path}")
