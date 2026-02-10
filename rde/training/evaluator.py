"""Evaluate fine-tuned models against multi-model baselines.

Answers Phase 6's core question: does the fine-tuned single model
produce reasoning that matches multi-model quality?
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..engine import DialecticalEngine
from ..models import ModelConfig, RecursionBudget
from ..utils.metrics import IndependenceReport, compute_all
from .models import (
    DistillationStrategy,
    EvaluationConfig,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


async def evaluate(config: EvaluationConfig) -> list[EvaluationResult]:
    """Run evaluation: fine-tuned single-model vs multi-model baseline.

    For each problem, runs both conditions and compares metrics.
    """
    results: list[EvaluationResult] = []

    for problem in config.problems:
        for strategy in config.strategies_to_evaluate:
            logger.info(
                "Evaluating %s / %s",
                problem.get("name", "?"),
                strategy.value,
            )

            ft_result, ft_report = await run_finetuned(
                problem, config.finetuned_model, strategy
            )
            bl_config = ModelConfig.model_validate(config.baseline_config) if config.baseline_config else ModelConfig()
            bl_result, bl_report = await run_baseline(problem, bl_config)

            deltas = compute_deltas(ft_report, bl_report)

            results.append(
                EvaluationResult(
                    strategy=strategy,
                    problem_name=problem.get("name", "unknown"),
                    finetuned_resolution=ft_result.resolution,
                    finetuned_confidence=ft_result.confidence.value,
                    finetuned_avg_divergence=ft_report.avg_reasoning_divergence if ft_report else 0.0,
                    finetuned_avg_jaccard=ft_report.avg_jaccard_distance if ft_report else 0.0,
                    finetuned_afdr_count=ft_report.agreement_for_different_reasons_count if ft_report else 0,
                    baseline_resolution=bl_result.resolution,
                    baseline_confidence=bl_result.confidence.value,
                    baseline_model_diversity=bl_report.model_diversity_score if bl_report else 0.0,
                    baseline_avg_divergence=bl_report.avg_reasoning_divergence if bl_report else 0.0,
                    baseline_avg_jaccard=bl_report.avg_jaccard_distance if bl_report else 0.0,
                    baseline_afdr_count=bl_report.agreement_for_different_reasons_count if bl_report else 0,
                    divergence_delta=deltas.get("divergence_delta", 0.0),
                    jaccard_delta=deltas.get("jaccard_delta", 0.0),
                )
            )

    return results


async def run_finetuned(
    problem: dict[str, str],
    finetuned_model: str,
    strategy: DistillationStrategy,
) -> tuple:
    """Run the fine-tuned model through RDE with all trace slots.

    Returns (EngineResult, IndependenceReport | None).
    """
    config = ModelConfig(
        trace_models=[finetuned_model, finetuned_model, finetuned_model],
        orchestrator_model=finetuned_model,
        arbiter_model=finetuned_model,
    )

    # For role-specific strategy, use orchestrator to assign roles
    use_orchestrator = strategy == DistillationStrategy.ROLE_SPECIFIC

    async with DialecticalEngine(config) as engine:
        result = await engine.run(
            prompt=problem["prompt"],
            use_orchestrator=use_orchestrator,
            budget=RecursionBudget(max_depth=1),
        )

    report = None
    if result.normalized_traces and len(result.normalized_traces) >= 2:
        report = await compute_all(result.normalized_traces, include_embeddings=False)

    return result, report


async def run_baseline(
    problem: dict[str, str],
    baseline_config: ModelConfig,
) -> tuple:
    """Run the multi-model baseline through RDE.

    Returns (EngineResult, IndependenceReport | None).
    """
    async with DialecticalEngine(baseline_config) as engine:
        result = await engine.run(
            prompt=problem["prompt"],
            use_orchestrator=True,
            budget=RecursionBudget(max_depth=1),
        )

    report = None
    if result.normalized_traces and len(result.normalized_traces) >= 2:
        report = await compute_all(result.normalized_traces, include_embeddings=False)

    return result, report


def compute_deltas(
    finetuned_report: IndependenceReport | None,
    baseline_report: IndependenceReport | None,
) -> dict[str, float]:
    """Compute metric deltas: finetuned - baseline.

    Positive delta = fine-tuned is MORE independent (better).
    Negative delta = fine-tuned is LESS independent (worse).
    """
    if finetuned_report is None or baseline_report is None:
        return {"divergence_delta": 0.0, "jaccard_delta": 0.0}

    return {
        "divergence_delta": (
            finetuned_report.avg_reasoning_divergence
            - baseline_report.avg_reasoning_divergence
        ),
        "jaccard_delta": (
            finetuned_report.avg_jaccard_distance
            - baseline_report.avg_jaccard_distance
        ),
    }


def format_evaluation_table(results: list[EvaluationResult]) -> str:
    """Format evaluation results as an ASCII comparison table."""
    if not results:
        return "No evaluation results."

    col_w = 14
    header = (
        f"{'Problem':<22} | {'Strategy':<15} | "
        f"{'FT Conf':<{col_w}} | {'BL Conf':<{col_w}} | "
        f"{'FT Div':>{col_w}} | {'BL Div':>{col_w}} | {'Delta':>{col_w}} | "
        f"{'FT Jacc':>{col_w}} | {'BL Jacc':>{col_w}} | {'Delta':>{col_w}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        div_delta = f"{r.divergence_delta:+.3f}"
        jac_delta = f"{r.jaccard_delta:+.3f}"
        row = (
            f"{r.problem_name:<22} | {r.strategy.value:<15} | "
            f"{r.finetuned_confidence:<{col_w}} | {r.baseline_confidence:<{col_w}} | "
            f"{r.finetuned_avg_divergence:>{col_w}.3f} | {r.baseline_avg_divergence:>{col_w}.3f} | {div_delta:>{col_w}} | "
            f"{r.finetuned_avg_jaccard:>{col_w}.3f} | {r.baseline_avg_jaccard:>{col_w}.3f} | {jac_delta:>{col_w}}"
        )
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


def save_evaluation_results(
    results: list[EvaluationResult],
    path: str | Path,
) -> None:
    """Save evaluation results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in results]
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %d evaluation results to %s", len(results), path)
