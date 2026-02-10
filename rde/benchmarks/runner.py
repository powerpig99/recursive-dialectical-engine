"""Benchmark runner with RLM baseline comparison.

Runs RDE on OOLONG, OOLONG-Pairs, and S-NIAH benchmarks and produces
comparison tables against the published results from Zhang et al. (2025).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine import DialecticalEngine
from ..models import ModelConfig, RecursionBudget
from .datasets import BenchmarkDataset, BenchmarkTask, OolongLoader, SNIAHLoader
from .scoring import exact_match, f1_pairs, oolong_accuracy

logger = logging.getLogger(__name__)


# RLM Paper Table 1 baselines (Zhang et al., arXiv:2512.24601)
# These are cited from the paper — NOT re-run
RLM_BASELINES: dict[str, dict[str, float]] = {
    "oolong": {
        "GPT-4o Base": 44.0,
        "RLM (GPT-4o)": 56.5,
        "Gemini-2.0-Flash Base": 31.5,
        "RLM (Gemini-2.0-Flash)": 50.0,
        "Qwen3-235B Base": 36.0,
        "RLM (Qwen3-235B)": 48.0,
    },
    "oolong_pairs": {
        "GPT-4o Base": 0.04,
        "RLM (GPT-4o)": 58.0,
        "Gemini-2.0-Flash Base": 0.0,
        "RLM (Gemini-2.0-Flash)": 37.5,
    },
    "sniah": {
        "GPT-4o Base": 93.3,
        "RLM (GPT-4o)": 96.0,
    },
}


@dataclass
class TaskResult:
    """Result from running a single benchmark task."""

    task_id: str
    predicted: Any
    ground_truth: Any
    score: float
    raw_output: str = ""
    latency_ms: float = 0.0
    llm_calls: int = 0


@dataclass
class BenchmarkResult:
    """Aggregate result from running a benchmark suite."""

    benchmark_name: str
    config_name: str
    task_results: list[TaskResult] = field(default_factory=list)
    aggregate_score: float = 0.0
    aggregate_metric: str = ""  # "accuracy" | "f1" | "exact_match"
    total_latency_ms: float = 0.0
    total_llm_calls: int = 0

    @property
    def num_tasks(self) -> int:
        return len(self.task_results)


class BenchmarkRunner:
    """Run RDE benchmarks and compare against RLM baselines."""

    def __init__(
        self,
        engine_config: ModelConfig,
        config_name: str = "RDE",
        execution_mode: str = "repl",
        use_orchestrator: bool = True,
        max_iterations: int = 1,
        max_cost_usd: float = 50.0,
    ) -> None:
        self.engine_config = engine_config
        self.config_name = config_name
        self.execution_mode = execution_mode
        self.use_orchestrator = use_orchestrator
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd

    async def run_oolong(
        self,
        context_size: int = 131_000,
        num_tasks: int | None = None,
    ) -> BenchmarkResult:
        """Run OOLONG benchmark (classification distribution)."""
        dataset = OolongLoader.load_trec_coarse(
            context_size_tokens=context_size,
            num_tasks=num_tasks or 50,
        )
        return await self._run_dataset(dataset, "accuracy")

    async def run_oolong_pairs(
        self,
        context_size: int = 131_000,
        num_tasks: int | None = None,
    ) -> BenchmarkResult:
        """Run OOLONG-Pairs benchmark (pairwise matching)."""
        dataset = OolongLoader.load_oolong_pairs(
            context_size_tokens=context_size,
            num_tasks=num_tasks or 20,
        )
        return await self._run_dataset(dataset, "f1")

    async def run_sniah(
        self,
        context_sizes: list[int] | None = None,
        tasks_per_size: int = 10,
    ) -> BenchmarkResult:
        """Run S-NIAH benchmark (needle in haystack)."""
        dataset = SNIAHLoader.load(
            context_sizes=context_sizes or [131_000],
            tasks_per_size=tasks_per_size,
        )
        return await self._run_dataset(dataset, "exact_match")

    async def run_all(self) -> dict[str, BenchmarkResult]:
        """Run all benchmarks."""
        results = {}
        results["oolong"] = await self.run_oolong()
        results["oolong_pairs"] = await self.run_oolong_pairs()
        results["sniah"] = await self.run_sniah()
        return results

    async def _run_dataset(
        self,
        dataset: BenchmarkDataset,
        metric_name: str,
    ) -> BenchmarkResult:
        """Run RDE on all tasks in a dataset."""
        task_results = []
        budget = RecursionBudget(max_cost_usd=self.max_cost_usd, max_total_calls=500)

        async with DialecticalEngine(self.engine_config) as engine:
            for task in dataset.tasks:
                logger.info("Running task %s", task.task_id)
                start = time.perf_counter()

                try:
                    # Combine context and query as the prompt
                    prompt = f"{task.context}\n\n---\nQUERY: {task.query}"

                    result = await engine.run(
                        prompt=prompt,
                        use_orchestrator=self.use_orchestrator,
                        max_iterations=self.max_iterations,
                        budget=budget,
                    )

                    elapsed_ms = (time.perf_counter() - start) * 1000

                    # Score the result
                    score = self._score_task(task, result.resolution)

                    task_results.append(TaskResult(
                        task_id=task.task_id,
                        predicted=result.resolution,
                        ground_truth=task.ground_truth,
                        score=score,
                        raw_output=result.resolution,
                        latency_ms=elapsed_ms,
                        llm_calls=budget.total_calls,
                    ))

                except Exception as e:
                    logger.error("Task %s failed: %s", task.task_id, e)
                    task_results.append(TaskResult(
                        task_id=task.task_id,
                        predicted=None,
                        ground_truth=task.ground_truth,
                        score=0.0,
                        raw_output=str(e),
                    ))

        # Aggregate
        scores = [tr.score for tr in task_results]
        aggregate = sum(scores) / len(scores) * 100 if scores else 0.0

        return BenchmarkResult(
            benchmark_name=dataset.name,
            config_name=self.config_name,
            task_results=task_results,
            aggregate_score=aggregate,
            aggregate_metric=metric_name,
            total_latency_ms=sum(tr.latency_ms for tr in task_results),
            total_llm_calls=sum(tr.llm_calls for tr in task_results),
        )

    def _score_task(self, task: BenchmarkTask, resolution: str) -> float:
        """Score a task based on its scoring function."""
        try:
            if task.scoring_fn == "oolong_accuracy":
                predicted = self._parse_category_counts(resolution)
                return oolong_accuracy(predicted, task.ground_truth)
            elif task.scoring_fn == "f1_pairs":
                predicted = self._parse_pairs(resolution)
                return f1_pairs(predicted, task.ground_truth)
            elif task.scoring_fn == "exact_match":
                return exact_match(resolution, task.ground_truth)
        except Exception as e:
            logger.warning("Scoring failed for %s: %s", task.task_id, e)
        return 0.0

    @staticmethod
    def _parse_category_counts(text: str) -> dict[str, float]:
        """Parse category count JSON from LLM output."""
        try:
            # Try to find JSON in the text
            import re
            match = re.search(r"\{[^}]+\}", text)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {}

    @staticmethod
    def _parse_pairs(text: str) -> set[tuple[str, str]]:
        """Parse pair list JSON from LLM output."""
        try:
            import re
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                pairs = json.loads(match.group())
                return {tuple(p) for p in pairs if len(p) == 2}
        except (json.JSONDecodeError, AttributeError):
            pass
        return set()

    @staticmethod
    def comparison_table(
        results: dict[str, BenchmarkResult],
        include_rlm: bool = True,
    ) -> str:
        """Generate a markdown comparison table against RLM baselines."""
        lines = []
        lines.append("| Method | OOLONG (Acc%) | OOLONG-Pairs (F1%) | S-NIAH (Acc%) |")
        lines.append("|--------|---------------|---------------------|---------------|")

        if include_rlm:
            # Add RLM baseline rows
            lines.append(
                f"| GPT-4o Base | "
                f"{RLM_BASELINES['oolong'].get('GPT-4o Base', '-'):.1f} | "
                f"{RLM_BASELINES['oolong_pairs'].get('GPT-4o Base', '-'):.2f} | "
                f"{RLM_BASELINES['sniah'].get('GPT-4o Base', '-'):.1f} |"
            )
            lines.append(
                f"| RLM (GPT-4o) [Zhang et al.] | "
                f"{RLM_BASELINES['oolong'].get('RLM (GPT-4o)', '-'):.1f} | "
                f"{RLM_BASELINES['oolong_pairs'].get('RLM (GPT-4o)', '-'):.1f} | "
                f"{RLM_BASELINES['sniah'].get('RLM (GPT-4o)', '-'):.1f} |"
            )

        # Add our results
        oolong = results.get("oolong")
        pairs = results.get("oolong_pairs")
        sniah = results.get("sniah")
        config_name = oolong.config_name if oolong else "RDE"

        oolong_score = f"{oolong.aggregate_score:.1f}" if oolong else "-"
        pairs_score = f"{pairs.aggregate_score:.1f}" if pairs else "-"
        sniah_score = f"{sniah.aggregate_score:.1f}" if sniah else "-"

        lines.append(f"| **{config_name}** | {oolong_score} | {pairs_score} | {sniah_score} |")

        return "\n".join(lines)

    @staticmethod
    def save_results(
        results: dict[str, BenchmarkResult],
        path: str | Path,
    ) -> None:
        """Save benchmark results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialized = {}
        for name, result in results.items():
            serialized[name] = {
                "benchmark_name": result.benchmark_name,
                "config_name": result.config_name,
                "aggregate_score": result.aggregate_score,
                "aggregate_metric": result.aggregate_metric,
                "num_tasks": result.num_tasks,
                "total_latency_ms": result.total_latency_ms,
                "total_llm_calls": result.total_llm_calls,
                "task_scores": [
                    {"task_id": tr.task_id, "score": tr.score}
                    for tr in result.task_results
                ],
            }

        path.write_text(json.dumps(serialized, indent=2))
        logger.info("Results saved to %s", path)
