"""Benchmark runner with RLM baseline comparison.

Runs RDE on OOLONG, OOLONG-Pairs, and S-NIAH benchmarks and produces
comparison tables against the published results from Zhang et al. (2025).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine import DialecticalEngine
from ..models import ModelConfig, RecursionBudget
from ..providers.router import ModelRouter
from .datasets import BenchmarkDataset, BenchmarkTask, OolongLoader, OolongPairsLoader, SNIAHLoader
from .scoring import exact_match, f1_pairs, oolong_accuracy, oolong_score

logger = logging.getLogger(__name__)


# RLM Paper Figure 1 baselines (Zhang et al., arXiv:2512.24601)
# Updated to GPT-5 results (paper revision, 2025). Read from Figure 1 at 131K context.
# These are cited from the paper — NOT re-run.
RLM_BASELINES: dict[str, dict[str, float]] = {
    "oolong": {
        "GPT-5 Base": 45.0,
        "RLM (GPT-5)": 57.0,
    },
    "oolong_pairs": {
        "GPT-5 Base": 0.0,
        "RLM (GPT-5)": 63.0,
    },
    "sniah": {
        "GPT-5 Base": 100.0,
        "RLM (GPT-5)": 100.0,
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
        num_traces: int | None = None,
    ) -> None:
        self.engine_config = engine_config
        self.config_name = config_name
        self.execution_mode = execution_mode
        self.use_orchestrator = use_orchestrator
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.num_traces = num_traces

    @property
    def _is_baseline(self) -> bool:
        """True when running a single-model, single-pass baseline (no orchestration)."""
        return not self.use_orchestrator and self.num_traces == 1 and self.execution_mode == "direct"

    async def run_oolong(
        self,
        context_len: int = 131_072,
        max_tasks: int | None = None,
    ) -> BenchmarkResult:
        """Run OOLONG benchmark from official oolongbench/oolong-synth dataset."""
        dataset = OolongLoader.load(
            context_len=context_len,
            max_tasks=max_tasks or 400,
        )
        return await self._run_dataset(dataset, "accuracy")

    async def run_oolong_pairs(
        self,
        context_size: int = 32_768,
        num_tasks: int | None = None,
    ) -> BenchmarkResult:
        """Run OOLONG-Pairs benchmark (20 pairwise queries from RLM paper)."""
        dataset = OolongPairsLoader.load(
            context_size_tokens=context_size,
            num_tasks=num_tasks or 20,
        )
        return await self._run_dataset(dataset, "f1")

    async def run_sniah(
        self,
        context_sizes: list[int] | None = None,
        tasks_per_size: int = 10,
    ) -> BenchmarkResult:
        """Run S-NIAH benchmark (RULER-style needle in haystack)."""
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
        """Run on all tasks in a dataset.

        For baseline configs (single-model, single-pass, no orchestration),
        calls the LLM directly with context in the system message and query in
        the user message — matching how OOLONG and S-NIAH papers evaluate.

        For RDE configs, routes through the full dialectical engine.
        """
        if self._is_baseline:
            return await self._run_dataset_baseline(dataset, metric_name)
        return await self._run_dataset_engine(dataset, metric_name)

    async def _run_dataset_baseline(
        self,
        dataset: BenchmarkDataset,
        metric_name: str,
    ) -> BenchmarkResult:
        """Run baseline: single LLM call per task with proper message formatting.

        Context goes in the system message; query goes in the user message.
        This matches how long-context benchmarks are evaluated in the literature.
        """
        task_results = []
        model = self.engine_config.trace_models[0]
        router = ModelRouter(self.engine_config)

        for task in dataset.tasks:
            logger.info("Running task %s (baseline, model=%s)", task.task_id, model)
            start = time.perf_counter()

            messages = [
                {"role": "system", "content": task.context},
                {"role": "user", "content": task.query},
            ]

            # Retry with backoff for rate limiting
            full_output = None
            last_error = None
            for attempt in range(4):
                try:
                    response = await router.complete(
                        messages=messages,
                        model=model,
                        temperature=0.0,
                        max_tokens=4096,
                    )
                    full_output = response.content
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    # Extract retry delay from API error if available
                    retry_match = _re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_str)
                    if "429" in error_str or "quota" in error_str.lower():
                        wait = int(retry_match.group(1)) + 5 if retry_match else 30 * (attempt + 1)
                        logger.warning("Rate limited on %s, waiting %ds (attempt %d/4)", task.task_id, wait, attempt + 1)
                        await asyncio.sleep(wait)
                    else:
                        break  # Non-retryable error

            elapsed_ms = (time.perf_counter() - start) * 1000

            if full_output is not None:
                score = self._score_task(task, full_output)
                task_results.append(TaskResult(
                    task_id=task.task_id,
                    predicted=full_output,
                    ground_truth=task.ground_truth,
                    score=score,
                    raw_output=full_output,
                    latency_ms=elapsed_ms,
                    llm_calls=1,
                ))
            else:
                logger.error("Task %s failed after retries: %s", task.task_id, last_error)
                task_results.append(TaskResult(
                    task_id=task.task_id,
                    predicted=None,
                    ground_truth=task.ground_truth,
                    score=0.0,
                    raw_output=str(last_error),
                ))

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

    async def _run_dataset_engine(
        self,
        dataset: BenchmarkDataset,
        metric_name: str,
    ) -> BenchmarkResult:
        """Run through the full dialectical engine (for RDE configurations)."""
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
                        num_traces=self.num_traces,
                        execution_mode=self.execution_mode,
                    )

                    elapsed_ms = (time.perf_counter() - start) * 1000

                    # Collect all raw outputs for scoring
                    full_output = result.resolution
                    if result.trace_results:
                        full_output = "\n".join(
                            tr.raw_output for tr in result.trace_results
                            if tr.raw_output
                        )

                    score = self._score_task(task, full_output)

                    task_results.append(TaskResult(
                        task_id=task.task_id,
                        predicted=result.resolution,
                        ground_truth=task.ground_truth,
                        score=score,
                        raw_output=full_output,
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
            if task.scoring_fn == "oolong":
                return oolong_score(resolution, task.ground_truth, task.answer_type)
            elif task.scoring_fn == "oolong_accuracy":
                # Legacy: category-count format
                predicted = self._parse_category_counts(resolution)
                return oolong_accuracy(predicted, task.ground_truth)
            elif task.scoring_fn == "f1_pairs":
                predicted = self._parse_user_pairs(resolution)
                return f1_pairs(predicted, task.ground_truth)
            elif task.scoring_fn == "exact_match":
                return exact_match(resolution, task.ground_truth)
        except Exception as e:
            logger.warning("Scoring failed for %s: %s", task.task_id, e)
        return 0.0

    @staticmethod
    def _parse_category_counts(text: str) -> dict[str, float]:
        """Parse category count JSON from LLM output."""
        import re

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text)

        # Try to find JSON object (greedy, handles multiline)
        try:
            match = re.search(r"\{[^{}]*\}", cleaned)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try multiline JSON with nested structure
        try:
            match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
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
    def _parse_user_pairs(text: str) -> set[tuple[str, str]]:
        """Parse user ID pairs from LLM output.

        OOLONG-Pairs format: (user_id_1, user_id_2) on each line.
        Also handles JSON array format: [["id1", "id2"], ...].
        """
        import re

        pairs: set[tuple[str, str]] = set()

        # Try tuple format: (12345, 67890)
        tuple_matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)
        if tuple_matches:
            for a, b in tuple_matches:
                pairs.add((str(min(int(a), int(b))), str(max(int(a), int(b)))))
            return pairs

        # Try JSON array format
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for p in parsed:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        a, b = str(p[0]), str(p[1])
                        pairs.add((min(a, b), max(a, b)))
                return pairs
        except (json.JSONDecodeError, AttributeError):
            pass

        return pairs

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
            # Add RLM baseline rows (GPT-5, from Zhang et al. Figure 1 at 131K)
            lines.append(
                f"| GPT-5 Base | "
                f"{RLM_BASELINES['oolong'].get('GPT-5 Base', '-'):.1f} | "
                f"{RLM_BASELINES['oolong_pairs'].get('GPT-5 Base', '-'):.1f} | "
                f"{RLM_BASELINES['sniah'].get('GPT-5 Base', '-'):.1f} |"
            )
            lines.append(
                f"| RLM (GPT-5) [Zhang et al.] | "
                f"{RLM_BASELINES['oolong'].get('RLM (GPT-5)', '-'):.1f} | "
                f"{RLM_BASELINES['oolong_pairs'].get('RLM (GPT-5)', '-'):.1f} | "
                f"{RLM_BASELINES['sniah'].get('RLM (GPT-5)', '-'):.1f} |"
            )

        # Add our results
        oolong = results.get("oolong")
        pairs = results.get("oolong_pairs")
        sniah = results.get("sniah")
        config_name = oolong.config_name if oolong else "RDE"

        oolong_val = f"{oolong.aggregate_score:.1f}" if oolong else "-"
        pairs_val = f"{pairs.aggregate_score:.1f}" if pairs else "-"
        sniah_val = f"{sniah.aggregate_score:.1f}" if sniah else "-"

        lines.append(f"| **{config_name}** | {oolong_val} | {pairs_val} | {sniah_val} |")

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
