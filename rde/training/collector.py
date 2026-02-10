"""Training data collector — runs RDE on problem sets and saves EngineResults."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from ..engine import DialecticalEngine
from ..models import ModelConfig, RecursionBudget
from ..utils.metrics import compute_all
from .models import CollectedResult, CollectionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expanded training problem set (20 diverse problems)
# ---------------------------------------------------------------------------

TRAINING_PROBLEMS: list[dict[str, str]] = [
    # Logic & probability
    {
        "name": "monty_hall",
        "prompt": "Should you switch doors in the Monty Hall problem? Prove your answer.",
    },
    {
        "name": "sleeping_beauty",
        "prompt": "In the Sleeping Beauty problem, what probability should Beauty assign to Heads on awakening? Defend your position.",
    },
    {
        "name": "prisoners_dilemma",
        "prompt": "In a one-shot Prisoner's Dilemma with no communication, should a rational agent cooperate or defect? Analyze from game theory and decision theory.",
    },
    {
        "name": "surprise_exam",
        "prompt": "A teacher announces a surprise exam next week. Can the students logically prove it cannot happen? Resolve the paradox.",
    },
    # Ethics & philosophy
    {
        "name": "trolley_problem",
        "prompt": "A runaway trolley will kill 5 people. You can divert it to kill 1 person instead. Should you? Analyze from multiple ethical frameworks.",
    },
    {
        "name": "experience_machine",
        "prompt": "Would it be rational to plug into Nozick's Experience Machine permanently? What does your answer reveal about the nature of value?",
    },
    {
        "name": "ship_of_theseus",
        "prompt": "If every plank of a ship is gradually replaced, is it the same ship? What does this tell us about identity?",
    },
    {
        "name": "repugnant_conclusion",
        "prompt": "In population ethics, is a world of 10 billion people living wonderful lives better than a world of 100 trillion people barely worth living? Address Parfit's Repugnant Conclusion.",
    },
    # Science & epistemology
    {
        "name": "p_vs_np",
        "prompt": "Is P equal to NP? What is the strongest evidence for each side?",
    },
    {
        "name": "climate_attribution",
        "prompt": "To what extent can recent extreme weather events be attributed to anthropogenic climate change? What are the key uncertainties?",
    },
    {
        "name": "consciousness",
        "prompt": "Can current AI systems be conscious? What would constitute evidence for or against machine consciousness?",
    },
    {
        "name": "dark_matter",
        "prompt": "Is dark matter a real substance or does it indicate our theory of gravity is wrong? Evaluate the evidence for MOND vs particle dark matter.",
    },
    # Causal reasoning
    {
        "name": "correlation_causation",
        "prompt": "Countries with more ice cream consumption have higher drowning rates. Explain this correlation. What causal frameworks help distinguish correlation from causation?",
    },
    {
        "name": "simpson_paradox",
        "prompt": "A treatment shows higher recovery rates in every subgroup but lower overall. How is this possible? What should a doctor recommend?",
    },
    {
        "name": "butterfly_effect",
        "prompt": "Can a butterfly flapping its wings in Brazil cause a tornado in Texas? Analyze the scientific basis and limits of sensitive dependence on initial conditions.",
    },
    # Ambiguous / multi-constraint
    {
        "name": "ai_alignment",
        "prompt": "Is the AI alignment problem fundamentally solvable? What are the strongest arguments for and against the possibility of aligned superintelligence?",
    },
    {
        "name": "free_will",
        "prompt": "Do humans have free will? Analyze from the perspectives of physics, neuroscience, and philosophy.",
    },
    {
        "name": "optimal_taxation",
        "prompt": "What is the optimal top marginal income tax rate? Analyze from economic efficiency, equity, and behavioral perspectives.",
    },
    # Code & formal reasoning
    {
        "name": "halting_problem",
        "prompt": "Explain why the halting problem is undecidable. Is this result practically relevant for real-world software?",
    },
    {
        "name": "godel_incompleteness",
        "prompt": "What are the implications of Godel's incompleteness theorems for mathematics, AI, and human cognition?",
    },
]


# ---------------------------------------------------------------------------
# Collection functions
# ---------------------------------------------------------------------------


async def collect(
    config: CollectionConfig,
    model_config: ModelConfig | None = None,
) -> list[CollectedResult]:
    """Run RDE on all problems, return and save CollectedResults."""
    model_config = model_config or ModelConfig()
    results: list[CollectedResult] = []

    async with DialecticalEngine(model_config) as engine:
        for problem in config.problems or TRAINING_PROBLEMS:
            for run_idx in range(config.runs_per_problem):
                result = await collect_single(
                    problem=problem,
                    engine=engine,
                    run_index=run_idx,
                    max_iterations=config.max_iterations,
                    max_depth=config.max_depth,
                    include_metrics=config.include_metrics,
                )
                results.append(result)
                logger.info(
                    "[%s] run=%d confidence=%s",
                    problem.get("name", "?"),
                    run_idx,
                    result.engine_result.get("confidence", "?"),
                )

    if config.output_path:
        save_results(results, config.output_path)

    return results


async def collect_single(
    problem: dict[str, str],
    engine: DialecticalEngine,
    run_index: int = 0,
    max_iterations: int = 1,
    max_depth: int = 3,
    include_metrics: bool = True,
) -> CollectedResult:
    """Run RDE on a single problem and wrap as CollectedResult."""
    budget = RecursionBudget(max_depth=max_depth)

    engine_result = await engine.run(
        prompt=problem["prompt"],
        use_orchestrator=True,
        max_iterations=max_iterations,
        budget=budget,
    )

    report_dict = None
    if include_metrics and engine_result.normalized_traces and len(engine_result.normalized_traces) >= 2:
        report = await compute_all(engine_result.normalized_traces, include_embeddings=False)
        report_dict = report.model_dump()

    return CollectedResult(
        problem_name=problem.get("name", "unknown"),
        problem_prompt=problem["prompt"],
        run_index=run_index,
        engine_result=engine_result.model_dump(),
        independence_report=report_dict,
        collection_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def save_results(results: list[CollectedResult], path: str | Path) -> None:
    """Append CollectedResults as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")
    logger.info("Saved %d results to %s", len(results), path)


def load_results(path: str | Path) -> list[CollectedResult]:
    """Load CollectedResults from a JSONL file."""
    path = Path(path)
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(CollectedResult.model_validate_json(line))
    return results
