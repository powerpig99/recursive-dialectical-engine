"""Training data models for Phase 6: Post-Training Exploration."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DistillationStrategy(str, Enum):
    """Which aspect of RDE reasoning to distill."""

    RESOLUTION = "resolution"
    FULL_DIALECTIC = "full_dialectic"
    ROLE_SPECIFIC = "role_specific"


class ProviderFormat(str, Enum):
    """Target fine-tuning provider format."""

    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    JSONL_GENERIC = "jsonl_generic"


class TrainingExample(BaseModel):
    """A single input/output pair for fine-tuning."""

    input_text: str
    output_text: str
    strategy: DistillationStrategy
    source_problem: str = ""
    source_run_index: int = 0
    metadata: dict = Field(default_factory=dict)


class CollectionConfig(BaseModel):
    """Configuration for the training data collection run."""

    problems: list[dict[str, str]] = Field(default_factory=list)
    output_path: str = "training_data/engine_results.jsonl"
    use_orchestrator: bool = True
    max_iterations: int = 1
    max_depth: int = 3
    include_metrics: bool = True
    runs_per_problem: int = 1


class CollectedResult(BaseModel):
    """An EngineResult with collection metadata."""

    problem_name: str
    problem_prompt: str
    run_index: int = 0
    engine_result: dict = Field(default_factory=dict)
    independence_report: Optional[dict] = None
    collection_timestamp: str = ""


class FormattingConfig(BaseModel):
    """Configuration for training data formatting."""

    input_path: str = "training_data/engine_results.jsonl"
    output_path: str = "training_data/formatted/"
    strategies: list[DistillationStrategy] = Field(
        default_factory=lambda: [DistillationStrategy.RESOLUTION]
    )
    provider_format: ProviderFormat = ProviderFormat.GEMINI
    min_trace_count: int = 2
    require_arbitration: bool = False


class EvaluationConfig(BaseModel):
    """Configuration for evaluating a fine-tuned model."""

    finetuned_model: str
    baseline_config: dict = Field(default_factory=dict)
    problems: list[dict[str, str]] = Field(default_factory=list)
    strategies_to_evaluate: list[DistillationStrategy] = Field(
        default_factory=lambda: [DistillationStrategy.RESOLUTION]
    )


class EvaluationResult(BaseModel):
    """Comparison result between fine-tuned and multi-model runs."""

    strategy: DistillationStrategy
    problem_name: str
    finetuned_resolution: str = ""
    finetuned_confidence: str = ""
    finetuned_avg_divergence: float = 0.0
    finetuned_avg_jaccard: float = 0.0
    finetuned_afdr_count: int = 0
    baseline_resolution: str = ""
    baseline_confidence: str = ""
    baseline_model_diversity: float = 0.0
    baseline_avg_divergence: float = 0.0
    baseline_avg_jaccard: float = 0.0
    baseline_afdr_count: int = 0
    divergence_delta: float = 0.0
    jaccard_delta: float = 0.0
