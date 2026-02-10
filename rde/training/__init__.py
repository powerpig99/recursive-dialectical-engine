"""Phase 6: Post-Training Exploration — distilling multi-model dialectical reasoning."""

from .collector import TRAINING_PROBLEMS, collect, collect_single, load_results, save_results
from .evaluator import compute_deltas, evaluate, format_evaluation_table
from .formatter import (
    format_all,
    format_full_dialectic,
    format_resolution,
    format_role_specific,
    save_formatted,
    to_anthropic_format,
    to_gemini_format,
    to_generic_jsonl,
)
from .models import (
    CollectedResult,
    CollectionConfig,
    DistillationStrategy,
    EvaluationConfig,
    EvaluationResult,
    FormattingConfig,
    ProviderFormat,
    TrainingExample,
)

__all__ = [
    "TRAINING_PROBLEMS",
    "CollectedResult",
    "CollectionConfig",
    "DistillationStrategy",
    "EvaluationConfig",
    "EvaluationResult",
    "FormattingConfig",
    "ProviderFormat",
    "TrainingExample",
    "collect",
    "collect_single",
    "compute_deltas",
    "evaluate",
    "format_all",
    "format_evaluation_table",
    "format_full_dialectic",
    "format_resolution",
    "format_role_specific",
    "load_results",
    "save_formatted",
    "save_results",
    "to_anthropic_format",
    "to_gemini_format",
    "to_generic_jsonl",
]
