"""Core data models for the Recursive Dialectical Engine."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConstraintLevel(str, Enum):
    """How much structure the orchestrator imposes on traces."""

    OPEN = "open"
    GUIDED = "guided"
    STRUCTURED = "structured"


class ConfidenceLevel(str, Enum):
    """Structural confidence in the resolution."""

    NECESSARY = "necessary"
    CONTINGENT = "contingent"
    UNRESOLVED = "unresolved"


class TraceConfig(BaseModel):
    """Configuration for a single trace, as output by the Orchestrator."""

    role: str
    perspective: str
    system_prompt: str
    context_strategy: str = "full"  # "full" | "partition:<spec>" | "search:<pattern>"
    temperature: float = 0.7
    model_preference: str = "any"  # "any" or specific model/family
    recursion_budget: int = 0
    can_recurse: bool = False


class OrchestratorOutput(BaseModel):
    """JSON output from the orchestrator LLM call."""

    problem_type: str
    decomposition_rationale: str
    constraint_level: ConstraintLevel
    traces: list[TraceConfig]


class TraceResult(BaseModel):
    """Raw output from a single trace execution."""

    trace_id: str
    role: str
    model_used: str
    raw_output: str
    extracted_answer: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    token_usage: dict = Field(default_factory=dict)


class NormalizedTrace(BaseModel):
    """A trace output normalized for cross-model comparison."""

    trace_id: str
    role: str
    model_family: str
    conclusion: str
    reasoning_chain: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_cited: list[str] = Field(default_factory=list)
    raw_output: str = ""


class ArbitrationResult(BaseModel):
    """Output from the Recursive Arbiter."""

    resolution: str
    causal_chain: str
    confidence: ConfidenceLevel
    interference_detected: list[str] = Field(default_factory=list)
    traces_adopted: list[str] = Field(default_factory=list)
    traces_rejected: list[str] = Field(default_factory=list)
    shadows: list[str] = Field(default_factory=list)
    raw_output: str = ""


class ConvergenceResult(BaseModel):
    """Output from the convergence/stopping check."""

    should_stop: bool
    reason: str
    next_framings: list[str] = Field(default_factory=list)
    shadow_report_for_human: str = ""


class RecursionBudget(BaseModel):
    """Tracks recursion budget across the call tree."""

    max_depth: int = 3
    max_total_calls: int = 20
    max_cost_usd: float = 10.0
    current_depth: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0

    def can_recurse(self) -> bool:
        """Check if another recursive call is allowed."""
        return (
            self.current_depth < self.max_depth
            and self.total_calls < self.max_total_calls
            and self.total_cost_usd < self.max_cost_usd
        )

    def child_budget(self) -> RecursionBudget:
        """Create a child budget at depth + 1, sharing the same counters."""
        return RecursionBudget(
            max_depth=self.max_depth,
            max_total_calls=self.max_total_calls,
            max_cost_usd=self.max_cost_usd,
            current_depth=self.current_depth + 1,
            total_calls=self.total_calls,
            total_cost_usd=self.total_cost_usd,
        )

    def record_call(self, cost_usd: float = 0.0) -> None:
        """Record a call against the budget."""
        self.total_calls += 1
        self.total_cost_usd += cost_usd

    @property
    def exhausted_reason(self) -> str:
        """Human-readable reason for budget exhaustion."""
        if self.current_depth >= self.max_depth:
            return f"Max depth {self.max_depth} reached"
        if self.total_calls >= self.max_total_calls:
            return f"Max calls {self.max_total_calls} reached"
        if self.total_cost_usd >= self.max_cost_usd:
            return f"Max cost ${self.max_cost_usd:.2f} reached"
        return "Budget available"


class CallTreeNode(BaseModel):
    """A node in the recursive call tree for visualization."""

    node_type: str  # "trace" | "arbiter" | "sub_lm" | "sub_dialectic"
    model: str = ""
    role: str = ""
    depth: int = 0
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    children: list[CallTreeNode] = Field(default_factory=list)


class EngineResult(BaseModel):
    """The final output from a complete RDE run."""

    resolution: str
    confidence: ConfidenceLevel
    shadows: list[str] = Field(default_factory=list)
    causal_chain: str = ""
    iterations: int = 1
    trace_results: list[TraceResult] = Field(default_factory=list)
    normalized_traces: list[NormalizedTrace] = Field(default_factory=list)
    arbitration: Optional[ArbitrationResult] = None
    total_latency_ms: float = 0.0
    consensus_reached: bool = False
    call_tree: Optional[CallTreeNode] = None


class ModelConfig(BaseModel):
    """Global model configuration for the engine."""

    # Orchestrator and Arbiter: strongest available
    orchestrator_model: str = "claude-opus-4-6"
    arbiter_model: str = "claude-opus-4-6"

    # Traces: diverse frontier models
    trace_models: list[str] = Field(
        default_factory=lambda: [
            "claude-sonnet-4-5-20250929",
            "gpt-5",
            "gemini-2.5-pro",
        ]
    )

    # Sub-LM: cheap, fast, high-volume
    sub_lm_models: list[str] = Field(
        default_factory=lambda: [
            "claude-haiku-4-5-20251001",
            "gpt-5-mini",
        ]
    )

    # Trace assignment strategy: round_robin | random | orchestrator
    trace_assignment: str = "round_robin"

    # Scaffolding preference: how much structure to impose on traces
    # "auto" = orchestrator decides, or specify "open" / "guided" / "structured"
    scaffolding_preference: str = "auto"

    # Temperatures for each component
    arbiter_temperature: float = 0.1
    orchestrator_temperature: float = 0.3
    sub_lm_temperature: float = 0.3

    # Local model path (for MLX provider)
    local_model_path: str = "~/Models/Qwen3-8B-4bit"
