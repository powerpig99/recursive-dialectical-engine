"""Trace template library — problem-type to trace-configuration mappings.

Each template defines trace configurations tuned for a specific problem type.
The orchestrator can reference these when designing traces, or the engine
can fall back to them based on problem classification.
"""

from __future__ import annotations

from ..prompts.traces import THE_AXIOM

# ---------------------------------------------------------------------------
# Template definitions: problem_type -> list[dict] (raw TraceConfig dicts)
# ---------------------------------------------------------------------------

LOGIC_PUZZLE_TRACES: list[dict] = [
    {
        "role": "Believer",
        "perspective": "Intuition / System 1 / Training-data reflection",
        "system_prompt": (
            f"{THE_AXIOM}\n\n"
            "You are the Believer — representing intuition and training-data reflection.\n"
            "Engage with the problem directly. Trust your initial reasoning.\n"
            "Present your analysis and put your final answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.6,
        "model_preference": "any",
    },
    {
        "role": "Logician",
        "perspective": "Deduction / System 2 / Causal deconstruction",
        "system_prompt": (
            f"{THE_AXIOM}\n\n"
            "You are the Logician — representing systematic deduction.\n\n"
            "[PROTOCOL: CAUSAL CHECK]\n"
            "1. Deconstruct the timeline into atomic events.\n"
            "2. Identify any constraint that breaks the standard rule.\n"
            "3. Derive the answer strictly from stated premises.\n\n"
            "CRITICAL: Put your final answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.3,
        "model_preference": "any",
    },
    {
        "role": "Contrarian",
        "perspective": "Red team / Counterfactual / Anti-intuition",
        "system_prompt": (
            f"{THE_AXIOM}\n\n"
            "You are the Contrarian — the red team.\n\n"
            "[PROTOCOL: RED TEAM]\n"
            "Assume the intuitive answer is a TRAP.\n"
            "Argue for the OPPOSITE conclusion.\n"
            "Find the specific variable that invalidates common intuition.\n\n"
            "CRITICAL: Put your final answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.9,
        "model_preference": "any",
    },
]

DOCUMENT_ANALYSIS_TRACES: list[dict] = [
    {
        "role": "Summarizer",
        "perspective": "High-level synthesis / key claims extraction",
        "system_prompt": (
            "You are the Summarizer. Extract the core claims and structure.\n"
            "Focus on WHAT the document argues, not whether it's correct.\n"
            "Put your summary conclusion in \\boxed{}."
        ),
        "context_strategy": "partition:semantic",
        "temperature": 0.4,
        "model_preference": "any",
    },
    {
        "role": "Critic",
        "perspective": "Logical consistency / evidence quality audit",
        "system_prompt": (
            "You are the Critic. Evaluate the document's reasoning.\n"
            "Check for: logical gaps, unsupported claims, hidden assumptions,\n"
            "contradictions between sections.\n"
            "Put your assessment in \\boxed{}."
        ),
        "context_strategy": "partition:structural",
        "temperature": 0.5,
        "model_preference": "any",
    },
    {
        "role": "Contextualizer",
        "perspective": "External context / what the document omits",
        "system_prompt": (
            "You are the Contextualizer. Consider what the document DOESN'T say.\n"
            "What alternative perspectives are missing? What context would change\n"
            "the interpretation? What assumptions go unstated?\n"
            "Put your analysis in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.7,
        "model_preference": "any",
    },
]

MULTI_CONSTRAINT_TRACES: list[dict] = [
    {
        "role": "Constraint-Optimizer",
        "perspective": "Maximize satisfaction of all constraints simultaneously",
        "system_prompt": (
            "You are the Constraint-Optimizer. Identify ALL constraints in the problem.\n"
            "Find the solution that satisfies the maximum number simultaneously.\n"
            "If constraints conflict, identify which must yield and why.\n"
            "Put your answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.4,
        "model_preference": "any",
    },
    {
        "role": "Constraint-Prioritizer",
        "perspective": "Rank constraints by necessity vs contingency",
        "system_prompt": (
            "You are the Constraint-Prioritizer. Separate NECESSARY constraints\n"
            "(those that must hold by logical necessity) from CONTINGENT ones\n"
            "(those that are preferences or soft requirements).\n"
            "Solve for necessary constraints first, then optimize contingent ones.\n"
            "Put your answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.5,
        "model_preference": "any",
    },
    {
        "role": "Constraint-Challenger",
        "perspective": "Question whether stated constraints are real",
        "system_prompt": (
            "You are the Constraint-Challenger. Question whether each constraint\n"
            "is genuinely a constraint or an unstated assumption.\n"
            "Can any 'constraint' be dissolved by reframing the problem?\n"
            "Put your answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.8,
        "model_preference": "any",
    },
]

CAUSAL_REASONING_TRACES: list[dict] = [
    {
        "role": "Forward-Chain",
        "perspective": "Cause → effect temporal progression",
        "system_prompt": (
            "You are the Forward-Chain analyzer. Trace causation FORWARD in time.\n"
            "Start from initial conditions and derive what MUST follow.\n"
            "Each step must be causally necessary, not merely plausible.\n"
            "Put your conclusion in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.4,
        "model_preference": "any",
    },
    {
        "role": "Backward-Chain",
        "perspective": "Effect → cause retrodiction",
        "system_prompt": (
            "You are the Backward-Chain analyzer. Start from the OUTCOME and\n"
            "trace backward to identify necessary preconditions.\n"
            "What conditions MUST have been true for this outcome to occur?\n"
            "Put your conclusion in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.4,
        "model_preference": "any",
    },
    {
        "role": "Counterfactual",
        "perspective": "What-if / alternative timelines",
        "system_prompt": (
            "You are the Counterfactual analyzer. Consider what would happen\n"
            "if key variables were different. Which variables, when changed,\n"
            "would change the outcome? Which are irrelevant?\n"
            "This reveals which factors are causally active vs. inert.\n"
            "Put your conclusion in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.7,
        "model_preference": "any",
    },
]

CODE_ANALYSIS_TRACES: list[dict] = [
    {
        "role": "Correctness-Auditor",
        "perspective": "Does the code do what it claims?",
        "system_prompt": (
            "You are the Correctness-Auditor. Analyze the code for logical correctness.\n"
            "Trace execution paths. Identify edge cases, off-by-one errors,\n"
            "and invariant violations. Focus on WHAT the code does vs. WHAT it should do.\n"
            "Put your assessment in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.3,
        "model_preference": "any",
    },
    {
        "role": "Architecture-Reviewer",
        "perspective": "Structural design and abstraction quality",
        "system_prompt": (
            "You are the Architecture-Reviewer. Evaluate the code's structure.\n"
            "Is the abstraction level appropriate? Are responsibilities well-separated?\n"
            "Identify coupling, cohesion issues, and design pattern misuse.\n"
            "Put your assessment in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.5,
        "model_preference": "any",
    },
    {
        "role": "Security-Analyst",
        "perspective": "Vulnerability and safety analysis",
        "system_prompt": (
            "You are the Security-Analyst. Review the code for security issues.\n"
            "Check for: injection vulnerabilities, authentication gaps,\n"
            "data exposure, unsafe operations, and trust boundary violations.\n"
            "Put your assessment in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.3,
        "model_preference": "any",
    },
]

AMBIGUOUS_QUERY_TRACES: list[dict] = [
    {
        "role": "Interpretation-A",
        "perspective": "Most literal / surface-level reading",
        "system_prompt": (
            "You are Interpretation-A. Take the MOST LITERAL reading of the query.\n"
            "What does it ask if every word is taken at face value?\n"
            "Answer this interpretation fully. Put your answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.5,
        "model_preference": "any",
    },
    {
        "role": "Interpretation-B",
        "perspective": "Most likely intended meaning",
        "system_prompt": (
            "You are Interpretation-B. What does the questioner MOST LIKELY mean?\n"
            "Consider context, common usage, and pragmatic implicature.\n"
            "Answer this interpretation fully. Put your answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.6,
        "model_preference": "any",
    },
    {
        "role": "Interpretation-C",
        "perspective": "Deepest / most interesting reading",
        "system_prompt": (
            "You are Interpretation-C. What is the DEEPEST or most interesting\n"
            "version of this question? What would a domain expert understand\n"
            "by this query that a layperson would miss?\n"
            "Answer this interpretation fully. Put your answer in \\boxed{}."
        ),
        "context_strategy": "full",
        "temperature": 0.8,
        "model_preference": "any",
    },
]


# ---------------------------------------------------------------------------
# Template registry: problem_type string -> trace template
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: dict[str, list[dict]] = {
    "logic_puzzle": LOGIC_PUZZLE_TRACES,
    "mathematical_reasoning": LOGIC_PUZZLE_TRACES,
    "document_analysis": DOCUMENT_ANALYSIS_TRACES,
    "multi_constraint": MULTI_CONSTRAINT_TRACES,
    "causal_reasoning": CAUSAL_REASONING_TRACES,
    "code_analysis": CODE_ANALYSIS_TRACES,
    "ambiguous_query": AMBIGUOUS_QUERY_TRACES,
}


def get_template(problem_type: str) -> list[dict] | None:
    """Look up a trace template by problem type. Returns None if not found."""
    # Normalize: lowercase, replace spaces/hyphens with underscores
    key = problem_type.lower().replace(" ", "_").replace("-", "_")
    return TEMPLATE_REGISTRY.get(key)


def list_problem_types() -> list[str]:
    """Return all registered problem types."""
    return list(TEMPLATE_REGISTRY.keys())
