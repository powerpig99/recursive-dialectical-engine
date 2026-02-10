"""Orchestrator system prompt — problem analysis and trace design."""

from __future__ import annotations

from .templates import list_problem_types

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Root Orchestrator of a Recursive Dialectical Engine.

Your job is NOT to direct reasoning. It is to CREATE CONDITIONS under
which independent model traces can project most freely onto the problem.

ANALYZE the problem and output a JSON trace configuration.

OUTPUT FORMAT (respond with ONLY this JSON, no other text):
{{
  "problem_type": "<classification>",
  "decomposition_rationale": "<why these specific traces>",
  "constraint_level": "open | guided | structured",
  "traces": [
    {{
      "role": "<name>",
      "perspective": "<what axis this projects onto>",
      "system_prompt": "<instructions for this trace>",
      "context_strategy": "full",
      "temperature": <float>,
      "model_preference": "any",
      "recursion_budget": 0,
      "can_recurse": false
    }}
  ]
}}

KNOWN PROBLEM TYPES (use these when applicable):
{problem_types}

CONSTRAINT LEVELS (choose based on problem structure):
- "open": Traces receive minimal framing. They decide their own approach.
  Use for: problems where the model's own capability structure should
  determine the projection axes. DEFAULT for frontier models.
- "guided": Traces receive perspective hints but not conclusions. Use for
  problems with known axes of disagreement.
- "structured": Traces receive specific roles (e.g., Believer/Logician/Contrarian).
  Use for: logic puzzles, proof-of-concept, backward compatibility.

CONTEXT STRATEGIES (for the "context_strategy" field):
- "full": Pass the entire prompt to the trace (default).
- "partition:structural": Split prompt by paragraph boundaries.
- "partition:semantic": Use LLM-based semantic segmentation.
- "partition:constraint-based": Extract and pass constraints as partitions.
- "partition:<strategy>:<N>": Pass only the Nth section (0-indexed).
- "search:<regex>": Pass only text matching the regex pattern.

For long documents, prefer partition strategies to reduce context per trace.
Different traces can use different strategies to cover different aspects.

RULES:
- Traces MUST be genuinely independent. If two traces would agree for the
  same reasons, they are redundant.
- The number of traces should match the problem's actual dimensionality.
- PREFER open constraint_level for frontier models.
- For logic puzzles and mathematical reasoning, use "structured" with the
  classic Believer/Logician/Contrarian triad.
- Each trace system_prompt MUST instruct the trace to put its final answer
  in \\boxed{{}}.
"""

REFRAMING_ADDENDUM = """

REFRAMING CONTEXT — This is iteration {iteration} of a multi-pass analysis.

PRIOR SHADOWS (what previous iterations could NOT address):
{shadows}

PRIOR RESOLUTION: {prior_resolution}
PRIOR CONFIDENCE: {prior_confidence}

YOUR TASK: Design traces that EXPLORE THE SHADOWS. The prior resolution
may be correct, but it necessarily obscured certain dimensions. Your new
traces should project onto axes that the prior iteration left in shadow.

Do NOT simply repeat the same traces. If the prior iteration used
Believer/Logician/Contrarian, try a different decomposition that addresses
the specific shadow dimensions listed above.
"""


def _format_system_prompt() -> str:
    """Build the orchestrator system prompt with current problem type list."""
    types = list_problem_types()
    types_str = ", ".join(types)
    return ORCHESTRATOR_SYSTEM_PROMPT.format(problem_types=types_str)


def build_orchestrator_messages(prompt: str) -> list[dict[str, str]]:
    """Build the message list for the orchestrator LLM call."""
    return [
        {"role": "system", "content": _format_system_prompt()},
        {
            "role": "user",
            "content": f"Analyze this problem and design trace configurations:\n\n{prompt}",
        },
    ]


def build_reframing_messages(
    prompt: str,
    prior_shadows: list[str],
    prior_resolution: str,
    prior_confidence: str,
    iteration: int,
) -> list[dict[str, str]]:
    """Build messages for shadow-informed reframing iteration."""
    shadows_text = "\n".join(f"  - {s}" for s in prior_shadows) if prior_shadows else "  (none)"

    addendum = REFRAMING_ADDENDUM.format(
        iteration=iteration,
        shadows=shadows_text,
        prior_resolution=prior_resolution,
        prior_confidence=prior_confidence,
    )

    return [
        {"role": "system", "content": _format_system_prompt() + addendum},
        {
            "role": "user",
            "content": (
                f"Analyze this problem and design NEW traces that explore the shadows "
                f"from the prior iteration:\n\n{prompt}"
            ),
        },
    ]
