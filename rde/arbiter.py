"""Arbiter — resolves trace disagreements via causal necessity."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .models import (
    ArbitrationResult,
    ConfidenceLevel,
    ConvergenceResult,
    NormalizedTrace,
    RecursionBudget,
)
from .prompts.arbiter import build_arbiter_messages
from .utils.extraction import extract_json_block

if TYPE_CHECKING:
    from .environment import ContextEnvironment
    from .providers.router import ModelRouter

logger = logging.getLogger(__name__)


class Arbiter:
    """Resolves disagreements between traces through causal necessity.

    Uses an LLM call for arbitration. Falls back to selecting the trace
    with the longest reasoning chain if the LLM call fails.

    When confidence is "unresolved" and budget remains, spawns
    sub-arbitration on interference_detected dimensions (Phase 3).
    """

    def __init__(self, router: ModelRouter, model: str, temperature: float = 0.1) -> None:
        self.router = router
        self.model = model
        self.temperature = temperature

    async def arbitrate(
        self,
        normalized_traces: list[NormalizedTrace],
        env: ContextEnvironment,
        budget: RecursionBudget | None = None,
    ) -> ArbitrationResult:
        """Resolve trace disagreements via LLM-based causal arbitration."""
        trace_summaries = [
            {
                "role": nt.role,
                "model": nt.model_family,
                "conclusion": nt.conclusion,
                "reasoning": "\n".join(nt.reasoning_chain) if nt.reasoning_chain else nt.raw_output,
            }
            for nt in normalized_traces
        ]

        try:
            messages = build_arbiter_messages(trace_summaries, env.prompt_var)
            response = await self.router.complete(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=4096,
            )

            json_str = extract_json_block(response.content)
            if not json_str:
                logger.warning("Arbiter produced no JSON; using fallback")
                return self._fallback_arbitration(normalized_traces)

            raw = json.loads(json_str)
            result = ArbitrationResult(
                resolution=raw.get("resolution", ""),
                causal_chain=raw.get("causal_chain", ""),
                confidence=ConfidenceLevel(raw.get("confidence", "contingent")),
                interference_detected=raw.get("interference_detected", []),
                traces_adopted=raw.get("traces_adopted", []),
                traces_rejected=raw.get("traces_rejected", []),
                shadows=raw.get("shadows", []),
                raw_output=response.content,
            )
            logger.info(
                "Arbiter resolved with confidence=%s: %s",
                result.confidence.value,
                result.resolution[:100],
            )

            # Phase 3: Recursive sub-arbitration on unresolved dimensions
            if (
                result.confidence == ConfidenceLevel.UNRESOLVED
                and result.interference_detected
                and budget
                and budget.can_recurse()
            ):
                result = await self._sub_arbitrate(result, env, budget)

            return result

        except Exception as e:
            logger.warning("Arbiter failed (%s); using fallback", e)
            return self._fallback_arbitration(normalized_traces)

    def _fallback_arbitration(self, traces: list[NormalizedTrace]) -> ArbitrationResult:
        """Fallback: select trace with longest reasoning chain."""
        best = max(traces, key=lambda t: len(t.reasoning_chain))
        return ArbitrationResult(
            resolution=best.conclusion,
            causal_chain="Fallback: selected trace with most detailed reasoning",
            confidence=ConfidenceLevel.CONTINGENT,
            traces_adopted=[best.role],
            traces_rejected=[t.role for t in traces if t.trace_id != best.trace_id],
            shadows=["Arbiter LLM call failed; resolution based on reasoning chain length only"],
        )

    async def _sub_arbitrate(
        self,
        result: ArbitrationResult,
        env: ContextEnvironment,
        budget: RecursionBudget,
    ) -> ArbitrationResult:
        """Spawn sub-arbitration on unresolved interference dimensions.

        Each dimension gets a mini-RDE that focuses on that specific axis.
        Results are composed back into the original arbitration.
        """
        from .trace import TraceExecutor

        logger.info(
            "Spawning sub-arbitration on %d interference dimensions",
            len(result.interference_detected),
        )

        sub_resolutions = []
        for dimension in result.interference_detected:
            if not budget.can_recurse():
                sub_resolutions.append(
                    f"{dimension}: [BUDGET EXHAUSTED: {budget.exhausted_reason}]"
                )
                continue

            sub_prompt = (
                f"Focus specifically on this dimension of the problem:\n\n"
                f"ORIGINAL PROBLEM:\n{env.prompt_var}\n\n"
                f"DIMENSION TO RESOLVE: {dimension}\n\n"
                f"The main arbitration could not resolve this dimension. "
                f"Analyze it independently and provide your conclusion."
            )

            executor = TraceExecutor(self.router, env, budget=budget)
            sub_result = await executor.spawn_sub_dialectic(sub_prompt, budget)
            sub_resolutions.append(f"{dimension}: {sub_result.resolution}")

            budget.record_call()

        # Compose sub-resolutions into the main result
        composed_resolution = result.resolution
        if sub_resolutions:
            composed_resolution += "\n\nSub-arbitration results:\n" + "\n".join(
                f"  - {sr}" for sr in sub_resolutions
            )

        return ArbitrationResult(
            resolution=composed_resolution,
            causal_chain=result.causal_chain,
            confidence=ConfidenceLevel.CONTINGENT,  # Upgraded from unresolved
            interference_detected=[],  # Addressed via sub-arbitration
            traces_adopted=result.traces_adopted,
            traces_rejected=result.traces_rejected,
            shadows=result.shadows + [
                f"Sub-arbitration on {len(sub_resolutions)} dimensions "
                f"(depth {budget.current_depth})"
            ],
            raw_output=result.raw_output,
        )

    def check_convergence(
        self,
        arbitration: ArbitrationResult,
        iteration: int,
        max_iterations: int,
        prior_shadows: list[list[str]] | None = None,
    ) -> ConvergenceResult:
        """Check if the engine should stop iterating.

        Stops when:
        - Budget exhausted (iteration >= max_iterations)
        - Confidence is "necessary" (causally resolved)
        - Shadows are irreducible (same shadows repeating)
        - No shadows to explore (clean resolution)

        Continues when:
        - Confidence is "contingent" or "unresolved" AND shadows exist
          AND budget remains AND shadows are novel
        """
        # Budget exhausted
        if iteration >= max_iterations:
            shadow_report = self._build_shadow_report(arbitration, prior_shadows)
            return ConvergenceResult(
                should_stop=True,
                reason=f"Budget exhausted after {iteration} iterations",
                shadow_report_for_human=shadow_report,
            )

        # Necessary confidence — causally resolved
        if arbitration.confidence == ConfidenceLevel.NECESSARY:
            return ConvergenceResult(
                should_stop=True,
                reason="Resolution achieved with necessary confidence",
            )

        # No shadows to explore
        if not arbitration.shadows:
            return ConvergenceResult(
                should_stop=True,
                reason="No shadows reported — resolution is clean",
            )

        # Check for shadow repetition (diminishing returns)
        if prior_shadows and self._shadows_are_repeating(arbitration.shadows, prior_shadows):
            shadow_report = self._build_shadow_report(arbitration, prior_shadows)
            return ConvergenceResult(
                should_stop=True,
                reason="Shadows are repeating — diminishing returns from further iteration",
                shadow_report_for_human=shadow_report,
            )

        # Continue: there are novel shadows to explore
        return ConvergenceResult(
            should_stop=False,
            reason="Novel shadows available for exploration",
            next_framings=arbitration.shadows,
        )

    def _shadows_are_repeating(
        self,
        current_shadows: list[str],
        prior_shadow_sets: list[list[str]],
    ) -> bool:
        """Check if current shadows substantially overlap with any prior set."""
        current_normalized = {s.lower().strip() for s in current_shadows}
        for prior_set in prior_shadow_sets:
            prior_normalized = {s.lower().strip() for s in prior_set}
            if not current_normalized or not prior_normalized:
                continue
            overlap = len(current_normalized & prior_normalized)
            # If >50% of current shadows appeared before, they're repeating
            if overlap / len(current_normalized) > 0.5:
                return True
        return False

    def _build_shadow_report(
        self,
        arbitration: ArbitrationResult,
        prior_shadows: list[list[str]] | None,
    ) -> str:
        """Build a human-readable shadow report across all iterations."""
        lines = ["Shadow Report (what this resolution necessarily obscures):"]
        if prior_shadows:
            for i, shadows in enumerate(prior_shadows):
                lines.append(f"\n  Iteration {i + 1} shadows:")
                for s in shadows:
                    lines.append(f"    - {s}")
        lines.append("\n  Final iteration shadows:")
        for s in arbitration.shadows:
            lines.append(f"    - {s}")
        return "\n".join(lines)
