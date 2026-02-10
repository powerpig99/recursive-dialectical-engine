"""Trace executor — runs a single dialectical trace via the model router."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from .models import EngineResult, RecursionBudget, TraceConfig, TraceResult
from .prompts.traces import build_trace_messages
from .utils.extraction import extract_boxed

if TYPE_CHECKING:
    from .environment import ContextEnvironment
    from .providers.router import ModelRouter

logger = logging.getLogger(__name__)


class TraceExecutor:
    """Executes a single trace against a model via the router."""

    def __init__(
        self,
        router: ModelRouter,
        environment: ContextEnvironment,
        budget: RecursionBudget | None = None,
    ) -> None:
        self.router = router
        self.env = environment
        self.budget = budget

    async def execute(self, config: TraceConfig, model: str) -> TraceResult:
        """Run a trace and return its result.

        Handles context strategy routing, error wrapping, and answer extraction.
        """
        trace_id = f"{config.role.lower()}_{uuid.uuid4().hex[:8]}"

        try:
            user_prompt = self._build_user_prompt(config)
            messages = build_trace_messages(config.system_prompt, user_prompt)

            response = await self.router.complete(
                messages=messages,
                model=model,
                temperature=config.temperature,
                max_tokens=4096,
            )

            # Track this LLM call against the budget
            if self.budget is not None:
                self.budget.record_call()

            extracted = extract_boxed(response.content)

            return TraceResult(
                trace_id=trace_id,
                role=config.role,
                model_used=response.model,
                raw_output=response.content,
                extracted_answer=extracted,
                latency_ms=response.latency_ms,
                token_usage=response.usage,
            )

        except Exception as e:
            return TraceResult(
                trace_id=trace_id,
                role=config.role,
                model_used=model,
                raw_output="",
                error=str(e),
            )

    def _build_user_prompt(self, config: TraceConfig) -> str:
        """Build user prompt based on context strategy.

        Strategies:
          - "full": pass the entire prompt
          - "partition:<strategy>": pass partitioned prompt
          - "partition:<strategy>:<index>": pass only the Nth partition
          - "search:<pattern>": pass regex search results
        """
        strategy = config.context_strategy

        if strategy == "full":
            return self.env.prompt_var

        if strategy.startswith("partition:"):
            rest = strategy.split(":", 1)[1]
            # Check for index: "semantic:0", "structural:2", etc.
            if ":" in rest:
                partition_strategy, idx_str = rest.rsplit(":", 1)
                try:
                    idx = int(idx_str)
                    parts = self.env.partition(partition_strategy)
                    if 0 <= idx < len(parts):
                        return f"[Section {idx + 1}/{len(parts)}]\n{parts[idx]}"
                    return "\n\n".join(parts)  # invalid index → all
                except ValueError:
                    # Not a number — treat the whole rest as strategy name
                    partition_strategy = rest
            else:
                partition_strategy = rest
            parts = self.env.partition(partition_strategy)
            return "\n\n".join(parts)

        if strategy.startswith("search:"):
            pattern = strategy.split(":", 1)[1]
            matches = self.env.search(pattern)
            if matches:
                return "Relevant excerpts:\n" + "\n".join(matches)
            return self.env.prompt_var  # fallback to full if no matches

        return self.env.prompt_var

    async def spawn_sub_dialectic(
        self,
        sub_prompt: str,
        budget: RecursionBudget | None = None,
    ) -> EngineResult:
        """Spawn a mini-RDE on a sub-problem.

        Creates a child engine with:
        - Reduced budget (depth + 1)
        - Independent traces (child cannot see parent)
        """
        effective_budget = budget or self.budget
        if effective_budget is None:
            effective_budget = RecursionBudget()

        if not effective_budget.can_recurse():
            logger.info(
                "Sub-dialectic blocked: %s", effective_budget.exhausted_reason
            )
            from .models import ConfidenceLevel

            return EngineResult(
                resolution=f"[BUDGET EXHAUSTED: {effective_budget.exhausted_reason}]",
                confidence=ConfidenceLevel.UNRESOLVED,
                shadows=[effective_budget.exhausted_reason],
            )

        child_budget = effective_budget.child_budget()

        # Import here to avoid circular imports
        from .engine import DialecticalEngine
        from .models import ModelConfig

        config = ModelConfig(
            orchestrator_model=self.router.config.orchestrator_model,
            arbiter_model=self.router.config.arbiter_model,
            trace_models=list(self.router.config.trace_models),
            sub_lm_models=list(self.router.config.sub_lm_models),
            trace_assignment=self.router.config.trace_assignment,
        )

        engine = DialecticalEngine(config)
        engine.router = self.router  # Share the router (and its providers)

        result = await engine.run(
            prompt=sub_prompt,
            use_orchestrator=False,
            max_iterations=1,
            budget=child_budget,
        )

        # Propagate budget consumption back to parent
        effective_budget.total_calls = child_budget.total_calls
        effective_budget.total_cost_usd = child_budget.total_cost_usd

        logger.info(
            "Sub-dialectic at depth %d resolved: %s (confidence=%s)",
            child_budget.current_depth,
            result.resolution[:80],
            result.confidence.value,
        )
        return result
