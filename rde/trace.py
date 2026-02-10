"""Trace executor — runs a single dialectical trace via the model router."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import TYPE_CHECKING

from .models import EngineResult, RecursionBudget, TraceConfig, TraceResult
from .prompts.repl import REPL_INITIAL_MESSAGE, REPL_ROLE_ADDENDUM, REPL_SYSTEM_PROMPT
from .prompts.traces import build_trace_messages
from .utils.extraction import extract_boxed

if TYPE_CHECKING:
    from .environment import ContextEnvironment
    from .providers.router import ModelRouter
    from .sandbox.base import ExecutionResult
    from .sandbox.repl_sandbox import REPLSandbox

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

        Handles context strategy routing, error wrapping, answer extraction,
        and fallback to an alternate model on failure.
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
            # Attempt fallback to a different model
            fallback_model = self.router.get_fallback_model(model)
            if fallback_model is not None:
                logger.warning(
                    "Trace %s failed with %s, falling back to %s: %s",
                    trace_id, model, fallback_model, e,
                )
                try:
                    user_prompt = self._build_user_prompt(config)
                    messages = build_trace_messages(config.system_prompt, user_prompt)
                    response = await self.router.complete(
                        messages=messages,
                        model=fallback_model,
                        temperature=config.temperature,
                        max_tokens=4096,
                    )
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
                        fallback_used=True,
                        original_model=model,
                    )
                except Exception as fallback_err:
                    logger.error(
                        "Fallback also failed for trace %s: %s",
                        trace_id, fallback_err,
                    )
                    return TraceResult(
                        trace_id=trace_id,
                        role=config.role,
                        model_used=model,
                        raw_output="",
                        error=f"Primary: {e}; Fallback ({fallback_model}): {fallback_err}",
                    )

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

    async def execute_repl(
        self,
        config: TraceConfig,
        model: str,
        repl_sandbox: REPLSandbox,
    ) -> TraceResult:
        """Run a trace in REPL mode: iterative code generation and execution.

        The LLM generates Python code. The code is executed in a REPLSandbox.
        Stdout/stderr is returned to the LLM as the next conversation turn.
        The loop continues until the LLM outputs FINAL(answer) or max_iterations.
        """
        trace_id = f"{config.role.lower()}_repl_{uuid.uuid4().hex[:8]}"
        max_iterations = config.max_repl_iterations
        total_latency_ms = 0.0
        total_usage: dict = {}

        system_prompt = self._build_repl_system_prompt(config)
        # Extract the query portion from the prompt (after "---\nQUERY:")
        prompt_text = self.env.prompt_var
        query_marker = "---\nQUERY:"
        if query_marker in prompt_text:
            query_text = prompt_text.split(query_marker, 1)[1].strip()
        else:
            query_text = prompt_text[-500:]  # Fallback: use last 500 chars
        initial_msg = REPL_INITIAL_MESSAGE.format(
            context_len=len(prompt_text),
            query=query_text,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_msg},
        ]

        last_output = ""
        final_answer = None
        start_time = time.perf_counter()

        try:
            for iteration in range(max_iterations):
                # Get LLM response (code generation)
                response = await self.router.complete(
                    messages=messages,
                    model=model,
                    temperature=config.temperature,
                    max_tokens=4096,
                )
                if self.budget is not None:
                    self.budget.record_call()

                total_latency_ms += response.latency_ms
                assistant_content = response.content
                last_output = assistant_content
                messages.append({"role": "assistant", "content": assistant_content})

                # Check for FINAL() answer
                final_answer = self._extract_final(assistant_content)
                if final_answer is not None:
                    break

                # Extract code block from response
                code = self._extract_code_block(assistant_content)
                if code is None:
                    messages.append({
                        "role": "user",
                        "content": "Please provide Python code in a ```python fenced block to continue your analysis.",
                    })
                    continue

                # Execute code in sandbox
                result = await repl_sandbox.execute(code)

                # Feed execution result back to LLM
                exec_output = self._format_execution_result(result)
                messages.append({"role": "user", "content": exec_output})

            total_elapsed_ms = (time.perf_counter() - start_time) * 1000

            return TraceResult(
                trace_id=trace_id,
                role=config.role,
                model_used=model,
                raw_output=last_output,
                extracted_answer=final_answer or extract_boxed(last_output),
                latency_ms=total_elapsed_ms,
                token_usage=total_usage,
            )

        except Exception as e:
            return TraceResult(
                trace_id=trace_id,
                role=config.role,
                model_used=model,
                raw_output=last_output,
                error=str(e),
            )

    def _build_repl_system_prompt(self, config: TraceConfig) -> str:
        """Build the system prompt for REPL-mode traces."""
        base = REPL_SYSTEM_PROMPT.format(context_len=len(self.env.prompt_var))
        addendum = REPL_ROLE_ADDENDUM.format(
            role=config.role,
            perspective=config.perspective,
            system_prompt=config.system_prompt,
        )
        return base + addendum

    @staticmethod
    def _extract_final(text: str) -> str | None:
        """Extract answer from FINAL(answer) marker in LLM output."""
        match = re.search(r"FINAL\((.+?)\)\s*$", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Also check for FINAL on its own line with content after
        match = re.search(r"^FINAL\((.+)\)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Extract Python code from markdown fenced code blocks."""
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: any code block
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _format_execution_result(result: ExecutionResult) -> str:
        """Format sandbox execution result as a user message for the LLM."""
        parts = []
        if result.stdout:
            parts.append(f"[stdout]\n{result.stdout}")
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        if result.timed_out:
            parts.append("[TIMED OUT]")
        if not parts:
            parts.append("[No output]")
        return "\n".join(parts)

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
