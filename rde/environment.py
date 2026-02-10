"""Context Environment — the externalized prompt REPL.

The prompt is NEVER passed directly as LM context. Instead, it's an
external object that traces interact with programmatically.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import CallTreeNode, RecursionBudget, TraceConfig, TraceResult
    from .providers.router import ModelRouter

logger = logging.getLogger(__name__)


class ContextEnvironment:
    """Persistent environment shared across traces and arbitration."""

    def __init__(
        self,
        prompt: str,
        max_iterations: int = 5,
        router: ModelRouter | None = None,
        budget: RecursionBudget | None = None,
        sub_lm_models: list[str] | None = None,
        sub_lm_temperature: float = 0.3,
    ):
        self.prompt_var: str = prompt
        self.workspace: dict = {}
        self.trace_results: dict[str, object] = {}
        self.iteration_history: list[dict] = []
        self.recursion_log: list[dict] = []
        self.current_iteration: int = 0
        self.max_iterations: int = max_iterations

        # Phase 3: recursion wiring
        self._router = router
        self._budget = budget
        self._sub_lm_models = sub_lm_models or []
        self._sub_lm_temperature = sub_lm_temperature
        self._call_tree_children: list[CallTreeNode] = []

        # Phase 4: partition cache
        self._partition_cache: dict[str, list[str]] = {}

    def peek(self, start: int, end: int) -> str:
        """View a character slice of the prompt without loading full context."""
        return self.prompt_var[start:end]

    def search(self, pattern: str) -> list[str]:
        """Regex search over the prompt. Returns all matches."""
        return re.findall(pattern, self.prompt_var)

    def partition(self, strategy: str = "structural") -> list[str]:
        """Decompose prompt into independent partitions.

        Returns cached results if available. For strategies requiring async
        LLM calls (semantic, constraint-based), call prepare_partitions()
        first — otherwise falls back to structural.
        """
        if strategy in self._partition_cache:
            return self._partition_cache[strategy]

        if strategy == "structural":
            parts = [p.strip() for p in re.split(r"\n\s*\n", self.prompt_var) if p.strip()]
            result = parts if parts else [self.prompt_var]
            self._partition_cache[strategy] = result
            return result

        # Unknown or async strategies without cached results → structural fallback
        return self.partition("structural")

    async def prepare_partitions(
        self,
        strategy: str,
        custom_fn: Callable[[str], list[str]] | None = None,
    ) -> list[str]:
        """Pre-compute partitions for strategies that require async work.

        Must be called before partition() for semantic/constraint-based strategies.
        Results are cached — subsequent partition() calls return cached results.
        """
        if strategy in self._partition_cache:
            return self._partition_cache[strategy]

        if strategy == "structural":
            return self.partition("structural")
        elif strategy == "semantic":
            result = await self._semantic_partition()
        elif strategy == "constraint-based":
            result = await self._constraint_partition()
        elif strategy == "custom":
            if custom_fn is None:
                raise ValueError("Custom partition requires a callable")
            result = custom_fn(self.prompt_var)
            if not result:
                result = [self.prompt_var]
        else:
            result = self.partition("structural")

        self._partition_cache[strategy] = result
        return result

    async def _semantic_partition(self) -> list[str]:
        """Use LLM sub-call to identify semantic boundaries in the prompt."""
        if self._router is None:
            logger.warning("No router for semantic partition; falling back to structural")
            return self.partition("structural")

        sub_prompt = (
            "Analyze this text and identify its major semantic sections. "
            "Output a JSON array of strings, where each string is one section "
            "of the original text. Preserve the original text exactly — just "
            "split it at natural semantic boundaries.\n\n"
            "TEXT:\n" + self.prompt_var
        )

        try:
            response = await self.spawn_sub_lm(sub_prompt)
            from .utils.extraction import extract_json_block

            json_str = extract_json_block(response)
            if json_str:
                parts = json.loads(json_str)
                if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                    return parts if parts else [self.prompt_var]
            logger.warning("Semantic partition: bad LLM response; falling back to structural")
            return self.partition("structural")
        except Exception:
            logger.warning("Semantic partition failed; falling back to structural", exc_info=True)
            return self.partition("structural")

    async def _constraint_partition(self) -> list[str]:
        """Use LLM sub-call to extract constraints from the prompt."""
        if self._router is None:
            logger.warning("No router for constraint partition; falling back to structural")
            return self.partition("structural")

        sub_prompt = (
            "Extract all distinct constraints, requirements, or conditions from this text. "
            "Output a JSON array of strings, each being one constraint "
            "stated or implied in the text.\n\n"
            "TEXT:\n" + self.prompt_var
        )

        try:
            response = await self.spawn_sub_lm(sub_prompt)
            from .utils.extraction import extract_json_block

            json_str = extract_json_block(response)
            if json_str:
                constraints = json.loads(json_str)
                if isinstance(constraints, list) and constraints:
                    return [str(c) for c in constraints]
            logger.warning("Constraint partition: bad LLM response; falling back to structural")
            return self.partition("structural")
        except Exception:
            logger.warning("Constraint partition failed; falling back to structural", exc_info=True)
            return self.partition("structural")

    async def spawn_sub_lm(self, sub_prompt: str, model: str | None = None) -> str:
        """Launch a recursive LM call on a sub-problem.

        Uses sub_lm_models (cheap, fast models) via the router.
        Logs each call to recursion_log and tracks budget.
        """
        if self._router is None:
            raise RuntimeError("Environment not wired to a router. Use engine context.")

        if self._budget and not self._budget.can_recurse():
            return f"[BUDGET EXHAUSTED: {self._budget.exhausted_reason}]"

        # Pick model: explicit > first available sub_lm
        if model is None:
            if self._sub_lm_models:
                model = self._sub_lm_models[0]
            else:
                return "[NO SUB-LM MODELS CONFIGURED]"

        start = time.perf_counter()
        response = await self._router.complete(
            messages=[
                {"role": "system", "content": "Answer concisely and precisely."},
                {"role": "user", "content": sub_prompt},
            ],
            model=model,
            temperature=self._sub_lm_temperature,
            max_tokens=2048,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Track budget
        if self._budget:
            self._budget.record_call()

        # Log
        log_entry = {
            "type": "sub_lm",
            "model": model,
            "prompt_preview": sub_prompt[:200],
            "output_preview": response.content[:200],
            "latency_ms": elapsed_ms,
            "depth": self._budget.current_depth if self._budget else 0,
        }
        self.recursion_log.append(log_entry)

        # Call tree node
        from .models import CallTreeNode

        self._call_tree_children.append(
            CallTreeNode(
                node_type="sub_lm",
                model=model,
                depth=self._budget.current_depth if self._budget else 0,
                input_summary=sub_prompt[:100],
                output_summary=response.content[:100],
                latency_ms=elapsed_ms,
            )
        )

        logger.debug("sub_lm call: model=%s latency=%.0fms", model, elapsed_ms)
        return response.content

    async def spawn_trace(self, trace_config: TraceConfig) -> TraceResult:
        """Launch a full dialectical trace that can itself recurse.

        Phase 1: raises NotImplementedError. Wired in Phase 3.
        """
        raise NotImplementedError("spawn_trace requires full recursion (Phase 3)")

    def store_iteration(self, results: dict, arbitration: dict) -> None:
        """Store completed iteration for history."""
        self.iteration_history.append(
            {
                "iteration": self.current_iteration,
                "trace_results": results,
                "arbitration": arbitration,
            }
        )
        self.current_iteration += 1
