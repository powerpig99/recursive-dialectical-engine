"""Orchestrator — designs trace configurations via LLM analysis."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .models import ArbitrationResult, OrchestratorOutput, TraceConfig
from .prompts.orchestrator import build_orchestrator_messages, build_reframing_messages
from .prompts.templates import get_template
from .prompts.traces import DEFAULT_STRUCTURED_TRACES
from .utils.extraction import extract_json_block

if TYPE_CHECKING:
    from .environment import ContextEnvironment
    from .providers.router import ModelRouter

logger = logging.getLogger(__name__)


class Orchestrator:
    """Analyzes a problem and designs trace configurations.

    Uses an LLM call to produce problem-specific trace configs.
    Falls back to template library or DEFAULT_STRUCTURED_TRACES on failure.
    """

    def __init__(self, router: ModelRouter, model: str, temperature: float = 0.3) -> None:
        self.router = router
        self.model = model
        self.temperature = temperature

    async def design_traces(self, env: ContextEnvironment) -> list[TraceConfig]:
        """Design traces for the given problem via LLM call.

        Returns a list of TraceConfig. Falls back to defaults on failure.
        """
        try:
            messages = build_orchestrator_messages(env.prompt_var)
            return await self._call_and_parse(messages)
        except Exception as e:
            logger.warning("Orchestrator failed (%s); falling back to defaults", e)
            return self._default_traces()

    async def design_traces_for_iteration(
        self,
        env: ContextEnvironment,
        prior_arbitration: ArbitrationResult,
        iteration: int,
    ) -> list[TraceConfig]:
        """Design traces informed by shadows from a prior iteration.

        Injects the prior iteration's shadows, resolution, and confidence
        into the orchestrator prompt so it designs traces that explore
        what was previously left in shadow.
        """
        try:
            messages = build_reframing_messages(
                prompt=env.prompt_var,
                prior_shadows=prior_arbitration.shadows,
                prior_resolution=prior_arbitration.resolution,
                prior_confidence=prior_arbitration.confidence.value,
                iteration=iteration,
            )
            return await self._call_and_parse(messages)
        except Exception as e:
            logger.warning(
                "Orchestrator reframing failed (%s); falling back to defaults", e
            )
            return self._default_traces()

    async def _call_and_parse(self, messages: list[dict[str, str]]) -> list[TraceConfig]:
        """Make the LLM call and parse the response into TraceConfigs."""
        response = await self.router.complete(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
        )

        json_str = extract_json_block(response.content)
        if not json_str:
            logger.warning("Orchestrator produced no JSON; falling back to defaults")
            return self._default_traces()

        raw = json.loads(json_str)
        output = OrchestratorOutput.model_validate(raw)

        # Try to use a template if the problem type matches one
        template = get_template(output.problem_type)
        if template and not output.traces:
            logger.info("Using template for problem type: %s", output.problem_type)
            return [TraceConfig.model_validate(t) for t in template]

        if not output.traces:
            logger.warning("Orchestrator returned no traces; falling back to defaults")
            return self._default_traces()

        logger.info(
            "Orchestrator designed %d traces for %s problem (constraint: %s)",
            len(output.traces),
            output.problem_type,
            output.constraint_level,
        )
        return output.traces

    def _default_traces(self) -> list[TraceConfig]:
        """Return the classic Believer/Logician/Contrarian triad."""
        return [TraceConfig.model_validate(t) for t in DEFAULT_STRUCTURED_TRACES]
