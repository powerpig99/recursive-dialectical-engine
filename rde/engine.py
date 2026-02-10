"""Dialectical Engine — the main orchestration loop.

Coordinates: ContextEnvironment → Orchestrator → parallel Traces →
Normalizer → consensus check → Arbiter → convergence check → reframe.

Supports multi-iteration shadow-informed reframing (Phase 2).
"""

from __future__ import annotations

import asyncio
import logging
import time

from .arbiter import Arbiter
from .environment import ContextEnvironment
from .models import CallTreeNode, ConfidenceLevel, EngineResult, ModelConfig, RecursionBudget, TraceConfig
from .normalizer import TraceNormalizer
from .orchestrator import Orchestrator
from .prompts.traces import DEFAULT_STRUCTURED_TRACES
from .providers.router import ModelRouter
from .trace import TraceExecutor

logger = logging.getLogger(__name__)


class DialecticalEngine:
    """Multi-model dialectical reasoning engine.

    Usage:
        async with DialecticalEngine(config) as engine:
            result = await engine.run("What is 2+2?")

    Multi-iteration mode:
        result = await engine.run("Complex problem", max_iterations=3)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.router: ModelRouter | None = None

    async def __aenter__(self) -> DialecticalEngine:
        self.router = ModelRouter(self.config)
        logger.info("Engine initialized with providers: %s", self.router.available_providers)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.router:
            await self.router.close()

    async def run(
        self,
        prompt: str,
        use_orchestrator: bool = True,
        max_iterations: int = 1,
        budget: RecursionBudget | None = None,
        num_traces: int | None = None,
        execution_mode: str | None = None,
    ) -> EngineResult:
        """Execute the full dialectical reasoning pipeline.

        Single iteration (max_iterations=1):
          orchestrate → trace → normalize → consensus? → arbiter → result

        Multi-iteration (max_iterations>1):
          Loop: orchestrate → trace → arbiter → convergence check →
          reframe with shadows → repeat until converged or budget exhausted.
        """
        assert self.router is not None, "Engine not initialized. Use 'async with' context manager."

        start_time = time.perf_counter()
        if budget is None:
            budget = RecursionBudget()
        env = ContextEnvironment(
            prompt,
            max_iterations=max_iterations,
            router=self.router,
            budget=budget,
            sub_lm_models=self.config.sub_lm_models,
            sub_lm_temperature=self.config.sub_lm_temperature,
        )
        root_tree = CallTreeNode(node_type="engine", depth=budget.current_depth)
        normalizer = TraceNormalizer()
        arbiter = Arbiter(self.router, self.config.arbiter_model, temperature=self.config.arbiter_temperature)
        orchestrator = Orchestrator(self.router, self.config.orchestrator_model, temperature=self.config.orchestrator_temperature) if use_orchestrator else None

        all_trace_results = []
        all_normalized = []
        shadow_history: list[list[str]] = []
        last_arbitration = None

        for iteration in range(max_iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, max_iterations)

            # Step 1: Design traces
            trace_configs = await self._design_traces(
                orchestrator, env, last_arbitration, iteration, num_traces=num_traces
            )
            # Apply execution_mode override if specified
            if execution_mode is not None:
                for tc in trace_configs:
                    tc.execution_mode = execution_mode
            logger.info("Designed %d traces (mode=%s)", len(trace_configs), execution_mode or "default")

            # Step 2: Assign models
            assigned_models = self.router.assign_trace_models(trace_configs)
            logger.info(
                "Assigned models: %s",
                [(tc.role, m) for tc, m in zip(trace_configs, assigned_models)],
            )

            # Step 2.5: Pre-compute partitions for non-structural strategies
            partition_strategies: set[str] = set()
            for tc in trace_configs:
                if tc.context_strategy.startswith("partition:"):
                    parts = tc.context_strategy.split(":", 2)
                    strategy = parts[1] if len(parts) >= 2 else "structural"
                    if strategy not in ("structural",):
                        partition_strategies.add(strategy)
            for strategy in partition_strategies:
                await env.prepare_partitions(strategy)

            # Step 3: Execute traces in parallel
            executor = TraceExecutor(self.router, env, budget=budget)
            tasks = []
            for config, model in zip(trace_configs, assigned_models):
                if config.execution_mode == "repl":
                    from .sandbox.repl_sandbox import REPLSandbox
                    repl_sandbox = REPLSandbox(env, self.router, budget)
                    tasks.append(executor.execute_repl(config, model, repl_sandbox))
                else:
                    tasks.append(executor.execute(config, model))
            trace_results = list(await asyncio.gather(*tasks))
            all_trace_results.extend(trace_results)

            # Step 4: Filter failed traces and normalize
            successful = [tr for tr in trace_results if tr.error is None]

            if not successful:
                logger.warning("All traces failed in iteration %d", iteration + 1)
                if last_arbitration:
                    # Use prior iteration's result
                    break
                total_ms = (time.perf_counter() - start_time) * 1000
                return EngineResult(
                    resolution="All traces failed",
                    confidence=ConfidenceLevel.UNRESOLVED,
                    shadows=["No trace produced output"],
                    trace_results=all_trace_results,
                    total_latency_ms=total_ms,
                )

            normalized = [normalizer.normalize(tr) for tr in successful]
            all_normalized.extend(normalized)

            # Step 5: Consensus check
            if normalizer.check_consensus(normalized):
                total_ms = (time.perf_counter() - start_time) * 1000
                logger.info("Consensus reached: %s", normalized[0].conclusion)
                root_tree.children.extend(env._call_tree_children)
                root_tree.latency_ms = total_ms
                return EngineResult(
                    resolution=normalized[0].conclusion,
                    confidence=ConfidenceLevel.NECESSARY,
                    trace_results=all_trace_results,
                    normalized_traces=all_normalized,
                    total_latency_ms=total_ms,
                    consensus_reached=True,
                    iterations=iteration + 1,
                    call_tree=root_tree,
                )

            # Step 6: Arbiter resolves disagreement
            last_arbitration = await arbiter.arbitrate(normalized, env, budget=budget)

            # Store iteration history
            env.store_iteration(
                results={tr.trace_id: tr.model_dump() for tr in trace_results},
                arbitration=last_arbitration.model_dump(),
            )
            shadow_history.append(last_arbitration.shadows)

            # Step 7: Convergence check (skip on last possible iteration)
            if iteration < max_iterations - 1:
                convergence = arbiter.check_convergence(
                    last_arbitration,
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                    prior_shadows=shadow_history[:-1] if len(shadow_history) > 1 else None,
                )
                if convergence.should_stop:
                    logger.info("Converged: %s", convergence.reason)
                    break
                logger.info(
                    "Continuing: %s (shadows to explore: %d)",
                    convergence.reason,
                    len(convergence.next_framings),
                )

        # Build final result from last arbitration
        total_ms = (time.perf_counter() - start_time) * 1000
        root_tree.children.extend(env._call_tree_children)
        root_tree.latency_ms = total_ms
        return EngineResult(
            resolution=last_arbitration.resolution,
            confidence=last_arbitration.confidence,
            shadows=last_arbitration.shadows,
            causal_chain=last_arbitration.causal_chain,
            iterations=env.current_iteration,
            trace_results=all_trace_results,
            normalized_traces=all_normalized,
            arbitration=last_arbitration,
            total_latency_ms=total_ms,
            call_tree=root_tree,
        )

    async def _design_traces(
        self,
        orchestrator: Orchestrator | None,
        env: ContextEnvironment,
        prior_arbitration,
        iteration: int,
        num_traces: int | None = None,
    ) -> list[TraceConfig]:
        """Design traces for this iteration, using reframing if applicable."""
        if orchestrator is None:
            traces = [TraceConfig.model_validate(t) for t in DEFAULT_STRUCTURED_TRACES]
            if num_traces is not None and num_traces < len(traces):
                traces = traces[:num_traces]
            return traces

        if iteration == 0 or prior_arbitration is None:
            return await orchestrator.design_traces(env)

        # Subsequent iterations: reframe based on prior shadows
        return await orchestrator.design_traces_for_iteration(
            env, prior_arbitration, iteration
        )
