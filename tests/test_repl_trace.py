"""Tests for REPL trace execution mode."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rde.environment import ContextEnvironment
from rde.models import RecursionBudget, TraceConfig
from rde.sandbox.repl_sandbox import REPLSandbox
from rde.trace import TraceExecutor


def _make_router_mock(responses: list[str]) -> MagicMock:
    """Create a mock router that returns responses in sequence."""
    router = MagicMock()
    mock_responses = []
    for text in responses:
        resp = MagicMock()
        resp.content = text
        resp.model = "mock-model"
        resp.latency_ms = 10.0
        resp.usage = {}
        resp.estimated_cost = 0.0
        mock_responses.append(resp)
    router.complete = AsyncMock(side_effect=mock_responses)
    return router


@pytest.fixture
def env():
    return ContextEnvironment("Test context for REPL trace execution.")


def test_extract_final():
    """_extract_final() parses FINAL(answer) markers."""
    assert TraceExecutor._extract_final("FINAL(42)") == "42"
    assert TraceExecutor._extract_final("Some text\nFINAL(the answer)") == "the answer"
    assert TraceExecutor._extract_final("No final here") is None
    assert TraceExecutor._extract_final("FINAL(multi\nline)") == "multi\nline"


def test_extract_code_block():
    """_extract_code_block() extracts Python code from markdown fences."""
    text = "Here is code:\n```python\nprint('hello')\n```\nDone."
    assert TraceExecutor._extract_code_block(text) == "print('hello')"

    # Generic code block
    text2 = "```\nx = 1\n```"
    assert TraceExecutor._extract_code_block(text2) == "x = 1"

    # No code block
    assert TraceExecutor._extract_code_block("just text") is None


def test_format_execution_result():
    """_format_execution_result() formats sandbox output for the LLM."""
    from rde.sandbox.base import ExecutionResult

    result = ExecutionResult(stdout="hello\n", stderr="", exit_code=0)
    formatted = TraceExecutor._format_execution_result(result)
    assert "[stdout]" in formatted
    assert "hello" in formatted

    # Error case
    result = ExecutionResult(stdout="", stderr="NameError: x", exit_code=1)
    formatted = TraceExecutor._format_execution_result(result)
    assert "[stderr]" in formatted
    assert "NameError" in formatted

    # Timeout case
    result = ExecutionResult(stdout="", stderr="", timed_out=True, exit_code=-1)
    formatted = TraceExecutor._format_execution_result(result)
    assert "TIMED OUT" in formatted

    # Empty case
    result = ExecutionResult(stdout="", stderr="", exit_code=0)
    formatted = TraceExecutor._format_execution_result(result)
    assert "No output" in formatted


async def test_repl_loop_with_final(env):
    """execute_repl() runs the REPL loop and extracts FINAL answer."""
    router = _make_router_mock([
        "Let me analyze:\n```python\nprint(len(context))\n```",
        "FINAL(42 characters)",
    ])
    env._router = router

    executor = TraceExecutor(router, env)
    config = TraceConfig(
        role="analyst",
        perspective="quantitative",
        system_prompt="Count things",
        execution_mode="repl",
        max_repl_iterations=5,
    )
    sandbox = REPLSandbox(env)
    result = await executor.execute_repl(config, "mock-model", sandbox)

    assert result.extracted_answer == "42 characters"
    assert result.error is None
    assert "repl" in result.trace_id


async def test_repl_max_iterations(env):
    """execute_repl() stops at max_repl_iterations."""
    # Always return code, never FINAL
    responses = [
        "```python\nprint('step')\n```",
    ] * 4
    router = _make_router_mock(responses)
    env._router = router

    executor = TraceExecutor(router, env)
    config = TraceConfig(
        role="analyst",
        perspective="iterative",
        system_prompt="Keep going",
        execution_mode="repl",
        max_repl_iterations=3,
    )
    sandbox = REPLSandbox(env)
    result = await executor.execute_repl(config, "mock-model", sandbox)

    # Should have called router.complete 3 times (max_repl_iterations)
    assert router.complete.call_count == 3
    assert result.extracted_answer is None  # Never reached FINAL


async def test_repl_no_code_prompt(env):
    """When LLM doesn't produce code, it gets prompted to write code."""
    router = _make_router_mock([
        "I'm thinking about this problem...",  # No code block
        "```python\nprint('ok')\n```",
        "FINAL(done)",
    ])
    env._router = router

    executor = TraceExecutor(router, env)
    config = TraceConfig(
        role="thinker",
        perspective="reflective",
        system_prompt="Think carefully",
        execution_mode="repl",
        max_repl_iterations=5,
    )
    sandbox = REPLSandbox(env)
    result = await executor.execute_repl(config, "mock-model", sandbox)

    assert result.extracted_answer == "done"
    # Should have 3 calls: no-code → code → FINAL
    assert router.complete.call_count == 3


async def test_repl_budget_tracking(env):
    """execute_repl() tracks budget correctly."""
    router = _make_router_mock([
        "```python\nprint('hi')\n```",
        "FINAL(answer)",
    ])
    env._router = router
    budget = RecursionBudget(max_total_calls=100)

    executor = TraceExecutor(router, env, budget=budget)
    config = TraceConfig(
        role="analyst",
        perspective="direct",
        system_prompt="Answer",
        execution_mode="repl",
    )
    sandbox = REPLSandbox(env)
    await executor.execute_repl(config, "mock-model", sandbox)

    assert budget.total_calls == 2  # Two router.complete calls


async def test_repl_error_in_code(env):
    """When code raises an error, stderr is fed back to the LLM."""
    router = _make_router_mock([
        "```python\n1/0\n```",
        "FINAL(caught error)",
    ])
    env._router = router

    executor = TraceExecutor(router, env)
    config = TraceConfig(
        role="debugger",
        perspective="careful",
        system_prompt="Fix errors",
        execution_mode="repl",
    )
    sandbox = REPLSandbox(env)
    result = await executor.execute_repl(config, "mock-model", sandbox)

    assert result.extracted_answer == "caught error"
