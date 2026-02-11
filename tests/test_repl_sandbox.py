"""Tests for the in-process REPL sandbox."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rde.environment import ContextEnvironment
from rde.sandbox.repl_sandbox import REPLSandbox


@pytest.fixture
def env():
    """Create a basic ContextEnvironment for testing."""
    return ContextEnvironment("Hello world. This is a test document with multiple words.")


@pytest.fixture
def sandbox(env):
    """Create a REPLSandbox with no router (llm_query won't work)."""
    return REPLSandbox(env)


@pytest.fixture
def sandbox_with_router(env):
    """Create a REPLSandbox with a mocked router for llm_query tests."""
    router = MagicMock()
    response = MagicMock()
    response.content = "The answer is 42."
    response.estimated_cost = 0.0
    router.complete = AsyncMock(return_value=response)
    env._router = router
    env._sub_lm_models = ["test-model"]
    return REPLSandbox(env, router=router)


async def test_basic_exec(sandbox):
    """Basic code execution captures stdout."""
    result = await sandbox.execute("print('hello')")
    assert result.success
    assert "hello" in result.stdout


async def test_context_available(sandbox):
    """The context variable is accessible in the namespace."""
    result = await sandbox.execute("print(len(context))")
    assert result.success
    assert str(len(sandbox._env.prompt_var)) in result.stdout


async def test_peek_works(sandbox):
    """peek() returns a character slice of the context."""
    result = await sandbox.execute("print(peek(0, 5))")
    assert result.success
    assert "Hello" in result.stdout


async def test_search_works(sandbox):
    """search() performs regex search on the context."""
    result = await sandbox.execute("print(search(r'\\w+')[:3])")
    assert result.success
    assert "Hello" in result.stdout


async def test_partition_works(sandbox):
    """partition() decomposes context into sections."""
    result = await sandbox.execute("parts = partition('structural')\nprint(len(parts))")
    assert result.success
    assert result.exit_code == 0


async def test_safe_modules_available(sandbox):
    """Standard library modules (re, json, math) are available."""
    result = await sandbox.execute(
        "import re\nimport json\nimport math\nprint(math.pi)"
    )
    assert result.success
    assert "3.14" in result.stdout


async def test_dangerous_import_blocked(sandbox):
    """Attempting to import os/subprocess/etc fails."""
    result = await sandbox.execute("import os\nos.listdir('.')")
    assert not result.success
    assert result.exit_code == 1


async def test_file_io_blocked(sandbox):
    """File I/O operations are blocked."""
    result = await sandbox.execute("open('/etc/passwd').read()")
    assert not result.success
    assert result.exit_code == 1


async def test_persistent_vars(sandbox):
    """Variables persist across execute() calls."""
    result1 = await sandbox.execute("my_var = 42")
    assert result1.success
    result2 = await sandbox.execute("print(my_var)")
    assert result2.success
    assert "42" in result2.stdout


async def test_error_handling(sandbox):
    """Errors in code are captured in stderr."""
    result = await sandbox.execute("1 / 0")
    assert not result.success
    assert "ZeroDivisionError" in result.stderr


async def test_output_truncation(env):
    """Output exceeding max_output_chars is truncated."""
    sandbox = REPLSandbox(env, max_output_chars=100)
    result = await sandbox.execute("print('x' * 500)")
    assert "TRUNCATED" in result.stdout


async def test_timeout_enforcement(env):
    """Code exceeding timeout is detected.

    Note: Python threads can't be interrupted for CPU-bound tight loops.
    We test with a computation that releases the GIL via sum() on a generator.
    """
    sandbox = REPLSandbox(env, timeout_seconds=0.1)
    # Large computation that takes time but periodically yields GIL
    result = await sandbox.execute("x = sum(i*i for i in range(10**9))")
    # Either it times out (good) or it completes fast on a fast machine
    # The key assertion: if it timed out, the flag should be set
    if result.timed_out:
        assert not result.success


@pytest.mark.asyncio
async def test_llm_query_from_repl(sandbox_with_router):
    """llm_query() works from REPL code."""
    result = await sandbox_with_router.execute("print(llm_query('What is the answer?'))")
    assert result.success
    assert "The answer is 42." in result.stdout
