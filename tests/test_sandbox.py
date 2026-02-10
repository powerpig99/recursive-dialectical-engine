"""Tests for the sandbox module."""

import pytest

from rde.sandbox import BaseSandbox, ExecutionResult, LocalSandbox


def test_execution_result_success():
    r = ExecutionResult(exit_code=0)
    assert r.success is True


def test_execution_result_failure():
    r = ExecutionResult(exit_code=1)
    assert r.success is False


def test_execution_result_timeout():
    r = ExecutionResult(exit_code=0, timed_out=True)
    assert r.success is False


@pytest.mark.asyncio
async def test_local_sandbox_python():
    async with LocalSandbox() as sb:
        result = await sb.execute("print('hello')", language="python")
    assert result.success
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_local_sandbox_bash():
    async with LocalSandbox() as sb:
        result = await sb.execute("echo world", language="bash")
    assert result.success
    assert "world" in result.stdout


@pytest.mark.asyncio
async def test_local_sandbox_error():
    async with LocalSandbox() as sb:
        result = await sb.execute("import sys; sys.exit(1)", language="python")
    assert not result.success
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_local_sandbox_timeout():
    async with LocalSandbox(max_timeout=1.0) as sb:
        result = await sb.execute(
            "import time; time.sleep(10)",
            language="python",
            timeout_seconds=0.5,
        )
    assert result.timed_out
    assert not result.success


@pytest.mark.asyncio
async def test_local_sandbox_unsupported_language():
    async with LocalSandbox() as sb:
        result = await sb.execute("code", language="rust")
    assert not result.success
    assert "Unsupported" in result.stderr


@pytest.mark.asyncio
async def test_local_sandbox_runtime_ms():
    async with LocalSandbox() as sb:
        result = await sb.execute("print(1+1)", language="python")
    assert result.runtime_ms > 0


def test_modal_sandbox_import_guard():
    """ModalSandbox raises ImportError if modal not installed."""
    try:
        from rde.sandbox.modal_sandbox import ModalSandbox

        ModalSandbox()
    except ImportError:
        pass  # Expected when modal is not installed


def test_e2b_sandbox_import_guard():
    """E2BSandbox raises ImportError if e2b not installed."""
    try:
        from rde.sandbox.e2b_sandbox import E2BSandbox

        E2BSandbox()
    except ImportError:
        pass  # Expected when e2b is not installed


def test_sandbox_exports():
    """Verify sandbox __init__ exports the right classes."""
    assert BaseSandbox is not None
    assert ExecutionResult is not None
    assert LocalSandbox is not None
