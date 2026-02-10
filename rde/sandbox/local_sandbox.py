"""Local subprocess sandbox using asyncio."""

from __future__ import annotations

import asyncio
import time

from .base import BaseSandbox, ExecutionResult


class LocalSandbox(BaseSandbox):
    """Execute code in a local subprocess with timeout."""

    def __init__(self, max_timeout: float = 60.0) -> None:
        self._max_timeout = max_timeout

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        effective_timeout = min(timeout_seconds, self._max_timeout)

        if language == "python":
            cmd = ["python3", "-c", code]
        elif language == "bash":
            cmd = ["bash", "-c", code]
        else:
            return ExecutionResult(
                stderr=f"Unsupported language: {language}",
                exit_code=1,
            )

        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                exit_code=proc.returncode or 0,
                runtime_ms=elapsed_ms,
            )
        except asyncio.TimeoutError:
            proc.kill()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                stderr=f"Execution timed out after {effective_timeout}s",
                exit_code=-1,
                timed_out=True,
                runtime_ms=elapsed_ms,
            )

    async def close(self) -> None:
        pass  # No persistent resources
