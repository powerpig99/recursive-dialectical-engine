"""In-process REPL sandbox for trace code execution.

Unlike LocalSandbox (subprocess), this runs LLM-generated code via exec()
in the current process with a restricted namespace. This enables callbacks
to ContextEnvironment methods (peek, search, partition, llm_query) without
cross-process IPC.

Follows the RLM paper's approach (Zhang et al., 2025): the LLM writes
Python code that interacts with the context through helper functions.
"""

from __future__ import annotations

import asyncio
import collections
import io
import itertools
import json
import logging
import math
import os
import re
import string
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any

from .base import ExecutionResult

if TYPE_CHECKING:
    from ..environment import ContextEnvironment
    from ..models import RecursionBudget
    from ..providers.router import ModelRouter

logger = logging.getLogger(__name__)


class _TimeoutError(Exception):
    """Raised when code execution exceeds the time limit."""


class REPLSandbox:
    """In-process sandbox that exposes ContextEnvironment methods to exec'd code.

    The namespace provides:
      - context: str — the full externalized prompt
      - peek(start, end) -> str — character slice of context
      - search(pattern) -> list[str] — regex search over context
      - partition(strategy) -> list[str] — decompose context
      - llm_query(prompt, model=None) -> str — recursive sub-LM call
      - Standard library: re, json, math, collections, itertools, string
    """

    # Modules whitelisted for use in generated code
    SAFE_MODULES = {
        "re": re,
        "json": json,
        "math": math,
        "collections": collections,
        "itertools": itertools,
        "string": string,
    }

    # Builtins allowed in the restricted namespace
    SAFE_BUILTINS = {
        "abs", "all", "any", "bool", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset",
        "hasattr", "hash", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next",
        "ord", "pow", "print", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip",
    }

    def __init__(
        self,
        environment: ContextEnvironment,
        router: ModelRouter | None = None,
        budget: RecursionBudget | None = None,
        max_output_chars: int = 50_000,
        timeout_seconds: float = 60.0,
        sandbox_mode: str | None = None,
    ) -> None:
        self._env = environment
        self._router = router
        self._budget = budget
        self._max_output = max_output_chars
        self._timeout = timeout_seconds
        # Persistent state across REPL iterations (like a real REPL)
        self._persistent_vars: dict[str, Any] = {}
        self._sandbox_mode = sandbox_mode or os.environ.get(
            "RDE_REPL_SANDBOX_MODE", "in_process"
        )
        if self._sandbox_mode not in ("in_process", "subprocess"):
            raise ValueError(
                f"Invalid sandbox_mode={self._sandbox_mode}. "
                "Use 'in_process' or 'subprocess'."
            )
        if self._sandbox_mode == "in_process":
            logger.warning(
                "REPL sandbox running in-process. This is not a security boundary. "
                "Set RDE_REPL_SANDBOX_MODE=subprocess for stronger isolation."
            )

        # Capture main event loop if available (for llm_query from worker threads)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fall back to the current event loop (may not be running yet).
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = None

        self._local_sandbox = None
        if self._sandbox_mode == "subprocess":
            from .local_sandbox import LocalSandbox

            self._local_sandbox = LocalSandbox(max_timeout=self._timeout)

    def _restricted_import(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Restricted __import__ that only allows whitelisted modules."""
        if name in self.SAFE_MODULES:
            return self.SAFE_MODULES[name]
        raise ImportError(f"Import of '{name}' is not allowed in the REPL sandbox")

    def _build_namespace(self) -> dict[str, Any]:
        """Build the restricted namespace for code execution."""
        # Safe builtins subset
        safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in self.SAFE_BUILTINS}
        safe_builtins["__import__"] = self._restricted_import

        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            # Context interaction
            "context": self._env.prompt_var,
            "peek": self._env.peek,
            "search": self._env.search,
            "partition": self._env.partition,
            "llm_query": self._sync_llm_query,
            # Safe modules
            **self.SAFE_MODULES,
        }

        # Restore persistent variables from previous iterations
        namespace.update(self._persistent_vars)

        return namespace

    def _sync_llm_query(self, prompt: str, model: str | None = None) -> str:
        """Synchronous wrapper for spawn_sub_lm.

        Bridges sync exec() context to async ContextEnvironment.spawn_sub_lm()
        using the running event loop.
        """
        if self._env._router is None:
            return "[NO ROUTER CONFIGURED]"

        loop = self._loop
        if loop is None or loop.is_closed():
            return "[NO EVENT LOOP — llm_query requires async context]"

        future = asyncio.run_coroutine_threadsafe(
            self._env.spawn_sub_lm(prompt, model),
            loop,
        )

        try:
            return future.result(timeout=self._timeout)
        except TimeoutError:
            return "[LLM QUERY TIMED OUT]"
        except Exception as e:
            return f"[LLM QUERY ERROR: {e}]"

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in the restricted namespace.

        Runs exec() in a thread to avoid blocking the event loop.
        Captures stdout via StringIO. Enforces timeout.
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if self._sandbox_mode == "subprocess":
            return await self._execute_subprocess(code)
        return await self._execute_in_process(code)

    async def _execute_in_process(self, code: str) -> ExecutionResult:
        namespace = self._build_namespace()
        stdout_capture = io.StringIO()
        start = time.perf_counter()

        def _run() -> tuple[str, str, int]:
            """Run exec() in a thread with stdout capture."""
            nonlocal namespace
            try:
                with redirect_stdout(stdout_capture):
                    exec(code, namespace)  # noqa: S102
                stdout = stdout_capture.getvalue()
                if len(stdout) > self._max_output:
                    stdout = stdout[: self._max_output] + "\n... [OUTPUT TRUNCATED]"
                return stdout, "", 0
            except Exception as e:
                stdout = stdout_capture.getvalue()
                return stdout, f"{type(e).__name__}: {e}", 1

        try:
            loop = asyncio.get_running_loop()
            stdout, stderr, exit_code = await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                stdout=stdout_capture.getvalue(),
                stderr=f"Execution timed out after {self._timeout}s",
                exit_code=-1,
                timed_out=True,
                runtime_ms=elapsed_ms,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Persist user-defined variables for next iteration
        self._save_persistent_vars(namespace)

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            runtime_ms=elapsed_ms,
        )

    async def _execute_subprocess(self, code: str) -> ExecutionResult:
        if self._local_sandbox is None:
            return ExecutionResult(
                stderr="Subprocess sandbox unavailable",
                exit_code=1,
            )

        wrapped_code = self._build_subprocess_code(code)
        result = await self._local_sandbox.execute(
            wrapped_code,
            language="python",
            timeout_seconds=self._timeout,
        )

        stdout = result.stdout
        if stdout:
            stdout, state = self._extract_state_from_stdout(stdout)
            if state:
                self._persistent_vars.update(state)

        if len(stdout) > self._max_output:
            stdout = stdout[: self._max_output] + "\n... [OUTPUT TRUNCATED]"

        return ExecutionResult(
            stdout=stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            runtime_ms=result.runtime_ms,
        )

    def _build_subprocess_code(self, code: str) -> str:
        """Wrap user code with helpers and state persistence for subprocess mode."""
        context_literal = json.dumps(self._env.prompt_var)
        prelude_lines = [
            "import re, json, math, collections, itertools, string",
            f"context = {context_literal}",
            "def peek(start, end):\n    return context[start:end]",
            "def search(pattern):\n    return re.findall(pattern, context)",
            "def partition(strategy='structural'):\n"
            "    if strategy == 'structural':\n"
            "        parts = [p.strip() for p in re.split(r'\\n\\s*\\n', context) if p.strip()]\n"
            "        return parts if parts else [context]\n"
            "    return partition('structural')",
            "def llm_query(prompt, model=None):\n    return '[LLM QUERY DISABLED IN SUBPROCESS]'",
        ]

        # Restore persistent vars (JSON-serializable only)
        for key, value in self._persistent_vars.items():
            if key.startswith("_"):
                continue
            try:
                encoded = json.dumps(value)
            except (TypeError, ValueError):
                continue
            prelude_lines.append(f"{key} = {encoded}")

        footer = [
            "import json as __rde_json",
            "__rde_state = {}",
            "__rde_skip = {'context', 'peek', 'search', 'partition', 'llm_query'}",
            "for __k, __v in globals().items():",
            "    if __k.startswith('_') or __k in __rde_skip:",
            "        continue",
            "    try:",
            "        __rde_json.dumps(__v)",
            "        __rde_state[__k] = __v",
            "    except Exception:",
            "        pass",
            "print('__RDE_STATE__START__')",
            "print(__rde_json.dumps(__rde_state))",
            "print('__RDE_STATE__END__')",
        ]

        return "\n".join(prelude_lines) + "\n" + code + "\n" + "\n".join(footer)

    @staticmethod
    def _extract_state_from_stdout(stdout: str) -> tuple[str, dict[str, Any] | None]:
        """Extract persisted state dump from stdout."""
        pattern = r"__RDE_STATE__START__\\n(.*?)\\n__RDE_STATE__END__"
        match = re.search(pattern, stdout, re.DOTALL)
        if not match:
            return stdout, None
        json_blob = match.group(1).strip()
        cleaned = re.sub(pattern, "", stdout, flags=re.DOTALL).strip()
        try:
            state = json.loads(json_blob)
            if isinstance(state, dict):
                return cleaned, state
        except Exception:
            pass
        return cleaned, None

    def _save_persistent_vars(self, namespace: dict[str, Any]) -> None:
        """Save user-defined variables from namespace for next iteration."""
        skip_keys = {
            "__builtins__", "context", "peek", "search", "partition",
            "llm_query", *self.SAFE_MODULES.keys(),
        }
        for key, value in namespace.items():
            if key.startswith("_"):
                continue
            if key in skip_keys:
                continue
            # Only persist serializable-ish types (not modules, functions from stdlib)
            if callable(value) and not isinstance(value, type):
                # Skip lambdas and functions defined in exec'd code — they may
                # reference the old namespace. Let them be redefined each time.
                continue
            self._persistent_vars[key] = value

    async def close(self) -> None:
        """Clear persistent state."""
        self._persistent_vars.clear()
