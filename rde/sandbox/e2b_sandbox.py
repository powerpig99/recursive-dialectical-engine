"""E2B cloud sandbox (requires e2b-code-interpreter package)."""

from __future__ import annotations

from .base import BaseSandbox, ExecutionResult


class E2BSandbox(BaseSandbox):
    """Execute code in an E2B cloud sandbox.

    Requires: pip install e2b-code-interpreter
    Configure: E2B_API_KEY env var
    """

    def __init__(self) -> None:
        try:
            import e2b_code_interpreter  # noqa: F401
        except ImportError:
            raise ImportError(
                "E2B sandbox requires the 'e2b-code-interpreter' package. "
                "Install with: pip install e2b-code-interpreter"
            ) from None

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        raise NotImplementedError(
            "E2B sandbox is a stub. Implement based on E2B API: "
            "https://e2b.dev/docs"
        )

    async def close(self) -> None:
        pass
