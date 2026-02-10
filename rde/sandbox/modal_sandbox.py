"""Modal cloud sandbox (requires modal package)."""

from __future__ import annotations

from .base import BaseSandbox, ExecutionResult


class ModalSandbox(BaseSandbox):
    """Execute code in a Modal cloud sandbox.

    Requires: pip install modal
    Configure: modal token new
    """

    def __init__(self) -> None:
        try:
            import modal  # noqa: F401
        except ImportError:
            raise ImportError(
                "Modal sandbox requires the 'modal' package. "
                "Install with: pip install modal"
            ) from None

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        raise NotImplementedError(
            "Modal sandbox is a stub. Implement based on Modal Sandbox API: "
            "https://modal.com/docs/reference/modal.Sandbox"
        )

    async def close(self) -> None:
        pass
