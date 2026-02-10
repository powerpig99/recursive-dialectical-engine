"""Abstract base class for code execution sandboxes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """Result of executing code in a sandbox."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    runtime_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class BaseSandbox(ABC):
    """Abstract interface for code execution sandboxes."""

    @abstractmethod
    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute code and return the result."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release sandbox resources."""
        ...

    async def __aenter__(self) -> BaseSandbox:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
