"""Runtime failures that must be separated from task-quality failures."""

from specagent.protocol.types import RuntimeErrorCode


class SpecAgentRuntimeError(RuntimeError):
    def __init__(self, code: RuntimeErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
