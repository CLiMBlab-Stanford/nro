"""Shared orchestration error types without service-side dependencies."""


class SchedulerError(RuntimeError):
    """Report a scheduler transport or service failure."""

    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        """Preserve the remote error category with its user-facing message."""
        super().__init__(message)
        self.error_type = error_type
