"""The one error type xmas raises for a failed call."""

from __future__ import annotations


class XmasError(RuntimeError):
    """A call that failed; the message reads '<source>: <what> (<code>)', e.g. 'llm: rate limited (429)'.

    Attributes:
        status: The HTTP-style status code, or ``None``.
        retry_after: Seconds the endpoint asked the caller to wait, or ``None``.
    """

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
