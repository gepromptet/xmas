"""Small helpers that are about nothing in particular."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import httpx


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Return a response's ``Retry-After`` header in seconds; only the delta-seconds form is read."""
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def error_message(response: httpx.Response) -> str:
    """Return the endpoint's own explanation, or the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and "message" in error:
        return str(error["message"])
    return str(body)


def timestamp() -> str:
    """Return the current time as ISO 8601 with an explicit UTC offset."""
    return datetime.now(UTC).isoformat()


def approximate_tokens(text: str) -> int:
    """Count whitespace words, the same undercount promptolution uses.

    Comparable across calls, not to a provider's meter.
    """
    return len(text.split())
