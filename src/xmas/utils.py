"""Small helpers that are about nothing in particular."""

from __future__ import annotations

import httpx


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` header in seconds; only the delta-seconds form is read."""
    raw = response.headers.get("Retry-After")
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
