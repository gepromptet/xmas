"""Assistant-token masking: the log-probabilities of a continuation inside an echoed sequence.

Only the assistant's own tokens count; the chat-template scaffolding around them was never the model's to predict.
"""

from __future__ import annotations

from collections.abc import Sequence


def assistant_span(tokens: Sequence[str], continuation: str) -> tuple[int, int]:
    """Locate a continuation inside an echoed token sequence.

    Walks backwards from the end until the concatenated tokens equal ``continuation``
    exactly; an approximate match is never returned.

    Args:
        tokens: Every token the backend echoed, in order.
        continuation: The exact text whose span is wanted.

    Returns:
        A half-open ``(start, end)`` index range into ``tokens``.

    Raises:
        ValueError: If the continuation does not align with token boundaries.
    """
    accumulated = ""
    for index in range(len(tokens) - 1, -1, -1):
        accumulated = tokens[index] + accumulated
        if len(accumulated) < len(continuation):
            continue
        if accumulated == continuation:
            return index, len(tokens)
        break
    raise ValueError("masking: continuation does not align with token boundaries")


def mask_logprobs(
    tokens: Sequence[str],
    logprobs: Sequence[float | None],
    continuation: str,
) -> list[float]:
    """Return the log-probabilities belonging to the continuation alone.

    Args:
        tokens: Every token the backend echoed, in order.
        logprobs: Their log-probabilities. The first is usually ``None``, since
            nothing conditions it.
        continuation: The exact text whose log-probabilities are wanted.

    Returns:
        One log-probability per continuation token.

    Raises:
        ValueError: If the continuation does not align with token boundaries, or
            any of its tokens lacks a log-probability.
    """
    if len(tokens) != len(logprobs):
        raise ValueError(f"masking: {len(tokens)} tokens but {len(logprobs)} log-probabilities")
    start, end = assistant_span(tokens, continuation)
    result = []
    for index in range(start, end):
        value = logprobs[index]
        if value is None:
            raise ValueError(f"masking: token {index} carries no log-probability")
        result.append(value)
    return result
