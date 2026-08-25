"""Assistant-token masking.

Teacher-forced scoring asks what log-probability a model assigns to a fixed
continuation. When the continuation is an assistant turn inside a chat template,
only the assistant's own tokens count: the template scaffolding around it was
never the model's to predict, and including it dilutes the score by a
prompt-dependent amount.

This is a standalone utility rather than a method on an LLM, because a probing
task, an attributor and a backend all need the same answer and none of them
should own it.
"""

from __future__ import annotations

from collections.abc import Sequence


def assistant_span(tokens: Sequence[str], continuation: str) -> tuple[int, int]:
    """Locate a continuation inside an echoed token sequence.

    The span is found by walking backwards from the end and accumulating tokens
    until the concatenation equals ``continuation`` exactly. An approximate match
    is never returned: a continuation that does not fall on token boundaries has
    no exact sequence log-probability, and saying so beats a plausible wrong
    number.

    Args:
        tokens: Every token the backend echoed, in order.
        continuation: The exact text whose span is wanted.

    Returns:
        A half-open ``(start, end)`` index range into ``tokens``.

    Raises:
        ValueError: If the continuation does not align with token boundaries.
    """
    if not continuation:
        raise ValueError("Cannot locate an empty continuation.")
    if not tokens:
        raise ValueError("Cannot locate a continuation in an empty token sequence.")

    accumulated = ""
    for index in range(len(tokens) - 1, -1, -1):
        accumulated = tokens[index] + accumulated
        if len(accumulated) < len(continuation):
            continue
        if accumulated == continuation:
            return index, len(tokens)
        break

    raise ValueError(
        "The continuation does not align with the backend's token boundaries, so it has "
        "no exact sequence log-probability. Scoring an approximation here would be "
        "indistinguishable from a correct answer downstream."
    )


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
        raise ValueError(f"Got {len(tokens)} tokens but {len(logprobs)} log-probabilities.")

    start, end = assistant_span(tokens, continuation)
    span = list(logprobs[start:end])
    missing = [index for index, value in enumerate(span, start=start) if value is None]
    if missing:
        raise ValueError(
            f"Tokens at positions {missing} carry no log-probability, so the continuation "
            "cannot be scored. The first prompt token is unconditioned by definition; a "
            "continuation starting there is not scoreable."
        )
    return [float(value) for value in span if value is not None]
