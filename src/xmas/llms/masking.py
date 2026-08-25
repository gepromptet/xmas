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
    raise NotImplementedError


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
    raise NotImplementedError
