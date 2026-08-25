"""The result type shared by every backend's ``score``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredSequence:
    """Per-token and aggregate log-probabilities for one scored continuation.

    Attributes:
        tokens: The continuation's tokens, in order.
        token_logprobs: Their log-probabilities, aligned to ``tokens``.
    """

    tokens: tuple[str, ...]
    token_logprobs: tuple[float, ...]

    @property
    def sum_logprob(self) -> float:
        """Total log-probability of the continuation."""
        raise NotImplementedError

    @property
    def mean_logprob(self) -> float:
        """Length-normalized log-probability, ``0.0`` for an empty continuation."""
        raise NotImplementedError

    @property
    def token_count(self) -> int:
        """Number of scored tokens."""
        raise NotImplementedError
