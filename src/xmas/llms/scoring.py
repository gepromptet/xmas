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
        return float(sum(self.token_logprobs))

    @property
    def mean_logprob(self) -> float:
        """Length-normalized log-probability, ``0.0`` for an empty continuation.

        Length normalization matters whenever continuations of different lengths
        are compared: a longer one accumulates more negative log-probability
        simply by being longer.
        """
        return self.sum_logprob / len(self.token_logprobs) if self.token_logprobs else 0.0

    @property
    def token_count(self) -> int:
        """Number of scored tokens."""
        return len(self.tokens)
