"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

from typing import Any

from promptolution.llms import api_llm as _promptolution

from xmas.llms.scoring import ScoredSequence

ENV_BASE_URL = "XMAS_BASE_URL"
"""Environment variable holding the endpoint base URL."""

ENV_API_KEY = "XMAS_API_KEY"
"""Environment variable holding the API key."""


class APILLM(_promptolution.APILLM):
    """An OpenAI-compatible model that can also score a fixed continuation.

    Generation is inherited untouched from promptolution: batching, concurrency,
    retries and token metering already live there, and reimplementing any of it
    here would create a second place for the same bug. What xmas adds is
    :meth:`score`.

    The constructor accepts ``model`` and ``base_url``, which is what the xmas
    surface reads like, and forwards them to promptolution's ``model_id`` and
    ``api_url``. Both spellings are accepted so that code written against either
    library keeps working.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            model: Model identifier, e.g. ``"llama-3.1-8b"``. Alias of the
                parent's ``model_id``.
            base_url: Endpoint base URL. Alias of the parent's ``api_url``.
                Defaults to the ``XMAS_BASE_URL`` environment variable.
            api_key: API key. Defaults to the ``XMAS_API_KEY`` environment
                variable. Never read from a file in the repository.
            **kwargs: Forwarded to promptolution's ``APILLM``.

        Raises:
            ValueError: If no model is given, or no base URL can be resolved.
        """
        raise NotImplementedError

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        This is teacher forcing: the continuation is fixed and scored, not
        generated. It calls the gateway's scoring endpoint, which sits next to
        ``base_url`` and needs a backend that echoes prompt log-probabilities.

        Args:
            context: The text the continuation is conditioned on.
            continuation: The exact text to score.

        Returns:
            The continuation's per-token and aggregate log-probabilities.

        Raises:
            ScoringUnsupportedError: If the endpoint reports that the configured
                backend cannot echo prompt log-probabilities.
            ValueError: If the continuation does not align with token boundaries.
        """
        raise NotImplementedError

    def score_batch(self, contexts: list[str], continuations: list[str]) -> list[ScoredSequence]:
        """Score several continuations, concurrently where the endpoint allows.

        Args:
            contexts: The conditioning texts.
            continuations: The texts to score, aligned to ``contexts``.

        Returns:
            One result per pair, in the order given.
        """
        raise NotImplementedError


class ScoringUnsupportedError(RuntimeError):
    """Raised when a backend cannot expose prompt log-probabilities.

    Carries the endpoint's own explanation, which names the models that would
    work, so the caller is not left guessing what to switch to.
    """
