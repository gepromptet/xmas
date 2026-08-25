"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

import os
from typing import Any

import httpx
from promptolution.llms import api_llm as _promptolution

from xmas.llms.scoring import ScoredSequence

ENV_BASE_URL = "XMAS_BASE_URL"
"""Environment variable holding the endpoint base URL."""

ENV_API_KEY = "XMAS_API_KEY"
"""Environment variable holding the API key."""

SCORE_PATH = "score"
"""Scoring endpoint, resolved relative to ``base_url``."""


class ScoringUnsupportedError(RuntimeError):
    """Raised when a backend cannot expose prompt log-probabilities.

    Carries the endpoint's own explanation, which names the models that would
    work, so the caller is not left guessing what to switch to.
    """


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
        model = model or kwargs.pop("model_id", None)
        base_url = base_url or kwargs.pop("api_url", None) or os.environ.get(ENV_BASE_URL)
        api_key = api_key or os.environ.get(ENV_API_KEY)

        if not model:
            raise ValueError("APILLM needs a model, e.g. APILLM(model='llama-3.1-8b').")
        if not base_url:
            raise ValueError(f"APILLM needs a base_url, or {ENV_BASE_URL} set in the environment.")

        self.base_url = base_url.rstrip("/")
        super().__init__(model_id=model, api_url=self.base_url, api_key=api_key, **kwargs)

    @property
    def model(self) -> str:
        """The model identifier, under the name the xmas surface uses."""
        return str(self.model_id)

    def _score_endpoint(self) -> str:
        """Return the absolute URL of the scoring endpoint."""
        return f"{self.base_url}/{SCORE_PATH}"

    def _headers(self) -> dict[str, str]:
        """Return the auth headers for a direct call."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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
            ValueError: If the continuation does not align with token boundaries,
                or the request is otherwise rejected.
        """
        if not continuation:
            raise ValueError("Cannot score an empty continuation.")

        response = httpx.post(
            self._score_endpoint(),
            json={"model": self.model, "prompt": context, "continuation": continuation},
            headers=self._headers(),
            timeout=self.call_timeout_s,
        )
        return self._parse_score(response)

    def score_batch(self, contexts: list[str], continuations: list[str]) -> list[ScoredSequence]:
        """Score several continuations.

        Sequential for now: the endpoint has no batch form, and issuing
        concurrent requests against a metered gateway is a decision to take
        deliberately rather than inherit.

        Args:
            contexts: The conditioning texts.
            continuations: The texts to score, aligned to ``contexts``.

        Returns:
            One result per pair, in the order given.
        """
        pairs = zip(contexts, continuations, strict=True)
        return [self.score(context, continuation) for context, continuation in pairs]

    @staticmethod
    def _parse_score(response: httpx.Response) -> ScoredSequence:
        """Turn a scoring response into a :class:`ScoredSequence`.

        Args:
            response: The HTTP response.

        Returns:
            The parsed result.

        Raises:
            ScoringUnsupportedError: On a 501 capability refusal.
            ValueError: On any other error status or a malformed body.
        """
        if response.status_code == httpx.codes.NOT_IMPLEMENTED:
            raise ScoringUnsupportedError(_error_message(response))
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise ValueError(f"Scoring failed ({response.status_code}): {_error_message(response)}")

        body = response.json()
        tokens = body.get("tokens")
        logprobs = body.get("token_logprobs")
        if not isinstance(tokens, list) or not isinstance(logprobs, list):
            raise ValueError(f"Scoring response is missing tokens/token_logprobs: {body!r}")

        return ScoredSequence(
            tokens=tuple(str(token) for token in tokens),
            token_logprobs=tuple(float(value) for value in logprobs),
        )


def _error_message(response: httpx.Response) -> str:
    """Extract the endpoint's own explanation, falling back to the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and "message" in error:
        return str(error["message"])
    return str(body)
