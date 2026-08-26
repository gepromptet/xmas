"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from promptolution.llms import api_llm as _promptolution

from xmas.llms.scoring import ScoredSequence
from xmas.llms.telemetry import GENERATE, SCORE, CallLog, CallRecord, approximate_tokens

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


class RateLimitedError(RuntimeError):
    """Raised when the endpoint refuses a call because the key is over its limit.

    Typed, and carrying ``retry_after``, because a limiter wall is the system
    working and the only correct response is to wait exactly as long as the
    endpoint asked. A caller that could not tell this apart from a malformed
    request would have to either give up or guess, and guessing is what turns
    one 429 into a retry storm.

    Attributes:
        retry_after: Seconds the endpoint asked the caller to wait, or ``None``
            when it did not say.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Initialize the error.

        Args:
            message: The endpoint's own explanation.
            retry_after: Value of the ``Retry-After`` header, in seconds.
        """
        super().__init__(message)
        self.retry_after = retry_after


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
        telemetry: CallLog | None = None,
        backend_tag: str = "gateway",
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
            telemetry: Collector for per-call statistics. ``None`` records
                nothing, which is the default so that ordinary use pays no
                attention to instrumentation.
            backend_tag: What this backend calls itself in a record. Calls from
                two serving paths must never be pooled into one statistic, and
                this is what keeps them separable afterwards.
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
        self.telemetry = telemetry
        self.backend_tag = backend_tag
        super().__init__(model_id=model, api_url=self.base_url, api_key=api_key, **kwargs)

    def _record(self, record: CallRecord) -> None:
        """Hand one call to the collector, if there is one."""
        if self.telemetry is not None:
            self.telemetry.record(record)

    def get_response(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None = None,
    ) -> list[str]:
        """Generate, timing the call and recording what it cost.

        The generation itself is promptolution's; this adds the measurement.
        Token counts are read from the parent's own counters, which split on
        whitespace, so they are an approximation and are reported as one.

        Args:
            prompts: Prompt or prompts to answer.
            system_prompts: System prompt or prompts.

        Returns:
            One response per prompt.
        """
        n_prompts = 1 if isinstance(prompts, str) else len(prompts)
        before = self.get_token_count()
        started = time.perf_counter()
        try:
            responses = super().get_response(prompts, system_prompts)
        except Exception as error:
            self._record(
                CallRecord(
                    operation=GENERATE,
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    n_prompts=n_prompts,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        latency = time.perf_counter() - started
        after = self.get_token_count()
        self._record(
            CallRecord(
                operation=GENERATE,
                model=self.model,
                backend=self.backend_tag,
                latency_s=latency,
                input_tokens=after["input_tokens"] - before["input_tokens"],
                output_tokens=after["output_tokens"] - before["output_tokens"],
                n_prompts=n_prompts,
            )
        )
        return responses

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

        started = time.perf_counter()
        try:
            response = httpx.post(
                self._score_endpoint(),
                json={"model": self.model, "prompt": context, "continuation": continuation},
                headers=self._headers(),
                timeout=self.call_timeout_s,
            )
            scored = self._parse_score(response)
        except Exception as error:
            self._record(
                CallRecord(
                    operation=SCORE,
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        self._record(
            CallRecord(
                operation=SCORE,
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                # A probe is billed for everything it sends, which is the prompt and
                # the continuation both, once per candidate-reference pair. It decodes
                # nothing, so it has no output tokens at all, and what it measures is
                # the continuation alone.
                input_tokens=approximate_tokens(context) + approximate_tokens(continuation),
                output_tokens=0,
                scored_tokens=scored.token_count,
            )
        )
        return scored

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
            RateLimitedError: On a 429, carrying ``Retry-After`` when the endpoint
                sent one.
            ValueError: On any other error status or a malformed body.
        """
        if response.status_code == httpx.codes.NOT_IMPLEMENTED:
            raise ScoringUnsupportedError(_error_message(response))
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise RateLimitedError(_error_message(response), retry_after=_retry_after(response))
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


def _retry_after(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` header in seconds, if the endpoint sent a usable one.

    Only the delta-seconds form is read. The HTTP-date form is legal and no
    endpoint in this stack sends it; parsing it speculatively would add a clock
    skew problem to a wait that is already conservative.

    Args:
        response: The HTTP response.

    Returns:
        The number of seconds to wait, or ``None``.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


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
