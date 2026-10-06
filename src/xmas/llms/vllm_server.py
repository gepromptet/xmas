"""Teacher-forced scoring against a self-hosted vLLM server.

This is an *API* backend, not the in-process vLLM backend the design principles
exclude: no ``vllm`` import, no torch, nothing but HTTP. It exists because the
two endpoints that can teacher-force do not agree on how to ask.
:class:`~xmas.llms.api_llm.APILLM` posts to the gateway's bespoke ``/score``
route; a vLLM server has no such route and instead echoes prompt
log-probabilities back from the OpenAI-compatible ``/completions`` route.

Subclassing rather than reimplementing keeps one object serving both operations
against one server, which is the property the experiment needs: generation and
scoring come from the same weights on the same stack, so a log-probability
ranking is never compared across vendors or quantizations.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from xmas.errors import XmasError
from xmas.llms.api_llm import APILLM
from xmas.llms.call_log import CallRecord
from xmas.llms.masking import mask_logprobs
from xmas.llms.scoring import ScoredSequence
from xmas.utils import approximate_tokens, error_message, retry_after_seconds, timestamp

ENV_BASE_URL = "XMAS_VLLM_BASE_URL"
"""Environment variable holding the vLLM server's base URL.

Separate from the gateway's ``XMAS_BASE_URL`` on purpose. The two endpoints serve
different weights, and an experiment that silently fell back from one to the
other would compare log-probabilities across stacks, which is the one thing the
serving decision exists to prevent.
"""

ENV_API_KEY = "XMAS_VLLM_API_KEY"
"""Environment variable holding the vLLM server's key, if it was started with one.

Also deliberately separate: the gateway's key is a credential for a metered
third-party service and has no business being sent to a cluster node.
"""

PLACEHOLDER_KEY = "EMPTY"
"""Bearer token used when the server was started without ``--api-key``.

vLLM ignores the header in that case, but the OpenAI client promptolution builds
refuses to construct without something, so a placeholder is what keeps a keyless
server usable. It is a stand-in for an absent credential, never a real one.
"""

DEFAULT_SCORE_BATCH = 32
"""Continuations sent per HTTP request when scoring many at once."""

BACKEND_TAG = "vllm-server"
"""What this backend calls itself in a :class:`~xmas.llms.call_log.CallRecord`.

Distinct from the gateway's tag so that probes from the two serving paths are
never pooled into one latency or cost statistic.
"""


class VLLMServerLLM(APILLM):
    """An OpenAI-compatible vLLM server that scores by echoing the prompt.

    Generation is inherited from :class:`~xmas.llms.api_llm.APILLM` untouched, so
    pointing ``base_url`` at the server is all a generation path needs. What this
    class replaces is :meth:`score`.

    The mechanism: send ``context + continuation`` as the prompt with
    ``echo=True``, ``logprobs=0`` and ``max_tokens=0``. The server decodes nothing
    and answers with one log-probability per prompt token. The continuation's own
    tokens are then cut out of that echo by
    :func:`~xmas.llms.masking.mask_logprobs`, which is the same utility any other
    backend uses and the reason the span is exact rather than approximate.

    Rendering is deliberately not this class's job. ``score`` takes a context
    string that is already whatever the model expects, chat template and think
    block included, because a scorer that applied a template would need a
    tokenizer and would stop being HTTP-only.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        score_batch_size: int = DEFAULT_SCORE_BATCH,
        **kwargs: Any,
    ) -> None:
        """Initialize the backend.

        Args:
            model: Model identifier as the server advertises it, e.g. ``"qwen3-8b"``.
            base_url: Server base URL, including the version prefix. Defaults to
                ``XMAS_VLLM_BASE_URL``, never to the gateway's variable.
            api_key: API key, if the server was started with one. Defaults to
                ``XMAS_VLLM_API_KEY``, and to a placeholder when that is unset.
            score_batch_size: Continuations per request in :meth:`score_many`.
            **kwargs: Forwarded to :class:`~xmas.llms.api_llm.APILLM`, including
                ``call_log``. ``backend_tag`` defaults to :data:`BACKEND_TAG`.

        Raises:
            ValueError: If ``score_batch_size`` is not positive, or no base URL
                can be resolved.
        """
        if score_batch_size < 1:
            raise ValueError(f"score_batch_size must be at least 1, got {score_batch_size}.")
        self.score_batch_size = score_batch_size

        resolved_url = base_url or kwargs.pop("api_url", None) or os.environ.get(ENV_BASE_URL)
        if not resolved_url:
            raise ValueError(f"VLLMServerLLM needs a base_url, or {ENV_BASE_URL} set in the environment.")
        resolved_key = api_key or os.environ.get(ENV_API_KEY) or PLACEHOLDER_KEY
        kwargs.setdefault("backend_tag", BACKEND_TAG)

        super().__init__(model=model, base_url=resolved_url, api_key=resolved_key, **kwargs)

    def _completions_endpoint(self) -> str:
        """Return the absolute URL of the echo-capable completions route."""
        return f"{self.api_url}/completions"

    def _payload(self, prompts: list[str]) -> dict[str, Any]:
        """Return the request body for an echo-only scoring call.

        ``max_tokens=0`` is what makes this a probe rather than a generation:
        the server prefills and stops. ``temperature`` is pinned to 0 so that
        nothing about sampling can enter a number that is supposed to be a
        property of the weights alone.

        Args:
            prompts: Full ``context + continuation`` strings.

        Returns:
            The JSON body.
        """
        return {
            "model": self.model,
            "prompt": prompts,
            "echo": True,
            "logprobs": 0,
            "max_tokens": 0,
            "temperature": 0.0,
        }

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        Args:
            context: The already-rendered text the continuation is conditioned on.
            continuation: The exact text to teacher-force.

        Returns:
            The continuation's per-token and aggregate log-probabilities.

        Raises:
            XmasError: On a 429 (``status`` 429), carrying ``Retry-After`` as ``retry_after`` when sent.
            ValueError: If the continuation is empty, does not align with the
                server's token boundaries, or the request is rejected.
        """
        return self.score_many([(context, continuation)])[0]

    def score_batch(self, contexts: list[str], continuations: list[str]) -> list[ScoredSequence]:
        """Score several continuations over the batched echo path.

        A vLLM server takes many prompts per request, so this delegates to
        :meth:`score_many`.

        Args:
            contexts: The conditioning texts.
            continuations: The texts to score, aligned to ``contexts``.

        Returns:
            One result per pair, in the order given.
        """
        return self.score_many(list(zip(contexts, continuations, strict=True)))

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[ScoredSequence]:
        """Score several continuations, batching them into as few requests as possible.

        A probe is prefill-only, so a batch of them is exactly the workload a
        server schedules well. Sending them one at a time would price the
        experiment's probe path at its worst case rather than its real one.

        Args:
            pairs: ``(context, continuation)`` pairs.

        Returns:
            One result per pair, in the order given.

        Raises:
            XmasError: On a 429 (``status`` 429), carrying ``Retry-After`` as ``retry_after`` when sent.
            ValueError: If any continuation is empty, does not align with token
                boundaries, or the request is rejected.
        """
        for _, continuation in pairs:
            if not continuation:
                raise ValueError("Cannot score an empty continuation.")

        results: list[ScoredSequence] = []
        for start in range(0, len(pairs), self.score_batch_size):
            chunk = list(pairs[start : start + self.score_batch_size])
            results.extend(self._score_chunk(chunk))
        return results

    def _score_chunk(self, chunk: list[tuple[str, str]]) -> list[ScoredSequence]:
        """Send one echo request and record it as one call carrying ``len(chunk)`` prompts.

        Accounting follows the gateway's probe record: billed input is context
        plus continuation for every pair, output is zero, and ``scored_tokens``
        is the exact count of continuation tokens the server returned.

        Args:
            chunk: The pairs this request carries.

        Returns:
            One result per pair, in the order given.
        """
        started_at = timestamp()
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                self._completions_endpoint(),
                json=self._payload([context + continuation for context, continuation in chunk]),
                headers=headers,
                timeout=self.call_timeout_s,
            )
            scored = self._parse_echo(response, chunk)
        except Exception as error:
            self.record(
                CallRecord(
                    operation="score",
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    n_prompts=len(chunk),
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                    ts=started_at,
                )
            )
            raise
        self.record(
            CallRecord(
                operation="score",
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                input_tokens=sum(
                    approximate_tokens(context) + approximate_tokens(continuation) for context, continuation in chunk
                ),
                output_tokens=0,
                scored_tokens=sum(result.token_count for result in scored),
                n_prompts=len(chunk),
                ts=started_at,
            )
        )
        return scored

    @staticmethod
    def _parse_echo(response: httpx.Response, pairs: list[tuple[str, str]]) -> list[ScoredSequence]:
        """Cut each continuation's own log-probabilities out of an echoed prompt.

        Args:
            response: The HTTP response.
            pairs: The pairs this response answers, in request order.

        Returns:
            One result per pair.

        Raises:
            XmasError: On a 429 (``status`` 429), carrying ``Retry-After`` as ``retry_after`` when sent.
            ValueError: On any other error status, a malformed body, a choice
                count that does not match the request, or a continuation that
                does not fall on token boundaries.
        """
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = retry_after_seconds(response.headers)
            message = f"llm: rate limited (429), retry after {retry_after}s: {error_message(response)}"
            raise XmasError(message, status=429, retry_after=retry_after)
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise ValueError(f"Scoring failed ({response.status_code}): {response.text}")

        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != len(pairs):
            raise ValueError(
                f"Scoring response carries {len(choices) if isinstance(choices, list) else 'no'} choices "
                f"for {len(pairs)} prompts. Without a one-to-one match a probe cannot be attributed "
                "to the continuation it belongs to."
            )

        # The server may answer out of order, and does index its choices, so the
        # index is honoured rather than the position assumed.
        ordered: list[dict[str, Any]] = [{}] * len(pairs)
        for position, choice in enumerate(choices):
            index = choice.get("index", position)
            if not isinstance(index, int) or not 0 <= index < len(pairs):
                raise ValueError(f"Scoring response carries an out-of-range choice index {index!r}.")
            ordered[index] = choice

        results = []
        for choice, (_, continuation) in zip(ordered, pairs, strict=True):
            logprobs = choice.get("logprobs") or {}
            tokens = logprobs.get("tokens")
            values = logprobs.get("token_logprobs")
            if not isinstance(tokens, list) or not isinstance(values, list):
                raise ValueError(
                    f"Scoring response is missing echoed tokens/token_logprobs: {choice!r}. The server "
                    "answered without `echo`, so there is nothing to teacher-force against."
                )
            span = mask_logprobs(tokens, values, continuation)
            results.append(
                ScoredSequence(
                    tokens=tuple(str(token) for token in tokens[len(tokens) - len(span) :]),
                    token_logprobs=tuple(span),
                )
            )
        return results
