"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

import httpx
from openai.types.chat import ChatCompletion
from promptolution.llms import api_llm as promptolution
from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from xmas.errors import XmasError
from xmas.llms.call_log import CallLog, CallRecord, approximate_tokens, timestamp
from xmas.llms.scoring import ScoredSequence
from xmas.tools import Tool, ToolCallRequest, complete_with_tools
from xmas.utils import error_message, retry_after_seconds

# Completion-id plumbing: _completion_id, _Completion and the _ainvoke_once /
# _ainvoke_with_retries overrides below exist only because promptolution's
# _ainvoke_with_retries returns the response text and drops the ChatCompletion,
# and with it the provider's completion id that CallRecord.response_id(s) carry.
# Temporary: delete all of it once promptolution returns the completion (or its id).
_completion_id: ContextVar[list[str | None] | None] = ContextVar("xmas_completion_id", default=None)
"""Per-request slot for the provider's completion id.

promptolution keeps only a completion's text, and each prompt of a batch runs as its
own asyncio task, so a context variable is the one place the id can travel from the
request that saw it to the response it belongs to without reimplementing the batch.
"""


class _Completion(str):
    """A response string that remembers which provider completion it came from."""

    completion_id: str | None

    def __new__(cls, text: str, completion_id: str | None) -> _Completion:
        """Wrap ``text``, attaching ``completion_id``."""
        instance = super().__new__(cls, text)
        instance.completion_id = completion_id
        return instance


class RateLimitedError(RuntimeError):
    """Raised by xmas.tools.mcp on a 429; goes once mcp raises XmasError.

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


class APILLM(promptolution.APILLM):
    """promptolution's APILLM plus :meth:`score`, tool turns and a per-call log.

    ``model`` and ``base_url`` are forwarded to promptolution's ``model_id`` and ``api_url``.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        call_log: CallLog | None = None,
        backend_tag: str = "gateway",
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            model: Model identifier, e.g. ``"llama-3.1-8b"``.
            base_url: Endpoint base URL. Defaults to ``XMAS_BASE_URL``.
            api_key: API key. Defaults to ``XMAS_API_KEY``.
            call_log: Collector for per-call statistics; ``None`` records nothing.
            backend_tag: What this backend calls itself in a record.
            **kwargs: Forwarded to promptolution's ``APILLM``.

        Raises:
            ValueError: If no model is given, or no base URL can be resolved.
        """
        base_url = base_url or os.environ.get("XMAS_BASE_URL")
        if not model:
            raise ValueError("APILLM needs a model, e.g. APILLM(model='llama-3.1-8b').")
        if not base_url:
            raise ValueError("APILLM needs a base_url, or XMAS_BASE_URL set in the environment.")
        self.call_log = call_log
        self.backend_tag = backend_tag
        api_key = api_key or os.environ.get("XMAS_API_KEY")
        super().__init__(model_id=model, api_url=base_url.rstrip("/"), api_key=api_key, **kwargs)

    def _record(self, record: CallRecord) -> None:
        """Hand one call to the collector, if there is one."""
        if self.call_log is not None:
            self.call_log.record(record)

    def _call_params(self) -> dict[str, Any]:
        """Return what a generation is sent besides the messages, merged as promptolution merges it."""
        params: dict[str, Any] = {"max_tokens": self.max_tokens}
        params.update(self._call_kwargs)
        # Headers can carry credentials, so they never reach a record.
        params.pop("extra_headers", None)
        return params

    async def _ainvoke_once(self, prompt: str, system_prompt: str) -> ChatCompletion:
        """Make one request, noting the provider's completion id for the caller."""
        completion = await super()._ainvoke_once(prompt, system_prompt)
        slot = _completion_id.get()
        if slot is not None:
            slot[0] = completion.id
        return completion

    async def _ainvoke_with_retries(self, prompt: str, system_prompt: str) -> str:
        """Run promptolution's retry loop and tag the answer with its completion id.

        The slot is overwritten by every attempt that reaches the provider, and the
        attempt that returns is the last one, so the id that survives belongs to the
        text that is returned.
        """
        slot: list[str | None] = [None]
        token = _completion_id.set(slot)
        try:
            content = await super()._ainvoke_with_retries(prompt, system_prompt)
        finally:
            _completion_id.reset(token)
        return _Completion(content, slot[0])

    def get_response(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None = None,
    ) -> list[str]:
        """Generate, timing the call and recording what it cost.

        The generation itself is promptolution's; this adds the measurement.
        Token counts are read from the parent's own counters, which split on
        whitespace, so they are an approximation and are reported as one.

        A batch is one call and stays one record. Its provenance (start time,
        sampling parameters, completion ids) is always recorded; its text only
        when the collector was built with ``capture_text=True``.

        Args:
            prompts: Prompt or prompts to answer.
            system_prompts: System prompt or prompts.

        Returns:
            One response per prompt.
        """
        n_prompts = 1 if isinstance(prompts, str) else len(prompts)
        params = self._call_params()
        before = self.get_token_count()
        started_at = timestamp()
        started = time.perf_counter()
        try:
            tagged = super().get_response(prompts, system_prompts)
        except Exception as error:
            self._record(
                CallRecord(
                    operation="generate",
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    n_prompts=n_prompts,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                    ts=started_at,
                    params=params,
                    **self._provenance(prompts, system_prompts, None, None),
                )
            )
            raise
        latency = time.perf_counter() - started
        after = self.get_token_count()
        response_ids = [getattr(response, "completion_id", None) for response in tagged]
        # Hand the caller ordinary strings, not the tagged subclass.
        responses = [str(response) for response in tagged]
        self._record(
            CallRecord(
                operation="generate",
                model=self.model,
                backend=self.backend_tag,
                latency_s=latency,
                input_tokens=after["input_tokens"] - before["input_tokens"],
                output_tokens=after["output_tokens"] - before["output_tokens"],
                n_prompts=n_prompts,
                ts=started_at,
                params=params,
                **self._provenance(prompts, system_prompts, responses, response_ids),
            )
        )
        return responses

    def _provenance(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None,
        responses: list[str] | None,
        response_ids: list[str | None] | None,
    ) -> dict[str, Any]:
        """Return the text and id fields of a generation record.

        Args:
            prompts: The prompts as the caller passed them.
            system_prompts: The system prompts as the caller passed them, resolved
                here exactly as promptolution resolves them before sending.
            responses: The responses, or ``None`` when the call failed.
            response_ids: The provider's completion ids, aligned to ``responses``.

        Returns:
            Keyword arguments for :class:`CallRecord`: singular fields for a
            one-prompt call, list fields for a batch. The text is present only when
            the collector asked for it.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        known_ids = response_ids if response_ids and any(i is not None for i in response_ids) else None
        capture = self.call_log is not None and self.call_log.capture_text
        if system_prompts is None:
            system_prompts = DEFAULT_SYS_PROMPT
        system_list = [system_prompts] * len(prompt_list) if isinstance(system_prompts, str) else list(system_prompts)

        if len(prompt_list) == 1:
            fields: dict[str, Any] = {"response_id": known_ids[0] if known_ids else None}
            if capture:
                fields["prompt"] = prompt_list[0]
                fields["system_prompt"] = system_list[0] if system_list else None
                fields["response"] = responses[0] if responses else None
            return fields
        fields = {"response_ids": known_ids}
        if capture:
            fields["prompts"] = prompt_list
            fields["system_prompts"] = system_list
            fields["responses"] = responses
        return fields

    def get_tool_response(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest]]:
        """Run one chat turn that may request tools.

        Scheduled on promptolution's background loop and semaphore, with its
        timeout and backoff, and recorded like :meth:`get_response`. Token counts
        are the provider's when it reports usage.

        Args:
            messages: OpenAI-shaped history (system, user, assistant with
                ``tool_calls``, tool).
            tools: Tools on offer. Empty omits the ``tools`` key, which forces a
                text answer.

        Returns:
            The message text (``""`` when the model only called tools) and the
            requested calls, in order.
        """
        return complete_with_tools(self, messages, tools)

    @property
    def model(self) -> str:
        """The model identifier, under the name the xmas surface uses."""
        return str(self.model_id)

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        Teacher forcing: the continuation is fixed and scored, not generated, through
        the gateway's ``/score`` endpoint next to ``base_url``.

        Args:
            context: The text the continuation is conditioned on.
            continuation: The exact text to score.

        Returns:
            The continuation's per-token and aggregate log-probabilities.

        Raises:
            ValueError: If the continuation is empty.
            XmasError: If the endpoint refuses (501 unsupported, 429 rate limited,
                any other 4xx/5xx) or the body is malformed.
        """
        if not continuation:
            raise ValueError("Cannot score an empty continuation.")

        started_at = timestamp()
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                f"{self.api_url}/score",
                json={"model": self.model, "prompt": context, "continuation": continuation},
                headers=headers,
                timeout=self.call_timeout_s,
            )
            scored = parse_score(response)
        except Exception as error:
            self._record(
                CallRecord(
                    operation="score",
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                    ts=started_at,
                )
            )
            raise
        self._record(
            CallRecord(
                operation="score",
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
                ts=started_at,
            )
        )
        return scored


def parse_score(response: httpx.Response) -> ScoredSequence:
    """Turn a scoring response into a :class:`ScoredSequence`.

    Raises:
        XmasError: On an error status or a body without tokens/token_logprobs.
    """
    status = response.status_code
    if status == 501:
        raise XmasError(f"llm: scoring unsupported (501): {error_message(response)}", status=501)
    if status == 429:
        retry_after = retry_after_seconds(response)
        message = f"llm: rate limited (429), retry after {retry_after}s: {error_message(response)}"
        raise XmasError(message, status=429, retry_after=retry_after)
    if status >= 400:
        raise XmasError(f"llm: scoring failed ({status}): {error_message(response)}", status=status)
    body = response.json()
    tokens = body.get("tokens")
    logprobs = body.get("token_logprobs")
    if not isinstance(tokens, list) or not isinstance(logprobs, list):
        raise XmasError(f"llm: scoring response missing tokens/token_logprobs: {body!r}")
    return ScoredSequence(
        tokens=tuple(str(token) for token in tokens),
        token_logprobs=tuple(float(value) for value in logprobs),
    )
