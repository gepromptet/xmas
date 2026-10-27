"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from contextvars import ContextVar
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from promptolution.llms import api_llm as promptolution
from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from xmas.budget import SLICE_S, CancelFlag, cancel_flag
from xmas.errors import MONEY_WORDS, EmptyResponse, ErrorKind, XmasError
from xmas.llms.call_log import CallLog, CallRecord
from xmas.llms.meta import CallMeta, TokenPrice, cut_meta, meta_from_completion, system_prompt_sent
from xmas.llms.parts import Part, prompt_messages, record_text
from xmas.llms.scoring import ScoredSequence
from xmas.media import Capabilities
from xmas.tools import Tool, ToolCallRequest, turn_usage
from xmas.trace import TokenUsage
from xmas.utils import approximate_tokens, error_message, retry_after_seconds, timestamp

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "APILLM",
    "assemble",
    "parse_score",
]


# promptolution's _ainvoke_with_retries returns only the text, so the provider's
# completion id travels to CallRecord.response_id(s) in this per-request slot (each
# prompt of a batch runs as its own task); a request outside that loop writes to the
# default slot, which no one reads.
completion_id: ContextVar[list[str | None]] = ContextVar("xmas_completion_id", default=[None])  # noqa: B039

STREAM = {"stream": True, "stream_options": {"include_usage": True}}
"""What a streamed request adds to the request, after its other keys."""


@dataclass
class Progress:
    """What one streamed attempt sent and has received so far, for the meta of a call cut before it finished."""

    messages: list[dict[str, Any]] | None = None
    text: list[str] = field(default_factory=list)
    model: str | None = None
    started: float = field(default_factory=time.perf_counter)


# ask's thread sets an attempt's progress; the request coroutine, scheduled from
# that thread, runs in a copy of its context and fills it.
current_progress: ContextVar[Progress | None] = ContextVar("xmas_progress", default=None)

IMAGE_MODALITIES = ("image", "text")
"""What an image request asks for: OpenRouter's ``modalities``, images first."""

DATA_URI = re.compile(r"data:([^;,]+);base64,(.*)", re.DOTALL)


class Completion(str):
    """A response string that remembers which provider completion it came from."""

    completion_id: str | None

    def __new__(cls, text: str, completion_id: str | None) -> Completion:
        """Wrap ``text``, attaching ``completion_id``."""
        instance = super().__new__(cls, text)
        instance.completion_id = completion_id
        return instance


class APILLM(promptolution.APILLM):
    """promptolution's APILLM plus :meth:`score`, tool turns, call metadata and a per-call log.

    :meth:`get_response` is promptolution's batching, concurrency and retries,
    timed and logged. :meth:`generate_with_meta` sends the same request and
    keeps what the provider reported (tokens, cost, finish reason, served
    model, completion id, and the chosen tokens' logprobs when ``call_kwargs``
    asked for them) as a :class:`~xmas.llms.meta.CallMeta`;
    :meth:`get_tool_response_meta` does the same for a tool turn,
    :meth:`generate_parts_with_meta` for a prompt with pictures or sound in it,
    and :meth:`generate_image_with_meta` for an answer with images. Each sends a
    permanent client error once and retries the rest with promptolution's
    attempt count and backoff: any 4xx but 408, 409 and 429, and any 4xx whose
    message or body names money (:data:`~xmas.errors.MONEY_WORDS`), is
    permanent. Each asks an empty answer again, unless the model was cut or
    refused; every paid attempt is summed into the meta.

    A call made in a run ends within a moment once the run is stopped
    (:func:`~xmas.budget.cancel_flag`): the request is cancelled, which closes
    its connection, and :class:`~xmas.errors.Stopped` carries what the call
    used. Only a streamed call (``stream=True``) keeps the answer received
    before the cut.

    Attributes:
        full_response: Whether the engine reads this model through
            :meth:`generate_with_meta` and :meth:`get_tool_response_meta`.
        stream: Whether requests ask for the answer as a stream of chunks.
        price: The price used for a call whose response reports no cost.
        accepts: The kinds the model reads, as stated: ``"text"``, and
            ``"image"`` or ``"audio"`` for a model that sees or hears.
        produces: The kinds the model answers with, as stated: ``"text"``, and
            ``"image"`` for a model that draws.
        system_prompt: The system prompt every call of this model is sent
            when its caller names none: the one given, else promptolution's
            ``DEFAULT_SYS_PROMPT``.
    """

    def __init__(  # noqa: PLR0913, PLR0917  the public constructor, kept for version 1 callers
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        call_log: CallLog | None = None,
        backend_tag: str = "gateway",
        full_response: bool | None = None,  # noqa: FBT001  public positional parameter of the constructor
        price: TokenPrice | None = None,
        accepts: Sequence[str] = ("text",),
        produces: Sequence[str] = ("text",),
        stream: bool = False,  # noqa: FBT001, FBT002  public positional parameter of the constructor
        seed: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            model: Model identifier, e.g. ``"llama-3.1-8b"``; promptolution's ``model_id``.
            base_url: Endpoint base URL, promptolution's ``api_url``. Defaults to ``XMAS_BASE_URL``.
            api_key: API key. Defaults to ``XMAS_API_KEY``.
            call_log: Collector for per-call statistics; ``None`` records nothing.
            backend_tag: What this backend calls itself in a record.
            full_response: ``True`` sends plain agents through
                :meth:`generate_with_meta`, ``False`` keeps promptolution's path,
                and ``None`` decides by class, once, at construction: on, unless
                the class overrides one of promptolution's call hooks or
                ``get_tool_response``, so a subclass keeps the path it hooked.
            price: USD per million tokens, used only when a response reports
                no cost. ``None`` leaves such a cost unknown.
            accepts: The kinds the model reads; a MAS sends media only to a
                model that accepts their kind.
            produces: The kinds the model answers with.
            stream: ``True`` adds ``"stream": true`` and ``"stream_options":
                {"include_usage": true}`` to each request and builds the
                answer from its chunks, so a cut call keeps its text so far;
                ``False`` sends promptolution's request unchanged.
            seed: Sent as ``seed`` on every request when set. It is merged into
                ``call_kwargs`` ahead of the given entries, so a ``seed`` in
                ``call_kwargs`` wins; both land in a record's ``params``.
            temperature: Sent as ``temperature`` on every request when set,
                merged as ``seed`` is.
            system_prompt: The system prompt sent on every call whose caller
                names none, in place of promptolution's ``DEFAULT_SYS_PROMPT``.
                A call's record and the agent's record keep the one sent.
            **kwargs: Forwarded to promptolution's ``APILLM``.

        Raises:
            ValueError: If no model is given, no base URL can be resolved, or ``accepts`` or ``produces`` is not
                a list of kinds a model may take or give.
        """
        base_url = base_url or os.environ.get("XMAS_BASE_URL")
        if not model:
            raise ValueError("APILLM needs a model, e.g. APILLM(model='llama-3.1-8b').")
        if not base_url:
            raise ValueError("APILLM needs a base_url, or XMAS_BASE_URL set in the environment.")
        self.call_log = call_log
        self.backend_tag = backend_tag
        self.full_response: bool = not overrides_call_path(type(self)) if full_response is None else full_response
        self.price = price
        self.stream = stream
        stated = Capabilities.parse({"accepts": accepts, "produces": produces}, "APILLM")
        self.accepts, self.produces = stated.accepts, stated.produces
        self.system_prompt: str = DEFAULT_SYS_PROMPT if system_prompt is None else system_prompt
        sampling = {key: value for key, value in (("seed", seed), ("temperature", temperature)) if value is not None}
        if sampling:
            kwargs["call_kwargs"] = {**sampling, **(kwargs.get("call_kwargs") or {})}
        api_key = api_key or os.environ.get("XMAS_API_KEY")
        super().__init__(model_id=model, api_url=base_url.rstrip("/"), api_key=api_key, **kwargs)

    def record(self, record: CallRecord) -> None:
        """Hand one call to the collector, if there is one."""
        if self.call_log is not None:
            self.call_log.record(record)

    def log_call(
        self, operation: str, started: float, started_at: str, *, error: BaseException | None = None, **fields: Any
    ) -> None:
        """Record one call that started at ``started``; ``error`` marks it failed."""
        if error is not None:
            fields.update(ok=False, error=f"{type(error).__name__}: {error}")
        self.record(
            CallRecord(
                operation=operation,
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                ts=started_at,
                **fields,
            )
        )

    def log_answer(
        self, started: float, started_at: str, meta: CallMeta, estimate: TokenUsage, **fields: Any
    ) -> TokenUsage:
        """Record an answered ``generate`` call with the provider's token counts, else ``estimate``; return them."""
        tokens = estimate if meta.tokens is None else meta.tokens
        fields.setdefault("params", self.call_params())
        fields.setdefault("logprobs", meta.logprobs)
        self.log_call(
            "generate",
            started,
            started_at,
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            **fields,
        )
        return tokens

    def call_params(self, system_prompt: str | None = None) -> dict[str, Any]:
        """Return what a generation is sent besides the messages, merged as promptolution merges it.

        ``system_prompt`` is the one sent, the model's own when ``None``; it
        is a ``system_prompt`` entry only when it is not promptolution's
        default, so a default model's params read as before.
        """
        params: dict[str, Any] = {"max_tokens": self.max_tokens}
        params.update(self._call_kwargs)
        # Headers can carry credentials, so they never reach a record.
        params.pop("extra_headers", None)
        if self.stream:
            params.update(STREAM)
        sent = self.system_prompt if system_prompt is None else system_prompt
        if sent != DEFAULT_SYS_PROMPT:
            params["system_prompt"] = sent
        return params

    async def _ainvoke_once(self, prompt: str, system_prompt: str) -> ChatCompletion:
        """Make promptolution's request, noting the provider's completion id for the caller."""
        completion = await self.send_once(self.build_request(prompt_messages(prompt, system_prompt)))
        completion_id.get()[0] = completion.id
        return completion

    async def _ainvoke_with_retries(self, prompt: str, system_prompt: str) -> str:
        """Run promptolution's retry loop and tag the answer with the id of the attempt that returned it."""
        slot: list[str | None] = [None]
        token = completion_id.set(slot)
        try:
            content = await super()._ainvoke_with_retries(prompt, system_prompt)
        finally:
            completion_id.reset(token)
        return Completion(content, slot[0])

    def get_response(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None = None,
    ) -> list[str]:
        """Generate through promptolution, timing the call and recording what it cost.

        Token counts are promptolution's whitespace-word approximation. A batch
        is one call and stays one record. Without ``system_prompts`` each
        prompt is sent the model's :attr:`system_prompt`.
        """
        if system_prompts is None:
            system_prompts = self.system_prompt
        n_prompts = 1 if isinstance(prompts, str) else len(prompts)
        params = self.call_params()
        before = self.get_token_count()
        started_at = timestamp()
        started = time.perf_counter()
        try:
            tagged = super().get_response(prompts, system_prompts)
        except Exception as error:
            self.log_call(
                "generate",
                started,
                started_at,
                error=error,
                n_prompts=n_prompts,
                params=params,
                **self.call_provenance(prompts, system_prompts, None, None),
            )
            raise
        after = self.get_token_count()
        response_ids = [getattr(response, "completion_id", None) for response in tagged]
        responses = [str(response) for response in tagged]  # plain strings, not the tagged subclass
        self.log_call(
            "generate",
            started,
            started_at,
            input_tokens=after["input_tokens"] - before["input_tokens"],
            output_tokens=after["output_tokens"] - before["output_tokens"],
            n_prompts=n_prompts,
            params=params,
            **self.call_provenance(prompts, system_prompts, responses, response_ids),
        )
        return responses

    def call_provenance(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None,
        responses: list[str] | None,
        response_ids: list[str | None] | None,
    ) -> dict[str, Any]:
        """Return a generation record's id fields, and its text when the collector asked for it.

        Singular fields for a one-prompt call, list fields for a batch; the
        system prompts are resolved as :meth:`get_response` resolves them.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        known_ids = response_ids if response_ids and any(i is not None for i in response_ids) else None
        capture = self.call_log is not None and self.call_log.capture_text
        if system_prompts is None:
            system_prompts = self.system_prompt
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
        """:meth:`get_tool_response_meta` without the metadata."""
        text, requests, _ = self.get_tool_response_meta(messages, tools)
        return text, requests

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest], CallMeta]:
        """Run one chat turn that may request tools, and keep its metadata.

        The request carries ``model``, ``messages``, ``max_tokens`` and the
        tools; sampling parameters are sent only when set on the model
        (``seed``, ``temperature``) or in ``call_kwargs``, so a route that
        refuses them still serves the call. Empty ``tools`` omits the key,
        which forces a text answer. The system prompt is the caller's, the
        first of ``messages``.

        Returns:
            The message text (``""`` when the model only called tools), the
            requested calls in order, and the turn's metadata.

        Raises:
            EmptyResponse: If no attempt's response had a choice, or the model
                was cut or refused before it wrote any text or call.
        """
        request = self.build_request(messages)
        if tools:
            request["tools"] = [spec.to_openai() for spec in tools]
        params = self.call_params(system_prompt_sent(messages) or DEFAULT_SYS_PROMPT)
        started_at = timestamp()
        started = time.perf_counter()
        try:
            completion, meta = self.ask_until_answered(request, "tool turn", tool_turn=True)
        except Exception as error:
            self.log_call("generate", started, started_at, error=error, params=params)
            raise
        message = completion.choices[0].message
        text = message.content or ""
        requests = [
            ToolCallRequest(call.id, call.function.name, call.function.arguments or "{}")
            for call in message.tool_calls or []
            if call.type == "function"
        ]
        estimate = turn_usage(approximate_tokens, messages, text, requests)
        tokens = self.log_answer(started, started_at, meta, estimate, params=params, response_id=completion.id)
        self.input_token_count += tokens.input_tokens
        self.output_token_count += tokens.output_tokens
        return text, requests, meta

    def generate_with_meta(self, prompt: str, system_prompt: str | None = None) -> tuple[str, CallMeta]:
        """Answer one prompt with promptolution's own request, and keep its metadata.

        One ``generate`` record per call, failed or not, whatever its attempts;
        the token counters move as after :meth:`get_response`. Without a
        ``system_prompt`` the model's own (:attr:`system_prompt`) is sent.

        Raises:
            EmptyResponse: If the answer has no content: at once, with kind
                ``length_cut`` or ``refusal``, when the model was cut or
                refused; otherwise after the last attempt.
            TimeoutError: If the call outlives ``gather_timeout_s + 5``.
        """
        _, text, meta = self.send_generation(prompt, system_prompt)
        return text, meta

    def generate_parts_with_meta(self, parts: Sequence[Part], system_prompt: str | None = None) -> tuple[str, CallMeta]:
        """Answer a prompt of text and media parts, as :meth:`generate_with_meta` answers text.

        The request has the same keys (``model``, ``messages``, ``max_tokens``,
        then ``call_kwargs``); the user content is the parts'
        :func:`~xmas.llms.parts.to_content`, or the plain string when no part is
        media. The record's prompt shows a marker per ref, never its bytes.

        Raises:
            ValueError: If a part cannot be sent (see :func:`~xmas.llms.parts.to_content`).
            EmptyResponse: As :meth:`generate_with_meta`.
            TimeoutError: As :meth:`generate_with_meta`.
        """
        _, text, meta = self.send_generation(parts, system_prompt)
        return text, meta

    def generate_image_with_meta(
        self, prompt: str | Sequence[Part], system_prompt: str | None = None
    ) -> tuple[str, list[tuple[bytes, str]], CallMeta]:
        """Ask for an answer with images, and keep its metadata.

        The request adds ``"modalities": ["image", "text"]`` after
        ``max_tokens`` (``call_kwargs`` still override it). The images are read
        from the message's ``images`` list of ``data:`` URIs, as OpenRouter
        sends them; images without text are an answer, not an empty one.

        Returns:
            The answer's text (``""`` when it has none), each image as
            ``(bytes, media type)`` in order, and the call's metadata.

        Raises:
            EmptyResponse: Of kind ``bad_format``, with the paid call's meta, if an image is not a base64
                ``data:`` URI; and if the answer has neither text nor images, as :meth:`generate_with_meta`.
            ValueError: If a part cannot be sent (see :func:`~xmas.llms.parts.to_content`).
            TimeoutError: As :meth:`generate_with_meta`.
        """
        completion, text, meta = self.send_generation(prompt, system_prompt, modalities=list(IMAGE_MODALITIES))
        try:
            images = [decode_image(url) for url in image_urls(completion.choices[0].message)]
        except XmasError as error:  # the call was paid: the meta goes with the error to the meter
            raise EmptyResponse(str(error), kind=ErrorKind.BAD_FORMAT, meta=meta) from None
        return text, images, meta

    def build_request(self, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        """Return promptolution's request for ``messages``, with ``extra`` keys before ``call_kwargs``."""
        request: dict[str, Any] = {"model": self.model_id, "messages": messages, "max_tokens": self.max_tokens}
        request.update(extra)
        request.update(self._call_kwargs)
        return request

    def send_generation(
        self, prompt: str | Sequence[Part], system_prompt: str | None, **extra: Any
    ) -> tuple[ChatCompletion, str, CallMeta]:
        """Send one generation with ``extra`` request keys, record it, and move the token counters by its text.

        ``system_prompt`` is the caller's, the model's own when ``None``.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt
        request = self.build_request(prompt_messages(prompt, system_prompt), **extra)
        shown = prompt if isinstance(prompt, str) else record_text(prompt)
        counted = prompt if isinstance(prompt, str) else "".join(part for part in prompt if isinstance(part, str))
        logged = {"params": self.call_params(system_prompt)}
        for key, value in extra.items():
            logged["params"].setdefault(key, value)
        started_at = timestamp()
        started = time.perf_counter()
        try:
            completion, meta = self.ask_until_answered(request, "model call", images="modalities" in extra)
        except Exception as error:
            fields = self.call_provenance(shown, system_prompt, None, None)
            self.log_call("generate", started, started_at, error=error, **logged, **fields)
            raise
        text = completion.choices[0].message.content or ""
        self.update_token_count([counted, system_prompt], [text])
        estimate = TokenUsage(approximate_tokens(counted) + approximate_tokens(system_prompt), approximate_tokens(text))
        fields = self.call_provenance(shown, system_prompt, [text], [completion.id])
        self.log_answer(started, started_at, meta, estimate, **logged, **fields)
        return completion, text, meta

    def ask_until_answered(
        self, request: dict[str, Any], what: str, *, tool_turn: bool = False, images: bool = False
    ) -> tuple[ChatCompletion, CallMeta]:
        """Send ``request`` until the response has an answer, summing the paid attempts into one meta.

        A plain answer needs content, or with ``images`` an image; a tool turn
        needs a choice, and its text may be empty when it calls tools or
        finished cleanly. An empty answer
        that was cut or refused raises at once, since the same request would
        only repeat it. Inside a metered call the waits end once the call's
        :func:`~xmas.budget.cancel_flag` is set (:meth:`wait_for_attempt`).
        The meta keeps the system prompt the request sent, when it is not
        promptolution's default.
        """
        paid: CallMeta | None = None
        sent = system_prompt_sent(request.get("messages") or [])
        flag = cancel_flag()
        for attempt in range(self.max_retries + 1):
            progress = Progress()
            token = current_progress.set(progress)
            try:
                future = asyncio.run_coroutine_threadsafe(self.send_with_retries(request), self._loop)
            finally:
                current_progress.reset(token)
            completion = self.wait_for_attempt(future, what, flag, progress, paid)
            meta = meta_from_completion(completion, time.perf_counter() - progress.started, self.price, sent)
            paid = meta if paid is None else paid.combine(meta)
            if completion.choices:
                message = completion.choices[0].message
                empty = (
                    message.content is None
                    and not (tool_turn and message.tool_calls)
                    and not (images and image_urls(message))
                )
                if empty and meta.error_kind is not None:
                    raise EmptyResponse(
                        f"Empty content from model (finish_reason={meta.finish_reason}, refused={meta.refused})",
                        kind=meta.error_kind,
                        meta=paid,
                    )
                if tool_turn or not empty:
                    return completion, paid
            if attempt < self.max_retries:
                delay = self.retry_base_delay_s * (2**attempt)
                if flag is None:
                    time.sleep(delay)
                elif flag.wait(delay):
                    flag.check(paid)
        raise EmptyResponse(f"Empty content from model after {self.max_retries + 1} attempts", meta=paid)

    def wait_for_attempt(
        self,
        future: Future[ChatCompletion],
        what: str,
        flag: CancelFlag | None,
        progress: Progress,
        paid: CallMeta | None,
    ) -> ChatCompletion:
        """Wait for one attempt until promptolution's batch deadline, checking ``flag`` every :data:`SLICE_S`.

        Raises:
            Stopped: Once ``flag`` is set; the request is cancelled, and the
                error carries the paid attempts plus the cut one's meta
                (:func:`~xmas.llms.meta.cut_meta`) and the text it received.
            TimeoutError: If the attempt outlives the deadline.
        """
        deadline = self.gather_timeout_s + 5.0  # as long as promptolution's _get_response waits for a batch
        ends = time.monotonic() + deadline
        while True:
            try:
                return future.result(timeout=deadline if flag is None else min(SLICE_S, ends - time.monotonic()))
            except FuturesTimeout:
                if flag is not None and flag.is_set():
                    self.stop_attempt(future, flag, progress, paid)
                if flag is None or time.monotonic() >= ends:
                    future.cancel()
                    raise TimeoutError(f"{what} timed out after {deadline}s") from None

    def stop_attempt(
        self, future: Future[ChatCompletion], flag: CancelFlag, progress: Progress, paid: CallMeta | None
    ) -> None:
        """Cancel a stopped attempt and raise :class:`~xmas.errors.Stopped` with what it used and received."""
        future.cancel()
        received = "".join(progress.text)
        tokens = None if progress.messages is None else turn_usage(approximate_tokens, progress.messages, received, [])
        cut = cut_meta(time.perf_counter() - progress.started, tokens, progress.model, self.price)
        flag.check(cut if paid is None else paid.combine(cut), received)

    async def send_with_retries(self, request: dict[str, Any]) -> ChatCompletion:
        """Make ``request``, retrying with promptolution's backoff unless the error is permanent."""
        attempt = 0
        while True:
            try:
                return await self.send_once(request)
            except Exception as exc:
                if is_permanent(exc) or attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.retry_base_delay_s * (2**attempt))
                attempt += 1

    async def send_once(self, request: dict[str, Any]) -> ChatCompletion:
        """Make ``request`` once, in a slot of promptolution's semaphore, within its per-call timeout."""
        async with self._sem:
            return await asyncio.wait_for(self.fetch_completion(request), timeout=self.call_timeout_s)

    async def fetch_completion(self, kwargs: dict[str, Any]) -> ChatCompletion:
        """Send ``kwargs``; when streaming, with :data:`STREAM` added, the chunks assembled (:func:`assemble`)."""
        if not self.stream:
            completion: ChatCompletion = await self.client.chat.completions.create(**kwargs)
            return completion
        progress = current_progress.get() or Progress()
        progress.messages = kwargs["messages"]
        progress.text.clear()  # a retry starts its answer anew
        chunks: list[ChatCompletionChunk] = []
        stream = await self.client.chat.completions.create(**{**kwargs, **STREAM})  # as call_params records it
        try:
            async for chunk in stream:
                chunks.append(chunk)
                progress.model = chunk.model or progress.model
                if chunk.choices and chunk.choices[0].delta.content:
                    progress.text.append(chunk.choices[0].delta.content)
        finally:
            await stream.close()
        return assemble(chunks)

    @property
    def model(self) -> str:
        """The model identifier, under the name the xmas surface uses."""
        return str(self.model_id)

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        Teacher forcing: the continuation is fixed and scored, not generated, through
        the gateway's ``/score`` endpoint next to ``base_url``.

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
            self.log_call("score", started, started_at, error=error)
            raise
        # A probe is billed for the prompt and the continuation both, decodes
        # nothing, and measures the continuation alone.
        self.log_call(
            "score",
            started,
            started_at,
            input_tokens=approximate_tokens(context) + approximate_tokens(continuation),
            output_tokens=0,
            scored_tokens=scored.token_count,
        )
        return scored


def assemble(chunks: Sequence[ChatCompletionChunk]) -> ChatCompletion:
    """Build the chat completion a streamed answer's chunks make, as the provider would have sent it whole.

    The text, refusal and each tool call's arguments are joined (calls by
    their index), OpenRouter's ``images`` collected; the finish reason, usage,
    a top-level ``cost``, the model and the id are the last ones sent. A
    stream with no text has ``None`` content, as a whole answer without any.
    """
    text: list[str] = []
    refusal: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    images: list[Any] = []
    body: dict[str, Any] = {"id": "", "object": "chat.completion", "created": 0, "model": ""}
    finish: str | None = None
    chose = False
    for chunk in chunks:
        body.update(
            id=chunk.id or body["id"], created=chunk.created or body["created"], model=chunk.model or body["model"]
        )
        extra = chunk.model_extra or {}
        if "cost" in extra:
            body["cost"] = extra["cost"]
        if chunk.usage is not None:
            body["usage"] = chunk.usage.model_dump()
        if not chunk.choices:
            continue
        chose = True
        choice = chunk.choices[0]
        delta = choice.delta
        text.append(delta.content or "")
        refusal.append(delta.refusal or "")
        images.extend((delta.model_extra or {}).get("images") or [])
        for call in delta.tool_calls or []:
            entry = calls.setdefault(
                call.index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
            )
            entry["id"] = call.id or entry["id"]
            entry["type"] = call.type or entry["type"]
            if call.function is not None:
                entry["function"]["name"] += call.function.name or ""
                entry["function"]["arguments"] += call.function.arguments or ""
        finish = choice.finish_reason or finish
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text) or None,
        "refusal": "".join(refusal) or None,
    }
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    if images:
        message["images"] = images
    body["choices"] = [{"index": 0, "finish_reason": finish, "message": message}] if chose else []
    return ChatCompletion.construct(**body)


def image_urls(message: ChatCompletionMessage) -> list[str]:
    """Return the image URLs of an answer's ``images`` list, which the client keeps as an extra field."""
    entries = (message.model_extra or {}).get("images") or []
    urls = []
    for entry in entries:
        image_url = entry.get("image_url") if isinstance(entry, dict) else None
        url = image_url.get("url") if isinstance(image_url, dict) else None
        if isinstance(url, str):
            urls.append(url)
    return urls


def decode_image(url: str) -> tuple[bytes, str]:
    """Return a ``data:`` URI's bytes and media type.

    Raises:
        XmasError: For any other URL, which xmas would have to fetch, or invalid base64.
    """
    found = DATA_URI.fullmatch(url)
    if found is None:
        raise XmasError(f"llm: an image answer is not a base64 data URI ({url[:40]}...); xmas reads no network")
    try:
        return base64.b64decode(found.group(2), validate=True), found.group(1)
    except binascii.Error as exc:
        raise XmasError(f"llm: an image answer's base64 is invalid: {exc}") from None


def is_permanent(exc: BaseException) -> bool:
    """Whether retrying cannot help: a 4xx other than 408, 409 and 429, or a 4xx that names money."""
    if not isinstance(exc, openai.APIStatusError) or not (
        HTTPStatus.BAD_REQUEST <= exc.status_code < HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        return False
    text = str(exc) if exc.body is None else f"{exc} {exc.body}"
    return exc.status_code not in (408, 409, 429) or MONEY_WORDS.search(text) is not None


CALL_PATH = (
    "get_response",
    "_get_response",
    "_aget_batch",
    "_ainvoke_with_retries",
    "_ainvoke_once",
    "get_tool_response",
)
"""The hooks whose override keeps a class on promptolution's path by default."""


def overrides_call_path(cls: type) -> bool:
    """Whether ``cls`` replaces a hook of :data:`CALL_PATH`."""
    return any(getattr(cls, name) is not getattr(APILLM, name) for name in CALL_PATH)


def parse_score(response: httpx.Response) -> ScoredSequence:
    """Turn a scoring response into a :class:`ScoredSequence`.

    Raises:
        XmasError: On an error status or a body without tokens/token_logprobs.
    """
    status = response.status_code
    if status == HTTPStatus.NOT_IMPLEMENTED:
        raise XmasError(f"llm: scoring unsupported (501): {error_message(response)}", status=501)
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        retry_after = retry_after_seconds(response.headers)
        message = f"llm: rate limited (429), retry after {retry_after}s: {error_message(response)}"
        raise XmasError(message, status=429, retry_after=retry_after)
    if status >= HTTPStatus.BAD_REQUEST:
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
