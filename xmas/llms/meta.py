"""What one model call cost, how it ended, and which model served it.

promptolution keeps only a completion's text; a :class:`CallMeta` keeps the
provider's token counts, the reported (or priced) cost, the finish reason, a
refusal flag, the served model, the completion's id, its chosen-token logprobs
when the request asked for them, and the system prompt it was sent, for the
agent's :class:`~xmas.trace.AgentRecord`. A cost that is not known is ``None``,
never ``0``: a budget can only bind on a number it can trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from xmas.errors import ErrorKind, kind_of_finish
from xmas.trace import TokenUsage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from openai.types.chat import ChatCompletion
    from openai.types.completion_usage import CompletionUsage

    from xmas.llms.parts import Part
    from xmas.tools import Tool, ToolCallRequest

__all__ = [
    "CallMeta",
    "Logprobs",
    "SupportsImages",
    "SupportsMeta",
    "SupportsParts",
    "TokenPrice",
    "chosen_logprobs",
    "cut_meta",
    "meta_from_completion",
    "supports_images",
    "supports_meta",
    "supports_parts",
    "system_prompt_of",
    "system_prompt_sent",
]

Logprobs = tuple[tuple[str, float], ...]
"""The chosen token and its log-probability, per generated token."""


@dataclass(frozen=True)
class TokenPrice:
    """A model's price in USD per million tokens, for providers that report no cost.

    Reasoning tokens are already part of the completion tokens, so they are not
    priced a second time. Tokens written to a prompt cache are priced as plain
    input.

    Attributes:
        input_per_m: USD per million uncached input tokens.
        output_per_m: USD per million output tokens, reasoning included.
        cached_input_per_m: USD per million input tokens read from a cache, or
            ``None`` to price them as plain input.
    """

    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None

    def cost_usd(self, tokens: TokenUsage) -> float:
        """Return the price in USD of one call's tokens."""
        cached = tokens.cached_tokens or 0
        # cache reads are part of the input (openai's schema); more of them than input
        # means the provider counted them apart, and the whole input is uncached
        uncached = tokens.input_tokens - cached if cached <= tokens.input_tokens else tokens.input_tokens
        cached_rate = self.input_per_m if self.cached_input_per_m is None else self.cached_input_per_m
        return (uncached * self.input_per_m + cached * cached_rate + tokens.output_tokens * self.output_per_m) / 1e6


@dataclass(frozen=True)
class CallMeta:
    """What the provider reported about one call, or about several combined.

    Attributes:
        latency_s: Wall seconds the call took.
        tokens: The provider's token counts, or ``None`` when it sent none.
        cost_usd: Reported or priced cost in USD, or ``None`` when unknown.
        finish_reason: Why the model stopped (``"stop"``, ``"length"``,
            ``"tool_calls"``, ``"content_filter"``, ...).
        refused: Whether the model refused to answer.
        served_model: The model the provider says served the call, which can
            differ from the name that was asked for.
        logprobs: The chosen token and its log-probability per generated
            token (``choices[0].logprobs.content``), when the request asked
            for logprobs; ``None`` when it did not or the provider sent none.
        response_id: The provider's completion id, or ``None`` when it sent none.
        system_prompt: The system prompt the call was sent, when it is not
            promptolution's default; ``None`` for the default.
    """

    latency_s: float
    tokens: TokenUsage | None = None
    cost_usd: float | None = None
    finish_reason: str | None = None
    refused: bool = False
    served_model: str | None = None
    logprobs: Logprobs | None = None
    response_id: str | None = None
    system_prompt: str | None = None

    @property
    def error_kind(self) -> ErrorKind | None:
        """The kind of a cut or refused answer, or ``None`` for a clean finish."""
        return kind_of_finish(self.finish_reason, self.refused)

    def combine(self, other: CallMeta) -> CallMeta:
        """Return the meta of this call followed by ``other``.

        Latency, tokens and cost are summed. An unknown count or cost on either
        side makes the sum unknown, because a partial sum would understate what
        was paid. How the pair ended (finish reason, refusal, served model,
        logprobs, completion id, system prompt) is ``other``'s, the later call.
        """
        tokens = None if self.tokens is None or other.tokens is None else self.tokens + other.tokens
        cost = None if self.cost_usd is None or other.cost_usd is None else self.cost_usd + other.cost_usd
        return CallMeta(
            latency_s=self.latency_s + other.latency_s,
            tokens=tokens,
            cost_usd=cost,
            finish_reason=other.finish_reason,
            refused=other.refused,
            served_model=other.served_model,
            logprobs=other.logprobs,
            response_id=other.response_id,
            system_prompt=other.system_prompt,
        )

    def record_fields(self) -> dict[str, Any]:
        """Return the :class:`~xmas.trace.AgentRecord` keywords this meta fills.

        ``latency_s`` is not among them: the engine times the whole agent, tool
        turns and tool calls included.
        """
        return {
            "cost_usd": self.cost_usd,
            "tokens": self.tokens,
            "finish_reason": self.finish_reason,
            "error_kind": self.error_kind,
            "served_model": self.served_model,
            "logprobs": self.logprobs,
            "response_id": self.response_id,
            "system_prompt": self.system_prompt,
        }


@runtime_checkable
class SupportsMeta(Protocol):
    """A model that returns call metadata next to its answers, when ``full_response`` is ``True``."""

    full_response: bool

    def generate_with_meta(self, prompt: str, system_prompt: str = DEFAULT_SYS_PROMPT) -> tuple[str, CallMeta]:
        """Return the answer to one prompt and the call's metadata."""
        ...

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest], CallMeta]:
        """Return one tool turn's text, requested calls and metadata; no ``tools`` forces a text answer."""
        ...


def supports_meta(llm: object) -> bool:
    """Whether the engine takes ``llm``'s metadata path: it has the methods and ``full_response`` is ``True``."""
    return isinstance(llm, SupportsMeta) and llm.full_response is True


@runtime_checkable
class SupportsParts(Protocol):
    """A model that reads pictures and sound in a prompt, when ``full_response`` is ``True``.

    A class that hooks the text path (such as one overriding ``_ainvoke_once``)
    is off by default, so its hook is never bypassed by a parts request.
    """

    full_response: bool

    def generate_parts_with_meta(
        self, parts: Sequence[Part], system_prompt: str = DEFAULT_SYS_PROMPT
    ) -> tuple[str, CallMeta]:
        """Return the answer to a prompt of text and media parts, and the call's metadata."""
        ...


@runtime_checkable
class SupportsImages(Protocol):
    """A model that answers with images, when ``full_response`` is ``True``."""

    full_response: bool

    def generate_image_with_meta(
        self, prompt: str | Sequence[Part], system_prompt: str = DEFAULT_SYS_PROMPT
    ) -> tuple[str, list[tuple[bytes, str]], CallMeta]:
        """Return the answer's text, its images as ``(bytes, media type)`` pairs, and the call's metadata."""
        ...


def supports_parts(llm: object) -> bool:
    """Whether ``llm`` can be sent media parts: it has the method and ``full_response`` is ``True``."""
    return isinstance(llm, SupportsParts) and llm.full_response is True


def supports_images(llm: object) -> bool:
    """Whether ``llm`` can answer with images: it has the method and ``full_response`` is ``True``."""
    return isinstance(llm, SupportsImages) and llm.full_response is True


def meta_from_completion(
    completion: ChatCompletion, latency_s: float, price: TokenPrice | None, system_prompt: str | None = None
) -> CallMeta:
    """Read a chat completion's metadata; ``system_prompt`` is what the request sent, as :func:`system_prompt_sent`.

    The cost is the first that is known of: ``usage.cost`` as the provider sent
    it, a top-level ``cost`` field some proxies add, the tokens priced at
    ``price``. Otherwise it is ``None``. Only non-negative numbers count as a
    reported cost.
    """
    usage = completion.usage
    tokens = provider_usage(usage) if usage is not None else None
    cost = reported_cost(usage.model_extra) if usage is not None else None
    if cost is None:
        cost = reported_cost(completion.model_extra)
    if cost is None and price is not None and tokens is not None:
        cost = price.cost_usd(tokens)

    choice = completion.choices[0] if completion.choices else None
    finish = None if choice is None or choice.finish_reason is None else str(choice.finish_reason)
    refused = bool(choice is not None and choice.message.refusal) or finish == "content_filter"
    return CallMeta(
        latency_s=latency_s,
        tokens=tokens,
        cost_usd=cost,
        finish_reason=finish,
        refused=refused,
        served_model=completion.model or None,
        logprobs=chosen_logprobs(completion),
        response_id=completion.id or None,
        system_prompt=system_prompt,
    )


def chosen_logprobs(completion: ChatCompletion) -> Logprobs | None:
    """Return the first choice's ``logprobs.content`` as ``(token, logprob)`` pairs, or ``None`` when there is none.

    A provider sends them only when the request asked (``logprobs: true``);
    the client does not validate responses, so an entry without a token or a
    number is read as none sent.
    """
    choice = completion.choices[0] if completion.choices else None
    content = None if choice is None or choice.logprobs is None else choice.logprobs.content
    if content is None:
        return None
    pairs = []
    for entry in content:
        token, logprob = getattr(entry, "token", None), getattr(entry, "logprob", None)
        if not isinstance(token, str) or isinstance(logprob, bool) or not isinstance(logprob, int | float):
            return None
        pairs.append((token, float(logprob)))
    return tuple(pairs)


def system_prompt_sent(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the system prompt ``messages`` open with, when it is not promptolution's default; else ``None``."""
    first = messages[0] if messages else None
    if first is None or first.get("role") != "system":
        return None
    content = first.get("content")
    return None if not isinstance(content, str) or content == DEFAULT_SYS_PROMPT else content


def system_prompt_of(llm: object) -> str:
    """Return the system prompt ``llm`` sends when a caller names none: its ``system_prompt``, else promptolution's."""
    stated = getattr(llm, "system_prompt", None)
    return stated if isinstance(stated, str) else DEFAULT_SYS_PROMPT


def cut_meta(
    latency_s: float, tokens: TokenUsage | None, served_model: str | None, price: TokenPrice | None
) -> CallMeta:
    """Return the meta of a call cut before it finished.

    ``tokens`` are an estimate, what was sent and the text received so far,
    or ``None`` when not known; the cost is theirs at ``price``, else unknown.
    A cut call has no finish reason.
    """
    cost = None if tokens is None or price is None else price.cost_usd(tokens)
    return CallMeta(latency_s=latency_s, tokens=tokens, cost_usd=cost, served_model=served_model)


def provider_usage(usage: CompletionUsage) -> TokenUsage | None:
    """Return the provider's counts, with cache reads and reasoning when it sent them.

    ``None`` when a count is missing: the client does not validate responses, so
    ``"usage": {}`` arrives as ``None`` counts whatever the types say.
    """
    input_tokens: object = usage.prompt_tokens
    output_tokens: object = usage.completion_tokens
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=None if prompt_details is None else prompt_details.cached_tokens,
        reasoning_tokens=None if completion_details is None else completion_details.reasoning_tokens,
    )


def reported_cost(extra: dict[str, Any] | None) -> float | None:
    """Return a ``cost`` field when it is a non-negative number, else ``None``.

    A bool is an int to Python but never a cost, and a string is not trusted.
    """
    value = (extra or {}).get("cost")
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
        return None
    return float(value)
