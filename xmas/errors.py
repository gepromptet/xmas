"""Typed errors: what kind of failure ended a call or a run.

xmas classifies exceptions rather than matching their words. :func:`classify`
reads an exception, and the exceptions it was raised from, and returns an
:class:`ErrorKind`, so callers branch on a closed set instead of matching words
in ``Trace.error``. A wrapper keeps its cause: a version 1 plain call's
``RuntimeError("LLM batch failed: ...")`` and a local tool's ``XmasError``
classify by what they wrap. :func:`status_for` turns a kind into the run's
:class:`RunStatus`.

This is a leaf module: at runtime it imports no other xmas module. ``httpx2``,
which openai 3.x and the MCP client raise from, is read when it is installed.
"""

from __future__ import annotations

import enum
import json
import re
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal

import httpx
import openai
import pydantic

if TYPE_CHECKING:  # pragma: no cover
    from xmas.llms.meta import CallMeta

__all__ = [
    "MODERATION_WORDS",
    "MONEY_WORDS",
    "RUN_ENDS",
    "BadFormat",
    "BudgetExceeded",
    "EmptyResponse",
    "ErrorKind",
    "RaceLost",
    "RunStatus",
    "Stopped",
    "Unanswered",
    "XmasError",
    "classify",
    "kind_of_finish",
    "kind_of_status",
    "status_for",
]


try:  # openai 3.x and the MCP client raise httpx2's errors, which httpx's types do not match
    import httpx2
except ImportError:  # pragma: no cover (httpx2 is optional here, not a dependency)
    HTTPX2_TIMEOUTS: tuple[type[BaseException], ...] = ()
    HTTPX2_OUTAGES: tuple[type[BaseException], ...] = ()
else:
    HTTPX2_TIMEOUTS = (httpx2.TimeoutException,)
    HTTPX2_OUTAGES = (httpx2.TransportError,)


class ErrorKind(enum.StrEnum):
    """Why a call or a run failed."""

    MONEY = "money"
    LENGTH_CUT = "length_cut"
    OUTAGE = "outage"
    BAD_FORMAT = "bad_format"
    REFUSAL = "refusal"
    STOPPED = "stopped"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    BUDGET = "budget"
    UNKNOWN = "unknown"


RUN_ENDS = frozenset({ErrorKind.BUDGET, ErrorKind.STOPPED})
"""The kinds that end a run: no ``on error`` edge routes them."""


class RunStatus(enum.StrEnum):
    """How a run ended, or ``paused`` while it waits for a person."""

    OK = "ok"
    FAILED = "failed"
    STOPPED = "stopped"
    OVER_BUDGET = "over_budget"
    PAUSED = "paused"


class XmasError(RuntimeError):
    """The error xmas raises itself; the message reads '<source>: <what> (<code>)', e.g. 'llm: rate limited (429)'.

    A plain ``XmasError`` is a failed call; its subclasses below name a run's
    typed failures.

    Attributes:
        status: The HTTP-style status code, or ``None``.
        retry_after: Seconds the endpoint asked the caller to wait, or ``None``.
        kind: What :func:`classify` returns for this error when it carries no
            ``status``.
        meta: The combined metadata of the paid attempts behind the error, so a
            run's meter can count what they cost, or ``None`` when not known.
    """

    kind: ErrorKind = ErrorKind.UNKNOWN
    meta: CallMeta | None = None
    pickled: tuple[str, ...] = ("message", "status", "retry_after")
    """The constructor's arguments, as attributes (``message`` is the message), that a pickle keeps."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by the constructor's arguments, which the message alone would lose."""
        return type(self), tuple(str(self) if name == "message" else getattr(self, name) for name in self.pickled)


class BudgetExceeded(XmasError):
    """A run reached a limit of its budget, so the next call was not started.

    Attributes:
        limit: The limit that bound: ``"cost_usd"``, ``"wall_s"``, ``"calls"``, or
            ``"cost_unknown"`` when a cost limit is set but a call reported no cost.
        spent: What the run had used of that limit. For ``"cost_unknown"``, the
            number of calls that reported no cost.
        allowed: The limit's value. For ``"cost_unknown"``, ``max_cost_usd``.
        calls: The calls the run had started.
    """

    kind = ErrorKind.BUDGET
    pickled = ("limit", "spent", "allowed", "calls")

    def __init__(
        self, limit: Literal["cost_usd", "wall_s", "calls", "cost_unknown"], spent: float, allowed: float, calls: int
    ) -> None:
        """Build the message that names the limit and the refused call."""
        if limit == "cost_unknown":
            uncosted = "1 call reported no cost and has" if spent == 1 else f"{spent:g} calls reported no cost and have"
            message = (
                f"cost_unknown: {uncosted} no price, "
                f"so max_cost_usd {allowed:g} cannot be checked before call {calls + 1}"
            )
        else:
            message = f"{limit} {spent:g} >= max {allowed:g} before call {calls + 1}"
        super().__init__(message)
        self.limit = limit
        self.spent = spent
        self.allowed = allowed
        self.calls = calls


class Stopped(XmasError):
    """A run was stopped on request; a caller's model wrapper raises it to end the run with status ``stopped``.

    Attributes:
        reason: Why the run was stopped, the error's message.
        meta: What the cut call had used, or ``None`` when not known.
        text: The answer received before the cut, ``""`` when none was.
    """

    kind = ErrorKind.STOPPED
    pickled = ("reason", "meta", "text")

    def __init__(self, reason: str = "stopped", meta: CallMeta | None = None, text: str = "") -> None:
        """Keep the reason as the message, and the cut call's meta and partial answer."""
        super().__init__(reason)
        self.reason = reason
        self.meta = meta
        self.text = text


class RaceLost(Stopped):
    """A ``first of`` contestant's call was cut because another contestant finished first.

    The engine keeps the contestant's record as lost, not failed; the run goes on.
    """


class Unanswered(XmasError):
    """A question of a run no one answers later (``run``, ``run_batch``) got no answer and has no default.

    It fails the run wherever the human action sits, a chain item or a module included.
    """


class EmptyResponse(XmasError):
    """A model answered with no content.

    The message starts with promptolution's words, "Empty content from model", so
    checks written against promptolution's error keep matching.

    Attributes:
        kind: Why the answer was empty, such as ``LENGTH_CUT`` or ``REFUSAL``.
    """

    pickled = ("message", "kind", "meta")

    def __init__(self, message: str, kind: ErrorKind = ErrorKind.UNKNOWN, meta: CallMeta | None = None) -> None:
        """Keep the message, the kind and the paid attempts' metadata."""
        super().__init__(message)
        self.kind = kind
        self.meta = meta


class BadFormat(XmasError):
    """A node's output did not have its declared form.

    An agent with an answer form raises it when its answer and its one retry
    both fail the form; a code node raises it when a write has the wrong type.

    Attributes:
        node: The node whose output was refused.
        problems: One line per refused attempt, first attempt first.
    """

    kind = ErrorKind.BAD_FORMAT
    pickled = ("node", "problems", "meta")

    def __init__(self, node: str, problems: tuple[str, ...], meta: CallMeta | None = None) -> None:
        """Build the message that names the node and every problem."""
        super().__init__(f"Bad format from {node!r}: " + "; after the retry: ".join(problems))
        self.node = node
        self.problems = tuple(problems)
        self.meta = meta


MONEY_WORDS = re.compile(
    r"key limit exceeded|requires more credits|insufficient credits|insufficient_quota"
    r"|exceeded your current quota|payment required|in_flight_budget_exhausted|insufficient_budget",
    re.IGNORECASE,
)
"""Provider texts that mean the key or account is out of money."""

MODERATION_WORDS = re.compile(r"flagged|moderation", re.IGNORECASE)
"""Provider texts that mean a moderation filter refused the request."""

MAX_DEPTH = 8


def classify(error: BaseException) -> ErrorKind:
    """Return the kind of an exception, reading the exceptions it came from too.

    The walk visits ``error``, then its ``__cause__`` and ``__context__``, nearest
    first, at most 8 deep, and visits each exception once, so a cycle ends it.
    The first rule that matches any visited exception wins:

    0. an :class:`XmasError` with a ``status``: :func:`kind_of_status` of it and
       the message, except that 401 and 403 (the source refused the caller) are
       ``REFUSAL`` when no money words appear;
    1. a ``kind`` attribute that is an :class:`ErrorKind`;
    2. ``openai.APIStatusError``: :func:`kind_of_status` of its status code and its
       message plus body;
    3. ``openai.APITimeoutError``, ``TimeoutError`` or ``TimeoutException`` of
       ``httpx`` or ``httpx2``: ``TIMEOUT`` (before rule 4, whose types they
       subclass);
    4. ``openai.APIConnectionError``, ``ConnectionError`` or ``TransportError``
       of ``httpx`` or ``httpx2``: ``OUTAGE``;
    5. ``pydantic.ValidationError`` or ``json.JSONDecodeError``: ``BAD_FORMAT``.

    Otherwise the kind is ``UNKNOWN``. Walking the chain is what classifies
    promptolution's ``RuntimeError("LLM batch failed: ...")`` by its cause.
    """
    seen: set[int] = set()
    queue: deque[tuple[BaseException, int]] = deque([(error, 0)])
    while queue:
        exc, depth = queue.popleft()
        if id(exc) in seen or depth >= MAX_DEPTH:
            continue
        seen.add(id(exc))
        kind = own_kind(exc)
        if kind is not None:
            return kind
        queue.extend((linked, depth + 1) for linked in (exc.__cause__, exc.__context__) if linked is not None)
    return ErrorKind.UNKNOWN


def own_kind(exc: BaseException) -> ErrorKind | None:
    """Return the kind of one exception by rules 0 to 5, or ``None`` if none matches."""
    kind = reported_kind(exc)
    return kind if kind is not None else kind_of_type(exc)


def reported_kind(exc: BaseException) -> ErrorKind | None:
    """Return the kind an exception reports by rules 0 to 2, its status or its ``kind``, or ``None``."""
    if isinstance(exc, XmasError) and exc.status is not None:
        status_kind = kind_of_status(exc.status, str(exc))
        if status_kind == ErrorKind.UNKNOWN and exc.status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
            return ErrorKind.REFUSAL
        return status_kind
    kind = getattr(exc, "kind", None)
    if isinstance(kind, ErrorKind):
        return kind
    if isinstance(exc, openai.APIStatusError):
        text = str(exc) if exc.body is None else f"{exc} {exc.body}"
        return kind_of_status(exc.status_code, text)
    return None


def kind_of_type(exc: BaseException) -> ErrorKind | None:
    """Return the kind of an exception by its type, rules 3 to 5, or ``None``."""
    if isinstance(exc, (openai.APITimeoutError, TimeoutError, httpx.TimeoutException, *HTTPX2_TIMEOUTS)):
        return ErrorKind.TIMEOUT
    if isinstance(exc, (openai.APIConnectionError, ConnectionError, httpx.TransportError, *HTTPX2_OUTAGES)):
        return ErrorKind.OUTAGE
    if isinstance(exc, pydantic.ValidationError | json.JSONDecodeError):
        return ErrorKind.BAD_FORMAT
    return None


def kind_of_status(status: int, text: str = "") -> ErrorKind:
    """Return the kind of an HTTP error status.

    402, or any 4xx whose text names money (403 "Key limit exceeded", 429
    "insufficient_quota"), is ``MONEY``; 429 is ``RATE_LIMITED``; 408 is
    ``TIMEOUT``; 500 and above are ``OUTAGE``; 400 or 403 naming moderation is
    ``REFUSAL``; anything else (401, 404, 422) is ``UNKNOWN``. ``text`` is the
    error's message and body, searched for provider words.
    """
    client_error = HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR
    if status == HTTPStatus.PAYMENT_REQUIRED or (client_error and MONEY_WORDS.search(text)):
        return ErrorKind.MONEY
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return ErrorKind.RATE_LIMITED
    if status == HTTPStatus.REQUEST_TIMEOUT:
        return ErrorKind.TIMEOUT
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return ErrorKind.OUTAGE
    if status in (HTTPStatus.BAD_REQUEST, HTTPStatus.FORBIDDEN) and MODERATION_WORDS.search(text):
        return ErrorKind.REFUSAL
    return ErrorKind.UNKNOWN


def kind_of_finish(
    finish_reason: str | None,
    refused: bool = False,  # noqa: FBT001, FBT002  the public signature; CallMeta passes it positionally
) -> ErrorKind | None:
    """Return the kind of an answer that ended badly but still arrived, or ``None`` if it ended well.

    ``LENGTH_CUT`` for the finish reason ``"length"``; ``REFUSAL`` for
    ``"content_filter"`` or when the provider flagged a refusal.
    """
    if finish_reason == "length":
        return ErrorKind.LENGTH_CUT
    if finish_reason == "content_filter" or refused:
        return ErrorKind.REFUSAL
    return None


def status_for(kind: ErrorKind | None) -> RunStatus:
    """Return a failed run's status: ``OVER_BUDGET`` for ``BUDGET``, ``STOPPED`` for ``STOPPED``, else ``FAILED``."""
    if kind == ErrorKind.BUDGET:
        return RunStatus.OVER_BUDGET
    if kind == ErrorKind.STOPPED:
        return RunStatus.STOPPED
    return RunStatus.FAILED
