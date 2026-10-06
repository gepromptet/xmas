"""Tools: functions an agent may call while it answers.

A tool is described once, as a :class:`Tool`: a name, a description and a JSON
Schema of its arguments, derived from a plain function by :func:`tool`. Where a
tool runs is a :class:`ToolSource`. Bare ``Tool`` objects run in-process
(:class:`LocalTools`); :class:`xmas.tools.mcp.MCPTools` runs them on an MCP
server. A MAS resolves ``Agent.tools`` names against its sources when it is
built, so a misspelled name fails before any model is called.

An agent with tools runs a bounded loop in its node: the model may request tool
calls, each result goes back as a ``role=tool`` message, and after
``max_tool_rounds`` such turns one last turn with the tools withheld forces a
text answer. Every call is kept as a :class:`~xmas.trace.ToolCall` on the
agent's record. A tool that fails is data (the model sees the error and may try
again); a source that refuses the caller ends the run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import typing
from collections.abc import Callable, Sequence
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

import openai
from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from xmas.llms.call_log import GENERATE, CallRecord, approximate_tokens, timestamp
from xmas.prompts import safe_format
from xmas.trace import AgentRecord, TokenUsage, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from openai.types.chat import ChatCompletion
    from promptolution.llms.base_llm import BaseLLM

    from xmas.agent import Agent
    from xmas.llms.api_llm import APILLM
    from xmas.mas import MAS
    from xmas.runtime import MASState

__all__ = [
    "TOOL_NAME",
    "LocalTools",
    "SupportsToolCalls",
    "Tool",
    "ToolAccessError",
    "ToolCallRequest",
    "ToolError",
    "ToolSource",
    "tool",
]

TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""Names the OpenAI ``tools`` payload accepts; checked when a Tool is built."""


class ToolError(RuntimeError):
    """A tool ran and failed, or was called wrongly.

    Fed back to the model as the call's result, never raised out of a run.
    """


class ToolAccessError(RuntimeError):
    """The tool source refused the caller (bad key, tool out of scope, budget spent).

    Ends the run; the message lands in ``Trace.error``.
    """


# --------------------------------------------------------------------- schema

_SECTION_RE = re.compile(r"^(Args|Arguments|Parameters)\s*:\s*$")
_PARAM_RE = re.compile(r"^\*{0,2}(?P<name>[A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(?P<text>.*)$")
_FIELD_PREFIX = "p_"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {param: description})`` for a cleaned docstring (Google ``Args:``)."""
    lines = doc.splitlines()
    summary_lines: list[str] = []
    for line in lines:
        if not line.strip() or _SECTION_RE.match(line.strip()):
            break
        summary_lines.append(line.strip())
    summary = " ".join(summary_lines)

    params: dict[str, str] = {}
    in_args = False
    header_indent = 0
    entry_indent: int | None = None
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if not in_args:
            if _SECTION_RE.match(stripped):
                in_args, header_indent, entry_indent, current = True, _indent(line), None, None
            continue
        if not stripped:
            continue
        indent = _indent(line)
        if indent <= header_indent:  # the next section (Returns:, Raises:, prose) ends Args
            in_args = False
            if _SECTION_RE.match(stripped):
                in_args, header_indent, entry_indent, current = True, indent, None, None
            continue
        if entry_indent is None:
            entry_indent = indent
        if indent == entry_indent and (match := _PARAM_RE.match(stripped)):
            current = match.group("name")
            params[current] = match.group("text").strip()
        elif indent > entry_indent and current is not None:
            params[current] = f"{params[current]} {stripped}".strip()
    return summary, params


def _qualname(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


def _build_model(fn: Callable[..., Any], descriptions: dict[str, str]) -> type[BaseModel]:
    """Build the pydantic arguments model of ``fn``, extra fields forbidden.

    Field names are prefixed internally and aliased to the real parameter names,
    so parameters such as ``json`` or ``schema`` cannot clash with ``BaseModel``.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot read the signature of {fn!r}") from exc
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(f"{_qualname(fn)}: *args/**kwargs are not allowed ({param.name})")
        if param.kind is param.POSITIONAL_ONLY:
            raise ValueError(f"{_qualname(fn)}: positional-only parameter {param.name!r}")
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise ValueError(f"{_qualname(fn)}: cannot resolve type hints: {exc}") from exc
    fields: dict[str, Any] = {}
    for param in params:
        if param.name not in hints:
            raise ValueError(f"{_qualname(fn)}: parameter {param.name!r} has no type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        extra: dict[str, Any] = {"alias": param.name}
        if descriptions.get(param.name):
            extra["description"] = descriptions[param.name]
        fields[_FIELD_PREFIX + param.name] = (hints[param.name], Field(default, **extra))
    try:
        model: type[BaseModel] = create_model(
            f"{getattr(fn, '__name__', 'tool')}_arguments",
            __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=False),
            **fields,
        )
    except Exception as exc:
        raise ValueError(f"{_qualname(fn)}: cannot build an arguments model: {exc}") from exc
    return model


def _schema(model: type[BaseModel], fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        schema = model.model_json_schema(by_alias=True)
    except Exception as exc:
        raise ValueError(f"{_qualname(fn)}: type hints are not JSON-Schema-able: {exc}") from exc
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    schema["required"] = list(schema.get("required", []))
    schema["additionalProperties"] = False
    return schema


# ----------------------------------------------------------------------- Tool


@dataclass(frozen=True)
class Tool:
    """A function an agent may call, described once.

    Attributes:
        name: Unique tool name, what ``Agent.tools`` refers to.
        description: What the model is told the tool does (the docstring summary).
        parameters: JSON Schema of the arguments object.
        fn: The implementation for a local tool, ``None`` for a remote one (called
            through its source).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any] | None = None
    _model: type[BaseModel] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Reject names the OpenAI ``tools`` payload would refuse."""
        if not TOOL_NAME.match(self.name):
            raise ValueError(f"Tool name {self.name!r} must match {TOOL_NAME.pattern}.")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``fn`` directly, so a decorated function stays a function.

        Raises:
            TypeError: If the tool is remote and has no local implementation.
        """
        if self.fn is None:
            raise TypeError(f"Tool {self.name!r} is remote; call it through its source.")
        return self.fn(*args, **kwargs)

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Coerce model-supplied arguments against the schema.

        A tool built by :func:`tool` validates (and coerces) with its pydantic
        model. A tool built from a foreign schema only checks required and
        unknown keys.

        Args:
            arguments: The decoded arguments object.

        Returns:
            Keyword arguments ready for ``fn(**kwargs)``.

        Raises:
            ToolError: If the arguments do not match.
        """
        if not isinstance(arguments, dict):
            raise ToolError(f"arguments must be an object, got {type(arguments).__name__}")
        if self._model is None:
            props = self.parameters.get("properties", {})
            missing = [key for key in self.parameters.get("required", []) if key not in arguments]
            if missing:
                raise ToolError(f"missing required argument(s): {', '.join(missing)}")
            if self.parameters.get("additionalProperties") is False:
                unexpected = [key for key in arguments if key not in props]
                if unexpected:
                    raise ToolError(f"unexpected argument(s): {', '.join(unexpected)}")
            return dict(arguments)
        try:
            instance = self._model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()
            )
            raise ToolError(f"invalid arguments for {self.name}: {problems}") from None
        return {info.alias or name: getattr(instance, name) for name, info in type(instance).model_fields.items()}

    def to_openai(self) -> dict[str, Any]:
        """Return the entry for an OpenAI chat-completions ``tools`` list."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...
@overload
def tool(*, name: str | None = None) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    fn: Callable[..., Any] | None = None, /, *, name: str | None = None
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Declare a function as a tool.

    Usable bare (``@tool``) or with a name (``@tool(name="add")``).

    Args:
        fn: A function with a type hint on every parameter and a docstring whose
            summary becomes the description. ``*args``, ``**kwargs`` and
            positional-only parameters are rejected. Google-style ``Args:``
            entries become the parameter descriptions.
        name: Tool name. Defaults to ``fn.__name__``.

    Returns:
        The :class:`Tool`, or a decorator producing one when only ``name`` is given.

    Raises:
        ValueError: If the function cannot be described (missing hint or
            docstring, unschemable type, bad name).
    """

    def describe(func: Callable[..., Any]) -> Tool:
        doc = inspect.getdoc(func)
        if not doc or not doc.strip():
            raise ValueError(f"{_qualname(func)}: a tool needs a docstring")
        summary, descriptions = _split_docstring(doc)
        if not summary:
            raise ValueError(f"{_qualname(func)}: the docstring has no summary paragraph")
        tool_name = name or getattr(func, "__name__", None)
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError(f"cannot derive a tool name for {func!r}; pass name=")
        model = _build_model(func, descriptions)
        return Tool(tool_name, summary, _schema(model, func), fn=func, _model=model)

    return describe if fn is None else describe(fn)


# ------------------------------------------------------------------- sources


@runtime_checkable
class ToolSource(Protocol):
    """Where a MAS resolves tool names.

    :class:`xmas.tools.mcp.MCPTools` is one; bare local Tools are wrapped in a
    :class:`LocalTools`.
    """

    def tools(self) -> list[Tool]:
        """Every tool this source offers. May do I/O once; the source caches it."""
        ...

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and return the text the model sees.

        Raises:
            ToolError: The tool failed; the message is fed back to the model.
            ToolAccessError: The caller may not use this source.
            RateLimitedError: The source refused on concurrency
                (``xmas.llms.RateLimitedError``).
        """
        ...


class LocalTools:
    """A ToolSource over in-process Tools; the MAS wraps bare Tool objects in one.

    ``call`` validates, then runs ``fn(**kwargs)``. Any exception becomes a
    :class:`ToolError`. A ``str`` result is returned as is, anything else as
    JSON (``default=str``), the same rule as serving-xmas's in-process executor.
    """

    def __init__(self, tools: Sequence[Tool]) -> None:
        """Initialize the source.

        Args:
            tools: Local tools, each with an ``fn``. Names must be unique.

        Raises:
            ValueError: If a tool has no ``fn`` or two tools share a name.
        """
        self._tools: dict[str, Tool] = {}
        for item in tools:
            if item.fn is None:
                raise ValueError(f"Tool {item.name!r} has no local implementation.")
            if item.name in self._tools and self._tools[item.name] is not item:
                raise ValueError(f"Two local tools are named {item.name!r}.")
            self._tools[item.name] = item

    def tools(self) -> list[Tool]:
        """Return the wrapped tools."""
        return list(self._tools.values())

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Validate the arguments, run the function, and stringify its result.

        Args:
            name: The tool to run.
            arguments: The decoded arguments object.

        Returns:
            The result as text.

        Raises:
            ToolError: If the tool is unknown, the arguments are invalid, or the
                function raised.
        """
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool {name!r}")
        kwargs = spec.validate(arguments)
        try:
            result = spec(**kwargs)
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        """Name the wrapped tools."""
        return f"LocalTools({sorted(self._tools)!r})"


ToolIndex = dict[str, tuple[Tool, ToolSource]]
"""Tools by name, each with the source that runs it."""


def index_tools(items: Sequence[ToolSource | Tool]) -> ToolIndex:
    """Index every tool offered by a MAS's ``tools=`` list by name.

    Bare Tools are grouped into one :class:`LocalTools`. The same object listed
    twice counts once; two different offers of one name are an error, with no
    shadowing and no priority order.

    Args:
        items: Tools and tool sources, mixed.

    Returns:
        The index.

    Raises:
        ValueError: If two sources offer the same name.
        TypeError: If an item is neither a Tool nor a ToolSource.
    """
    local: list[Tool] = []
    sources: list[ToolSource] = []
    for item in items:
        if isinstance(item, Tool):
            if not any(seen is item for seen in local):
                local.append(item)
        elif isinstance(item, ToolSource):
            if not any(seen is item for seen in sources):
                sources.append(item)
        else:
            raise TypeError(f"tools=[...] takes Tool objects and tool sources, not {type(item).__name__}.")
    if local:
        sources.insert(0, LocalTools(local))
    index: ToolIndex = {}
    for source in sources:
        for spec in source.tools():
            if spec.name in index:
                raise ValueError(f"Tool {spec.name!r} is offered by both {index[spec.name][1]!r} and {source!r}.")
            index[spec.name] = (spec, source)
    return index


def select_tools(index: ToolIndex, agent: Agent, llm: BaseLLM | None) -> ToolIndex:
    """Return the tools one agent may call, checking that its model can call them.

    Args:
        index: Every tool the MAS offers, from :func:`index_tools`.
        agent: The agent.
        llm: The agent's model, or ``None`` when none is bound yet.

    Returns:
        The agent's tools, keyed by name, in the order the agent lists them.

    Raises:
        ValueError: If a name is unknown, or the model cannot call tools.
    """
    unknown = [name for name in agent.tools if name not in index]
    if unknown:
        raise ValueError(f"Agent {agent.name!r} uses unknown tools {unknown}; available: {sorted(index)}.")
    if llm is not None and not isinstance(llm, SupportsToolCalls):
        raise ValueError(
            f"Agent {agent.name!r} uses tools, but its model {type(llm).__name__} has no get_tool_response."
        )
    return {name: index[name] for name in agent.tools}


# ------------------------------------------------------------------ model side


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call the model asked for.

    Attributes:
        id: The provider's call id, echoed back as ``tool_call_id``.
        name: The tool name.
        arguments: The raw JSON arguments, as the model sent them.
    """

    id: str
    name: str
    arguments: str


@runtime_checkable
class SupportsToolCalls(Protocol):
    """A model that can take one chat turn with tools on offer."""

    def get_tool_response(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest]]:
        """Run one chat turn that may request tools.

        Args:
            messages: OpenAI-shaped history (system, user, assistant with
                ``tool_calls``, tool).
            tools: Tools on offer. Empty forces a text answer.

        Returns:
            The message text (``""`` when the model only called tools) and the
            requested calls, in order.
        """
        ...


_RETRYABLE_4XX = frozenset({408, 409, 429})
"""Client-error statuses a retry can fix: request timeout, conflict, rate limit."""


def _is_permanent(exc: Exception) -> bool:
    """Whether retrying cannot help: a 4xx answer other than 408, 409 and 429.

    The status rule the openai client applies to its own retries. Connection
    errors, timeouts and 5xx stay retryable.
    """
    return (
        isinstance(exc, openai.APIStatusError)
        and 400 <= exc.status_code < 500
        and exc.status_code not in _RETRYABLE_4XX
    )


async def _create_with_retries(llm: APILLM, kwargs: dict[str, Any]) -> ChatCompletion:
    """One chat-completions call with promptolution's timeout and backoff.

    A permanent client error (400, 401, 403, 404, 422, ...) is raised at once
    instead of being sent ``max_retries`` more times.
    """
    last: Exception | None = None
    for attempt in range(llm.max_retries + 1):
        try:
            async with llm._sem:
                completion: ChatCompletion = await asyncio.wait_for(
                    llm.client.chat.completions.create(**kwargs), timeout=llm.call_timeout_s
                )
            return completion
        except Exception as exc:
            if _is_permanent(exc):
                raise
            last = exc
            if attempt < llm.max_retries:
                await asyncio.sleep(llm.retry_base_delay_s * (2**attempt))
    assert last is not None
    raise last


def complete_with_tools(
    llm: APILLM, messages: list[dict[str, Any]], tools: Sequence[Tool]
) -> tuple[str, list[ToolCallRequest]]:
    """Implement :meth:`xmas.llms.APILLM.get_tool_response`.

    Runs on the model's own background loop and semaphore, and records one
    ``GENERATE`` call with the provider's exact token counts when it reports them.

    Args:
        llm: The model.
        messages: OpenAI-shaped history.
        tools: Tools on offer. Empty omits the ``tools`` key.

    Returns:
        The message text and the requested calls, in order.
    """
    kwargs: dict[str, Any] = {"model": llm.model_id, "messages": messages, "max_tokens": llm.max_tokens}
    kwargs.update(llm._call_kwargs)
    if tools:
        kwargs["tools"] = [spec.to_openai() for spec in tools]
    params = llm._call_params()
    started_at = timestamp()
    started = time.perf_counter()
    future = asyncio.run_coroutine_threadsafe(_create_with_retries(llm, kwargs), llm._loop)
    try:
        try:
            completion = future.result(timeout=llm.gather_timeout_s + 5.0)
        except FuturesTimeout:
            future.cancel()
            raise TimeoutError(f"tool turn timed out after {llm.gather_timeout_s + 5.0}s") from None
    except Exception as error:
        llm._record(
            CallRecord(
                operation=GENERATE,
                model=llm.model,
                backend=llm.backend_tag,
                latency_s=time.perf_counter() - started,
                ok=False,
                error=f"{type(error).__name__}: {error}",
                ts=started_at,
                params=params,
            )
        )
        raise
    message = completion.choices[0].message
    text = message.content or ""
    requests = [
        ToolCallRequest(call.id, call.function.name, call.function.arguments or "{}")
        for call in message.tool_calls or []
        if call.type == "function"
    ]
    usage = completion.usage
    if usage is not None:
        input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
    else:
        input_tokens = sum(approximate_tokens(str(item.get("content") or "")) for item in messages)
        output_tokens = approximate_tokens(text) + sum(approximate_tokens(r.arguments) for r in requests)
    identifier = getattr(completion, "id", None)
    llm.input_token_count += input_tokens
    llm.output_token_count += output_tokens
    llm._record(
        CallRecord(
            operation=GENERATE,
            model=llm.model,
            backend=llm.backend_tag,
            latency_s=time.perf_counter() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ts=started_at,
            params=params,
            response_id=identifier if isinstance(identifier, str) else None,
        )
    )
    return text, requests


# ---------------------------------------------------------------- engine side


def _wire(request: ToolCallRequest) -> dict[str, Any]:
    """Return a request as the ``tool_calls`` entry of an assistant message."""
    return {"id": request.id, "type": "function", "function": {"name": request.name, "arguments": request.arguments}}


def _execute(tools: ToolIndex, request: ToolCallRequest) -> ToolCall:
    """Run one requested call; every failure but a refusal becomes a failed ToolCall.

    ``ToolAccessError`` and ``RateLimitedError`` propagate and end the run.
    """
    try:
        arguments = json.loads(request.arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolCall(request.name, {"_raw": request.arguments}, f"invalid JSON arguments: {exc}", failed=True)
    if not isinstance(arguments, dict):
        return ToolCall(request.name, {"_raw": request.arguments}, "invalid JSON arguments: not an object", failed=True)
    entry = tools.get(request.name)
    if entry is None:
        return ToolCall(request.name, arguments, f"unknown tool; available: {', '.join(tools)}", failed=True)
    spec, source = entry
    try:
        result = source.call(spec.name, arguments)
    except ToolError as exc:
        return ToolCall(request.name, arguments, str(exc), failed=True)
    return ToolCall(request.name, arguments, result)


def run_tool_loop(
    llm: SupportsToolCalls,
    rendered: str,
    tools: ToolIndex,
    max_rounds: int,
    system_prompt: str,
    count_tokens: Callable[[str], float],
) -> tuple[str, tuple[ToolCall, ...], TokenUsage]:
    """Answer one rendered prompt, letting the model call tools.

    At most ``max_rounds`` turns may request tools. Calls within a turn run in
    order. If the model is still asking after that, one more turn with the tools
    withheld forces a text answer.

    Args:
        llm: The agent's model.
        rendered: The rendered prompt.
        tools: The agent's tools.
        max_rounds: Turns that may request tools.
        system_prompt: The system message, as a plain agent's call would send.
        count_tokens: Token counter for the agent's model.

    Returns:
        The final text, the calls made, and the approximate usage over all turns.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rendered},
    ]
    specs = [spec for spec, _ in tools.values()]
    calls: list[ToolCall] = []
    usage = TokenUsage()

    def turn(offered: list[Tool]) -> tuple[str, list[ToolCallRequest]]:
        nonlocal usage
        sent = sum(int(count_tokens(str(item.get("content") or ""))) for item in messages)
        text, requests = llm.get_tool_response(messages, offered)
        received = int(count_tokens(text)) + sum(int(count_tokens(r.arguments)) for r in requests)
        usage = usage + TokenUsage(input_tokens=sent, output_tokens=received)
        return text, requests

    for _ in range(max_rounds):
        text, requests = turn(specs)
        if not requests:
            return text, tuple(calls), usage
        messages.append({"role": "assistant", "content": text or None, "tool_calls": [_wire(r) for r in requests]})
        for request in requests:
            call = _execute(tools, request)
            calls.append(call)
            messages.append({"role": "tool", "tool_call_id": request.id, "content": call.result})
    text, _ = turn([])
    return text, tuple(calls), usage


def tool_node(
    mas: MAS, agent: Agent, llm: BaseLLM, count_tokens: Callable[[str], float]
) -> Callable[[MASState], dict[str, Any]]:
    """Build the node the engine runs for an agent with tools.

    The same contract as the engine's plain node (render, answer, record), with
    the single model call replaced by :func:`run_tool_loop`. Each visit starts a
    fresh loop; nothing carries over between visits of a cyclic agent.

    Args:
        mas: The system, which resolves the agent's tools and bounds the rounds.
        agent: The agent to wrap.
        llm: Its model.
        count_tokens: Token counter for the model.

    Returns:
        A function mapping the incoming state to this agent's state updates.
    """
    tools = mas.resolve_tools(agent)
    if not isinstance(llm, SupportsToolCalls):  # resolve_tools checked it; this narrows the type
        raise ValueError(f"Agent {agent.name!r} uses tools, but its model cannot call them.")
    caller: SupportsToolCalls = llm

    def node(state: MASState) -> dict[str, Any]:
        prompts = state.get("prompts") or {}
        template = prompts.get(agent.name, agent.prompt)

        values = state.get("values") or {}
        inputs = {name: str(values.get(name) or "") for name in agent.inputs}
        rendered = safe_format(template, inputs)
        response, calls, usage = run_tool_loop(
            caller, rendered, tools, mas.max_tool_rounds, DEFAULT_SYS_PROMPT, count_tokens
        )

        record = AgentRecord(
            agent=agent.name,
            step=-1,
            prompt=template,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: response},
            tool_calls=calls,
            usage=usage,
            model=agent.model,
            raw_response=response,
        )
        return {"values": {agent.output: response}, "records": [record], "visits": {agent.name: 1}}

    return node
