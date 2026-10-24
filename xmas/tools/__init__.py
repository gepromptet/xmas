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

The same registry serves steps (version 2): a :class:`~xmas.nodes.Step` names a
tool in ``uses`` and :func:`run_step` runs it with values from the graph. A tool
reads its parameters (:attr:`Tool.reads`) and writes :attr:`Tool.writes`. A step
that writes a picture, a sound or a file returns a :class:`~pathlib.Path` (copied
into the store) or a :class:`~xmas.media.Ref`; a remote one sends media blocks
(:meth:`ToolSource.call_media`).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, overload

from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, ValidationError

from xmas.errors import BadFormat, BudgetExceeded, Stopped, XmasError
from xmas.media import store
from xmas.schemas import RETRY_NOTE
from xmas.tools.schema import (
    arguments_from,
    arguments_model,
    describe_validation_error,
    function_name,
    json_schema,
    split_docstring,
    valid_tool_name,
)
from xmas.trace import TokenUsage, ToolCall
from xmas.values import TEXT, ValueType, check_value, coerce_media, media_kind, parse_type, type_of_hint

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.agent import Agent
    from xmas.budget import RunMeter, VisitMeter
    from xmas.llms.meta import CallMeta, SupportsMeta
    from xmas.llms.parts import Part

__all__ = [
    "LocalTools",
    "Tool",
    "ToolCallRequest",
    "ToolIndex",
    "ToolSource",
    "execute_tool_call",
    "index_tools",
    "run_step",
    "run_tool_loop",
    "select_tools",
    "tool",
]


def return_type(fn: Callable[..., Any] | None) -> ValueType:
    """Return the type a callable's single write has: text for a ``str`` return hint or a remote tool, else json."""
    if fn is None:
        return TEXT
    try:
        hint = typing.get_type_hints(fn, include_extras=True).get("return")
    except Exception:  # noqa: BLE001  get_type_hints evaluates arbitrary annotation strings; a hint that fails says nothing
        hint = None
    return type_of_hint(hint)


@dataclass(frozen=True, init=False)
class Tool:
    """A function an agent may call, described once.

    Attributes:
        name: Unique tool name, what ``Agent.tools`` refers to.
        description: What the model is told the tool does (the docstring summary).
        parameters: JSON Schema of the arguments object.
        fn: The implementation for a local tool, ``None`` for a remote one (called
            through its source).
        writes: What the tool writes when it runs as a step, ``{write name:
            type}``. It may be given as type words; ``None`` is one write named
            like the tool: text when ``fn``'s return hint is ``str`` or the tool
            is remote, json otherwise.
        params: Arguments bound by :meth:`bind`, passed on every call and left
            out of ``parameters``.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any] | None
    arguments_model: type[BaseModel] | None = field(compare=False, repr=False)
    writes: dict[str, ValueType]
    params: dict[str, Any]

    def __init__(  # noqa: PLR0913, PLR0917  the public constructor, kept for version 1 callers
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any] | None = None,
        arguments_model: type[BaseModel] | None = None,
        writes: Mapping[str, ValueType | str] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Reject names the OpenAI ``tools`` payload would refuse, and read the write types."""
        if not valid_tool_name(name):
            raise ValueError(
                f"Tool name {name!r} must match the OpenAI rule: letters, digits, _ and -, at most 64 characters."
            )
        given = {name: return_type(fn)} if writes is None else writes
        if not given:
            raise ValueError(f"Tool {name!r} needs at least one write.")
        typed = {key: vtype if isinstance(vtype, ValueType) else parse_type(vtype) for key, vtype in given.items()}
        values = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "fn": fn,
            "arguments_model": arguments_model,
            "writes": typed,
            "params": dict(params or {}),
        }
        for key, value in values.items():
            object.__setattr__(self, key, value)  # frozen

    @property
    def reads(self) -> dict[str, ValueType]:
        """The type each parameter reads when the tool runs as a step: text for a ``string``, else json."""
        properties = self.parameters.get("properties") or {}
        return {name: type_of_hint(schema) for name, schema in properties.items()}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``fn`` directly, so a decorated function stays a function; a remote tool raises ``TypeError``."""
        if self.fn is None:
            raise TypeError(f"Tool {self.name!r} is remote; call it through its source.")
        return self.fn(*args, **{**self.params, **kwargs})

    def bind(self, **params: Any) -> Tool:
        """Return this tool with ``params`` fixed: every call passes them, and the model and the graph never see them.

        Raises:
            ValueError: For a name that is not one of the tool's parameters.
        """
        properties = self.parameters.get("properties") or {}
        unknown = sorted(set(params) - set(properties))
        if unknown:
            raise ValueError(f"Tool {self.name!r} has no parameters {unknown}; it takes {sorted(properties)}.")
        schema = dict(self.parameters)
        schema["properties"] = {key: value for key, value in properties.items() if key not in params}
        if "required" in schema:
            schema["required"] = [key for key in schema["required"] if key not in params]
        return dataclasses.replace(self, parameters=schema, params={**self.params, **params})

    def renamed(self, name: str) -> Tool:
        """Return this tool under another name."""
        return dataclasses.replace(self, name=name)

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return model-supplied arguments as keyword arguments for ``fn``, or raise ``XmasError``.

        A tool built by :func:`tool` validates (and coerces) with its pydantic
        model. A tool built from a foreign schema only checks required and
        unknown keys. Bound :attr:`params` are added and win.
        """
        if not isinstance(arguments, dict):
            raise XmasError(f"arguments must be an object, got {type(arguments).__name__}")
        if self.arguments_model is None:
            props = self.parameters.get("properties", {})
            missing = [key for key in self.parameters.get("required", []) if key not in arguments]
            if missing:
                raise XmasError(f"missing required argument(s): {', '.join(missing)}")
            if self.parameters.get("additionalProperties") is False:
                unexpected = [key for key in arguments if key not in props]
                if unexpected:
                    raise XmasError(f"unexpected argument(s): {', '.join(unexpected)}")
            return {**arguments, **self.params}
        try:
            instance = self.arguments_model.model_validate({**arguments, **self.params})
        except ValidationError as exc:
            raise XmasError(f"invalid arguments for {self.name}: {describe_validation_error(exc)}") from None
        return arguments_from(instance)

    def to_openai(self) -> dict[str, Any]:
        """Return the entry for an OpenAI chat-completions ``tools`` list."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


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


class ToolSource:
    """Where a MAS resolves tool names.

    :class:`xmas.tools.mcp.MCPTools` is one; bare local Tools are wrapped in a
    :class:`LocalTools`.
    """

    def tools(self) -> list[Tool]:
        """Every tool this source offers. May do I/O once; the source caches it."""
        raise NotImplementedError

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and return the text the model sees.

        Raises:
            XmasError: The tool failed, and the message is fed back to the model;
                or, with ``status`` 401, 403 or 429, the source refused the caller
                and the run ends.
        """
        raise NotImplementedError

    def call_media(self, name: str, arguments: dict[str, Any]) -> list[tuple[bytes, str]]:  # noqa: ARG002
        """Run one tool and return the media it sent, as ``(bytes, media type)`` blocks in order.

        Raises:
            XmasError: Always, unless the source sends media; otherwise as :meth:`call`.
        """
        raise XmasError(f"{self!r} returns text only, so {name!r} cannot write media")


class LocalTools(ToolSource):
    """A ToolSource over in-process Tools; the MAS wraps bare Tool objects in one.

    ``call`` validates, then runs ``fn(**kwargs)``. Any exception becomes an
    :class:`~xmas.errors.XmasError`. A ``str`` result is returned as is, anything
    else as JSON (``default=str``), the same rule as serving-xmas's in-process executor.
    """

    def __init__(self, tools: Sequence[Tool]) -> None:
        """Wrap local tools; ``ValueError`` if one has no ``fn`` or two share a name."""
        self.by_name: dict[str, Tool] = {}
        for item in tools:
            if item.fn is None:
                raise ValueError(f"Tool {item.name!r} has no local implementation.")
            if item.name in self.by_name and self.by_name[item.name] is not item:
                raise ValueError(f"Two local tools are named {item.name!r}.")
            self.by_name[item.name] = item

    def tools(self) -> list[Tool]:
        """Return the wrapped tools."""
        return list(self.by_name.values())

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Validate the arguments, run the function, and return its result as text.

        Raises:
            XmasError: If the tool is unknown, the arguments are invalid, or the
                function raised. ``BudgetExceeded``, ``Stopped`` and LangGraph's
                own signals (``GraphBubbleUp``) pass through unwrapped.
        """
        spec = self.by_name.get(name)
        if spec is None:
            raise XmasError(f"unknown tool {name!r}")
        kwargs = spec.validate(arguments)
        try:
            result = spec(**kwargs)
        except (BudgetExceeded, Stopped, GraphBubbleUp):
            raise
        except Exception as exc:
            raise XmasError(f"{type(exc).__name__}: {exc}") from exc
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        """Name the wrapped tools."""
        return f"LocalTools({sorted(self.by_name)!r})"


ToolIndex = dict[str, tuple[Tool, ToolSource]]
"""Tools by name, each with the source that runs it."""


@overload
def tool(fn: Callable[..., Any], /, *, name: str | None = None, writes: Mapping[str, str] | None = None) -> Tool: ...
@overload
def tool(
    *, name: str | None = None, writes: Mapping[str, str] | None = None
) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    fn: Callable[..., Any] | None = None, /, *, name: str | None = None, writes: Mapping[str, str] | None = None
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Declare a function as a tool.

    Usable bare (``@tool``) or with keywords (``@tool(name="add")``,
    ``@tool(writes={"counts": "json"})``).

    Args:
        fn: A function with a type hint on every parameter and a docstring whose
            summary becomes the description. ``*args``, ``**kwargs`` and
            positional-only parameters are rejected. Google-style ``Args:``
            entries become the parameter descriptions.
        name: Tool name. Defaults to ``fn.__name__``.
        writes: What the tool writes as a step, ``{write name: type}``. Defaults
            to one write named like the tool: text when the return hint is
            ``str``, else json. Several writes need ``fn`` to return a mapping
            with exactly those keys.

    Raises:
        ValueError: If the function cannot be described (missing hint or
            docstring, unschemable type, bad name).
    """

    def describe(func: Callable[..., Any]) -> Tool:
        doc = inspect.getdoc(func)
        if not doc or not doc.strip():
            raise ValueError(f"{function_name(func)}: a tool needs a docstring")
        summary, descriptions = split_docstring(doc)
        if not summary:
            raise ValueError(f"{function_name(func)}: the docstring has no summary paragraph")
        tool_name = name or getattr(func, "__name__", None)
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError(f"cannot derive a tool name for {func!r}; pass name=")
        model = arguments_model(func, descriptions)
        return Tool(tool_name, summary, json_schema(model, func), fn=func, arguments_model=model, writes=writes)

    return describe if fn is None else describe(fn)


def index_tools(items: Sequence[ToolSource | Tool]) -> ToolIndex:
    """Index every tool offered by a MAS's ``tools=`` list by name.

    Bare Tools are grouped into one :class:`LocalTools`. The same object listed
    twice counts once; two different offers of one name are a ``ValueError``,
    with no shadowing and no priority order.
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
    """Return the tools one agent may call, in its order.

    A tool that reads media is refused: a model would pass a made-up ref, and
    the tool would read whatever file it names.

    Raises:
        ValueError: For an unknown name, a tool that reads media, or a model that cannot call tools.
    """
    unknown = [name for name in agent.tools if name not in index]
    if unknown:
        raise ValueError(f"Agent {agent.name!r} uses unknown tools {unknown}; available: {sorted(index)}.")
    for name in agent.tools:
        for parameter, vtype in index[name][0].reads.items():
            if media_kind(vtype) is not None:
                raise ValueError(
                    f"Agent {agent.name!r} uses tool {name!r}, which reads {vtype} {parameter!r}; a model passes "
                    "no media to a tool, so run it as a step."
                )
    if llm is not None and not hasattr(llm, "get_tool_response"):
        raise ValueError(
            f"Agent {agent.name!r} uses tools, but its model {type(llm).__name__} has no get_tool_response."
        )
    return {name: index[name] for name in agent.tools}


def run_step(spec: Tool, source: ToolSource, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a callable as a step and return ``{write: value}``, keyed by its own write names.

    A local tool's ``fn`` is called here, not through :meth:`LocalTools.call`,
    which would turn its result into text, and what it raises passes unchanged;
    a remote tool runs through ``source.call``, and a json write is the JSON its
    text holds. One write takes the whole result; several need a mapping with
    exactly those keys. A media write takes a ref, or a :class:`~pathlib.Path`
    whose bytes are copied into :func:`~xmas.media.store`, so a step that
    rewrites one path per item leaves earlier refs intact; a remote tool's one
    media write is its :meth:`~ToolSource.call_media` blocks, stored (exactly one
    for a single value).

    Raises:
        XmasError: If the arguments do not match the parameters, or a remote
            tool failed or was refused (``status`` 401, 403 or 429).
        BadFormat: If the result does not fit the writes; it names the callable.
    """
    writes = spec.writes
    kwargs = spec.validate(arguments)
    only = next(iter(writes.values())) if len(writes) == 1 else None
    if spec.fn is not None:
        result = spec.fn(**kwargs)
    elif only is not None and media_kind(only) is not None:
        result = stored_blocks(spec.name, source.call_media(spec.name, kwargs), only)
    else:
        text = source.call(spec.name, kwargs)
        if only == TEXT:
            result = text
        else:
            try:
                result = json.loads(text)
            except json.JSONDecodeError as exc:
                raise BadFormat(spec.name, (f"the result is not JSON: {exc}",)) from exc
    if len(writes) > 1 and not (isinstance(result, Mapping) and set(result) == set(writes)):
        got = sorted(result) if isinstance(result, Mapping) else type(result).__name__
        raise BadFormat(spec.name, (f"expected a mapping with keys {sorted(writes)}, got {got}",))
    produced = {next(iter(writes)): result} if len(writes) == 1 else dict(result)
    media, problems = store(), []
    for name, value in produced.items():
        try:
            produced[name] = coerce_media(value, writes[name], store=media)
        except ValueError as error:  # a returned Path that is no file
            raise BadFormat(spec.name, (f"{name}: {error}",)) from error
        if (problem := check_value(produced[name], writes[name])) is not None:
            problems.append(f"{name}: {problem}")
    if problems:
        raise BadFormat(spec.name, ("; ".join(problems),))
    return produced


def stored_blocks(name: str, blocks: list[tuple[bytes, str]], vtype: ValueType) -> Any:
    """Store a remote tool's media blocks: the one ref for a single value, every ref for a list."""
    if vtype.item is None and len(blocks) != 1:
        raise BadFormat(name, (f"expected one {vtype} block, got {len(blocks)}",))
    media = store()
    refs = [media.put(data, media_type) for data, media_type in blocks]
    return refs if vtype.item is not None else refs[0]


def turn_usage(
    count: Callable[[str], float], messages: list[dict[str, Any]], text: str, requests: Sequence[ToolCallRequest]
) -> TokenUsage:
    """Estimate one tool turn's tokens with ``count``: every message's text in, the text and call arguments out."""
    from xmas.llms.parts import content_text  # noqa: PLC0415  here, not at the top: xmas.llms imports this module

    return TokenUsage(
        input_tokens=sum(int(count(content_text(item.get("content") or ""))) for item in messages),
        output_tokens=int(count(text)) + sum(int(count(request.arguments)) for request in requests),
    )


def run_tool_loop(  # noqa: PLR0913, PLR0917  the public signature, kept for version 1 callers
    llm: Any,
    rendered: str | Sequence[Part],
    tools: ToolIndex,
    max_rounds: int,
    system_prompt: str,
    count_tokens: Callable[[str], float],
    meter: RunMeter | VisitMeter,
    check: Callable[[str], str | None] | None = None,
) -> tuple[str, tuple[ToolCall, ...], TokenUsage, CallMeta | None]:
    """Answer one rendered prompt, letting the model call tools.

    At most ``max_rounds`` turns may request tools; calls within a turn run in
    order. A prompt of parts with media in it is the first user message's
    content (:func:`~xmas.llms.parts.to_content`); later turns are text. If
    the model is still asking after that, one more turn with the tools
    withheld forces a text answer. With ``check``, a final answer it
    names a problem with stays in the conversation, and one more turn, tools
    withheld, asks again with :data:`~xmas.schemas.RETRY_NOTE`; that answer is
    returned unchecked. Every turn is checked against ``meter`` before it starts.

    Returns:
        The final text, the calls made, the approximate usage over all turns
        (``count_tokens``), and the turns' combined metadata, or ``None`` when
        the model gives none (see :func:`xmas.llms.meta.supports_meta`).

    Raises:
        BudgetExceeded: If the meter refuses a turn.
        EmptyResponse: If a turn came back empty after its retries.
        XmasError: If a tool source refused the caller (``status`` 401, 403 or 429).
    """
    from xmas.llms.meta import supports_meta  # noqa: PLC0415  here, not at the top: xmas.llms imports this module
    from xmas.llms.parts import prompt_messages  # noqa: PLC0415  the same import cycle

    conversation = ToolConversation(
        llm,
        cast("SupportsMeta", llm) if supports_meta(llm) else None,
        meter,
        count_tokens,
        prompt_messages(rendered, system_prompt),
    )
    specs = [spec for spec, _ in tools.values()]
    calls: list[ToolCall] = []
    for _ in range(max_rounds):
        text, requests = conversation.turn(specs)
        if not requests:
            return conversation.answer(text, check, calls)
        calls.extend(conversation.run_calls(tools, text, requests))
    text, _ = conversation.turn([])
    return conversation.answer(text, check, calls)


@dataclass
class ToolConversation:
    """The messages of one tool loop, and the usage and metadata its turns have added up.

    Attributes:
        llm: The model answering.
        with_meta: The same model when it reports call metadata, else ``None``.
        meter: Checks every turn before it starts.
        count_tokens: Estimates the tokens of a text.
        messages: The conversation so far, sent whole on every turn.
        usage: The approximate usage over the turns so far.
        meta: The turns' combined metadata, or ``None`` when the model gives none.
    """

    llm: Any
    with_meta: SupportsMeta | None
    meter: RunMeter | VisitMeter
    count_tokens: Callable[[str], float]
    messages: list[dict[str, Any]]
    usage: TokenUsage = field(default_factory=TokenUsage)
    meta: CallMeta | None = None

    def ask(self, offered: list[Tool]) -> tuple[tuple[str, list[ToolCallRequest]], CallMeta | None]:
        """Send the conversation with ``offered`` tools; return the turn and its metadata, if the model gives any."""
        if self.with_meta is None:
            return self.llm.get_tool_response(self.messages, offered), None
        text, requests, call_meta = self.with_meta.get_tool_response_meta(self.messages, offered)
        return (text, requests), call_meta

    def turn(self, offered: list[Tool]) -> tuple[str, list[ToolCallRequest]]:
        """Take one metered turn and add its usage and metadata."""
        (text, requests), turn_meta = self.meter.metered(lambda: self.ask(offered), llm=self.llm)
        if turn_meta is not None:
            self.meta = turn_meta if self.meta is None else self.meta.combine(turn_meta)
        self.usage = self.usage + turn_usage(self.count_tokens, self.messages, text, requests)
        return text, requests

    def checked(self, text: str, check: Callable[[str], str | None] | None) -> str:
        """Return ``text``, or the answer to one more turn, tools withheld, when ``check`` names a problem with it."""
        problem = None if check is None else check(text)
        if problem is None:
            return text
        self.messages.append({"role": "assistant", "content": text})
        self.messages.append({"role": "user", "content": RETRY_NOTE.format(problem=problem)})
        return self.turn([])[0]

    def answer(
        self, text: str, check: Callable[[str], str | None] | None, calls: list[ToolCall]
    ) -> tuple[str, tuple[ToolCall, ...], TokenUsage, CallMeta | None]:
        """Return the checked final text, the calls made, and the usage and metadata of every turn."""
        final = self.checked(text, check)
        return final, tuple(calls), self.usage, self.meta

    def run_calls(self, tools: ToolIndex, text: str, requests: list[ToolCallRequest]) -> list[ToolCall]:
        """Add a turn's tool requests to the conversation, run them in order, and add each result."""
        wired = [
            {"id": r.id, "type": "function", "function": {"name": r.name, "arguments": r.arguments}} for r in requests
        ]
        self.messages.append({"role": "assistant", "content": text or None, "tool_calls": wired})
        calls = []
        for request in requests:
            call = execute_tool_call(tools, request)
            calls.append(call)
            self.messages.append({"role": "tool", "tool_call_id": request.id, "content": call.result})
        return calls


def execute_tool_call(tools: ToolIndex, request: ToolCallRequest) -> ToolCall:
    """Run one requested call; every failure but a refusal becomes a failed ToolCall.

    An XmasError with ``status`` 401, 403 or 429 is a refusal: it propagates and ends the run.
    So do :class:`~xmas.errors.BudgetExceeded` and :class:`~xmas.errors.Stopped`,
    which end a run wherever they are raised.
    """
    try:
        arguments = json.loads(request.arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolCall(request.name, {"raw": request.arguments}, f"invalid JSON arguments: {exc}", failed=True)
    if not isinstance(arguments, dict):
        return ToolCall(request.name, {"raw": request.arguments}, "invalid JSON arguments: not an object", failed=True)
    entry = tools.get(request.name)
    if entry is None:
        return ToolCall(request.name, arguments, f"unknown tool; available: {', '.join(tools)}", failed=True)
    spec, source = entry
    try:
        result = source.call(spec.name, arguments)
    except XmasError as exc:
        if exc.status in (401, 403, 429) or isinstance(exc, BudgetExceeded | Stopped):
            raise
        return ToolCall(request.name, arguments, str(exc), failed=True)
    return ToolCall(request.name, arguments, result)
