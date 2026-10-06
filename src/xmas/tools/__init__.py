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

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, overload

from pydantic import BaseModel, ValidationError

from xmas.errors import XmasError
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

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.agent import Agent


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
    arguments_model: type[BaseModel] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Reject names the OpenAI ``tools`` payload would refuse."""
        if not valid_tool_name(self.name):
            raise ValueError(
                f"Tool name {self.name!r} must match the OpenAI rule: letters, digits, _ and -, at most 64 characters."
            )

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
            XmasError: If the arguments do not match.
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
            return dict(arguments)
        try:
            instance = self.arguments_model.model_validate(arguments)
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


class LocalTools(ToolSource):
    """A ToolSource over in-process Tools; the MAS wraps bare Tool objects in one.

    ``call`` validates, then runs ``fn(**kwargs)``. Any exception becomes an
    :class:`~xmas.errors.XmasError`. A ``str`` result is returned as is, anything
    else as JSON (``default=str``), the same rule as serving-xmas's in-process executor.
    """

    def __init__(self, tools: Sequence[Tool]) -> None:
        """Initialize the source.

        Args:
            tools: Local tools, each with an ``fn``. Names must be unique.

        Raises:
            ValueError: If a tool has no ``fn`` or two tools share a name.
        """
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
        """Validate the arguments, run the function, and stringify its result.

        Args:
            name: The tool to run.
            arguments: The decoded arguments object.

        Returns:
            The result as text.

        Raises:
            XmasError: If the tool is unknown, the arguments are invalid, or the
                function raised.
        """
        spec = self.by_name.get(name)
        if spec is None:
            raise XmasError(f"unknown tool {name!r}")
        kwargs = spec.validate(arguments)
        try:
            result = spec(**kwargs)
        except Exception as exc:
            raise XmasError(f"{type(exc).__name__}: {exc}") from exc
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        """Name the wrapped tools."""
        return f"LocalTools({sorted(self.by_name)!r})"


ToolIndex = dict[str, tuple[Tool, ToolSource]]
"""Tools by name, each with the source that runs it."""


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
            raise ValueError(f"{function_name(func)}: a tool needs a docstring")
        summary, descriptions = split_docstring(doc)
        if not summary:
            raise ValueError(f"{function_name(func)}: the docstring has no summary paragraph")
        tool_name = name or getattr(func, "__name__", None)
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError(f"cannot derive a tool name for {func!r}; pass name=")
        model = arguments_model(func, descriptions)
        return Tool(tool_name, summary, json_schema(model, func), fn=func, arguments_model=model)

    return describe if fn is None else describe(fn)


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
    if llm is not None and not hasattr(llm, "get_tool_response"):
        raise ValueError(
            f"Agent {agent.name!r} uses tools, but its model {type(llm).__name__} has no get_tool_response."
        )
    return {name: index[name] for name in agent.tools}


def run_tool_loop(
    llm: Any,
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
        llm: The agent's model; it has a ``get_tool_response(messages, tools)``.
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

    def turn(offered: list[Tool]) -> tuple[str, list[ToolCallRequest], TokenUsage]:
        sent = sum(int(count_tokens(str(item.get("content") or ""))) for item in messages)
        text, requests = llm.get_tool_response(messages, offered)
        received = int(count_tokens(text)) + sum(int(count_tokens(r.arguments)) for r in requests)
        return text, requests, TokenUsage(input_tokens=sent, output_tokens=received)

    for _ in range(max_rounds):
        text, requests, turn_usage = turn(specs)
        usage = usage + turn_usage
        if not requests:
            return text, tuple(calls), usage
        wired = [
            {"id": r.id, "type": "function", "function": {"name": r.name, "arguments": r.arguments}} for r in requests
        ]
        messages.append({"role": "assistant", "content": text or None, "tool_calls": wired})
        for request in requests:
            call = execute_tool_call(tools, request)
            calls.append(call)
            messages.append({"role": "tool", "tool_call_id": request.id, "content": call.result})
    text, _, turn_usage = turn([])
    return text, tuple(calls), usage + turn_usage


def execute_tool_call(tools: ToolIndex, request: ToolCallRequest) -> ToolCall:
    """Run one requested call; every failure but a refusal becomes a failed ToolCall.

    An XmasError with ``status`` 401, 403 or 429 is a refusal: it propagates and ends the run.
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
        if exc.status in (401, 403, 429):
            raise
        return ToolCall(request.name, arguments, str(exc), failed=True)
    return ToolCall(request.name, arguments, result)
