"""From a plain function to a tool description: docstring parsing and the pydantic arguments model."""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator

__all__ = [
    "args_sections",
    "arguments_from",
    "arguments_model",
    "describe_validation_error",
    "docstring_summary",
    "function_name",
    "indent_of",
    "is_args_header",
    "json_schema",
    "parse_args_section",
    "split_docstring",
    "valid_tool_name",
]

MAX_TOOL_NAME_LENGTH = 64
"""The longest tool name the OpenAI ``tools`` payload accepts."""


def valid_tool_name(name: str) -> bool:
    """Whether the OpenAI ``tools`` payload accepts ``name``: ASCII letters, digits, ``_`` and ``-``, 1 to 64 long."""
    return 0 < len(name) <= MAX_TOOL_NAME_LENGTH and all((c.isascii() and c.isalnum()) or c in "_-" for c in name)


def split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {param: description})`` for a cleaned docstring (Google ``Args:``)."""
    lines = doc.splitlines()
    params: dict[str, str] = {}
    for section in args_sections(lines):
        params.update(parse_args_section(section))
    return docstring_summary(lines), params


def docstring_summary(lines: list[str]) -> str:
    """Join the docstring lines before the first blank line or parameter section."""
    summary_lines: list[str] = []
    for line in lines:
        if not line.strip() or is_args_header(line.strip()):
            break
        summary_lines.append(line.strip())
    return " ".join(summary_lines)


def args_sections(lines: list[str]) -> Iterator[list[str]]:
    """Yield the non-blank lines of each parameter section, its header left out.

    A line indented no deeper than the header (``Returns:``, ``Raises:``, prose)
    ends the section, and opens a new one if it is a parameter header itself.
    """
    section: list[str] | None = None
    header_indent = 0
    for line in lines:
        stripped = line.strip()
        if section is None:
            if is_args_header(stripped):
                section, header_indent = [], indent_of(line)
            continue
        if not stripped:
            continue
        if indent_of(line) > header_indent:
            section.append(line)
            continue
        yield section
        section = [] if is_args_header(stripped) else None
        header_indent = indent_of(line)
    if section is not None:
        yield section


def parse_args_section(section: list[str]) -> dict[str, str]:
    """Return ``{param: description}`` from one section's lines; deeper lines continue the entry above."""
    params: dict[str, str] = {}
    entry_indent: int | None = None
    current: str | None = None
    for line in section:
        stripped = line.strip()
        indent = indent_of(line)
        if entry_indent is None:
            entry_indent = indent
        head, sep, text = stripped.partition(":")
        name = head.lstrip("*").split("(")[0].strip()
        if indent == entry_indent and sep and name.isidentifier():
            current = name
            params[current] = text.strip()
        elif indent > entry_indent and current is not None:
            params[current] = f"{params[current]} {stripped}".strip()
    return params


def is_args_header(stripped: str) -> bool:
    """Whether a stripped docstring line opens the parameter section."""
    return stripped.endswith(":") and stripped[:-1].strip() in ("Args", "Arguments", "Parameters")


def indent_of(line: str) -> int:
    """Count a line's leading whitespace."""
    return len(line) - len(line.lstrip())


def function_name(fn: Callable[..., Any]) -> str:
    """Name a callable in an error message."""
    return getattr(fn, "__qualname__", None) or repr(fn)


def arguments_model(fn: Callable[..., Any], descriptions: dict[str, str]) -> type[BaseModel]:
    """Build the pydantic arguments model of ``fn``, extra fields forbidden."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot read the signature of {fn!r}") from exc
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(f"{function_name(fn)}: *args/**kwargs are not allowed ({param.name})")
        if param.kind is param.POSITIONAL_ONLY:
            raise ValueError(f"{function_name(fn)}: positional-only parameter {param.name!r}")
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise ValueError(f"{function_name(fn)}: cannot resolve type hints: {exc}") from exc
    fields: dict[str, Any] = {}
    for param in params:
        if param.name not in hints:
            raise ValueError(f"{function_name(fn)}: parameter {param.name!r} has no type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        extra: dict[str, Any] = {"alias": param.name}
        if descriptions.get(param.name):
            extra["description"] = descriptions[param.name]
        # Prefixed and aliased back, so parameters named json or schema cannot shadow BaseModel attributes.
        fields["p_" + param.name] = (hints[param.name], Field(default, **extra))
    try:
        model: type[BaseModel] = create_model(
            f"{getattr(fn, '__name__', 'tool')}_arguments",
            __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=False),
            **fields,
        )
    except Exception as exc:
        raise ValueError(f"{function_name(fn)}: cannot build an arguments model: {exc}") from exc
    return model


def json_schema(model: type[BaseModel], fn: Callable[..., Any]) -> dict[str, Any]:
    """Return the arguments model as the JSON Schema the model is shown, titles dropped."""
    try:
        schema = model.model_json_schema(by_alias=True)
    except Exception as exc:
        raise ValueError(f"{function_name(fn)}: type hints are not JSON-Schema-able: {exc}") from exc
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    schema["additionalProperties"] = False
    return schema


def arguments_from(instance: BaseModel) -> dict[str, Any]:
    """Return a validated arguments model as keyword arguments under the real parameter names."""
    return {info.alias or name: getattr(instance, name) for name, info in type(instance).model_fields.items()}


def describe_validation_error(exc: ValidationError) -> str:
    """Join pydantic's errors into one line, ``location: message`` each."""
    problems = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        problems.append(f"{location}: {err['msg']}")
    return "; ".join(problems)
