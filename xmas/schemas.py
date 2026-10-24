"""Answer forms: the JSON a json agent must answer with, checked without vendor help.

A form is written in the README's dialect, one entry per field::

    fits: {type: bool, means: the draft makes one claim}
    action: {type: text, one of: [publish, shorten]}
    checks: {type: list of object, fields: {comment: {type: int}, reason: {type: text, optional: true}}}

``type`` is ``text``, ``bool``, ``int``, ``number``, ``json`` or ``object``, or
``list of <type>``; ``fields`` describes an object (or a list's objects);
``one of`` lists the allowed values (of a list's items, for a list); ``means``
is what the model is told; ``optional`` lets the field be left out. Fields the
form does not name are kept. Coercion is pydantic's lax mode, so ``"yes"`` is
``true`` and ``"3"`` is ``3``.

:func:`compile_form` turns a form into an :class:`AnswerForm`, whose
:meth:`~AnswerForm.parse` reads an answer and whose
:meth:`~AnswerForm.format_block` is the text appended to the agent's prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model

from xmas.errors import ErrorKind
from xmas.values import LIST_PREFIX

__all__ = [
    "RETRY_NOTE",
    "AnswerForm",
    "FormError",
    "compile_form",
]


RETRY_NOTE = "Your previous answer did not match the required format: {problem}\nAnswer again with only the JSON."
"""Appended to the prompt of the one retry; ``{problem}`` is the first problem."""

BASE: dict[str, Any] = {"text": str, "bool": bool, "int": int, "number": float, "json": Any}
WORDS = {"text": "text", "bool": "true or false", "int": "integer", "number": "number", "json": "any JSON"}
OBJECT = "object"
KEYS = frozenset({"type", "means", "optional", "one of", "fields"})
FENCE = re.compile(r"\A```[\w-]*[ \t]*\n(.*)\n[ \t]*```\Z", re.DOTALL)
ANY_JSON: TypeAdapter[Any] = TypeAdapter(Any)


class FormError(ValueError):
    """An answer does not match its form; ``problem`` names every problem on one line."""

    kind = ErrorKind.BAD_FORMAT

    def __init__(self, problem: str) -> None:
        """Keep the problem as the message."""
        super().__init__(problem)
        self.problem = problem


class AnswerForm:
    """A compiled answer form; ``fields`` is a plain copy of the form, or ``None`` for any JSON."""

    fields: dict[str, dict[str, Any]] | None

    def __init__(self, fields: Mapping[str, Mapping[str, Any]] | None) -> None:
        """Check the form, naming the field that breaks the dialect, and build its pydantic model."""
        self.fields = None if fields is None else plain(fields, "")
        self.model = None if self.fields is None else build_model(self.fields, "answer")

    def parse(self, text: str) -> Any:
        """Read an answer, raising :class:`FormError` when it does not match.

        Whitespace and at most one surrounding Markdown code fence are stripped;
        the rest must be JSON matching the form. Without a form any JSON passes.
        A form's object has the form's fields first (an optional field left out
        is ``None``), then any extra fields.
        """
        body = text.strip()
        if match := FENCE.match(body):
            body = match.group(1)
        try:
            if self.model is None:
                return ANY_JSON.validate_json(body)
            return self.model.model_validate_json(body).model_dump(mode="json", by_alias=True)
        except ValidationError as error:
            raise FormError(
                "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in error.errors())
            ) from error

    def format_block(self) -> str:
        """Return the format instructions appended to the agent's prompt.

        For the form ``{fits: {type: bool, means: it fits}}`` this is::

            Answer with one JSON object and nothing else. Its fields:
            - fits (true or false): it fits

        A field with ``one of`` reads ``(one of "a", "b")``, an optional one
        adds ``, optional``, and an object's fields follow on lines indented two
        spaces more. Without a form it is
        ``Answer with one JSON value and nothing else.``
        """
        if self.fields is None:
            return "Answer with one JSON value and nothing else."
        if not self.fields:
            return "Answer with one JSON object and nothing else."
        return "\n".join(["Answer with one JSON object and nothing else. Its fields:", *format_lines(self.fields, "")])

    def paths(self) -> tuple[str, ...]:
        """Return every field path in the form, nested objects as ``a.b``.

        Fields of a list's objects are not paths, since a path cannot index a
        list. Without a form there are none.
        """
        return () if self.fields is None else tuple(form_paths(self.fields, ""))

    def choices(self, path: str) -> tuple[Any, ...] | None:
        """Return the ``one of`` values of a field path, or ``None`` when it has none or is not in the form."""
        fields = self.fields
        spec: dict[str, Any] | None = None
        for name in path.split("."):
            spec = None if fields is None else fields.get(name)
            if spec is None:
                return None
            fields = spec.get("fields") if spec["type"] == OBJECT else None
        return None if spec is None or "one of" not in spec else tuple(spec["one of"])

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return f"AnswerForm({self.fields!r})"


def compile_form(fields: Mapping[str, Mapping[str, Any]] | None) -> AnswerForm:
    """Compile an answer form; ``None`` accepts any JSON and ``{}`` any JSON object."""
    return AnswerForm(fields)


def plain(fields: Mapping[Any, Any], where: str) -> dict[str, dict[str, Any]]:
    """Check one level of a form and return a plain copy of it."""
    if not isinstance(fields, Mapping):
        raise ValueError(f"Answer form{where}: fields must be a mapping, got {type(fields).__name__}.")
    return {name: plain_field(name, spec, f"{where}.{name}" if where else str(name)) for name, spec in fields.items()}


def plain_field(name: Any, spec: Any, path: str) -> dict[str, Any]:
    """Check one field at ``path`` and return a plain copy of it, nested fields included."""
    base = field_base(name, spec, path)
    kind = spec["type"]
    copy: dict[str, Any] = {"type": kind}
    if "means" in spec:
        copy["means"] = str(spec["means"])
    if "optional" in spec:
        if not isinstance(spec["optional"], bool):
            raise ValueError(f"Answer form field {path!r}: optional must be true or false.")
        copy["optional"] = spec["optional"]
    if "one of" in spec:
        choices = spec["one of"]
        if not isinstance(choices, list) or not choices or base in (OBJECT, "json"):
            raise ValueError(f"Answer form field {path!r}: one of needs a non-empty list on a plain type.")
        copy["one of"] = list(choices)
    if "fields" in spec:
        if base != OBJECT:
            raise ValueError(f"Answer form field {path!r}: only an object has fields, not {kind!r}.")
        copy["fields"] = plain(spec["fields"], path)
    return copy


def field_base(name: Any, spec: Any, path: str) -> str:
    """Check a field's name, keys and type, and return the base type under any ``list of`` words."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Answer form: field name {name!r} is not a non-empty string.")
    if not isinstance(spec, Mapping):
        raise ValueError(f"Answer form field {path!r}: expected a mapping such as {{type: text}}, got {spec!r}.")
    unknown = set(spec) - KEYS
    if unknown:
        raise ValueError(f"Answer form field {path!r}: unknown keys {sorted(unknown)}; known: {sorted(KEYS)}.")
    kind = spec.get("type")
    base = item_kind(kind)
    if base is None:
        raise ValueError(
            f"Answer form field {path!r}: unknown type {kind!r}; "
            "known: text, bool, int, number, json, object, list of <type>."
        )
    return base


def item_kind(kind: Any) -> str | None:
    """Return the base type under any ``list of`` words, or ``None`` if unknown."""
    while isinstance(kind, str) and kind.startswith(LIST_PREFIX):
        kind = kind[len(LIST_PREFIX) :].strip()
    return kind if isinstance(kind, str) and (kind in BASE or kind == OBJECT) else None


def build_model(fields: dict[str, dict[str, Any]], name: str) -> type[BaseModel]:
    """Build the pydantic model of one level, fields aliased as in ``tools._build_model``."""
    made: dict[str, Any] = {}
    for index, (key, spec) in enumerate(fields.items()):
        default = None if spec.get("optional") else ...
        extra = {"description": spec["means"]} if "means" in spec else {}
        made[f"f{index}"] = (annotation(spec, spec["type"], f"{name}.{key}"), Field(default, alias=key, **extra))
    model: type[BaseModel] = create_model(name, __config__=ConfigDict(extra="allow"), **made)
    return model


def annotation(spec: dict[str, Any], kind: str, name: str) -> Any:
    """Return the Python type of a field whose (remaining) type words are ``kind``."""
    if kind.startswith(LIST_PREFIX):
        return list[annotation(spec, kind[len(LIST_PREFIX) :].strip(), name)]  # type: ignore[misc]
    if "one of" in spec:
        return Literal[tuple(spec["one of"])]
    if kind == OBJECT:
        return build_model(spec["fields"], name) if "fields" in spec else dict[str, Any]
    return BASE[kind]


def phrase(spec: dict[str, Any], kind: str) -> str:
    """Return the words for a field's type, such as ``list of text`` or ``one of "a", "b"``."""
    if kind.startswith(LIST_PREFIX):
        return LIST_PREFIX + phrase(spec, kind[len(LIST_PREFIX) :].strip())
    if "one of" in spec:
        return "one of " + ", ".join(json.dumps(choice, ensure_ascii=False) for choice in spec["one of"])
    return OBJECT if kind == OBJECT else WORDS[kind]


def format_lines(fields: dict[str, dict[str, Any]], indent: str) -> list[str]:
    """Return the format block's lines for one level of a form."""
    lines: list[str] = []
    for key, spec in fields.items():
        words = phrase(spec, spec["type"]) + (", optional" if spec.get("optional") else "")
        means = f": {spec['means']}" if "means" in spec else ""
        lines.append(f"{indent}- {key} ({words}){means}")
        if "fields" in spec:
            lines.extend(format_lines(spec["fields"], indent + "  "))
    return lines


def form_paths(fields: dict[str, dict[str, Any]], prefix: str) -> list[str]:
    """Return the paths of one level of a form and of its nested objects."""
    paths: list[str] = []
    for key, spec in fields.items():
        path = prefix + key
        paths.append(path)
        if spec["type"] == OBJECT and "fields" in spec:
            paths.extend(form_paths(spec["fields"], path + "."))
    return paths
