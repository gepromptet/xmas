"""Typed values: the type words of spec version 2, and how values render and check.

A value in a run's state is plain data: text is a ``str``, json is plain JSON
data (``dict`` with ``str`` keys, ``list``, ``str``, ``int``, ``float``,
``bool``, ``None``), and an image, audio, document, table or file is a
:class:`~xmas.media.Ref`. The type lives in the system, never on the value, so
a version 1 state of strings stays a valid version 2 state of text.

It imports no xmas module but :mod:`xmas.media`.
"""

from __future__ import annotations

import json
import math
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xmas.media import MEDIA_KINDS, Kind, Ref, Store, fits_kind

__all__ = [
    "FILE",
    "JSON",
    "KINDS",
    "LIST_PREFIX",
    "MEDIA",
    "TEXT",
    "ValueType",
    "check_value",
    "coerce_media",
    "fits",
    "media_kind",
    "parse_type",
    "plain",
    "render_value",
    "type_of_hint",
]


KINDS = ("text", "json", *MEDIA_KINDS)
"""Every value kind of the format; ``list of <kind>`` builds on them."""

LIST = "list"
LIST_PREFIX = "list of "
"""How a list type starts, in value types and in answer forms alike."""


@dataclass(frozen=True)
class ValueType:
    """The type of one value: ``kind`` is one of :data:`KINDS`, or ``"list"`` with ``item`` its items' type."""

    kind: str
    item: ValueType | None = None

    def __post_init__(self) -> None:
        """Refuse unknown kinds and an item on anything but a list."""
        if self.kind == LIST:
            if self.item is None:
                raise ValueError("A list type needs an item type.")
        elif self.kind not in KINDS:
            raise ValueError(f"Unknown value type {self.kind!r}; known: {', '.join(KINDS)}, list of <type>.")
        elif self.item is not None:
            raise ValueError(f"Only a list has an item type, not {self.kind!r}.")

    def __str__(self) -> str:
        """Return the type as the spec writes it, such as ``list of text``."""
        return f"{LIST_PREFIX}{self.item}" if self.item is not None else self.kind


TEXT = ValueType("text")
"""A ``str``."""

JSON = ValueType("json")
"""Plain JSON data."""

FILE = ValueType("file")
"""Any :class:`~xmas.media.Ref`."""

MEDIA = tuple(ValueType(kind) for kind in MEDIA_KINDS)
"""The media types: image, audio, document, table and file, each a :class:`~xmas.media.Ref`."""


def parse_type(text: str) -> ValueType:
    """Read a type as the spec writes it, such as ``"json"`` or ``"list of text"``; ValueError if it names none."""
    if not isinstance(text, str):
        raise ValueError(f"A value type is a string such as 'text', not {text!r}.")
    words = text.strip()
    if words.startswith(LIST_PREFIX):
        return ValueType(LIST, parse_type(words[len(LIST_PREFIX) :]))
    return ValueType(words)


def media_kind(vtype: ValueType) -> str | None:
    """Return the media kind of a media type or of a list's innermost items, else ``None``."""
    while vtype.item is not None:
        vtype = vtype.item
    return vtype.kind if vtype.kind in MEDIA_KINDS else None


def type_of_hint(hint: Any) -> ValueType:
    """Return the value type of a callable's parameter or return value.

    ``str`` (``Annotated`` unwrapped) is text, ``Annotated[Ref, Kind(k)]`` (such
    as :data:`~xmas.media.Image`) is ``k``, a bare ``Ref`` is file, a ``list``
    of a media type is a list of it, and anything else is json, so a
    ``str | None`` return is json (``None`` is a json value). A JSON-schema
    mapping, as a tool describes its parameters, is the kind its
    ``"x-xmas-kind"`` names (on ``items`` for a list), text for
    ``{"type": "string"}``, either of them when it may also be null (a
    ``str | None`` parameter), and json otherwise.
    """
    if isinstance(hint, dict):
        return type_of_schema(hint)
    if typing.get_origin(hint) is typing.Annotated:
        hint, *extras = typing.get_args(hint)
        if hint is Ref:
            return ValueType(next((extra.kind for extra in reversed(extras) if isinstance(extra, Kind)), "file"))
    if hint is Ref:
        return FILE
    if typing.get_origin(hint) is list and len(args := typing.get_args(hint)) == 1:
        item = type_of_hint(args[0])
        if media_kind(item) is not None:
            return ValueType(LIST, item)
    return TEXT if hint is str else JSON


def type_of_schema(schema: dict[str, Any]) -> ValueType:
    """Read a parameter's JSON schema as :func:`type_of_hint` describes."""
    options = [
        option
        for option in schema.get("anyOf") or [schema]
        if not (isinstance(option, dict) and option.get("type") == "null")
    ]
    if len(options) == 1 and isinstance(option := options[0], dict):
        media = media_of_schema(option)
        if media is not None:
            return media
    kinds: set[Any] = set()
    for option in schema.get("anyOf") or [schema]:
        kind = option.get("type") if isinstance(option, dict) else None
        kinds.update(kind if isinstance(kind, list) else [kind])
    return TEXT if "string" in kinds and kinds <= {"string", "null"} else JSON


def media_of_schema(schema: dict[str, Any]) -> ValueType | None:
    """Return the media type a schema marks, a list of one for an array of marked items, else ``None``."""
    kind = schema.get("x-xmas-kind")
    if kind in MEDIA_KINDS:
        return ValueType(kind)
    items = schema.get("items")
    if schema.get("type") == "array" and isinstance(items, dict):
        item = type_of_schema(items)  # items may be null too, as a gathered list's failed ones are
        return ValueType(LIST, item) if media_kind(item) is not None else None
    return None


def render_value(value: Any, vtype: ValueType, *, present: bool = True) -> str:
    """Render a value as the text a prompt, a record or a trace output shows.

    Text renders as itself; json and lists render as one line of
    ``json.dumps(value, ensure_ascii=False)`` with the keys in the order the
    value has them, so ``0``, ``false``, ``[]`` and a json ``null`` show as
    JSON. A ref renders as the same line of :meth:`~xmas.media.Ref.to_dict`,
    a list of refs as one JSON list of them with ``null`` items kept: a model
    sees where a file is, never its bytes. A missing value renders as ``""``:
    the caller says so with ``present=False``, or passes ``None`` for a text or
    media value.
    """
    if not present or (value is None and (vtype == TEXT or vtype in MEDIA)):
        return ""
    if vtype == TEXT:
        return value if isinstance(value, str) else str(value)
    return json.dumps(plain(value) if media_kind(vtype) is not None else value, ensure_ascii=False)


def plain(value: Any) -> Any:
    """Return ``value`` with every :class:`~xmas.media.Ref` in it, at any depth, as its dict."""
    if isinstance(value, Ref):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def coerce_media(value: Any, vtype: ValueType, *, store: Store | None = None) -> Any:
    """Turn a :class:`~pathlib.Path` given for a media type, or for items of a list of one, into a ref.

    Without ``store`` the file is referenced where it is
    (:meth:`~xmas.media.Ref.from_path`, its sha256 checked on every read);
    with one, its bytes are copied into the store, so rewriting the path later
    leaves the ref intact. Every other value passes unchanged, for
    :func:`check_value` to judge.
    """
    if media_kind(vtype) is None:
        return value
    if vtype.item is not None:
        return [coerce_media(item, vtype.item, store=store) for item in value] if isinstance(value, list) else value
    if not isinstance(value, Path):
        return value
    ref = Ref.from_path(value)
    return ref if store is None else store.put(ref.read_bytes(), ref.media_type, ref.name)


def check_value(value: Any, vtype: ValueType) -> str | None:
    """Return ``None`` when a value fits its type, else one line such as ``"not JSON at counts.words: set"``.

    A list checks every item and allows ``null`` items, which a gathered list
    holds where an item failed; a bad item is named by its index, such as
    ``"expected text at 2, got int"``. A media value is a
    :class:`~xmas.media.Ref` of its kind (``"expected image, got a Ref of
    text/csv"``); a file takes any ref.
    """
    return fit_problem(value, vtype, "")


def fits(value: ValueType, wanted: ValueType) -> bool:
    """Whether a value of type ``value`` may be read where ``wanted`` is declared.

    The same type, a list as json, or any media value as a file.
    """
    return value == wanted or (wanted == JSON and value.item is not None) or (wanted == FILE and value in MEDIA)


def fit_problem(value: Any, vtype: ValueType, path: str) -> str | None:
    """Return where ``value`` stops fitting ``vtype``, or ``None``."""
    where = f" at {path}" if path else ""
    if vtype.item is not None:
        if not isinstance(value, list):
            return f"expected {vtype}{where}, got {type(value).__name__}"
        for index, item in enumerate(value):
            problem = None if item is None else fit_problem(item, vtype.item, join(path, index))
            if problem is not None:
                return problem
        return None
    if vtype == TEXT:
        return None if isinstance(value, str) else f"expected text{where}, got {type(value).__name__}"
    if vtype in MEDIA:
        return media_problem(value, vtype.kind, where)
    return json_problem(value, path)


def media_problem(value: Any, kind: str, where: str) -> str | None:
    """Return why ``value`` is not a ref of ``kind``, or ``None``."""
    if isinstance(value, Ref):
        return None if fits_kind(value, kind) else f"expected {kind}{where}, got a Ref of {value.media_type}"
    if isinstance(value, (bytes, bytearray)):
        return f"expected {kind}{where}, got bytes; return a Path or a Ref, Ref.from_bytes stores bytes"
    return f"expected {kind}{where}, got {type(value).__name__}"


def json_problem(value: Any, path: str) -> str | None:
    """Return where ``value`` stops being plain JSON data, or ``None``."""
    where = f" at {path}" if path else ""
    if value is None or isinstance(value, (str, bool, int)):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else f"not JSON{where}: {value!r}"
    if isinstance(value, list):
        problems = (json_problem(item, join(path, index)) for index, item in enumerate(value))
        return next((problem for problem in problems if problem is not None), None)
    if isinstance(value, dict):
        return mapping_problem(value, path)
    return f"not JSON{where}: {type(value).__name__}"


def mapping_problem(value: dict[Any, Any], path: str) -> str | None:
    """Return the first key of ``value`` that is not a string, or the first problem of its items, or ``None``."""
    for key, item in value.items():
        if not isinstance(key, str):
            where = f" at {path}" if path else ""
            return f"not JSON{where}: key {key!r} is not a string"
        problem = json_problem(item, join(path, key))
        if problem is not None:
            return problem
    return None


def join(path: str, step: int | str) -> str:
    """Return ``path`` one item or key deeper, such as ``counts.words``."""
    return f"{path}.{step}" if path else str(step)
