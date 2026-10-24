"""Edges of spec version 2: the chain strings and one-key mappings a spec writes.

A spec lists its edges as items of two forms::

    - writer -> count -> judge                                # a chain of plain edges
    - judge -> writer: {when: not verdict.fits, max_visits: 3}  # one conditional edge
    - judge -> apology: else
    - count -> apology: {on error: [timeout, outage]}

    - judge -> publish: {switch: verdict.action}              # case defaults to the target
    - judge -> hold: {switch: verdict.action, case: wait}

    - outline -> quick_notes: first of                       # a race: the first to finish wins
    - outline -> careful_notes: first of

:func:`parse_edges` reads them into :class:`Edge` objects, one per arrow, and
:func:`dump_edges` writes them back canonically. Whether the predicates, paths
and error kinds make sense, and whether a pair of nodes is listed twice, is
:func:`xmas.check.check_system`'s question.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "ANY_ERROR",
    "ARROW",
    "Edge",
    "EdgeKind",
    "dump_edges",
    "parse_edges",
]


ARROW = "->"
"""What separates the names of a chain."""

EDGE_ENDS = 2
"""The nodes one edge joins: a chain names at least two, a conditional edge exactly two."""

ANY_ERROR = "any"
"""The ``on error`` spec that routes every error kind."""

ON_ERROR = "on error"


class EdgeKind(StrEnum):
    """What decides whether an edge is taken."""

    PLAIN = "plain"
    """Always taken."""
    WHEN = "when"
    """Taken when its predicate holds."""
    ELSE = "else"
    """Taken when no ``when``, ``while`` or ``switch`` edge of its source is."""
    WHILE = "while"
    """A ``when`` edge back into a loop, taken at most ``max_visits`` times."""
    SWITCH = "switch"
    """Taken when the switched path equals ``case``."""
    ERROR = "error"
    """Taken when the source fails with one of ``errors``."""
    FIRST = "first of"
    """One of a race: every ``first of`` edge of a source is taken, and the target that finishes first wins."""


@dataclass(frozen=True)
class Edge:
    """One edge between two nodes.

    ``when`` is the predicate of a ``when`` or ``while`` edge, and
    ``max_visits`` how often a ``while`` edge's target may run. ``switch`` is
    the path a ``switch`` edge compares and ``case`` the value it takes, the
    target name by default. ``errors`` are the error kinds an ``error`` edge
    routes, or ``("any",)``.
    """

    source: str
    target: str
    kind: EdgeKind = EdgeKind.PLAIN
    when: str | None = None
    max_visits: int | None = None
    switch: str | None = None
    case: str | int | float | bool | None = None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check that the fields fit the kind, filling a switch's default case."""
        if not self.source or not self.target:
            raise ValueError(f"An edge needs two node names, got {self.source!r} -> {self.target!r}.")
        kind = EdgeKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "errors", tuple(self.errors))
        if kind is EdgeKind.SWITCH and self.case is None:
            object.__setattr__(self, "case", self.target)
        needs = {
            "when": kind in (EdgeKind.WHEN, EdgeKind.WHILE),
            "max_visits": kind is EdgeKind.WHILE,
            "switch": kind is EdgeKind.SWITCH,
            "case": kind is EdgeKind.SWITCH,
            "errors": kind is EdgeKind.ERROR,
        }
        for name, needed in needs.items():
            value = getattr(self, name)
            given = value is not None if name in ("case", "max_visits") else bool(value)  # case 0 is a case
            if given != needed:
                state = "needs" if needed else "takes no"
                raise ValueError(f"Edge {self.label}: a {kind.value} edge {state} {name!r}.")
        if self.max_visits is not None and self.max_visits < 1:
            raise ValueError(f"Edge {self.label}: max_visits must be a positive integer, got {self.max_visits!r}.")

    @property
    def label(self) -> str:
        """The edge as a spec names it, ``"a -> b"``."""
        return f"{self.source} {ARROW} {self.target}"

    def spec(self) -> str | dict[str, Any] | None:
        """Return the spec of a conditional edge as a spec writes it, ``None`` for a plain edge."""
        if self.kind is EdgeKind.PLAIN:
            return None
        if self.kind in (EdgeKind.ELSE, EdgeKind.FIRST):
            return self.kind.value
        if self.kind is EdgeKind.ERROR:
            return {ON_ERROR: ANY_ERROR if self.errors == (ANY_ERROR,) else list(self.errors)}
        if self.kind is EdgeKind.SWITCH:
            data: dict[str, Any] = {"switch": self.switch}
            if self.case != self.target:
                data["case"] = self.case
            return data
        data = {"when": self.when}
        if self.max_visits is not None:
            data["max_visits"] = self.max_visits
        return data


def split_names(text: str) -> list[str]:
    """Split ``"a -> b -> c"`` into its names."""
    names = [part.strip() for part in text.split(ARROW)]
    if len(names) < EDGE_ENDS or not all(names):
        raise ValueError(f"Edge {text!r} must name two or more nodes joined by '{ARROW}'.")
    return names


def conditional(source: str, target: str, spec: Any) -> Edge:
    """Build one edge from the spec of a one-key mapping."""
    where = f"{source} {ARROW} {target}"
    if spec in (EdgeKind.ELSE.value, EdgeKind.FIRST.value):
        return Edge(source, target, EdgeKind(spec))
    if not isinstance(spec, dict):
        raise ValueError(f"Edge {where!r}: its spec is 'else', 'first of' or a mapping, not {spec!r}.")
    deciders = [key for key in ("when", "switch", ON_ERROR) if key in spec]
    if len(deciders) != 1:
        raise ValueError(f"Edge {where!r} needs exactly one of 'when', 'switch' or 'on error', got {sorted(spec)}.")
    allowed = {"when": {"when", "max_visits"}, "switch": {"switch", "case"}, ON_ERROR: {ON_ERROR}}[deciders[0]]
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Edge {where!r} has unknown keys: {sorted(unknown)}.")
    if "when" in spec:
        kind = EdgeKind.WHILE if "max_visits" in spec else EdgeKind.WHEN
        visits = spec.get("max_visits")
        if visits is not None and (not isinstance(visits, int) or isinstance(visits, bool)):
            raise ValueError(f"Edge {where!r}: max_visits must be a positive integer, got {visits!r}.")
        return Edge(source, target, kind, when=text(spec["when"], where), max_visits=visits)
    if "switch" in spec:
        case = spec.get("case")
        return Edge(source, target, EdgeKind.SWITCH, switch=text(spec["switch"], where), case=case)
    errors = spec[ON_ERROR]
    if isinstance(errors, str):
        errors = [errors]
    if not isinstance(errors, list) or not errors or not all(isinstance(kind, str) and kind for kind in errors):
        raise ValueError(f"Edge {where!r}: 'on error' takes 'any', an error kind or a list of them, not {errors!r}.")
    return Edge(source, target, EdgeKind.ERROR, errors=tuple(errors))


def text(value: Any, where: str) -> str:
    """Return a predicate or path, refusing anything but non-empty text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Edge {where!r}: expected text, got {value!r}.")
    return value.strip()


def parse_edges(items: Iterable[Any]) -> list[Edge]:
    """Read a spec's edge list into edges, in the order written; a chain gives one edge per arrow.

    The items are chain strings such as ``"a -> b -> c"`` and one-key mappings
    such as ``{"a -> b": {"when": "..."}}``, ``{"a -> b": "else"}`` or
    ``{"a -> b": "first of"}``; anything else raises ValueError.
    """
    if isinstance(items, (str, dict)):
        raise ValueError("Edges are a list of chains and one-key mappings.")
    edges: list[Edge] = []
    for item in items:
        if isinstance(item, str):
            names = split_names(item)
            edges.extend(Edge(source, target) for source, target in pairwise(names))
        elif isinstance(item, dict) and len(item) == 1:
            ((key, spec),) = item.items()
            names = split_names(key) if isinstance(key, str) else []
            if len(names) != EDGE_ENDS:
                raise ValueError(f"A conditional edge names one source and one target, not {key!r}.")
            edges.append(conditional(names[0], names[1], spec))
        else:
            raise ValueError(f"Edge {item!r} must be a chain such as 'a -> b' or a one-key mapping.")
    return edges


def dump_edges(edges: Sequence[Edge]) -> list[str | dict[str, Any]]:
    """Write edges as a spec lists them, canonically.

    Edges keep their order; consecutive plain edges that continue one another
    (``a -> b`` then ``b -> c``) merge into one chain; defaults are left out.
    The caller passes declared edges only, so implicit start and end wiring is
    never written. Dumping what :func:`parse_edges` reads back from a dump
    gives the same list.
    """
    items: list[str | dict[str, Any]] = []
    chain: list[str] = []
    for edge in edges:
        plain = edge.kind is EdgeKind.PLAIN
        if plain and chain and chain[-1] == edge.source:
            chain.append(edge.target)
            continue
        if chain:
            items.append(f" {ARROW} ".join(chain))
        chain = [edge.source, edge.target] if plain else []
        if not plain:
            items.append({edge.label: edge.spec()})
    if chain:
        items.append(f" {ARROW} ".join(chain))
    return items
