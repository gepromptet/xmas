"""``for each`` chains: a group of nodes that runs once per item of a list.

A node joins a chain by naming an item and the list it comes from, ``for each:
page in plan.slides``. Nodes with the same item and list that are joined by
edges among themselves make one chain; :func:`find_chains` finds them and their
head, the node every edge from outside lands on. Outside the chain, the chain is
its head: :func:`collapse` drops the edges inside it and makes its exits leave
from the head, and a value written inside, of type ``T``, is a ``list of T``
outside, one entry per item in item order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xmas.edges import Edge, EdgeKind

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS
    from xmas.values import ValueType

__all__ = [
    "ITEM_BUDGET_KEYS",
    "Chain",
    "ChainShape",
    "ForEach",
    "collapse",
    "find_chains",
]


ITEM_BUDGET_KEYS = ("cost_usd", "wall_s")
"""What an item budget may limit: an item's summed cost in USD and its summed latency in seconds."""

NAME = r"[A-Za-z_][A-Za-z0-9_]*"
PATH = rf"{NAME}(?:\.{NAME})*"
FORM = re.compile(rf"\A\s*({NAME})\s+in\s+({PATH})\s*\Z")
SPEC_KEYS = {"item": "item", "in": "over", "at once": "at_once", "key": "key", "item budget": "item_budget"}


@dataclass(frozen=True)
class ForEach:
    """Where a chain's items come from, and the chain's options.

    Attributes:
        item: The name an item has inside the chain.
        over: The list: a value name plus ``.field`` steps, such as
            ``plan.slides``.
        at_once: How many items run at the same time; ``None`` for no cap.
        key: The field of a json item that names it on the records, such as
            ``id``; ``None`` names items by their index only.
        item_budget: ``{"cost_usd": ..., "wall_s": ...}`` limits per item, which
            flag an item that passes them and stop nothing.
    """

    item: str
    over: str
    at_once: int | None = None
    key: str | None = None
    item_budget: Mapping[str, float] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        """Refuse a malformed item, list or option, and keep a plain copy of the item budget."""
        item: Any = self.item
        over: Any = self.over
        if not (
            isinstance(item, str) and isinstance(over, str) and re.fullmatch(NAME, item) and re.fullmatch(PATH, over)
        ):
            raise ValueError(
                f"for each {item!r} in {over!r}: the item is a name and the list a value name with '.field' steps, "
                "such as 'page in plan.slides'."
            )
        at_once: Any = self.at_once
        if at_once is not None and (isinstance(at_once, bool) or not isinstance(at_once, int) or at_once < 1):
            raise ValueError(f"for each {self}: at once must be a positive integer, got {at_once!r}.")
        key: Any = self.key
        if key is not None and not (isinstance(key, str) and re.fullmatch(NAME, key)):
            raise ValueError(f"for each {self}: key names a field of the item, got {key!r}.")
        budget: Any = self.item_budget
        if budget is None:
            return
        if not (isinstance(budget, Mapping) and budget and set(budget) <= set(ITEM_BUDGET_KEYS)) or not all(
            isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit > 0 for limit in budget.values()
        ):
            raise ValueError(
                f"for each {self}: item budget maps {list(ITEM_BUDGET_KEYS)} to positive numbers, got {budget!r}."
            )
        object.__setattr__(self, "item_budget", dict(budget))

    @property
    def root(self) -> str:
        """The value the list is read from."""
        return self.over.split(".", 1)[0]

    @property
    def fields(self) -> tuple[str, ...]:
        """The field steps from the root to the list; empty when the root is the list."""
        return tuple(self.over.split(".")[1:])

    @property
    def options(self) -> dict[str, Any]:
        """The chain-wide options that are set, as a spec writes them."""
        budget = None if self.item_budget is None else dict(self.item_budget)
        given: dict[str, Any] = {"at once": self.at_once, "key": self.key, "item budget": budget}
        return {name: value for name, value in given.items() if value is not None}

    def __str__(self) -> str:
        """Return the form as a spec writes it, ``"page in plan.slides"``."""
        return f"{self.item} in {self.over}"

    def to_spec(self) -> str | dict[str, Any]:
        """Return the short form, or the mapping ``{item, in, ...}`` with the set options when there are any."""
        options = self.options
        return {"item": self.item, "in": self.over, **options} if options else str(self)

    @classmethod
    def parse(cls, spec: str | Mapping[str, Any]) -> ForEach:
        """Read ``"page in plan.slides"``, or ``{item: page, in: plan.slides}`` with any of the options."""
        if isinstance(spec, str):
            match = FORM.match(spec)
            if match is None:
                raise ValueError(f"for each {spec!r} must read '<item> in <list>', such as 'page in plan.slides'.")
            return cls(match[1], match[2])
        if not isinstance(spec, Mapping):
            raise ValueError(f"for each is a text such as 'page in plan.slides' or a mapping, not {spec!r}.")
        unknown, missing = sorted(set(spec) - set(SPEC_KEYS)), sorted({"item", "in"} - set(spec))
        if unknown or missing:
            problem = f"has unknown keys {unknown}" if unknown else f"is missing the keys {missing}"
            raise ValueError(f"for each {dict(spec)!r} {problem}; it takes {list(SPEC_KEYS)}.")
        return cls(**{SPEC_KEYS[key]: value for key, value in spec.items()})


@dataclass(frozen=True)
class ChainShape:
    """One chain as :func:`find_chains` finds it in the edges.

    Attributes:
        head: The node every edge from outside the chain lands on.
        nodes: The chain's nodes, in the order they were given.
        each: The head's form, which holds the chain-wide options.
        internal: The edges between the chain's nodes, in declared order.
        exits: The edges from a chain node to a node outside, in declared order;
            none when the chain feeds the end.
    """

    head: str
    nodes: tuple[str, ...]
    each: ForEach
    internal: tuple[Edge, ...]
    exits: tuple[Edge, ...]


@dataclass(frozen=True)
class Chain:
    """A checked chain: what the engine runs once per item.

    Attributes:
        head: The chain's head, which names it.
        nodes: The chain's nodes.
        each: The head's form.
        inputs: What one item's run starts from: ``{item: type}`` and each value
            from outside that a chain node reads, with its type.
        writes: Each value the chain writes, with its type for one item; the
            gathered value outside is a list of it.
        system: The chain's nodes as a system of their own, which one item runs.
    """

    head: str
    nodes: tuple[str, ...]
    each: ForEach
    inputs: dict[str, ValueType]
    writes: dict[str, ValueType]
    system: MAS


def find_chains(
    nodes: Sequence[str], edges: Sequence[Edge], each: Mapping[str, ForEach]
) -> tuple[list[ChainShape], list[str]]:
    """Group the nodes that carry a form into chains, and say what is wrong with each group.

    Nodes with an equal item and list that edges join make one chain; the same
    form on two groups no edge joins makes two. A chain has one head: every
    edge from outside lands on it, and so does the implicit start (a node with
    no incoming edge but ``while`` edges). Its exits are ``plain`` or ``else``
    edges to one node outside. The options sit on the head only. A sound group
    becomes a :class:`ChainShape`; the others give problems instead.
    """
    group_of = {name: name for name in nodes if name in each}

    def find(name: str) -> str:
        while group_of[name] != name:
            name = group_of[name]
        return name

    same = [(e.source, e.target) for e in edges if e.source in group_of and e.target in group_of]
    for source, target in same:
        left, right = each[source], each[target]
        if (left.item, left.over) == (right.item, right.over):
            group_of[find(target)] = find(source)
    groups: dict[str, list[str]] = {}
    for name in group_of:
        groups.setdefault(find(name), []).append(name)

    shapes: list[ChainShape] = []
    problems: list[str] = []
    for members in groups.values():
        inside = set(members)
        entries = {e.target for e in edges if e.target in inside and e.source not in inside}
        entries |= {m for m in members if not any(e.target == m and e.kind is not EdgeKind.WHILE for e in edges)}
        if len(entries) != 1:
            problems.append(
                f"chain {members} ({each[members[0]]}) is entered at {sorted(entries) or 'no node'}; a chain has "
                "one head, which every edge from outside it, the start included, lands on"
            )
            continue
        (head,) = entries
        exits = tuple(e for e in edges if e.source in inside and e.target not in inside)
        found = [
            f"chain {head!r}: the {e.kind.value} edge {e.label} leaves it; route it inside the chain"
            for e in exits
            if e.kind not in (EdgeKind.PLAIN, EdgeKind.ELSE)
        ]
        targets = sorted({e.target for e in exits})
        if len(targets) > 1:
            found.append(f"chain {head!r} leaves to {targets}; a chain leaves to one node")
        found += [
            f"chain {head!r}: {m!r} sets {', '.join(each[m].options)}, which only the head {head!r} sets"
            for m in members
            if m != head and each[m].options
        ]
        problems += found
        if not found:
            internal = tuple(e for e in edges if e.source in inside and e.target in inside)
            shapes.append(ChainShape(head, tuple(members), each[head], internal, exits))
    return shapes, problems


def collapse(shapes: Sequence[ChainShape], edges: Sequence[Edge]) -> tuple[Edge, ...]:
    """Return the edges with every chain as its head: edges inside dropped, exits leaving the head as plain edges."""
    head_of = {name: shape.head for shape in shapes for name in shape.nodes}
    collapsed: list[Edge] = []
    for edge in edges:
        source = head_of.get(edge.source)
        if source is None:
            collapsed.append(edge)
        elif head_of.get(edge.target) != source and Edge(source, edge.target) not in collapsed:
            collapsed.append(Edge(source, edge.target))
    return tuple(collapsed)
