"""Predicates: the conditions on ``when``, ``while`` and ``switch`` edges.

A predicate is a small expression over the run's values, such as
``not verdict.fits and visits < 3``. It is parsed once, when the system is built,
into a tree of tuples, and evaluated after the deciding node ran. Nothing is ever
passed to ``eval``: the tokenizer knows numbers, quoted strings without escapes,
names and ``== != <= >= < > ( ) [ ] , .``, and any other character is refused
with its position.

Grammar::

    expr := conj ('or' conj)*
    conj := neg ('and' neg)*
    neg  := 'not' neg | comp
    comp := atom [op atom]          op: == != < <= > >= in, not in
    atom := number | string | true | false | null | '[' literals ']'
          | name '(' args ')' | path | '(' expr ')'
    path := name ('.' name)*

Built-ins: ``exists(path)``, ``len(x)``, ``visits`` (the deciding node's visits),
``visits(node)``, ``cost`` (USD so far) and ``elapsed`` (seconds so far). Any other
call names a registered callable that returns a bool. A missing path is ``None``;
an ordering comparison with ``None``, or between a number and a string, is false;
a bare path is read by its truthiness.

This is a leaf module: it imports no other xmas module.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BUILTINS",
    "KEYWORDS",
    "MAX_DEPTH",
    "MAX_LENGTH",
    "Context",
    "Predicate",
    "PredicateError",
    "parse_predicate",
]


MAX_LENGTH = 500
"""The longest predicate text, in characters."""

MAX_DEPTH = 32
"""The deepest nesting of ``not``, brackets and calls."""

BUILTINS = frozenset({"exists", "len", "visits", "cost", "elapsed"})
"""Names that are built-ins, never values or registered callables."""

KEYWORDS = frozenset({"and", "or", "not", "in", "true", "false", "null"})
"""The words of the grammar itself."""

LITERALS = {"true": True, "false": False, "null": None}
ORDER = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
COMPARISONS = frozenset({"==", "!=", "in", "not in", *ORDER})
TOKEN = re.compile(
    r"""(?P<space>\s+)
    |(?P<number>-?\d+(?:\.\d+)?)
    |(?P<string>"[^"]*"|'[^']*')
    |(?P<name>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<op>==|!=|<=|>=|<|>|\(|\)|\[|\]|,|\.)""",
    re.VERBOSE,
)
MISSING = object()


class PredicateError(ValueError):
    """A predicate that cannot be parsed or evaluated.

    A ``ValueError``, like every other refusal of a system's shape, because
    ``check_system`` parses each predicate when the system is built; evaluation
    raises it only for a call no registered callable answers. A problem at one
    place in the text names its zero-based character position.
    """

    def __init__(self, message: str, position: int | None = None) -> None:
        """Add the position, when there is one, to the message."""
        super().__init__(message if position is None else f"{message} at position {position}")


@dataclass(frozen=True)
class Context:
    """What a predicate can see when it is evaluated.

    ``visits`` counts each node's runs, the deciding node's current visit
    included; ``node`` is the deciding node, which bare ``visits`` refers to;
    ``cost_usd`` and ``elapsed_s`` are ``None`` when unknown; ``call`` calls a
    registered callable by name and returns its bool, or is ``None`` when no
    callable is registered.
    """

    values: Mapping[str, Any]
    visits: Mapping[str, int] = field(default_factory=dict)
    node: str = ""
    cost_usd: float | None = None
    elapsed_s: float | None = None
    call: Callable[[str, tuple[Any, ...]], bool] | None = None


@dataclass(frozen=True)
class Predicate:
    """A parsed predicate: its ``text`` and its syntax ``tree``.

    The tree is nested tuples: ``("lit", value)``, ``("path", names)``,
    ``("call", name, args)``, ``("exists", names)``, ``("len", arg)``,
    ``("visits", node or None)``, ``("cost",)``, ``("elapsed",)``,
    ``("not", arg)``, ``("and", args)``, ``("or", args)`` and
    ``("cmp", op, left, right)``.
    """

    text: str
    tree: tuple[Any, ...]

    def paths(self) -> frozenset[tuple[str, ...]]:
        """Return every path the predicate reads, as tuples of names."""
        return frozenset(node[1] for node in walk(self.tree) if node[0] in ("path", "exists"))

    def calls(self) -> frozenset[tuple[str, int]]:
        """Return each call of a registered callable as ``(name, number of arguments)``."""
        return frozenset((node[1], len(node[2])) for node in walk(self.tree) if node[0] == "call")

    def evaluate(self, ctx: Context) -> bool:
        """Return whether the predicate holds in ``ctx``; a call that is not registered or returns no bool raises."""
        return bool(evaluate_tree(self.tree, ctx))


def parse_predicate(text: str) -> Predicate:
    """Parse a predicate, raising :class:`PredicateError` with the position of the first problem."""
    if len(text) > MAX_LENGTH:
        raise PredicateError(f"predicate is {len(text)} characters long, {MAX_LENGTH} at most")
    parser = Parser(text, tokenize(text))
    if not parser.tokens:
        raise PredicateError("predicate is empty")
    tree = parser.expr()
    if parser.index < len(parser.tokens):
        _, value, position = parser.tokens[parser.index]
        raise PredicateError(f"unexpected {value!r}", position)
    return Predicate(text, tree)


def tokenize(text: str) -> list[tuple[str, str, int]]:
    """Split ``text`` into ``(kind, text, position)`` tokens, spaces dropped."""
    tokens: list[tuple[str, str, int]] = []
    position = 0
    while position < len(text):
        match = TOKEN.match(text, position)
        if match is None:
            raise PredicateError(f"unexpected character {text[position]!r}", position)
        kind = match.lastgroup or ""
        if kind != "space":
            tokens.append((kind, match.group(), position))
        position = match.end()
    return tokens


class Parser:
    """Recursive descent over the tokens, one method per grammar rule."""

    def __init__(self, text: str, tokens: list[tuple[str, str, int]]) -> None:
        """Start before the first token."""
        self.text = text
        self.tokens = tokens
        self.index = 0
        self.depth = 0

    def peek(self) -> str | None:
        """Return the next token's text, or ``None`` at the end."""
        return self.tokens[self.index][1] if self.index < len(self.tokens) else None

    def take(self, expected: str | None = None) -> tuple[str, str, int]:
        """Consume the next token, which must be ``expected`` when given."""
        if self.index >= len(self.tokens):
            want = "a value" if expected is None else repr(expected)
            raise PredicateError(f"predicate ends early, expected {want}", len(self.text))
        token = self.tokens[self.index]
        if expected is not None and token[1] != expected:
            raise PredicateError(f"expected {expected!r}, got {token[1]!r}", token[2])
        self.index += 1
        return token

    def nested(self, position: int, parse: Callable[[], Any]) -> Any:
        """Run ``parse`` one level of nesting deeper, refusing the 33rd level."""
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise PredicateError(f"predicate nests deeper than {MAX_DEPTH}", position)
        parsed = parse()
        self.depth -= 1
        return parsed

    def items(self, close: str, item: Callable[[], Any]) -> list[Any]:
        """Parse ``item (',' item)*`` up to and including ``close``."""
        found: list[Any] = []
        while self.peek() not in (close, None):
            if found:
                self.take(",")
            found.append(item())
        self.take(close)
        return found

    def joined(self, word: str, part: Callable[[], tuple[Any, ...]]) -> tuple[Any, ...]:
        """Parse ``part (word part)*``."""
        parts = [part()]
        while self.peek() == word:
            self.take()
            parts.append(part())
        return parts[0] if len(parts) == 1 else (word, tuple(parts))

    def expr(self) -> tuple[Any, ...]:
        """Parse ``conj ('or' conj)*``."""
        return self.joined("or", self.conj)

    def conj(self) -> tuple[Any, ...]:
        """Parse ``neg ('and' neg)*``."""
        return self.joined("and", self.neg)

    def neg(self) -> tuple[Any, ...]:
        """Parse ``'not' neg | comp``."""
        if self.peek() != "not":
            return self.comp()
        return ("not", self.nested(self.take()[2], self.neg))

    def comp(self) -> tuple[Any, ...]:
        """Parse ``atom [op atom]``, reading ``not in`` as one operator."""
        left = self.atom()
        op = self.peek()
        if op == "not" and self.index + 1 < len(self.tokens) and self.tokens[self.index + 1][1] == "in":
            self.take()
            op = "not in"
        elif op not in COMPARISONS:
            return left
        self.take()
        return ("cmp", op, left, self.atom())

    def atom(self) -> tuple[Any, ...]:
        """Parse a literal, a list, a call, a path or a bracketed expression."""
        kind, value, position = self.take()
        if (literal := literal_value(kind, value)) is not MISSING:
            return ("lit", literal)
        if value == "(":
            inner: tuple[Any, ...] = self.nested(position, self.expr)
            self.take(")")
            return inner
        if value == "[":
            return ("lit", self.nested(position, lambda: self.items("]", self.literal)))
        return self.named(kind, value, position)

    def named(self, kind: str, value: str, position: int) -> tuple[Any, ...]:
        """Parse what starts with a name: a call, ``cost``, ``elapsed``, ``visits`` or a path."""
        if kind != "name" or value in KEYWORDS:
            raise PredicateError(f"unexpected {value!r}", position)
        if value.startswith("__"):
            raise PredicateError(f"names starting '__' belong to xmas: {value!r}", position)
        if self.peek() == "(":
            return self.call(value, position)
        if value in ("cost", "elapsed"):
            return (value,)
        if value == "visits":
            return ("visits", None)
        if value in BUILTINS:
            raise PredicateError(f"{value} needs an argument in brackets", position)
        return ("path", self.path(value))

    def literal(self) -> Any:
        """Parse one literal of a list."""
        kind, value, position = self.take()
        if (literal := literal_value(kind, value)) is MISSING:
            raise PredicateError(f"a list holds literals only, got {value!r}", position)
        return literal

    def path(self, first: str) -> tuple[str, ...]:
        """Parse the rest of ``name ('.' name)*``."""
        names = [first]
        while self.peek() == ".":
            self.take()
            kind, value, position = self.take()
            if kind != "name" or value in KEYWORDS:
                raise PredicateError(f"expected a name after '.', got {value!r}", position)
            names.append(value)
        return tuple(names)

    def call(self, name: str, position: int) -> tuple[Any, ...]:
        """Parse ``'(' args ')'`` after ``name``, checking the built-ins' arguments."""
        args: list[tuple[Any, ...]] = self.nested(self.take("(")[2], lambda: self.items(")", self.expr))
        if name in ("cost", "elapsed"):
            raise PredicateError(f"{name} takes no arguments", position)
        if name in BUILTINS and len(args) != 1:
            raise PredicateError(f"{name}() takes one argument, got {len(args)}", position)
        if name == "len":
            return ("len", args[0])
        if name in ("exists", "visits") and (args[0][0] != "path" or (name == "visits" and len(args[0][1]) != 1)):
            what = "a path" if name == "exists" else "a node name"
            raise PredicateError(f"{name}() takes {what}", position)
        if name == "exists":
            return ("exists", args[0][1])
        if name == "visits":
            return ("visits", args[0][1][0])
        return ("call", name, tuple(args))


def literal_value(kind: str, value: str) -> Any:
    """Return the value of a number, string, ``true``, ``false`` or ``null`` token, else ``MISSING``."""
    if kind == "number":
        return float(value) if "." in value else int(value)
    if kind == "string":
        return value[1:-1]
    return LITERALS.get(value, MISSING) if kind == "name" else MISSING


def walk(tree: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """Return every subtree of ``tree``, ``tree`` included."""
    head = tree[0]
    if head in ("not", "len"):
        children: tuple[tuple[Any, ...], ...] = (tree[1],)
    elif head in ("and", "or"):
        children = tree[1]
    elif head == "cmp":
        children = (tree[2], tree[3])
    elif head == "call":
        children = tree[2]
    else:
        children = ()
    found = [tree]
    for child in children:
        found.extend(walk(child))
    return found


def evaluate_tree(tree: tuple[Any, ...], ctx: Context) -> Any:
    """Return the value of one subtree in ``ctx``: a literal, a figure of the run, else by :data:`EVALUATORS`."""
    head = tree[0]
    if head == "lit":
        return tree[1]
    if head in RUN_FIGURES:
        return getattr(ctx, RUN_FIGURES[head])
    return EVALUATORS[head](tree, ctx)


def evaluate_path(tree: tuple[Any, ...], ctx: Context) -> Any:
    """Return the value at a path, ``None`` when nothing is there."""
    value = resolve(tree[1], ctx.values)
    return None if value is MISSING else value


def evaluate_len(tree: tuple[Any, ...], ctx: Context) -> Any:
    """Return the length of a text, list or object, ``None`` for anything else."""
    value = evaluate_tree(tree[1], ctx)
    return len(value) if isinstance(value, str | list | tuple | dict) else None


def evaluate_call(tree: tuple[Any, ...], ctx: Context) -> bool:
    """Call a registered function on the arguments' values; it must return a bool."""
    name, args = tree[1], tuple(evaluate_tree(arg, ctx) for arg in tree[2])
    if ctx.call is None:
        raise PredicateError(f"unknown function {name!r}")
    result = ctx.call(name, args)
    if not isinstance(result, bool):
        raise PredicateError(f"callable {name!r} returned {type(result).__name__}, not bool")
    return result


RUN_FIGURES = {"cost": "cost_usd", "elapsed": "elapsed_s"}
"""The built-ins that read a figure of the run, and the :class:`Context` field each reads."""

EVALUATORS: dict[str, Callable[[tuple[Any, ...], Context], Any]] = {
    "path": evaluate_path,
    "exists": lambda tree, ctx: resolve(tree[1], ctx.values) is not MISSING,
    "len": evaluate_len,
    "visits": lambda tree, ctx: ctx.visits.get(ctx.node if tree[1] is None else tree[1], 0),
    "not": lambda tree, ctx: not evaluate_tree(tree[1], ctx),
    "and": lambda tree, ctx: all(evaluate_tree(part, ctx) for part in tree[1]),
    "or": lambda tree, ctx: any(evaluate_tree(part, ctx) for part in tree[1]),
    "cmp": lambda tree, ctx: compare(tree[1], evaluate_tree(tree[2], ctx), evaluate_tree(tree[3], ctx)),
    "call": evaluate_call,
}
"""How each kind of subtree evaluates, by the head the parser gives it."""


def resolve(path: tuple[str, ...], values: Mapping[str, Any]) -> Any:
    """Follow ``path`` through ``values`` and nested mappings, or return ``MISSING``."""
    current: Any = values
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return MISSING
        current = current[name]
    return current


def compare(op: str, left: Any, right: Any) -> bool:
    """Apply a comparison: ordering only between two numbers or two strings."""
    if op == "==":
        return equal(left, right)
    if op == "!=":
        return not equal(left, right)
    if op in ("in", "not in"):
        if isinstance(right, list | tuple):
            found = any(equal(left, item) for item in right)
        else:  # a substring of a text, a key of an object
            found = isinstance(right, str | Mapping) and isinstance(left, str) and left in right
        return found if op == "in" else not found
    numbers = all(isinstance(side, int | float) and not isinstance(side, bool) for side in (left, right))
    if not numbers and not (isinstance(left, str) and isinstance(right, str)):
        return False
    return bool(ORDER[op](left, right))


def equal(left: Any, right: Any) -> bool:
    """Compare like JSON: a bool never equals a number, and lists and objects compare item by item."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    return bool(left == right)
