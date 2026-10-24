"""Predicates: the grammar, the built-ins, and what is refused."""

from __future__ import annotations

import re
from typing import Any

import pytest

from xmas.predicates import Context, PredicateError, parse_predicate

VALUES: dict[str, Any] = {
    "verdict": {"fits": False, "action": "shorten", "score": 7, "problems": ["long", "vague"]},
    "draft": "A short draft.",
    "count": 0,
    "nothing": None,
    "flags": {"a": True, "b": [False]},
    "ones": {"a": 1, "b": [0]},
    "same": {"b": [False], "a": True},
}


def is_long(name: str, args: tuple[Any, ...]) -> bool:
    assert name == "is_long"
    return len(args[0]) > 10


CTX = Context(VALUES, {"judge": 2, "writer": 3}, "judge", cost_usd=0.25, elapsed_s=None, call=is_long)

EVALUATE = [
    ("not verdict.fits", True, {"verdict"}),
    ("verdict.action == 'shorten' and verdict.score >= 7", True, {"verdict"}),
    ('verdict.action in ["publish", "hold"]', False, {"verdict"}),
    ("'vague' in verdict.problems and 'x' not in draft", True, {"verdict", "draft"}),
    ("visits < 3 and visits(writer) == 3", True, set()),
    ("cost <= 0.25 and not (elapsed > 10)", True, set()),
    ("exists(verdict.score) and not exists(verdict.missing) and exists(nothing)", True, {"verdict", "nothing"}),
    ("len(draft) == 14 and len(verdict.problems) == 2 and len(nothing) == null", True, {"draft", "verdict", "nothing"}),
    ("verdict.missing > 1 or verdict.missing < 1 or draft > 3", False, {"verdict", "draft"}),
    ("verdict.missing == null and nothing == null and verdict.deep.deeper == null", True, {"verdict", "nothing"}),
    ("count or nothing or verdict.fits or verdict.missing", False, {"count", "nothing", "verdict"}),
    ("true == 1 or false == 0 or [1, 'a'] != [1, 'a']", False, set()),
    ("flags != ones and flags == same", True, {"flags", "ones", "same"}),  # a bool never equals a number, nested too
    ("is_long(draft) and not is_long('short')", True, {"draft"}),
    ("verdict.score > -1.5 and verdict.action < 'z' and (count == 0 or count == 1)", True, {"verdict", "count"}),
]

REFUSED = [
    ("__import__('os')", "names starting '__' belong to xmas"),
    ("frobnicate(draft)", "unknown function 'frobnicate'"),
    ("verdict.fits; draft", "unexpected character ';' at position 12"),
    ("len(draft", "predicate ends early, expected ')' at position 9"),
    ("(" * 33 + "true" + ")" * 33, "nests deeper than 32 at position 32"),
    ("x == " + "'a' or x == " * 50 + "'b'", "characters long, 500 at most"),
]


@pytest.mark.parametrize(("text", "expected", "roots"), EVALUATE, ids=range(len(EVALUATE)))
def test_a_predicate_evaluates_and_names_its_roots(text: str, expected: bool, roots: set[str]) -> None:
    predicate = parse_predicate(text)
    assert predicate.evaluate(CTX) is expected
    assert {path[0] for path in predicate.paths()} == roots
    assert predicate.calls() == ({("is_long", 1)} if "is_long" in text else set())


@pytest.mark.parametrize(("text", "problem"), REFUSED, ids=range(len(REFUSED)))
def test_a_predicate_is_refused_when_parsed_or_evaluated(text: str, problem: str) -> None:
    """An unknown function is refused when evaluated, everything else when parsed."""
    with pytest.raises(PredicateError, match=re.escape(problem)) as raised:
        parse_predicate(text).evaluate(Context(VALUES))
    assert isinstance(raised.value, ValueError)
