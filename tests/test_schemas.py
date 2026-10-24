"""Answer forms: the README dialect parsed, refused with every problem named, and its exact format block."""

from __future__ import annotations

import re
from typing import Any

import pytest

from xmas.schemas import RETRY_NOTE, FormError, compile_form

VERDICT: dict[str, Any] = {
    "fits": {"type": "bool", "means": "the slide makes its claim at a glance"},
    "problems": {"type": "list of text", "means": "one line per problem"},
    "action": {"type": "text", "one of": ["post", "hold"]},
    "checks": {"type": "list of object", "fields": {"comment": {"type": "int"}, "reason": {"type": "text"}}},
}
VERDICT_BLOCK = """\
Answer with one JSON object and nothing else. Its fields:
- fits (true or false): the slide makes its claim at a glance
- problems (list of text): one line per problem
- action (one of "post", "hold")
- checks (list of object)
  - comment (integer)
  - reason (text)"""

NESTED: dict[str, Any] = {
    "slide": {
        "type": "object",
        "means": "the slide as drawn",
        "fields": {
            "title": {"type": "text"},
            "size": {
                "type": "object",
                "fields": {"w": {"type": "number"}, "unit": {"type": "text", "one of": ["px", "pt"], "optional": True}},
            },
        },
    },
    "score": {"type": "int"},
}
NESTED_BLOCK = """\
Answer with one JSON object and nothing else. Its fields:
- slide (object): the slide as drawn
  - title (text)
  - size (object)
    - w (number)
    - unit (one of "px", "pt", optional)
- score (integer)"""

GOOD = {
    "extra fields kept": (
        VERDICT,
        '{"fits": true, "problems": [], "action": "post", "checks": [{"comment": 1, "reason": "ok"}], "extra": 1}',
        {"fits": True, "problems": [], "action": "post", "checks": [{"comment": 1, "reason": "ok"}], "extra": 1},
    ),
    "fenced and coerced": (
        VERDICT,
        '  ```json\n{"fits": "yes", "problems": ["x"], "action": "hold", "checks": []}\n```\n',
        {"fits": True, "problems": ["x"], "action": "hold", "checks": []},
    ),
    "nested object": (
        NESTED,
        '{"slide": {"title": "Results", "size": {"w": "1920"}}, "score": "2"}',
        {"slide": {"title": "Results", "size": {"w": 1920.0, "unit": None}}, "score": 2},
    ),
}


@pytest.mark.parametrize("case", list(GOOD))
def test_an_answer_in_the_form_parses(case: str) -> None:
    fields, answer, value = GOOD[case]
    assert compile_form(fields).parse(answer) == value


BAD = {
    "prose": (VERDICT, 'Sure! {"fits": true}', ["<root>"]),
    "wrong types": (
        VERDICT,
        '{"fits": true, "problems": "none", "action": "prefer A", "checks": [{"comment": "one"}]}',
        ["problems", "action", "checks.0.comment", "checks.0.reason"],
    ),
    "nested": (NESTED, '```\n{"slide": {"title": 1, "size": {"w": 1}}}\n```', ["slide.title", "score"]),
}


@pytest.mark.parametrize("case", list(BAD))
def test_an_answer_outside_the_form_names_every_field_that_is_wrong(case: str) -> None:
    """The problem is one line, ``path: message`` per field, joined by ``; ``."""
    fields, answer, paths = BAD[case]
    with pytest.raises(FormError) as caught:
        compile_form(fields).parse(answer)
    assert re.findall(r"(?:^|; )([\w.<>]+): ", caught.value.problem) == paths


def test_the_retry_note_quotes_the_problem() -> None:
    assert RETRY_NOTE.format(problem="fits: Field required") == (
        "Your previous answer did not match the required format: fits: Field required\nAnswer again with only the JSON."
    )


def test_a_form_names_its_fields_paths_and_choices() -> None:
    verdict = compile_form(VERDICT)
    assert verdict.format_block() == VERDICT_BLOCK
    assert verdict.paths() == ("fits", "problems", "action", "checks")
    assert verdict.choices("action") == ("post", "hold")
    assert verdict.choices("fits") is None

    nested = compile_form(NESTED)
    assert nested.format_block() == NESTED_BLOCK
    assert nested.paths() == ("slide", "slide.title", "slide.size", "slide.size.w", "slide.size.unit", "score")
    assert nested.choices("slide.size.unit") == ("px", "pt")
