"""Version 2 modules in the engine: a whole inner system runs as one node, on the outer run's meter.

The models are scripted doubles.
"""

from __future__ import annotations

from typing import Any

import pytest

from xmas import MAS, Agent, Budget, ErrorKind, MASModule, TokenUsage
from xmas.edges import Edge, EdgeKind

from .conftest import MockLLM

TIMEOUT = TimeoutError("the model took too long")


def shorten_and_post(answers: list[str | Exception], **kwargs: Any) -> MAS:
    """Writer, then the module ``shorten`` (a one-agent inner system), then poster; a timeout routes to apology."""
    llm = MockLLM(answers=answers, meta=True)
    cutter = Agent("cutter", "Cut this to half its length: {page}", output="short", inputs=["page"])
    inner = MAS([cutter], inputs={"page": "text"}, llm=llm)
    return MAS(
        [
            Agent("writer", "Write about {query}", output="draft"),
            Agent("poster", "Post {draft}", output="post", inputs=["draft"]),
            Agent("apology", "Sorry about {query}", output="post"),
        ],
        [
            ("writer", "shorten"),
            ("shorten", "poster"),
            Edge("shorten", "apology", EdgeKind.ERROR, errors=("timeout",)),
        ],
        modules=[MASModule("shorten", inner, reads={"page": "draft"}, writes={"short": "draft"})],
        state={"draft": {"merge": "last"}, "post": {"merge": "last"}},
        llm=llm,
        **kwargs,
    )


def test_a_box_maps_its_values_in_and_out_and_nests_its_records() -> None:
    outer = shorten_and_post(["A long draft about cats.", "Cats.", "Posted."])
    trace = outer.run("cats")
    assert trace.error is None
    assert trace.output == "Posted."

    # the outer draft goes in as the inner page; the inner short comes back as the draft
    (box,) = trace["shorten"]
    assert box.kind == "module"
    assert box.route == ("poster",)
    assert box.inputs == {"draft": "A long draft about cats."}
    assert box.outputs == {"draft": "Cats."}
    assert trace["poster"][0].inputs == {"draft": "Cats."}
    assert box.inner is not None
    assert box.inner.query == '{"page": "A long draft about cats."}'
    assert box.inner.output == "Cats."
    assert [r.rendered_prompt for r in box.inner.records] == ["Cut this to half its length: A long draft about cats."]
    assert box.cost_usd == pytest.approx(0.01)
    assert box.tokens == TokenUsage(10, 2)
    assert box.usage == box.inner.records[0].usage

    # the trace keeps the inner records inside the module, and walk() names the path
    assert [r.agent for r in trace.records] == ["writer", "shorten", "poster"]
    assert [(path, r.agent) for path, r in trace.walk()] == [
        ((), "writer"),
        ((), "shorten"),
        (("shorten",), "cutter"),
        ((), "poster"),
    ]
    data = trace.to_dict()["records"][1]
    assert data["kind"] == "module"
    assert [r["agent"] for r in data["inner"]["records"]] == ["cutter"]
    assert "inner" not in trace.to_dict()["records"][0]


def test_a_box_inside_a_box_names_both_in_the_path() -> None:
    again = shorten_and_post(["A long draft about cats.", "Cats.", "Posted.", "Echoed."])
    wrapper = MAS(
        [Agent("echo", "Echo {post}", output="echoed", inputs=["post"])],
        [("all", "echo")],
        modules=[MASModule("all", again, reads={"query": "query"}, writes={"post": "post"})],
        llm=again.llm,
    )
    nested = wrapper.run("cats")
    assert nested.error is None
    assert nested.output == "Echoed."
    assert [(path, r.agent) for path, r in nested.walk()] == [
        ((), "all"),
        (("all",), "writer"),
        (("all",), "shorten"),
        (("all", "shorten"), "cutter"),
        (("all",), "poster"),
        ((), "echo"),
    ]


def test_the_outer_budget_counts_the_inner_calls() -> None:
    limited = shorten_and_post(["A long draft about cats.", "Cats.", "Posted."], budget=Budget(max_calls=2))
    failed = limited.run("cats")
    assert failed.error_kind == ErrorKind.BUDGET
    assert [r.agent for r in failed.records] == ["writer", "shorten"]
    assert failed.health is not None
    assert failed.health.budget["calls"] == {"spent": 2, "limit": 2}


def test_an_inner_failure_fails_the_box_with_its_kind_and_the_outer_error_edge_routes_it() -> None:
    routed_mas = shorten_and_post(["A long draft about cats.", TIMEOUT, "Sorry."])
    routed = routed_mas.run("cats")
    assert routed.error is None
    assert routed.output == "Sorry."
    (failed_box,) = routed["shorten"]
    assert failed_box.kind == "module"
    assert failed_box.error_kind == ErrorKind.TIMEOUT
    assert failed_box.error == "TimeoutError: the model took too long"
    assert failed_box.route == ("apology",)
    assert [r.agent for r in routed.records] == ["writer", "shorten", "apology"]
    assert failed_box.inner is not None
    assert failed_box.inner.error == "TimeoutError: the model took too long"
    assert [(r.agent, r.error_kind) for r in failed_box.inner.records] == [("cutter", ErrorKind.TIMEOUT)]


def test_a_failed_box_keeps_its_inner_trace_and_cost() -> None:
    """The inner records before the failure stay in the failed module's record.

    One unknown cost makes its sum unknown.
    """
    llm = MockLLM(answers=[], meta=True)
    inner = MAS(
        [
            Agent("cutter", "Cut {page}", output="short", inputs=["page"]),
            Agent("polish", "Polish {short}", output="final", inputs=["short"]),
        ],
        [("cutter", "polish")],
        inputs={"page": "text"},
        llm=llm,
    )
    outer = MAS(
        [Agent("writer", "Write {query}", output="draft"), Agent("apology", "Sorry {query}", output="post")],
        [("writer", "shorten"), Edge("shorten", "apology", EdgeKind.ERROR, errors=("timeout",))],
        modules=[MASModule("shorten", inner, reads={"page": "draft"}, writes={"final": "post"})],
        state={"post": {"merge": "last"}},
        llm=llm,
    )
    for _ in range(2):  # a second run on the same inner system shows only its own records
        llm.answers = ["A draft.", "Cut.", TIMEOUT, "Sorry."]
        trace = outer.run("cats")
        assert trace.error is None
        assert trace.output == "Sorry."
        (box,) = trace["shorten"]
        assert box.inner is not None
        assert [r.agent for r in box.inner.records] == ["cutter", "polish"]
        assert box.inner.records[1].error_kind == ErrorKind.TIMEOUT
        assert box.inner.error_kind == ErrorKind.TIMEOUT
        assert box.inner.records[0].cost_usd == pytest.approx(0.01)  # the cutter's paid call stays in the record
        assert box.cost_usd is None
        assert box.tokens is None  # polish's cost is unknown, so the module's is too
        assert trace.cost_usd is None
        assert [(path, r.agent) for path, r in trace.walk()][1:4] == [
            ((), "shorten"),
            (("shorten",), "cutter"),
            (("shorten",), "polish"),
        ]
