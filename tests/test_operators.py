"""Version 2 operators in the engine: when, else, while, switch, merges, joins and ``on error``.

The models are scripted doubles.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from xmas import END, MAS, Agent, Budget, ErrorKind
from xmas.edges import Edge, EdgeKind
from xmas.errors import BudgetExceeded
from xmas.nodes import Step
from xmas.routes import RouteGraph
from xmas.tools import tool

from .conftest import VERDICT, MockLLM, worked_example

OK = '{"fits": true, "action": "publish"}'
NO = '{"fits": false, "action": "publish"}'


def test_a_while_edge_loops_back_until_the_switch_routes_on() -> None:
    trace = worked_example(iter([NO, NO, OK])).run("cats")
    assert trace.error is None
    assert [r.agent for r in trace.records] == ["writer", "count", "judge"] * 3 + ["publish"]
    assert [r.route for r in trace["judge"]] == [("writer",), ("writer",), ("publish",)]
    assert [r.route for r in trace.records if r.agent != "judge"] == [None] * 7
    assert not any(r.capped for r in trace.records)
    assert trace["writer"][1].inputs["verdict"] == NO
    assert trace["writer"][0].inputs["verdict"] == ""  # a first-visit gap renders empty, so nothing is unfilled
    assert trace.output == "Published: A short draft about cats."
    assert trace.health is not None
    assert trace.health.loops == {"judge -> writer": {"taken": 2, "cap_hit": False}}
    assert trace.to_dict()["records"][2]["route"] == ["writer"]
    assert "capped" not in trace.to_dict()["records"][2]


def test_a_loop_at_its_cap_is_capped_and_takes_the_else_route() -> None:
    trace = worked_example(itertools.repeat(NO)).run("cats")
    assert [r.agent for r in trace.records] == ["writer", "count", "judge"] * 3 + ["apology"]
    assert [(r.route, r.capped) for r in trace["judge"]] == [(("writer",), False)] * 2 + [(("apology",), True)]
    assert trace.output == "Sorry, no short answer about cats."
    assert trace.health is not None
    assert trace.health.loops == {"judge -> writer": {"taken": 2, "cap_hit": True}}
    assert sorted({(f.signal, f.node) for f in trace.health.flags}) == [
        ("stuck", "count"),
        ("stuck", "judge"),
        ("stuck", "writer"),
    ]
    assert trace.to_dict()["records"][8]["capped"] is True
    assert trace.to_dict()["health"]["loops"] == {"judge -> writer": {"taken": 2, "cap_hit": True}}


def test_a_switch_value_outside_the_cases_takes_else() -> None:
    trace = worked_example(iter(['{"fits": true, "action": "shorten"}'])).run("cats")
    assert [r.agent for r in trace.records] == ["writer", "count", "judge", "apology"]
    assert trace["judge"][0].route == ("apology",)


def test_the_first_when_edge_that_holds_wins_in_declared_order() -> None:
    llm = MockLLM(lambda p: "12" if p.startswith("Score") else f"done: {p}")
    scorer = Agent("scorer", "Score {query}", output="score", output_type="json")
    high, low = Agent("high", "High {score}", output="note", inputs=["score"]), Agent("low", "Low", output="note")
    mas = MAS(
        [scorer, high, low, Agent("other", "Other", output="note")],
        [
            Edge("scorer", "high", EdgeKind.WHEN, when="score > 10"),
            Edge("scorer", "low", EdgeKind.WHEN, when="score > 5"),
            Edge("scorer", "other", EdgeKind.ELSE),
        ],
        state={"note": {"merge": "last"}},
        llm=llm,
    )
    trace = mas.run("x")
    assert [r.agent for r in trace.records] == ["scorer", "high"]
    assert trace.output == "done: High 12"


def test_a_merge_inside_a_loop_joins_a_parallel_chain_that_runs_once() -> None:
    """A loop whose when/else merge ``m`` feeds ``fin``, which also waits for the chain s1 -> s2 -> s3."""
    drafts = iter(["A rather long first draft.", "Short.", "Short."])

    def answer(prompt: str) -> str:
        return next(drafts) if prompt.startswith("Draft") else prompt.split(":", maxsplit=1)[0]

    @tool(writes={"verdict": "json"})
    def check(draft: str) -> dict[str, bool]:
        """Say whether a draft is long."""
        return {"long": len(draft) > 10}

    agents = [
        Agent("s1", "S1: {query}", output="one", inputs=["query"]),
        Agent("s2", "S2: {one}", output="two", inputs=["one"]),
        Agent("s3", "S3: {two}", output="three", inputs=["two"]),
        Agent("w", "Draft {query}", output="draft", inputs=["query"]),
        Agent("a", "A: {draft}", output="cut", inputs=["draft"]),
        Agent("m", "M: {draft}", output="note", inputs=["draft"]),
        Agent("fin", "Fin: {three} {note}", output="final", inputs=["three", "note"]),
    ]
    mas = MAS(
        agents,
        [
            ("s1", "s2"),
            ("s2", "s3"),
            ("s3", "fin"),
            ("w", "c"),
            Edge("c", "a", EdgeKind.WHEN, when="verdict.long"),
            Edge("c", "m", EdgeKind.ELSE),
            ("a", "m"),
            Edge("m", "w", EdgeKind.WHILE, when="true", max_visits=3),
            Edge("m", "fin", EdgeKind.ELSE),
        ],
        steps=[Step("c", "check")],
        llm=MockLLM(answer),
        tools=[check],
    )
    trace = mas.run("cats")
    assert trace.error is None
    names = [r.agent for r in trace.records]
    assert names.count("fin") == 1
    assert names[-1] == "fin"
    assert names.count("m") == 3
    assert names.count("w") == 3
    assert names.count("a") == 1
    assert names.index("s3") < names.index("fin")
    assert [r.route for r in trace["c"]] == [("a",), ("m",), ("m",)]
    assert [(r.route, r.capped) for r in trace["m"]] == [(("w",), False)] * 2 + [(("fin",), True)]
    assert trace["fin"][0].inputs == {"three": "S3", "note": "M"}
    assert trace.output == "Fin"
    assert set(names) == {a.name for a in agents} | {"c"}  # a relay runs, but leaves no record


def fetch_and_join() -> MAS:
    """``fetch`` (failing as its query says) and ``side`` feed the join ``j``; a timeout of fetch goes to ``fb``."""

    @tool(writes={"page": "text"})
    def fetch(query: str) -> str:
        """Fetch a page, or fail as the query says."""
        if query == "slow":
            raise TimeoutError("the server took too long")
        if query == "down":
            raise ConnectionError("the server is down")
        return f"page on {query}"

    @tool(writes={"page": "text"})
    def fallback(query: str) -> str:
        """Give a cached page."""
        return f"cached page on {query}"

    @tool(writes={"extra": "text"})
    def side(query: str) -> str:
        """Give a side note."""
        return f"note on {query}"

    joiner = Agent("j", "Join {page} and {extra}", output="answer", inputs=["page", "extra"])
    return MAS(
        [joiner],
        [
            ("fetch", "j"),
            Edge("fetch", "fb", EdgeKind.ERROR, errors=("timeout",)),
            ("fb", "j"),
            ("side", "j"),
        ],
        steps=[Step("fetch", "fetch"), Step("fb", "fallback"), Step("side", "side")],
        state={"page": {"merge": "last"}},
        llm=MockLLM(lambda p: p.upper()),
        tools=[fetch, fallback, side],
    )


def test_a_node_that_succeeds_takes_its_plain_route() -> None:
    ok = fetch_and_join().run("cats")
    assert ok.error is None
    assert ok.output == "JOIN PAGE ON CATS AND NOTE ON CATS"
    assert sorted(r.agent for r in ok.records) == ["fetch", "j", "side"]
    assert ok["fetch"][0].route == ("j",)


def test_a_failure_of_a_routed_kind_takes_its_error_route() -> None:
    """A failed record without writes, and the join still runs once."""
    routed = fetch_and_join().run("slow")
    assert routed.error is None
    assert routed.error_kind is None
    assert routed.output == "JOIN CACHED PAGE ON SLOW AND NOTE ON SLOW"
    assert [r.agent for r in routed.records].count("j") == 1
    (failed,) = routed["fetch"]
    assert failed.error == "TimeoutError: the server took too long"
    assert failed.error_kind == ErrorKind.TIMEOUT
    assert failed.outputs == {}
    assert failed.route == ("fb",)
    assert failed.kind == "step"
    assert failed.latency_s is not None
    assert failed.to_dict()["error"] == "TimeoutError: the server took too long"
    assert routed.health is not None
    assert [(f.signal, f.node) for f in routed.health.flags] == [("error_routed", "fetch")]


def test_a_failure_without_a_route_fails_the_run_and_keeps_the_failed_record() -> None:
    unrouted = fetch_and_join().run("down")
    assert unrouted.error_kind == ErrorKind.OUTAGE
    assert unrouted.output == ""
    assert [(r.agent, r.error_kind) for r in unrouted.records if r.error] == [("fetch", ErrorKind.OUTAGE)]
    assert unrouted["fetch"][0].route is None


JUDGE = Agent("judge", "Judge {query}", output="verdict", output_type="json", schema=VERDICT)
SORRY = Agent("sorry", "Sorry about {query}", output="verdict_text")


def test_a_bad_format_is_routed_with_the_cost_of_both_paid_attempts() -> None:
    llm = MockLLM(answers=["not json", "still not json", "We are sorry."], meta=True)
    mas = MAS([JUDGE, SORRY], [Edge("judge", "sorry", EdgeKind.ERROR, errors=("bad_format",))], llm=llm)
    trace = mas.run("cats")
    assert trace.error is None
    assert trace.output == "We are sorry."
    (failed,) = trace["judge"]
    assert failed.error_kind == ErrorKind.BAD_FORMAT
    assert failed.cost_usd is not None
    assert abs(failed.cost_usd - 0.02) < 1e-9
    assert failed.route == ("sorry",)
    assert trace.health is not None
    assert [f.signal for f in trace.health.flags] == ["bad_format", "error_routed"]


def test_a_node_whose_only_out_edge_is_an_error_edge_ends_the_run_when_it_succeeds() -> None:
    llm = MockLLM(answers=[OK], meta=True)
    good = MAS([JUDGE, SORRY], [Edge("judge", "sorry", EdgeKind.ERROR, errors=("bad_format",))], llm=llm)
    assert sorted(good.graph.predecessors(END)) == ["judge", "sorry"]
    assert good.to_dict()["edges"] == [{"judge -> sorry": {"on error": ["bad_format"]}}]  # the END wiring is implicit
    trace = good.run("cats")
    assert trace.error is None
    assert [r.agent for r in trace.records] == ["judge"]
    assert trace.output == OK
    assert trace["judge"][0].route == (END,)


def test_a_budget_refusal_is_never_routed() -> None:
    """Not even by ``any``, and no edge may name it or a stop."""
    llm = MockLLM(answers=["unused"], meta=True)
    for kind in ("budget", "stopped"):
        with pytest.raises(ValueError, match=kind):
            MAS([JUDGE, SORRY], [Edge("judge", "sorry", EdgeKind.ERROR, errors=(kind,))], llm=llm)
    mas = MAS(
        [JUDGE, SORRY],
        [Edge("judge", "sorry", EdgeKind.ERROR, errors=("any",))],
        llm=llm,
        budget=Budget(max_calls=0),
    )
    trace = mas.run("cats")
    assert trace.error_kind == ErrorKind.BUDGET
    assert trace.records == ()
    assert trace.error is not None
    assert trace.error.startswith(BudgetExceeded.__name__)


def test_a_failing_decision_is_a_failed_visit() -> None:
    """A predicate's callable that raises or answers a non-bool fails the visit, which keeps its paid record."""

    @tool
    def long_enough(draft: str) -> bool:
        """Say whether a draft is long enough."""
        return "yes"  # type: ignore[return-value]  # a non-bool: the predicate refuses it

    writer = Agent("writer", "Write {query}", output="draft", inputs=["query"])
    done = Agent("done", "Done {draft}", output="final", inputs=["draft"])
    again = Agent("again", "Again {query}", output="final", inputs=["query"])

    def build(*error_edges: Edge) -> MAS:
        edges = [
            Edge("writer", "done", EdgeKind.WHEN, when="long_enough(draft)"),
            Edge("writer", "again", EdgeKind.ELSE),
            *error_edges,
        ]
        state = {"final": {"merge": "last"}}
        return MAS(
            [writer, done, again],
            edges,
            state=state,
            tools=[long_enough],
            llm=MockLLM(answers=["a draft"] * 3, meta=True),
        )

    unrouted = build().run("cats")
    assert unrouted.error is not None
    assert "returned str, not bool" in unrouted.error
    (failed,) = unrouted.records
    assert failed.agent == "writer"
    assert failed.route is None
    assert failed.error is not None
    assert failed.cost_usd == pytest.approx(0.01)
    assert unrouted.cost_usd == pytest.approx(0.01)

    sorry = Agent("sorry", "Sorry {query}", output="final", inputs=["query"])
    mas = MAS(
        [writer, done, again, sorry],
        [
            Edge("writer", "done", EdgeKind.WHEN, when="long_enough(draft)"),
            Edge("writer", "again", EdgeKind.ELSE),
            Edge("writer", "sorry", EdgeKind.ERROR, errors=("any",)),
        ],
        state={"final": {"merge": "last"}},
        tools=[long_enough],
        llm=MockLLM(answers=["a draft", "We are sorry."], meta=True),
    )
    routed = mas.run("cats")
    assert routed.error is None
    assert routed.output == "We are sorry."
    assert [(r.agent, r.route, r.outputs) for r in routed.records] == [
        ("writer", ("sorry",), {}),
        ("sorry", None, {"final": "We are sorry."}),
    ]
    assert routed.cost_usd == pytest.approx(0.02)

    # a callable called with the wrong number of arguments is refused when the system is built
    with pytest.raises(ValueError, match=r"calls 'long_enough' with 2 arguments, but it takes 1 \(draft\)"):
        MAS(
            [writer, done, again],
            [
                Edge("writer", "done", EdgeKind.WHEN, when="long_enough(draft, query)"),
                Edge("writer", "again", EdgeKind.ELSE),
            ],
            state={"final": {"merge": "last"}},
            tools=[long_enough],
            llm=MockLLM(answers=[], meta=True),
        )


def test_a_falsy_switch_case_is_a_case() -> None:
    """A switch case of 0, false or "" is a case like any other, and 0 is not false."""

    @tool(writes={"score": "json"})
    def score_of(query: str) -> dict[str, Any]:
        """Score a query: its length minus four, or false for an empty one."""
        return {"n": len(query) - 4 if query else False}

    def agent(name: str) -> Agent:
        return Agent(name, f"{name}: {{query}}", output="label", inputs=["query"])

    mas = MAS(
        [agent("zero"), agent("no"), agent("other")],
        [
            Edge("score", "zero", EdgeKind.SWITCH, switch="score.n", case=0),
            Edge("score", "no", EdgeKind.SWITCH, switch="score.n", case=False),
            Edge("score", "other", EdgeKind.ELSE),
        ],
        steps=[Step("score", "score_of")],
        state={"label": {"merge": "last"}},
        tools=[score_of],
        llm=MockLLM(lambda p: p.split(":")[0]),
    )
    assert [mas.run(query).output for query in ("cats", "", "kittens")] == ["zero", "no", "other"]
    restored = MAS.from_dict(dict(mas.to_dict()), llm=MockLLM(lambda p: p), tools=[score_of])
    assert [edge.case for edge in restored.edges] == [0, False, None]


# Shapes of fan-in: (nodes, edges as (source, target, kind)) -> what each node with several inputs is, or the refusal.
KINDS: dict[str, dict[str, Any]] = {
    "plain": {},
    "when": {"kind": EdgeKind.WHEN, "when": "flag"},
    "switch": {"kind": EdgeKind.SWITCH, "switch": "verdict.action"},
    "else": {"kind": EdgeKind.ELSE},
    "while": {"kind": EdgeKind.WHILE, "when": "flag", "max_visits": 3},
    "error": {"kind": EdgeKind.ERROR, "errors": ("any",)},
}
SHAPES: dict[str, tuple[str, list[tuple[str, str, str]], dict[str, str] | str]] = {
    "loop with an exit": (
        "writer render measure judge assemble sign_off",
        [("writer", "render", "plain"), ("render", "measure", "plain"), ("measure", "judge", "plain"),
         ("judge", "writer", "while"), ("judge", "assemble", "else"), ("assemble", "sign_off", "plain")],
        {},
    ),
    "when/else diamond": (
        "a b c m", [("a", "b", "when"), ("a", "c", "else"), ("b", "m", "plain"), ("c", "m", "plain")], {"m": "merge"}
    ),
    "parallel diamond": (
        "a b c j", [("a", "b", "plain"), ("a", "c", "plain"), ("b", "j", "plain"), ("c", "j", "plain")],
        {"j": "join, 2 groups"},
    ),
    "on-error fallback into a join": (
        "f q1 q2 fb j",
        [("f", "q1", "plain"), ("f", "q2", "plain"), ("q1", "j", "plain"), ("q1", "fb", "error"),
         ("fb", "j", "plain"), ("q2", "j", "plain")],
        {"j": "join, 2 groups"},
    ),
    "ambiguous": (
        "f b x d e j",
        [("f", "b", "plain"), ("f", "x", "plain"), ("x", "d", "when"), ("x", "e", "else"),
         ("b", "j", "plain"), ("d", "j", "plain")],
        "node 'j' is neither a merge nor a join: d -> j does not run on every pass that runs b -> j",
    ),
    "loop exit joins a parallel chain": (
        "w c a b m s1 s3 fin",
        [("w", "c", "plain"), ("c", "a", "when"), ("c", "b", "else"), ("a", "m", "plain"), ("b", "m", "plain"),
         ("m", "w", "while"), ("m", "fin", "else"), ("s1", "s3", "plain"), ("s3", "fin", "plain")],
        {"m": "merge", "fin": "join, 2 groups"},
    ),
    "parallel join with an inner diamond": (
        "f b x y z w j",
        [("f", "b", "plain"), ("f", "x", "plain"), ("x", "y", "when"), ("x", "z", "else"),
         ("y", "w", "plain"), ("z", "w", "plain"), ("w", "j", "plain"), ("b", "j", "plain")],
        {"w": "merge", "j": "join, 2 groups"},
    ),
    "switch to two terminals": (
        "judge post hold", [("judge", "post", "switch"), ("judge", "hold", "else")], {END: "merge"}
    ),
}  # fmt: skip


@pytest.mark.parametrize("shape", list(SHAPES))
def test_each_fan_in_shape_is_a_merge_or_a_join(shape: str) -> None:
    names, triples, expected = SHAPES[shape]
    edges = [Edge(source, target, **KINDS[kind]) for source, target, kind in triples]
    joins, refusals = RouteGraph(names.split(), edges).classify()
    if isinstance(expected, str):
        assert any(problem.startswith(expected) for problem in refusals.values())
        return
    assert not refusals
    assert {n: j.kind if j.kind == "merge" else f"join, {len(j.groups)} groups" for n, j in joins.items()} == expected
    relayed = {name: tuple(r is not None for r in join.relays) for name, join in joins.items() if join.kind == "join"}
    assert relayed == {
        "parallel diamond": {"j": (False, False)},
        "on-error fallback into a join": {"j": (True, False)},  # q1's ok route or fb
        "loop exit joins a parallel chain": {"fin": (True, False)},  # m's else route
        "parallel join with an inner diamond": {"j": (False, False)},
    }.get(shape, {})
