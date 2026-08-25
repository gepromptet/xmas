"""Cyclic topologies: how they are detected, how they compile, how they stop.

A topology optimizer proposes cycles whether or not anyone asked it to, so the
engine's behaviour on one has to be pinned rather than assumed. Before this
package a cycle constructed and ran but could not terminate: the edge back and
the edge onward both fired every pass, and the run died at the recursion limit.
"""

from __future__ import annotations

import networkx as nx
import pytest

import xmas._engine as engine
from xmas import MAS, Agent
from xmas.graph import END, SENTINELS, START, back_edges, build_graph, forward_predecessors, join_nodes

from .conftest import MockLLM

LOOP = [(START, "solver"), ("solver", "checker"), ("checker", "solver"), ("checker", END)]
"""solver drafts, checker decides whether to send it back. The canonical retry."""


def loop_agents() -> list[Agent]:
    """Two agents wired as a draft-and-review loop."""
    return [
        Agent("solver", "solve {query} {feedback}", output="draft", inputs=["query", "feedback"]),
        Agent("checker", "check {draft}", output="feedback", inputs=["draft"]),
    ]


def scripted(responses: list[str]) -> MockLLM:
    """A model that returns each response once, in order, then repeats the last."""
    remaining = list(responses)

    def respond(prompt: str) -> str:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return MockLLM(respond)


# ------------------------------------------------------------------ detection


def test_a_directed_acyclic_topology_has_no_back_edges() -> None:
    graph = build_graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
    assert back_edges(graph) == set()


@pytest.mark.parametrize(
    ("agents", "edges"),
    [
        (["a", "b"], [(START, "a"), ("a", "b"), ("b", "a"), ("b", END)]),
        (["a", "b", "c"], [(START, "a"), ("a", "b"), ("b", "c"), ("c", "a"), ("c", END)]),
        (["a", "b", "c"], [(START, "a"), ("a", "b"), ("b", "c"), ("a", "c"), ("c", "b"), ("c", END)]),
        (["a"], [(START, "a"), ("a", "a"), ("a", END)]),
    ],
)
def test_removing_the_back_edges_leaves_an_acyclic_topology(agents: list[str], edges: list[tuple[str, str]]) -> None:
    graph = build_graph(agents, edges)
    stripped = nx.DiGraph(graph)
    stripped.remove_edges_from(back_edges(graph))
    assert not nx.is_directed_acyclic_graph(graph)
    assert nx.is_directed_acyclic_graph(stripped)


def test_the_back_edge_is_the_one_running_against_the_flow() -> None:
    # a -> b -> c is the flow; c -> b is the retry, so c owns the decision.
    graph = build_graph(["a", "b", "c"], [(START, "a"), ("a", "b"), ("b", "c"), ("a", "c"), ("c", "b"), ("c", END)])
    assert back_edges(graph) == {("c", "b")}


def test_join_detection_ignores_back_edges() -> None:
    graph = build_graph(*(["solver", "checker"], LOOP))
    # solver is fed by START and by checker's back edge, but those never arrive
    # together, so solver is not a join.
    assert set(graph.predecessors("solver")) == {START, "checker"}
    assert forward_predecessors(graph, "solver") == {START}
    assert join_nodes(graph) == set()


# -------------------------------------------------------------------- running


def test_a_cycle_terminates_when_the_agent_stops_asking_for_another_pass() -> None:
    llm = scripted(["draft-1", f"{engine.CONTINUE_MARKER} not yet", "draft-2", "looks good"])
    mas = MAS(agents=loop_agents(), edges=LOOP, llm=llm, max_steps=25)

    trace = mas.run("2+2?")

    assert trace.error is None
    assert [record.agent for record in trace.records] == ["solver", "checker", "solver", "checker"]
    assert trace.output == "looks good"


def test_a_cycle_runs_once_when_no_agent_asks_to_continue() -> None:
    mas = MAS(agents=loop_agents(), edges=LOOP, llm=scripted(["fine"]), max_steps=25)

    trace = mas.run("2+2?")

    assert trace.error is None
    assert [record.agent for record in trace.records] == ["solver", "checker"]


def test_a_prompt_that_always_loops_stops_at_the_visit_cap() -> None:
    # The failure an optimizer eventually proposes: a prompt that emits the
    # marker unconditionally. The run must still end, and end with an answer.
    mas = MAS(agents=loop_agents(), edges=LOOP, llm=scripted([f"{engine.CONTINUE_MARKER} again"]), max_steps=50)

    trace = mas.run("2+2?")

    assert trace.error is None
    visits = [record.agent for record in trace.records].count("checker")
    assert visits == engine.MAX_NODE_VISITS


def test_an_unequal_depth_join_still_runs_once_inside_a_cyclic_topology() -> None:
    # c is a genuine join (a and b both feed it, at different depths) and also
    # sits on a cycle. The deferral has to survive the cycle.
    agents = [
        Agent("a", "a {query}", output="oa"),
        Agent("b", "b {oa}", output="ob", inputs=["oa"]),
        Agent("c", "c {oa} {ob}", output="oc", inputs=["oa", "ob"]),
    ]
    edges = [(START, "a"), ("a", "b"), ("b", "c"), ("a", "c"), ("c", "b"), ("c", END)]
    mas = MAS(agents=agents, edges=edges, llm=scripted(["done"]), max_steps=25)

    trace = mas.run("q")

    assert trace.error is None
    assert [record.agent for record in trace.records].count("c") == 1


def test_counting_back_edges_as_joins_defers_an_agent_that_should_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    # Defect reproduction. `join_nodes` used to count every predecessor, which on
    # a cyclic topology defers the agent at the top of the loop. Restore that and
    # the execution order changes, which is what pins the fix in place.
    agents = [
        Agent("a", "a {query}", output="oa"),
        Agent("b", "b {oa}", output="ob", inputs=["oa"]),
        Agent("c", "c {oa} {ob}", output="oc", inputs=["oa", "ob"]),
    ]
    edges = [(START, "a"), ("a", "b"), ("b", "c"), ("a", "c"), ("c", "b"), ("c", END)]

    def order() -> list[str]:
        mas = MAS(agents=[agent.copy() for agent in agents], edges=edges, llm=scripted(["done"]), max_steps=25)
        return [record.agent for record in mas.run("q").records]

    correct = order()

    def naive(graph: nx.DiGraph[str]) -> set[str]:
        return {node for node in set(graph.nodes) - SENTINELS if graph.in_degree(node) > 1}

    monkeypatch.setattr(engine, "join_nodes", naive)
    assert naive(build_graph(["a", "b", "c"], edges)) == {"b", "c"}
    assert order() != correct
