"""Unequal-depth fan-in must run a join node exactly once per query.

This is the defect the WP0 recon found latent in the seed engine (commit
``a2edfbc`` on the mas-optimization side). It does not bite a hand-written graph
with balanced branches, which is why it survived there, but every mutated
topology produces the shape, and topology mutation is the point of xmas.
"""

from __future__ import annotations

import pytest

from xmas import MAS, Agent
from xmas.graph import build_graph, join_nodes

from .conftest import MockLLM


def unequal_depth_mas(llm: MockLLM) -> MAS:
    """``a -> b -> c`` together with ``a -> c``.

    ``c``'s two inputs arrive in different super-steps: ``a``'s immediately and
    ``b``'s one step later. Without deferral ``c`` fires on each arrival.
    """
    return MAS(
        [
            Agent("a", prompt="A", output="from_a"),
            Agent("b", prompt="B", inputs=["from_a"], output="from_b"),
            Agent("c", prompt="C", inputs=["from_a", "from_b"], output="prediction"),
        ],
        edges=[("a", "b"), ("b", "c"), ("a", "c")],
        llm=llm,
    )


def test_join_nodes_identifies_the_unequal_depth_node() -> None:
    graph = build_graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
    assert join_nodes(graph) == {"c"}


def test_a_join_node_runs_exactly_once(mock_llm: MockLLM) -> None:
    mas = unequal_depth_mas(mock_llm)
    trace = mas.run("q")

    assert trace.error is None
    assert len(trace["c"]) == 1, "the join node must execute once per query"
    assert len(trace["a"]) == 1
    assert len(trace["b"]) == 1


def test_the_join_node_sees_both_upstream_outputs(mock_llm: MockLLM) -> None:
    """Running once is only correct if it runs *late* enough to see everything."""
    mas = unequal_depth_mas(mock_llm)
    trace = mas.run("q")

    (record,) = trace["c"]
    assert record.inputs["from_a"], "c ran before a's output was available"
    assert record.inputs["from_b"], "c ran before b's output was available"


def test_without_deferral_the_join_node_runs_twice(monkeypatch: pytest.MonkeyPatch, mock_llm: MockLLM) -> None:
    """Prove the deferral is load-bearing rather than incidentally harmless.

    With :func:`xmas.graph.join_nodes` neutered, the same topology reproduces the
    original defect. If someone deletes the ``defer`` argument, the test above
    keeps passing only if this one starts failing, so the pair pins the fix.
    """
    monkeypatch.setattr("xmas._engine.join_nodes", lambda graph: set())

    mas = unequal_depth_mas(mock_llm)
    trace = mas.run("q")

    assert len(trace["c"]) == 2, "expected the un-deferred defect to reproduce"


def test_balanced_fan_in_still_runs_once(mock_llm: MockLLM) -> None:
    """The equal-depth case the seed repo's graphs used, kept as a regression."""
    mas = MAS(
        [
            Agent("x", prompt="X", output="from_x"),
            Agent("y", prompt="Y", output="from_y"),
            Agent("z", prompt="Z", inputs=["from_x", "from_y"], output="prediction"),
        ],
        edges=[("x", "z"), ("y", "z")],
        llm=mock_llm,
    )
    trace = mas.run("q")

    assert len(trace["z"]) == 1
    (record,) = trace["z"]
    assert record.inputs["from_x"] and record.inputs["from_y"]
