"""Topology: the graph a MAS executes, and the questions asked of it.

The topology is a ``networkx.DiGraph`` whose nodes are agent names plus two
sentinels. Keeping it a plain nx graph is deliberate: a topology optimizer is
then an ordinary graph-rewriting component and needs nothing from xmas to work.
"""

from __future__ import annotations

import networkx as nx

START = "__start__"
"""Source sentinel. Every agent with no predecessor is wired from it."""

END = "__end__"
"""Terminal sentinel. Every agent with no successor is wired to it."""

SENTINELS = frozenset({START, END})
"""The node names reserved by xmas; an agent may not use either."""


def build_graph(agents: list[str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Build the topology for a set of agents and the edges between them.

    Agents with no incoming edge are connected from :data:`START`, and agents with
    no outgoing edge to :data:`END`, so the caller writes only the edges that
    carry meaning.

    Args:
        agents: Agent names, which become the non-sentinel nodes.
        edges: Directed ``(source, target)`` pairs between agent names. Sentinels
            may be named explicitly to override the implicit wiring.

    Returns:
        The topology, including both sentinels.

    Raises:
        ValueError: If an edge names an unknown agent, or an agent name collides
            with a sentinel.
    """
    raise NotImplementedError


def validate_graph(graph: nx.DiGraph) -> None:
    """Check that a topology is runnable.

    A runnable topology has exactly one source sentinel and one terminal
    sentinel, every agent reachable from :data:`START`, and every agent reaching
    :data:`END`. An unreachable agent is an error rather than a warning, because
    silently never running is the failure mode hardest to notice in a result.

    Args:
        graph: The topology to check.

    Raises:
        ValueError: If the topology is not runnable, naming the offending nodes.
    """
    raise NotImplementedError


def agent_names(graph: nx.DiGraph) -> list[str]:
    """Return the non-sentinel nodes, in deterministic order.

    Args:
        graph: The topology.

    Returns:
        Agent names, topologically sorted where the graph is acyclic and sorted
        by name otherwise, so serialization is stable.
    """
    raise NotImplementedError


def join_nodes(graph: nx.DiGraph) -> set[str]:
    """Return the agents that must wait for every inbound path before running.

    A node fed by predecessors at unequal depth would otherwise execute once per
    wave of arriving inputs: in ``A -> C`` together with ``A -> B -> C``, C sees
    A's write in one step and B's in the next, and runs twice for one query. The
    seed engine has this defect latent, and every mutated topology triggers it,
    so the set computed here is what the executor defers on.

    Args:
        graph: The topology.

    Returns:
        Names of agents with more than one distinct predecessor.
    """
    raise NotImplementedError


def terminal_agent(graph: nx.DiGraph) -> str:
    """Return the single agent feeding :data:`END`.

    Args:
        graph: The topology.

    Returns:
        The terminal agent's name, whose output is the MAS output.

    Raises:
        ValueError: If the topology has zero or several terminal agents.
    """
    raise NotImplementedError
