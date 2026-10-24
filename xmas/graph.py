"""Topology: the graph a MAS executes, and the questions asked of it.

The topology is a ``networkx.DiGraph`` whose nodes are node names, agents and
(version 2) steps alike, plus two sentinels. The functions here speak of agents
because version 1 has no other node; they treat every node the same. Keeping it
a plain nx graph is deliberate: a topology optimizer is then an ordinary
graph-rewriting component and needs nothing from xmas to work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Collection

__all__ = [
    "END",
    "SENTINELS",
    "START",
    "agent_names",
    "ancestors",
    "back_edges",
    "build_graph",
    "forward_graph",
    "forward_predecessors",
    "join_nodes",
    "loop_problems",
    "terminal_agent",
    "validate_graph",
]


START = "__start__"
"""Source sentinel. Every agent with no predecessor is wired from it."""

END = "__end__"
"""Terminal sentinel. Every agent with no successor is wired to it."""

SENTINELS = (START, END)
"""The node names reserved by xmas; an agent may not use either."""


def build_graph(
    agents: list[str],
    edges: list[tuple[str, str]],
    back: Collection[tuple[str, str]] = (),
    errors: Collection[tuple[str, str]] = (),
) -> nx.DiGraph[str]:
    """Build the topology for a set of agents and the edges between them.

    Agents with no incoming edge are connected from :data:`START`, and agents with
    no outgoing edge to :data:`END`, so the caller writes only the edges that
    carry meaning. An edge in ``back`` (a version 2 ``while`` edge) closes a
    loop and does not count as incoming: the loop's first node is still wired
    from :data:`START`. An edge in ``errors`` (a version 2 ``on error`` edge)
    is taken only when a visit fails, so a node whose out-edges are all error
    edges is wired to :data:`END` as well: a visit that succeeds ends there.
    ``agents`` holds every node name, steps included; an edge may name a
    sentinel to override the implicit wiring. An edge naming an unknown node,
    or a node named like a sentinel, raises ValueError.
    """
    reserved = [name for name in agents if name in SENTINELS]
    if reserved:
        raise ValueError(f"Agent names {sorted(reserved)} are reserved sentinels.")
    duplicates = {name for name in agents if agents.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate agent names: {sorted(duplicates)}.")

    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from([START, *agents, END])

    known = {*agents, *SENTINELS}
    for source, target in edges:
        for node in (source, target):
            if node not in known:
                raise ValueError(f"Edge ({source!r}, {target!r}) names unknown agent {node!r}.")
        graph.add_edge(source, target)

    # Wire the loose ends. Doing it here rather than asking the caller for it is
    # what lets `edges=[("solver", "checker")]` describe a whole system.
    for agent in agents:
        if not any((source, agent) not in back for source in graph.predecessors(agent)):
            graph.add_edge(START, agent)
        if not any((agent, target) not in errors for target in graph.successors(agent)):
            graph.add_edge(agent, END)

    return graph


def validate_graph(graph: nx.DiGraph[str]) -> None:
    """Check that a topology is runnable.

    A runnable topology has exactly one source sentinel and one terminal
    sentinel, every agent reachable from :data:`START`, and every agent reaching
    :data:`END`. An unreachable agent is an error rather than a warning, because
    silently never running is the failure mode hardest to notice in a result.
    The ValueError names the offending nodes.
    """
    for sentinel in (START, END):
        if sentinel not in graph:
            raise ValueError(f"Topology is missing the {sentinel!r} sentinel.")
    if graph.in_degree(START) != 0:
        raise ValueError(f"{START!r} must have no incoming edges.")
    if graph.out_degree(END) != 0:
        raise ValueError(f"{END!r} must have no outgoing edges.")

    agents = set(graph.nodes).difference(SENTINELS)
    if not agents:
        raise ValueError("Topology has no agents.")

    # The networkx stubs type these as taking Graph[object], which DiGraph[str] is not.
    unreachable = agents - nx.descendants(graph, START)  # type: ignore[arg-type]
    if unreachable:
        raise ValueError(f"Agents unreachable from {START!r}: {sorted(unreachable)}.")

    dead_ends = agents - nx.ancestors(graph, END)  # type: ignore[arg-type]
    if dead_ends:
        raise ValueError(f"Agents that never reach {END!r}: {sorted(dead_ends)}.")


def agent_names(graph: nx.DiGraph[str]) -> list[str]:
    """Return the non-sentinel nodes, topologically sorted when acyclic and by name otherwise."""
    agents = set(graph.nodes).difference(SENTINELS)
    if nx.is_directed_acyclic_graph(graph):
        return [node for node in nx.topological_sort(graph) if node in agents]
    return sorted(agents)


def back_edges(graph: nx.DiGraph[str]) -> set[tuple[str, str]]:
    """Return the edges whose removal makes the topology acyclic.

    These are the cycle-closing edges of a depth-first traversal from
    :data:`START`: an edge to a node that is still open on the search stack. They
    are the only edges a run may decline to follow, so identifying them is what
    turns a cyclic topology from an infinite loop into a retry loop. See
    :mod:`xmas.runtime` for how they compile.

    Which of a cycle's edges is the closing one is a real decision, not
    bookkeeping: the back edge is the conditional one, so it decides which agent
    owns the loop. Depth-first from :data:`START` with successors taken in sorted
    order picks the edge that runs against the flow, so in ``a -> b -> c`` with
    ``c -> b``, it is ``c -> b`` that becomes the retry and ``c`` that decides.
    """
    found: set[tuple[str, str]] = set()
    finished: set[str] = set()
    for root in [START, *sorted(graph.nodes)]:
        if root in finished:
            continue
        # Iterative DFS. `open_nodes` is the current stack as a set, so an edge
        # into it is an edge into an ancestor, which is the definition of a back
        # edge. Recursion is avoided because a mutated topology can be deep.
        # Successors are reversed because `pop` takes from the tail and the
        # traversal visits them in ascending order.
        stack: list[tuple[str, list[str]]] = [(root, sorted(graph.successors(root), reverse=True))]
        open_nodes = {root}
        while stack:
            node, pending = stack[-1]
            if not pending:
                stack.pop()
                open_nodes.discard(node)
                finished.add(node)
                continue
            target = pending.pop()
            if target in open_nodes:
                found.add((node, target))
            elif target not in finished:
                stack.append((target, sorted(graph.successors(target), reverse=True)))
                open_nodes.add(target)
    return found


def forward_graph(graph: nx.DiGraph[str], back: Collection[tuple[str, str]]) -> nx.DiGraph[str]:
    """Return a copy of the topology without its loop-closing edges: one pass of a version 2 run."""
    forward = graph.copy()
    forward.remove_edges_from(back)
    return forward


def loop_problems(graph: nx.DiGraph[str], back: Collection[tuple[str, str]] = ()) -> list[str]:
    """Return a problem for each cycle no ``while`` edge (``back``) closes and each ``while`` edge closing no cycle."""
    forward = forward_graph(graph, back)
    problems = [
        f"a loop needs a while edge: the edge {source} -> {target} closes a cycle"
        for source, target in sorted(back_edges(forward))
    ]
    for source, target in back:
        if source != target and not nx.has_path(forward, target, source):
            problems.append(f"the while edge {source} -> {target} closes no loop; make it a when edge")
    return problems


def ancestors(graph: nx.DiGraph[str], node: str) -> set[str]:
    """Return every node with a path to ``node``, sentinels included; ``node`` itself when it sits on a cycle."""
    seen: set[str] = set()
    stack = list(graph.predecessors(node))
    while stack:
        if (current := stack.pop()) not in seen:
            seen.add(current)
            stack.extend(graph.predecessors(current))
    return seen


def forward_predecessors(graph: nx.DiGraph[str], node: str) -> set[str]:
    """Return a node's predecessors excluding those that reach it by a back edge."""
    back = back_edges(graph)
    return {source for source in graph.predecessors(node) if (source, node) not in back}


def join_nodes(graph: nx.DiGraph[str]) -> set[str]:
    """Return the agents that must wait for every inbound path before running.

    A node fed by predecessors at unequal depth would otherwise execute once per
    wave of arriving inputs: in ``A -> C`` together with ``A -> B -> C``, C sees
    A's write in one step and B's in the next, and runs twice for one query, so
    the executor defers on the set computed here.

    Back edges are excluded from the count. A node at the top of a retry loop is
    fed both from upstream and from the agent that decided to loop, but those two
    never arrive in the same pass, and deferring on the second would hold the node
    until an input that only exists if it runs first.
    """
    back = back_edges(graph)
    return {
        agent
        for agent in set(graph.nodes).difference(SENTINELS)
        if sum((source, agent) not in back for source in graph.predecessors(agent)) > 1
    }


def terminal_agent(graph: nx.DiGraph[str]) -> str:
    """Return the single agent feeding :data:`END`, whose output is the MAS output; ValueError for none or several."""
    terminals = sorted(graph.predecessors(END))
    if not terminals:
        raise ValueError(f"No agent feeds {END!r}, so the MAS has no output.")
    if len(terminals) > 1:
        raise ValueError(
            f"Several agents feed {END!r} ({terminals}), so the MAS output is ambiguous. "
            "Add a final agent that consumes them."
        )
    return terminals[0]
