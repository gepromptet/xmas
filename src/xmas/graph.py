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


def _reachable(graph: nx.DiGraph[str], source: str, *, forward: bool) -> set[str]:
    """Return every node reachable from ``source``, following edges either way.

    Written out rather than calling ``nx.descendants``/``nx.ancestors`` because
    the networkx stubs type those as taking ``Graph[object]``, which a
    ``DiGraph[str]`` is not. A six-line traversal keeps the public signatures
    honest about nodes being agent names.

    Args:
        graph: The topology.
        source: Node to start from.
        forward: Follow successors when true, predecessors when false.

    Returns:
        The reachable nodes, excluding ``source`` itself.
    """
    step = graph.successors if forward else graph.predecessors
    seen: set[str] = set()
    stack = list(step(source))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(step(node))
    return seen


def build_graph(agents: list[str], edges: list[tuple[str, str]]) -> nx.DiGraph[str]:
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
    reserved = SENTINELS.intersection(agents)
    if reserved:
        raise ValueError(f"Agent names {sorted(reserved)} are reserved sentinels.")
    duplicates = {name for name in agents if agents.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate agent names: {sorted(duplicates)}.")

    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from([START, *agents, END])

    known = set(agents) | SENTINELS
    for source, target in edges:
        for node in (source, target):
            if node not in known:
                raise ValueError(f"Edge ({source!r}, {target!r}) names unknown agent {node!r}.")
        graph.add_edge(source, target)

    # Wire the loose ends. Doing it here rather than asking the caller for it is
    # what lets `edges=[("solver", "checker")]` describe a whole system.
    for agent in agents:
        if graph.in_degree(agent) == 0:
            graph.add_edge(START, agent)
        if graph.out_degree(agent) == 0:
            graph.add_edge(agent, END)

    return graph


def validate_graph(graph: nx.DiGraph[str]) -> None:
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
    for sentinel in (START, END):
        if sentinel not in graph:
            raise ValueError(f"Topology is missing the {sentinel!r} sentinel.")
    if graph.in_degree(START) != 0:
        raise ValueError(f"{START!r} must have no incoming edges.")
    if graph.out_degree(END) != 0:
        raise ValueError(f"{END!r} must have no outgoing edges.")

    agents = set(graph.nodes) - SENTINELS
    if not agents:
        raise ValueError("Topology has no agents.")

    unreachable = agents - _reachable(graph, START, forward=True)
    if unreachable:
        raise ValueError(f"Agents unreachable from {START!r}: {sorted(unreachable)}.")

    dead_ends = agents - _reachable(graph, END, forward=False)
    if dead_ends:
        raise ValueError(f"Agents that never reach {END!r}: {sorted(dead_ends)}.")


def agent_names(graph: nx.DiGraph[str]) -> list[str]:
    """Return the non-sentinel nodes, in deterministic order.

    Args:
        graph: The topology.

    Returns:
        Agent names, topologically sorted where the graph is acyclic and sorted
        by name otherwise, so serialization is stable.
    """
    agents = set(graph.nodes) - SENTINELS
    if nx.is_directed_acyclic_graph(graph):
        return [node for node in nx.topological_sort(graph) if node in agents]
    return sorted(agents)


def _pending(graph: nx.DiGraph[str], node: str) -> list[str]:
    """Return a node's successors ready to be consumed by ``pop``.

    Reversed, because ``pop`` takes from the tail and the traversal is specified
    to visit successors in ascending order.

    Args:
        graph: The topology.
        node: The node whose successors are wanted.

    Returns:
        The successors, in descending order.
    """
    return sorted(graph.successors(node), reverse=True)


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

    Args:
        graph: The topology.

    Returns:
        The cycle-closing edges. Empty for a directed acyclic graph.
    """
    found: set[tuple[str, str]] = set()
    finished: set[str] = set()
    for root in [START, *sorted(graph.nodes)]:
        if root not in graph or root in finished:
            continue
        # Iterative DFS. `open_nodes` is the current stack as a set, so an edge
        # into it is an edge into an ancestor, which is the definition of a back
        # edge. Recursion is avoided because a mutated topology can be deep.
        stack: list[tuple[str, list[str]]] = [(root, _pending(graph, root))]
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
                stack.append((target, _pending(graph, target)))
                open_nodes.add(target)
    return found


def forward_predecessors(graph: nx.DiGraph[str], node: str) -> set[str]:
    """Return a node's predecessors excluding those that reach it by a back edge.

    Args:
        graph: The topology.
        node: The node to inspect.

    Returns:
        Predecessors on acyclic paths only.
    """
    back = back_edges(graph)
    return {source for source in graph.predecessors(node) if (source, node) not in back}


def join_nodes(graph: nx.DiGraph[str]) -> set[str]:
    """Return the agents that must wait for every inbound path before running.

    A node fed by predecessors at unequal depth would otherwise execute once per
    wave of arriving inputs: in ``A -> C`` together with ``A -> B -> C``, C sees
    A's write in one step and B's in the next, and runs twice for one query. The
    seed engine has this defect latent, and every mutated topology triggers it,
    so the set computed here is what the executor defers on.

    Back edges are excluded from the count. A node at the top of a retry loop is
    fed both from upstream and from the agent that decided to loop, but those two
    never arrive in the same pass, and deferring on the second would hold the node
    until an input that only exists if it runs first.

    Args:
        graph: The topology.

    Returns:
        Names of agents with more than one distinct forward predecessor.
    """
    agents = set(graph.nodes) - SENTINELS
    back = back_edges(graph)
    counts = {agent: sum(1 for source in graph.predecessors(agent) if (source, agent) not in back) for agent in agents}
    return {agent for agent, count in counts.items() if count > 1}


def terminal_agent(graph: nx.DiGraph[str]) -> str:
    """Return the single agent feeding :data:`END`.

    Args:
        graph: The topology.

    Returns:
        The terminal agent's name, whose output is the MAS output.

    Raises:
        ValueError: If the topology has zero or several terminal agents.
    """
    terminals = sorted(graph.predecessors(END))
    if not terminals:
        raise ValueError(f"No agent feeds {END!r}, so the MAS has no output.")
    if len(terminals) > 1:
        raise ValueError(
            f"Several agents feed {END!r} ({terminals}), so the MAS output is ambiguous. "
            "Add a final agent that consumes them."
        )
    return terminals[0]
