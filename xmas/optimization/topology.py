# The operators are adapted from the graph_ops, mutation and crossover modules
# of the architecture_evolution package in tritthart/mas-optimization.
"""Graph operators for topology search.

The operators work on the *agent* edge set, by name: the edges between agents,
with the ``START`` and ``END`` wiring left to :func:`xmas.graph.build_graph`,
which reconnects loose ends on every construction. A candidate is legal exactly
when a ``MAS`` can be built from it, the same check the engine applies, so the
sampler, the operators and the executor cannot disagree about what a runnable
graph is. Several agents may be fed from ``START``, as a fan-in-to-consensus
shape needs; a second agent reaching ``END`` makes the output ambiguous, so the
single-terminal rule is enforced.

A version 2 system adds what no operator may touch (:class:`Fixed`): frozen
nodes and the endpoints of conditional edges keep every edge around them, steps
and boxes are never removed, and the system keeps the nodes it ends in. The
operators then move plain edges between free nodes only, and whether a
candidate runs is :func:`~xmas.check.check_system`'s answer, asked when the
candidate is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from xmas.graph import END, START, back_edges, build_graph, terminal_agent, validate_graph

if TYPE_CHECKING:
    import random

__all__ = [
    "KEEP_FROM_FIRST",
    "KEEP_FROM_SECOND",
    "MAX_ATTEMPTS",
    "Edges",
    "Fixed",
    "TopologyBudget",
    "add_agent",
    "add_edge",
    "crossover",
    "cycles",
    "is_buildable",
    "is_valid",
    "mas_edges",
    "mutate",
    "next_name",
    "remove_agent",
    "remove_edge",
    "repair",
    "rewire_edge",
    "sources",
    "terminals",
    "within_budget",
]


Edges = list[tuple[str, str]]
"""An agent-level edge set: the edges a ``MAS`` is constructed from."""


@dataclass(frozen=True)
class TopologyBudget:
    """The bounds a searched topology has to stay inside.

    One object, consulted by the sampler, the mutation operators, the crossover
    and the search loop alike, so that "legal" means one thing.

    The node bounds count every node, steps and boxes included, so a system
    whose code nodes already fill ``max_agents`` cannot grow.

    Attributes:
        min_agents: Fewest nodes a candidate may have.
        max_agents: Most nodes a candidate may have.
        min_edges: Fewest node-to-node edges. ``None`` means ``nodes - 1``,
            the least a connected chain needs.
        max_edges: Most node-to-node edges.
        max_cycles: How many non-trivial strongly connected components are
            allowed. Zero, the default, searches directed acyclic graphs only.
    """

    min_agents: int = 2
    max_agents: int = 6
    min_edges: int | None = None
    max_edges: int = 12
    max_cycles: int = 0

    def __post_init__(self) -> None:
        """Reject a budget that no topology could satisfy."""
        if self.min_agents < 1:
            raise ValueError("min_agents must be at least 1.")
        if self.max_agents < self.min_agents:
            raise ValueError("max_agents must be at least min_agents.")
        if self.max_edges < self.max_agents - 1:
            raise ValueError("max_edges must leave room to connect max_agents, so at least max_agents - 1.")
        if self.min_edges is not None and self.min_edges > self.max_edges:
            raise ValueError("min_edges must not exceed max_edges.")
        if self.max_cycles < 0:
            raise ValueError("max_cycles must not be negative.")

    def floor(self, agents: int) -> int:
        """Return the minimum edge count for a given number of agents."""
        return self.min_edges if self.min_edges is not None else max(agents - 1, 0)


@dataclass(frozen=True)
class Fixed:
    """What a mutation of a version 2 system leaves where it is.

    Passing one (rather than ``None``) also switches on version 2's validity
    rules: the candidate keeps the parent's ends, the cycles a ``while`` edge
    closes are not counted, and runnability is left to the system's own check.

    Attributes:
        nodes: Frozen nodes, conditional-edge endpoints and chain members. No operator adds,
            removes or moves an edge touching one, or removes the node.
        code: Steps and boxes. Their plain edges may move, but only agents are
            invented and dropped.
        terminals: The nodes the parent ends in, and so every candidate.
        conditional: The conditional edges, as pairs.
    """

    nodes: frozenset[str] = frozenset()
    code: frozenset[str] = frozenset()
    terminals: frozenset[str] = frozenset()
    conditional: frozenset[tuple[str, str]] = frozenset()

    def holds(self, edge: tuple[str, str]) -> bool:
        """Whether an edge touches a fixed node, so that no operator may change it."""
        return edge[0] in self.nodes or edge[1] in self.nodes


def terminals(names: list[str], edges: Edges) -> list[str]:
    """Return the agents with no outgoing edge, which are the ones feeding END."""
    with_successors = {source for source, _ in edges}
    return [name for name in names if name not in with_successors]


def cycles(names: list[str], edges: Edges) -> int:
    """Count the non-trivial strongly connected components among the agents."""
    return sum(1 for component in nx.strongly_connected_components(digraph(names, edges)) if len(component) > 1)


def digraph(names: list[str], edges: Edges) -> nx.DiGraph[str]:
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from(edges)
    return graph


def sources(names: list[str], edges: Edges) -> list[str]:
    """Return the agents with no incoming edge, which are the ones fed from START."""
    with_predecessors = {target for _, target in edges}
    return [name for name in names if name not in with_predecessors]


def mas_edges(names: list[str], edges: Edges, entry: str | None = None) -> Edges:
    """Return the edge list a ``MAS`` is built from, naming the boundary where it must be named.

    :func:`xmas.graph.build_graph` wires ``START`` to every agent with no
    predecessor and ``END`` to every agent with no successor, which covers
    everything acyclic. It covers nothing about a retry loop. In
    ``solver -> checker -> solver`` every agent has a predecessor, so nothing is
    wired from ``START`` and the graph is unreachable; every agent also has a
    successor, so nothing is wired to ``END`` and the graph never finishes. That
    shape is the canonical retry loop, and it is exactly what topology search is
    for, so rather than excluding it, both ends are named here.

    The exit is the agent that closes the cycle, which is the same agent the
    engine gives the loop-or-continue decision to. Anything else would put the
    decision and the way out in different places. ``entry``, normally the
    parent's, is the preferred entry, ignored if it is not among ``names`` or
    the topology already has a source.
    """
    if not names:
        return list(edges)

    wiring = list(edges)
    if not sources(names, edges):
        wiring = [(START, entry if entry in names else min(names)), *wiring]
    if not terminals(names, edges):
        wiring = [*wiring, (exit_agent(names, wiring), END)]
    return wiring


def exit_agent(names: list[str], wiring: Edges) -> str:
    """Return the source of a cycle-closing edge of ``wiring``, else the last agent by name."""
    closing = sorted(back_edges(digraph([START, *names], wiring)))
    return closing[0][0] if closing else max(names)


def is_buildable(names: list[str], edges: Edges, entry: str | None = None) -> bool:
    """Whether a ``MAS`` could be constructed from this candidate, by the core's own predicate."""
    try:
        graph = build_graph(list(names), mas_edges(names, edges, entry))
        validate_graph(graph)
        terminal_agent(graph)
    except ValueError:
        return False
    return True


def within_budget(names: list[str], edges: Edges, budget: TopologyBudget, fixed: Fixed | None = None) -> bool:
    """Check a candidate against the budget's numeric bounds only; ``fixed`` is a version 2 parent's."""
    if not budget.min_agents <= len(names) <= budget.max_agents:
        return False
    if not budget.floor(len(names)) <= len(edges) <= budget.max_edges:
        return False
    if fixed is not None:
        # Version 2 may end in several alternatives (switch branches that join
        # at the end), so the rule is "the parent's ends", and no fixed or code
        # node may have gone missing. A loop closed by a `while` edge is the
        # parent's, not the search's, so only plain cycles count.
        if not fixed.nodes | fixed.code <= set(names) or set(terminals(names, edges)) != fixed.terminals:
            return False
        return cycles(names, [edge for edge in edges if edge not in fixed.conditional]) <= budget.max_cycles
    # Zero terminals is legal: a fully cyclic topology has its exit named by
    # `mas_edges`. Two or more is not, because the system's output would be
    # ambiguous and a MAS refuses to build one.
    if len(terminals(names, edges)) > 1:
        return False
    return cycles(names, edges) <= budget.max_cycles


def is_valid(
    names: list[str], edges: Edges, budget: TopologyBudget, entry: str | None = None, fixed: Fixed | None = None
) -> bool:
    """Check a candidate against both the budget and the core's runnability rules.

    The single predicate the sampler, the operators and the search loop all use,
    so that "legal topology" means one thing across the package. A version 2
    candidate (``fixed`` given) is checked against the budget here and by
    :func:`~xmas.check.check_system` when it is built, which knows its
    conditional edges, joins and reads.
    """
    return within_budget(names, edges, budget, fixed) and (fixed is not None or is_buildable(names, edges, entry))


# ------------------------------------------------------------------ operators


def add_edge(names: list[str], edges: Edges, rng: random.Random, fixed: Fixed | None = None) -> Edges:
    """Add one edge between two nodes that do not already have one, neither of them fixed."""
    present = set(edges)
    held = fixed.nodes if fixed is not None else frozenset()
    options = [
        (source, target)
        for source in names
        for target in names
        if source != target and (source, target) not in present and source not in held and target not in held
    ]
    if not options:
        return list(edges)
    return sorted([*edges, rng.choice(options)])


def remove_edge(names: list[str], edges: Edges, rng: random.Random, fixed: Fixed | None = None) -> Edges:  # noqa: ARG001
    """Remove one edge that touches no fixed node."""
    movable = [edge for edge in edges if fixed is None or not fixed.holds(edge)]
    if not movable:
        return list(edges)
    dropped = rng.choice(movable)
    return [edge for edge in edges if edge != dropped]


def rewire_edge(names: list[str], edges: Edges, rng: random.Random, fixed: Fixed | None = None) -> Edges:
    """Move one edge somewhere else, keeping the edge count unchanged."""
    return add_edge(names, remove_edge(names, edges, rng, fixed), rng, fixed)


def add_agent(
    names: list[str], edges: Edges, rng: random.Random, fixed: Fixed | None = None
) -> tuple[list[str], Edges]:
    """Insert a new agent upstream of the terminal.

    The new agent is wired before the current terminal and, where there is one to
    spare, after a randomly chosen other agent, so it arrives with somewhere to
    read from and somewhere to write to and the candidate stays connected
    without a repair pass.

    The predecessor is never the terminal itself, which would close a cycle
    between the two and fail an acyclic budget. With no other agent available
    the new one has no predecessor, and becomes the entry.

    A version 2 system keeps its ends, so there the new agent goes beside a
    movable edge ``a -> b`` instead, as ``a -> new -> b``: it reads what ``a``
    writes, ``b`` gains it as an input, and no terminal or cycle appears.
    """
    name = next_name(names)
    if fixed is not None:
        movable = [edge for edge in edges if not fixed.holds(edge)]
        if not movable:
            return list(names), list(edges)
        source, target = rng.choice(movable)
        return [*names, name], sorted({*edges, (source, name), (name, target)})
    terminal = terminals(names, edges)
    if not terminal:
        return list(names), list(edges)

    grown = {*edges, (name, terminal[0])}
    others = [candidate for candidate in names if candidate != terminal[0]]
    if others:
        grown.add((rng.choice(others), name))
    return [*names, name], sorted(grown)


def remove_agent(
    names: list[str], edges: Edges, rng: random.Random, fixed: Fixed | None = None
) -> tuple[list[str], Edges]:
    """Drop an agent, bridging its predecessors to its successors.

    Bridging rather than simply deleting is what keeps the rest of the graph
    connected. Fixed nodes, steps and boxes are
    never dropped, nor an agent next to a fixed node, whose edge is fixed too.
    """
    terminal = terminals(names, edges)
    removable = [name for name in names if name not in terminal]
    if fixed is not None:
        removable = [
            name
            for name in removable
            if name not in fixed.nodes | fixed.code and not any(fixed.holds(edge) for edge in edges if name in edge)
        ]
    if len(names) <= 1 or not removable:
        return list(names), list(edges)

    victim = rng.choice(removable)
    predecessors = [source for source, target in edges if target == victim]
    successors = [target for source, target in edges if source == victim]
    survivors = [name for name in names if name != victim]
    kept = [(source, target) for source, target in edges if victim not in (source, target)]
    bridged = {(source, target) for source in predecessors for target in successors if source != target}
    return survivors, sorted({*kept, *bridged})


def next_name(names: list[str]) -> str:
    """Return an agent name not already in use."""
    index = len(names)
    while f"agent_{index}" in names:
        index += 1
    return f"agent_{index}"


# ------------------------------------------------------------------ mutation


def mutate(  # noqa: PLR0913, PLR0917  the public operator signature, kept for version 1 callers
    names: list[str],
    edges: Edges,
    budget: TopologyBudget,
    rng: random.Random,
    operations: int = 1,
    entry: str | None = None,
    fixed: Fixed | None = None,
) -> tuple[list[str], Edges]:
    """Apply a sequence of operators, retrying the sequence if it lands outside the budget.

    Retries the whole sequence of ``operations`` operators rather than the last
    one, and falls back to the parent unchanged when none of the attempts
    works. A tightly bounded search space produces a lot of invalid
    intermediates, and retrying only the last step biases the walk toward
    whichever operator happens to be cheapest to satisfy. ``entry`` is the
    parent's, preferred for a child whose entry is inside a cycle.
    """
    for _ in range(3):
        current_names, current_edges = list(names), list(edges)
        for _ in range(max(1, operations)):
            current_names, current_edges = apply_one(current_names, current_edges, budget, rng, fixed)
        if is_valid(current_names, current_edges, budget, entry, fixed):
            return current_names, current_edges
    return list(names), list(edges)


def apply_one(
    names: list[str],
    edges: Edges,
    budget: TopologyBudget,
    rng: random.Random,
    fixed: Fixed | None = None,
) -> tuple[list[str], Edges]:
    """Pick one operator that the budget still has room for, and apply it."""
    edge_ops = []
    if len(edges) < budget.max_edges:
        edge_ops.append(add_edge)
    if len(edges) > budget.floor(len(names)):
        edge_ops.append(remove_edge)
    edge_ops.append(rewire_edge)

    node_ops = []
    if len(names) < budget.max_agents:
        node_ops.append(add_agent)
    if len(names) > budget.min_agents:
        node_ops.append(remove_agent)

    choice = rng.randrange(len(edge_ops) + len(node_ops))
    if choice < len(edge_ops):
        return list(names), edge_ops[choice](names, edges, rng, fixed)
    return node_ops[choice - len(edge_ops)](names, edges, rng, fixed)


# ----------------------------------------------------------------- crossover

KEEP_FROM_FIRST = 0.75
KEEP_FROM_SECOND = 0.6
"""Inheritance rates: the first parent contributes more."""

MAX_ATTEMPTS = 10
"""Retries before giving up on recombination.

Falling back to a fresh random graph *within* an attempt turns most crossovers
into reseeds in a tightly constrained space, which defeats the point of having
two parents. So the whole recombination is retried, and only a total failure
falls back.
"""


def crossover(  # noqa: PLR0913, PLR0917  the public operator signature, kept for version 1 callers
    first: tuple[list[str], Edges],
    second: tuple[list[str], Edges],
    budget: TopologyBudget,
    rng: random.Random,
    entry: str | None = None,
    fixed: Fixed | None = None,
) -> tuple[list[str], Edges]:
    """Recombine two topologies into one that inherits from both.

    A version 2 child keeps the first parent's fixed edges and inherits only
    movable ones. It is not repaired, because the repair reshapes the graph
    around one terminal; an attempt outside the rules is retried instead.
    With no legal attempt, the child is the first parent.
    """
    for _ in range(MAX_ATTEMPTS):
        names = list(rng.choice([first[0], second[0]]))
        known = set(names)
        inherited = {
            edge for edge in first[1] if edge[0] in known and edge[1] in known and rng.random() < KEEP_FROM_FIRST
        } | {edge for edge in second[1] if edge[0] in known and edge[1] in known and rng.random() < KEEP_FROM_SECOND}
        if fixed is None:
            child = repair(names, sorted(inherited), budget, rng)
        else:
            held = {edge for edge in first[1] if fixed.holds(edge) and edge[0] in known and edge[1] in known}
            child = sorted(held | {edge for edge in inherited if not fixed.holds(edge)})
        if is_valid(names, child, budget, entry, fixed):
            return names, child
    return list(first[0]), list(first[1])


def repair(names: list[str], edges: Edges, budget: TopologyBudget, rng: random.Random) -> Edges:
    """Make an inherited edge set into something a MAS can be built from.

    Recombination cuts edges without regard for what the result has to satisfy,
    so the child arrives with self-loops, several agents reaching ``END``, agents
    that reach it through nothing, and an edge count anywhere at all. This walks
    it back inside the rules in that order, because each step can undo the last
    if they are done the other way round.
    """
    current = [(source, target) for source, target in edges if source != target]
    terminal = pick_terminal(names, current, rng)
    current = [(source, target) for source, target in current if source != terminal]

    # Every other agent needs a way onward, and a way that ends at the terminal.
    for name in names:
        if name != terminal and not any(source == name for source, _ in current):
            current.append((name, terminal))
    current = connect_to_terminal(names, current, terminal)

    if budget.max_cycles == 0:
        current = break_cycles(names, current, terminal)

    while len(current) > budget.max_edges:
        shorter = remove_edge(names, current, rng)
        if len(terminals(names, shorter)) != 1 or shorter == current:
            break
        current = shorter
    while len(current) < budget.floor(len(names)):
        longer = add_edge(names, current, rng)
        if longer == current:
            break
        current = [(source, target) for source, target in longer if source != terminal]

    return sorted(set(current))


def pick_terminal(names: list[str], edges: Edges, rng: random.Random) -> str:
    """Choose the one agent that will feed END, preferring an inherited terminal."""
    inherited = terminals(names, edges)
    return inherited[0] if inherited else rng.choice(sorted(names))


def connect_to_terminal(names: list[str], edges: Edges, terminal: str) -> Edges:
    """Add edges until every agent has a path to the terminal."""
    current = list(edges)
    while True:
        graph = digraph(names, current)
        stranded = [name for name in names if name != terminal and not nx.has_path(graph, name, terminal)]
        if not stranded:
            return current
        current.append((stranded[0], terminal))


def break_cycles(names: list[str], edges: Edges, terminal: str) -> Edges:
    """Drop edges until the topology is acyclic, keeping the terminal reachable."""
    current = list(edges)
    while True:
        try:
            cycle = nx.find_cycle(digraph(names, current))
        except nx.NetworkXNoCycle:
            return connect_to_terminal(names, current, terminal)
        source, target = cycle[-1][0], cycle[-1][1]
        current = [edge for edge in current if edge != (source, target)]
