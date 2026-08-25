# Operators adapted from tritthart/mas-optimization, mas_lib/optimization/
# architecture_evolution/{graph_ops,mutation,crossover}.py
"""Graph operators for topology search.

The seed's operators work on integer-labelled graphs where node ``0`` is the
single source and ``max(nodes)`` the single terminal, and relabel on every call.
xmas keeps agent names instead, and expresses the boundary with the ``START`` and
``END`` sentinels, so these operators work on the *agent* edge set: the edges
between agents, with the sentinel wiring left to :func:`xmas.graph.build_graph`,
which reconnects loose ends on every construction.

That substitution buys the validity predicate for free. A candidate is legal
exactly when a ``MAS`` can be built from it, which is the same check the engine
already applies, so the sampler, the operators and the executor cannot disagree
about what a runnable graph is.

Two differences from the seed follow from it. xmas permits several agents to be
fed from ``START``, which the seed's single-source rule forbids and which the
canonical fan-in-to-consensus shape needs. And the single-terminal rule is
enforced rather than assumed, because a second agent reaching ``END`` makes the
system's output ambiguous and xmas refuses to build one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from xmas.graph import END, SENTINELS, START, back_edges, build_graph, terminal_agent, validate_graph

Edges = list[tuple[str, str]]
"""An agent-level edge set: the edges a ``MAS`` is constructed from."""


@dataclass(frozen=True)
class TopologyBudget:
    """The bounds a searched topology has to stay inside.

    One object, consulted by the sampler, the mutation operators, the crossover
    and the search loop alike, so that "legal" means one thing. Ported from the
    seed's ``GraphSamplingConfig``, minus the fields that only its prompt
    pipeline used.

    Attributes:
        min_agents: Fewest agents a candidate may have.
        max_agents: Most agents a candidate may have.
        min_edges: Fewest agent-to-agent edges. ``None`` means ``agents - 1``,
            the least a connected chain needs.
        max_edges: Most agent-to-agent edges.
        max_cycles: How many non-trivial strongly connected components are
            allowed. Zero, the default, searches directed acyclic graphs only,
            which is also the seed's default.
    """

    min_agents: int = 2
    max_agents: int = 6
    min_edges: int | None = None
    max_edges: int = 12
    max_cycles: int = 0

    def __post_init__(self) -> None:
        """Reject a budget that no topology could satisfy.

        Raises:
            ValueError: If the bounds are inconsistent.
        """
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


def agent_edges(graph: nx.DiGraph[str]) -> Edges:
    """Return the edges between agents, dropping the sentinel wiring.

    Args:
        graph: A MAS topology.

    Returns:
        The agent-to-agent edges, sorted so that a candidate is reproducible.
    """
    return sorted(
        (source, target) for source, target in graph.edges if source not in SENTINELS and target not in SENTINELS
    )


def terminals(names: list[str], edges: Edges) -> list[str]:
    """Return the agents with no outgoing edge, which are the ones feeding END."""
    with_successors = {source for source, _ in edges}
    return [name for name in names if name not in with_successors]


def cycles(names: list[str], edges: Edges) -> int:
    """Count the non-trivial strongly connected components among the agents."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from(edges)
    return sum(1 for component in nx.strongly_connected_components(graph) if len(component) > 1)


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
    decision and the way out in different places.

    Args:
        names: The candidate's agent names.
        edges: Its agent-level edge set.
        entry: Preferred entry agent, normally inherited from the parent. Ignored
            if it is not among ``names`` or if the topology already has a source.

    Returns:
        The edge list, with explicit sentinel edges added where needed.
    """
    if not names:
        return list(edges)

    wiring = list(edges)
    if not sources(names, edges):
        wiring = [(START, entry if entry in names else sorted(names)[0]), *wiring]
    if not terminals(names, edges):
        wiring = [*wiring, (_exit_agent(names, wiring), END)]
    return wiring


def _exit_agent(names: list[str], wiring: Edges) -> str:
    """Return the agent that should be able to leave a fully cyclic topology.

    Args:
        names: The candidate's agent names.
        wiring: The edges so far, including the explicit ``START`` edge.

    Returns:
        The source of a cycle-closing edge, or the last agent by name if the
        classification finds none.
    """
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from([START, *names])
    graph.add_edges_from(wiring)
    closing = sorted(back_edges(graph))
    return closing[0][0] if closing else sorted(names)[-1]


def is_buildable(names: list[str], edges: Edges, entry: str | None = None) -> bool:
    """Check whether a ``MAS`` could be constructed from this candidate.

    Delegates to the core's own predicate rather than restating it. The seed kept
    a separate ``is_valid_source_terminal_graph``, which the executor then had to
    be trusted to agree with; asking the thing that will run the candidate is one
    fewer definition to keep in step.

    Args:
        names: The candidate's agent names.
        edges: Its agent-level edge set.
        entry: Preferred entry agent.

    Returns:
        Whether the topology is runnable.
    """
    try:
        graph = build_graph(list(names), mas_edges(names, edges, entry))
        validate_graph(graph)
        terminal_agent(graph)
    except ValueError:
        return False
    return True


def within_budget(names: list[str], edges: Edges, budget: TopologyBudget) -> bool:
    """Check a candidate against the budget's numeric bounds only.

    Args:
        names: The candidate's agent names.
        edges: Its agent-level edge set.
        budget: The bounds.

    Returns:
        Whether the candidate is inside the budget.
    """
    if not budget.min_agents <= len(names) <= budget.max_agents:
        return False
    if not budget.floor(len(names)) <= len(edges) <= budget.max_edges:
        return False
    # Zero terminals is legal: a fully cyclic topology has its exit named by
    # `mas_edges`. Two or more is not, because the system's output would be
    # ambiguous and a MAS refuses to build one.
    if len(terminals(names, edges)) > 1:
        return False
    return cycles(names, edges) <= budget.max_cycles


def is_valid(names: list[str], edges: Edges, budget: TopologyBudget, entry: str | None = None) -> bool:
    """Check a candidate against both the budget and the core's runnability rules.

    The single predicate the sampler, the operators and the search loop all use,
    so that "legal topology" means one thing across the package.

    Args:
        names: The candidate's agent names.
        edges: Its agent-level edge set.
        budget: The bounds.
        entry: Preferred entry agent.

    Returns:
        Whether the candidate is legal.
    """
    return within_budget(names, edges, budget) and is_buildable(names, edges, entry)


# ------------------------------------------------------------------ operators


def add_edge(names: list[str], edges: Edges, rng: random.Random) -> Edges:
    """Add one edge between two agents that do not already have one."""
    present = set(edges)
    options = [
        (source, target) for source in names for target in names if source != target and (source, target) not in present
    ]
    if not options:
        return list(edges)
    return sorted([*edges, rng.choice(options)])


def remove_edge(names: list[str], edges: Edges, rng: random.Random) -> Edges:  # noqa: ARG001
    """Remove one edge."""
    if not edges:
        return list(edges)
    dropped = rng.choice(edges)
    return [edge for edge in edges if edge != dropped]


def rewire_edge(names: list[str], edges: Edges, rng: random.Random) -> Edges:
    """Move one edge somewhere else, keeping the edge count unchanged."""
    return add_edge(names, remove_edge(names, edges, rng), rng)


def add_agent(names: list[str], edges: Edges, rng: random.Random) -> tuple[list[str], Edges]:
    """Insert a new agent downstream of an existing one.

    The new agent is wired after a randomly chosen predecessor and before the
    current terminal, which is the seed's shape: a node arrives with somewhere to
    read from and somewhere to write to, so the candidate stays connected without
    a repair pass.
    """
    name = next_name(names)
    terminal = terminals(names, edges)
    if not terminal:
        return list(names), list(edges)
    predecessor = rng.choice([candidate for candidate in names if candidate != terminal[0]] or names)
    return [*names, name], sorted({*edges, (predecessor, name), (name, terminal[0])})


def remove_agent(names: list[str], edges: Edges, rng: random.Random) -> tuple[list[str], Edges]:
    """Drop an agent, bridging its predecessors to its successors.

    Bridging rather than simply deleting is what keeps the rest of the graph
    connected, and it is the seed's behaviour.
    """
    terminal = terminals(names, edges)
    removable = [name for name in names if name not in terminal]
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


def mutate(
    names: list[str],
    edges: Edges,
    budget: TopologyBudget,
    rng: random.Random,
    operations: int = 1,
    entry: str | None = None,
) -> tuple[list[str], Edges]:
    """Apply a sequence of operators, retrying the sequence if it lands outside the budget.

    Retries the whole sequence rather than the last operator, and falls back to
    the parent unchanged when none of the attempts works, which is the seed's
    semantics. A tightly bounded search space produces a lot of invalid
    intermediates, and retrying only the last step biases the walk toward
    whichever operator happens to be cheapest to satisfy.

    Args:
        names: The parent's agent names.
        edges: The parent's agent-level edge set.
        budget: The bounds a candidate must stay inside.
        rng: Source of randomness.
        operations: How many operators to apply in one mutation step.
        entry: The parent's entry agent, preferred for the child when the child's
            entry is inside a cycle and has to be named.

    Returns:
        The mutated names and edges, or the parent's unchanged.
    """
    for _ in range(3):
        current_names, current_edges = list(names), list(edges)
        for _ in range(max(1, operations)):
            current_names, current_edges = _apply_one(current_names, current_edges, budget, rng)
        if is_valid(current_names, current_edges, budget, entry):
            return current_names, current_edges
    return list(names), list(edges)


def _apply_one(
    names: list[str],
    edges: Edges,
    budget: TopologyBudget,
    rng: random.Random,
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
        return list(names), edge_ops[choice](names, edges, rng)
    return node_ops[choice - len(edge_ops)](names, edges, rng)


# ----------------------------------------------------------------- crossover

KEEP_FROM_FIRST = 0.75
KEEP_FROM_SECOND = 0.6
"""Inheritance rates from the seed. The first parent contributes more."""

MAX_ATTEMPTS = 10
"""Retries before giving up on recombination.

The seed's comment on this is worth keeping: falling back to a fresh random graph
*within* an attempt turns most crossovers into reseeds in a tightly constrained
space, which defeats the point of having two parents. So the whole recombination
is retried, and only a total failure falls back.
"""


def crossover(
    first: tuple[list[str], Edges],
    second: tuple[list[str], Edges],
    budget: TopologyBudget,
    rng: random.Random,
    entry: str | None = None,
) -> tuple[list[str], Edges]:
    """Recombine two topologies into one that inherits from both.

    Args:
        first: One parent's names and edges.
        second: The other parent's.
        budget: The bounds the child must stay inside.
        rng: Source of randomness.
        entry: Preferred entry agent for the child.

    Returns:
        The child's names and edges, falling back to the first parent when no
        attempt produced something legal.
    """
    for _ in range(MAX_ATTEMPTS):
        names = list(rng.choice([first[0], second[0]]))
        known = set(names)
        inherited = {
            edge for edge in first[1] if edge[0] in known and edge[1] in known and rng.random() < KEEP_FROM_FIRST
        } | {edge for edge in second[1] if edge[0] in known and edge[1] in known and rng.random() < KEEP_FROM_SECOND}
        child = repair(names, sorted(inherited), budget, rng)
        if is_valid(names, child, budget, entry):
            return names, child
    return list(first[0]), list(first[1])


def repair(names: list[str], edges: Edges, budget: TopologyBudget, rng: random.Random) -> Edges:
    """Make an inherited edge set into something a MAS can be built from.

    Recombination cuts edges without regard for what the result has to satisfy,
    so the child arrives with self-loops, several agents reaching ``END``, agents
    that reach it through nothing, and an edge count anywhere at all. This walks
    it back inside the rules in that order, because each step can undo the last
    if they are done the other way round.

    Args:
        names: The child's agent names.
        edges: The inherited edges.
        budget: The bounds.
        rng: Source of randomness.

    Returns:
        A repaired edge set.
    """
    current = [(source, target) for source, target in edges if source != target]
    terminal = _pick_terminal(names, current, rng)
    current = [(source, target) for source, target in current if source != terminal]

    # Every other agent needs a way onward, and a way that ends at the terminal.
    for name in names:
        if name != terminal and not any(source == name for source, _ in current):
            current.append((name, terminal))
    current = _connect_to_terminal(names, current, terminal)

    if budget.max_cycles == 0:
        current = _break_cycles(names, current, terminal)

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


def _pick_terminal(names: list[str], edges: Edges, rng: random.Random) -> str:
    """Choose the one agent that will feed END, preferring an inherited terminal."""
    inherited = terminals(names, edges)
    return inherited[0] if inherited else rng.choice(sorted(names))


def _connect_to_terminal(names: list[str], edges: Edges, terminal: str) -> Edges:
    """Add edges until every agent has a path to the terminal."""
    current = list(edges)
    while True:
        graph: nx.DiGraph[str] = nx.DiGraph()
        graph.add_nodes_from(names)
        graph.add_edges_from(current)
        stranded = [name for name in names if name != terminal and not nx.has_path(graph, name, terminal)]
        if not stranded:
            return current
        current.append((stranded[0], terminal))


def _break_cycles(names: list[str], edges: Edges, terminal: str) -> Edges:
    """Drop edges until the topology is acyclic, keeping the terminal reachable."""
    current = list(edges)
    while True:
        graph: nx.DiGraph[str] = nx.DiGraph()
        graph.add_nodes_from(names)
        graph.add_edges_from(current)
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            return _connect_to_terminal(names, current, terminal)
        source, target = cycle[-1][0], cycle[-1][1]
        current = [edge for edge in current if edge != (source, target)]
