"""The topology updater: proposals that change the graph rather than the prompts."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from xmas.agent import Agent
from xmas.edges import Edge, EdgeKind
from xmas.graph import SENTINELS, START, ancestors, build_graph
from xmas.mas import MAS
from xmas.optimization.synthesis import LLMPromptSynthesizer, MechanicalSynthesizer, Rewrite
from xmas.optimization.topology import Edges, Fixed, TopologyBudget, crossover, mas_edges, mutate, terminals
from xmas.optimization.updaters import weakest
from xmas.spec import VERSION_2

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "NEW_AGENT_PROMPT",
    "GraphUpdater",
]


NEW_AGENT_PROMPT = "You are {name}. Do your part of the task and pass on your result."
"""Seed text for an agent a mutation invented, before its prompt is written."""


class GraphUpdater:
    """Proposes candidates that differ from a system in topology.

    Three things happen for every candidate, and skipping any one of them makes the search
    look like it is working while it is not:

    1. the edge set is mutated (:mod:`xmas.optimization.topology`);
    2. every agent's declared inputs are rewired to its new predecessors, because
       an agent that keeps its old inputs never reads the new edge;
    3. the prompts of the agents whose inputs changed are rewritten, because a
       prompt that never names a variable renders without it.

    A version 2 system mutates around what it holds still (see
    :class:`~xmas.optimization.topology.Fixed`): steps and boxes stay steps and
    boxes, human nodes, frozen agents and conditional edges stay as they are,
    and a candidate its own check refuses is dropped rather than returned. A
    candidate keeps the parent's asker and run store.

    Attributes:
        budget: The bounds every candidate stays inside.
        synthesizer: Writes prompts for agents whose inputs changed.
        operations: Operators applied per mutation step.
        rng: Source of randomness, seeded so a search reproduces.
    """

    def __init__(
        self,
        synthesizer: LLMPromptSynthesizer | MechanicalSynthesizer,
        budget: TopologyBudget | None = None,
        *,
        operations: int = 1,
        seed: int = 42,
    ) -> None:
        """Initialize the updater.

        Args:
            synthesizer: Writes prompts for rewired agents. Required rather than
                defaulted, because the two available strategies differ in whether
                a run is reproducible and in whether it costs model calls, and
                neither is the obvious choice to make silently.
            budget: The bounds. Defaults to a small acyclic budget.
            operations: Operators applied per mutation step.
            seed: Seed for the mutation and crossover draws.
        """
        self.budget = budget if budget is not None else TopologyBudget()
        self.synthesizer = synthesizer
        self.operations = operations
        self.rng = random.Random(seed)

    # ------------------------------------------------------------- proposing

    def propose(self, mas: MAS, credit: dict[str, list[float]], n: int) -> list[MAS]:
        """Propose up to ``n`` runnable candidates differing from ``mas`` in topology.

        Credit steers where the change lands. Candidates are drawn generously and
        then ranked by whether they touched the agent that earned the least,
        because an edge added somewhere the system was already doing fine is a
        wasted evaluation.
        """
        if n <= 0:
            return []
        names, edges = self.names_of(mas), self.edges_of(mas)
        entry, fixed = self.entry_of(mas), self.fixed_of(mas)

        drawn: dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], tuple[list[str], Edges]] = {}
        for _ in range(n * 6):
            candidate = mutate(names, edges, self.budget, self.rng, self.operations, entry, fixed)
            if candidate[0] == names and candidate[1] == edges:
                continue  # the mutation fell back to the parent
            # Deduplicated, because the operators repeat themselves in a small
            # space and evaluating one topology twice buys nothing.
            drawn.setdefault((tuple(candidate[0]), tuple(candidate[1])), candidate)

        weakest = self.weakest_agent(mas, credit)
        ranked = sorted(drawn.values(), key=lambda candidate: not self.touches(weakest, edges, candidate[1]))
        return self.build(mas, ranked, n)

    def recombine(self, first: MAS, second: MAS, n: int) -> list[MAS]:
        """Propose up to ``n`` runnable candidates inheriting topology from two parents.

        A version 2 child keeps what ``first`` holds still.
        """
        parents = ((self.names_of(first), self.edges_of(first)), (self.names_of(second), self.edges_of(second)))
        entry, fixed = self.entry_of(first), self.fixed_of(first)
        children = [crossover(parents[0], parents[1], self.budget, self.rng, entry, fixed) for _ in range(n)]
        return self.build(first, children, n)

    # -------------------------------------------------------------- building

    def build(self, mas: MAS, candidates: Iterable[tuple[list[str], Edges]], n: int) -> list[MAS]:
        """Build up to ``n`` candidates, best first, dropping those a version 2 check refuses.

        The operators know a version 2 system's budget and what it holds still,
        not its reads, types and joins, so :func:`~xmas.check.check_system` has
        the last word. A version 1 candidate the operators passed always builds,
        so a refusal there is a defect and is raised.
        """
        built: list[MAS] = []
        for names, edges in candidates:
            if len(built) == n:
                break
            try:
                built.append(self.rebuild(mas, names, edges))
            except ValueError:
                if mas.version != VERSION_2:
                    raise
        return built

    def rebuild(self, mas: MAS, names: list[str], edges: Edges) -> MAS:
        """Build a runnable system from a mutated topology; ``ValueError`` if a version 2 candidate does not check.

        Steps, boxes and human nodes come over as they are, and so do the agents whose
        predecessors did not change, frozen ones among them; a version 2 system
        also keeps its conditional edges and its named ``START``/``END`` edges,
        in the parent's order.
        """
        wiring: Sequence[tuple[str, str] | Edge]
        if mas.version == VERSION_2:
            kept = self.wiring_of(mas, names, edges)
            inputs, wiring = self.reads_for(mas, names, edges, kept), kept
        else:
            wiring = mas_edges(names, edges, self.entry_of(mas))
            inputs = self.inputs_for(names, wiring, {name: self.output_of(mas, name) for name in names})

        agents: dict[str, Agent] = {}
        pending: list[Rewrite] = []
        for name in inputs:
            parent = mas.agents.get(name)
            if parent is not None and sorted(parent.inputs) == sorted(inputs[name]):
                agents[name] = parent.copy()  # its reads stay as they are, renamed ones included
                continue
            if parent is None:
                parent = Agent(name, NEW_AGENT_PROMPT.format(name=name), output=self.output_of(mas, name))
            agents[name] = agent = parent.copy(inputs=inputs[name])
            pending.append(Rewrite(agent=name, prompt=agent.prompt, output=agent.output, inputs=inputs[name]))

        for request, written in zip(pending, self.synthesizer.rewrite(pending), strict=True):
            agents[request.agent] = agents[request.agent].copy(prompt=written)

        return MAS(
            agents=list(agents.values()),
            edges=wiring,
            llm=mas.llm,
            llms=mas.llms,
            max_steps=mas.max_steps,
            default_model=mas.default_model,
            tools=mas.tools,
            max_tool_rounds=mas.max_tool_rounds,
            max_visits=mas.max_visits,
            budget=mas.budget,
            asker=mas.asker,
            store=mas.store,
            **mas.v2_kwargs(),
        )

    def output_of(self, mas: MAS, name: str) -> str:
        """Return the variable an agent of a candidate writes: the parent's, or ``{name}_out`` for an invented one."""
        parent = mas.agents.get(name)
        return parent.output if parent is not None else f"{name}_out"

    def inputs_for(self, names: list[str], wiring: Edges, outputs: dict[str, str]) -> dict[str, list[str]]:
        """Work out what each agent of a version 1 candidate reads, from who now feeds it.

        The topology decides, and the prompt is written to match: deriving the
        inputs from a prompt's placeholders would let a prompt rewrite silently
        rewire the dataflow. ``outputs`` is each agent's variable, since a
        surviving agent keeps the one it had. ``wiring`` includes the sentinels.
        """
        explicit = {target for _, target in wiring}
        resolved: dict[str, list[str]] = {}
        for name in names:
            upstream = sorted({outputs[source] for source, target in wiring if target == name and source in outputs})
            from_start = any(source == START and target == name for source, target in wiring)
            if from_start or name not in explicit:
                upstream = ["query", *upstream]
            resolved[name] = upstream or ["query"]
        return resolved

    def wiring_of(self, mas: MAS, names: list[str], edges: Edges) -> list[Edge]:
        """Return a version 2 candidate's edges as the ``MAS`` takes them.

        The parent's ``Edge`` objects are reused wherever their pair survives,
        so conditional edges keep their kind, predicate and declared order (a
        ``when`` group is decided in that order); its edges from ``START`` or to
        ``END`` stay while their node does; new pairs are appended as plain edges.
        """
        pairs, present = set(edges), {*names, *SENTINELS}
        kept = [
            edge
            for edge in mas.edges
            if (edge.source, edge.target) in pairs
            or ({edge.source, edge.target}.intersection(SENTINELS) and {edge.source, edge.target} <= present)
        ]
        known = {(edge.source, edge.target) for edge in kept}
        return [*kept, *(Edge(source, target) for source, target in edges if (source, target) not in known)]

    def reads_for(self, mas: MAS, names: list[str], edges: Edges, wiring: list[Edge]) -> dict[str, list[str]]:
        """Work out what each agent of a version 2 candidate reads, invented ones included, in node order.

        An agent whose predecessors did not change keeps its reads, which covers
        every frozen agent and every endpoint of a conditional edge. Any other
        agent keeps the reads still satisfiable (an input, or written by a node
        that now runs before it) and gains every value its new predecessors
        write; one left with nothing reads the system's inputs. Steps, boxes and
        human nodes are not rewired.
        """
        code = {*mas.steps, *mas.boxes, *mas.humans}
        writes: dict[str, set[str]] = {name: {self.output_of(mas, name)} for name in names if name not in code}
        agents = list(writes)
        for value, writers in mas.checked().writers.items():
            for node in code.intersection(writers):
                writes.setdefault(node, set()).add(value)

        graph = build_graph(names, [(edge.source, edge.target) for edge in wiring])
        before = self.edges_of(mas)
        resolved: dict[str, list[str]] = {}
        for name in agents:
            parent = mas.agents.get(name)
            feeding = sorted(source for source, target in edges if target == name)
            if parent is not None and feeding == sorted(source for source, target in before if target == name):
                resolved[name] = list(parent.inputs)
                continue
            upstream = ancestors(graph, name)
            available = set(mas.inputs).union(*(writes[node] for node in upstream if node in writes))
            kept = [value for value in (parent.inputs if parent is not None else []) if value in available]
            fresh = sorted(set().union(*(writes.get(source, set()) for source in feeding)) - set(kept))
            resolved[name] = [*kept, *fresh] or list(mas.inputs)
        return resolved

    # --------------------------------------------------------------- helpers

    def names_of(self, mas: MAS) -> list[str]:
        """Return a system's node names: agents, then steps, boxes and human nodes."""
        return [*mas.agents, *mas.steps, *mas.boxes, *mas.humans]

    def edges_of(self, mas: MAS) -> Edges:
        """Return a system's node-level edges as pairs, conditional ones included, without the sentinel wiring."""
        pairs = {(edge.source, edge.target) for edge in mas.edges}
        return sorted((source, target) for source, target in pairs if not {source, target}.intersection(SENTINELS))

    def fixed_of(self, mas: MAS) -> Fixed | None:
        """Return what no mutation of a version 2 system may change, or ``None`` for version 1.

        Frozen nodes, human nodes, the ends of conditional edges and every
        member of a ``for each`` chain are held, so no mutation breaks a chain
        or moves a person.
        """
        if mas.version != VERSION_2:
            return None
        conditional = [edge for edge in mas.edges if edge.kind is not EdgeKind.PLAIN]
        held = set(mas.frozen).union(mas.each, mas.humans, *({edge.source, edge.target} for edge in conditional))
        return Fixed(
            nodes=frozenset(held.difference(SENTINELS)),
            code=frozenset([*mas.steps, *mas.boxes, *mas.humans]),
            terminals=frozenset(terminals(self.names_of(mas), self.edges_of(mas))),
            conditional=frozenset((edge.source, edge.target) for edge in conditional),
        )

    def entry_of(self, mas: MAS) -> str | None:
        """Return the agent a run starts at, used when a candidate must name its entry."""
        fed = sorted(mas.graph.successors(START))
        return fed[0] if fed else None

    def weakest_agent(self, mas: MAS, credit: dict[str, list[float]]) -> str | None:
        """Return the least-credited agent a candidate may rewire, or ``None`` when there is none to go on.

        A frozen agent, or one at a conditional edge (:meth:`fixed_of`), is never the target.
        """
        fixed = self.fixed_of(mas)
        held = frozenset() if fixed is None else fixed.nodes
        return weakest([name for name in mas.agents if name not in held], credit)

    def touches(self, agent: str | None, before: Edges, after: Edges) -> bool:
        """Whether a candidate changed the edges around ``agent``; always, with no agent to aim at."""
        if agent is None:
            return True
        changed = set(before) ^ set(after)
        return any(agent in edge for edge in changed) or agent not in {name for edge in after for name in edge}
