"""The topology updater: proposals that change the graph rather than the prompts."""

from __future__ import annotations

import math
import random

from xmas.agent import Agent
from xmas.graph import START
from xmas.mas import MAS
from xmas.optimization.synthesis import LLMPromptSynthesizer, MechanicalSynthesizer, Rewrite
from xmas.optimization.topology import Edges, TopologyBudget, crossover, mas_edges, mutate

NEW_AGENT_PROMPT = "You are {name}. Do your part of the task and pass on your result."
"""Seed text for an agent a mutation invented, before its prompt is written."""


class GraphUpdater:
    """Proposes candidates that differ from a system in topology.

    Three things happen for every candidate, and skipping any one of them makes the search
    look like it is working while it is not:

    1. the edge set is mutated, by the operators ported from the seed;
    2. every agent's declared inputs are rewired to its new predecessors, because
       an agent that keeps its old inputs never reads the new edge;
    3. the prompts of the agents whose inputs changed are rewritten, because a
       prompt that never names a variable renders without it.

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
        """Propose candidate systems differing from ``mas`` in topology.

        Credit steers where the change lands. Candidates are drawn generously and
        then ranked by whether they touched the agent that earned the least,
        because an edge added somewhere the system was already doing fine is a
        wasted evaluation.

        Args:
            mas: The current system.
            credit: Per-agent credit from an attributor, possibly empty.
            n: How many candidates to return.

        Returns:
            Up to ``n`` candidates, every one of them runnable.
        """
        if n <= 0:
            return []
        names, edges = list(mas.agents), self.edges_of(mas)
        entry = self.entry_of(mas)

        drawn: dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], tuple[list[str], Edges]] = {}
        for _ in range(n * 6):
            candidate = mutate(names, edges, self.budget, self.rng, self.operations, entry)
            if candidate[0] == names and candidate[1] == edges:
                continue  # the mutation fell back to the parent
            # Deduplicated, because the operators repeat themselves in a small
            # space and evaluating one topology twice buys nothing.
            drawn.setdefault((tuple(candidate[0]), tuple(candidate[1])), candidate)

        weakest = self.weakest_agent(mas, credit)
        ranked = sorted(drawn.values(), key=lambda candidate: not self.touches(weakest, edges, candidate[1]))
        return [self.rebuild(mas, candidate[0], candidate[1]) for candidate in ranked[:n]]

    def recombine(self, first: MAS, second: MAS, n: int) -> list[MAS]:
        """Propose candidates that inherit topology from two parents.

        Args:
            first: One parent.
            second: The other.
            n: How many candidates to return.

        Returns:
            Up to ``n`` candidates, every one of them runnable.
        """
        parents = ((list(first.agents), self.edges_of(first)), (list(second.agents), self.edges_of(second)))
        entry = self.entry_of(first)
        children = [crossover(parents[0], parents[1], self.budget, self.rng, entry) for _ in range(n)]
        return [self.rebuild(first, names, edges) for names, edges in children]

    # -------------------------------------------------------------- building

    def rebuild(self, mas: MAS, names: list[str], edges: Edges) -> MAS:
        """Build a runnable system from a mutated topology.

        Args:
            mas: The parent, whose agents and models are inherited where they survive.
            names: The candidate's agent names.
            edges: Its agent-level edge set.

        Returns:
            The candidate as a system.
        """
        wiring = mas_edges(names, edges, self.entry_of(mas))
        outputs = {name: self.output_of(mas, name) for name in names}
        inputs = self.inputs_for(names, wiring, outputs)

        agents: dict[str, Agent] = {}
        pending: list[Rewrite] = []
        for name in names:
            parent = mas.agents.get(name)
            output = outputs[name]
            prompt = parent.prompt if parent is not None else NEW_AGENT_PROMPT.format(name=name)
            model = parent.model if parent is not None else None
            tools = parent.tools if parent is not None else None
            agents[name] = Agent(name, prompt, output=output, inputs=inputs[name], tools=tools, model=model)
            if parent is None or sorted(parent.inputs) != sorted(inputs[name]):
                pending.append(Rewrite(agent=name, prompt=prompt, output=output, inputs=inputs[name]))

        for request, written in zip(pending, self.synthesizer.rewrite(pending), strict=True):
            agents[request.agent] = agents[request.agent].copy(prompt=written)

        return MAS(
            agents=[agents[name] for name in names],
            edges=wiring,
            llm=mas.llm,
            llms=mas.llms,
            max_steps=mas.max_steps,
            default_model=mas.default_model,
            tools=mas.tools,
            max_tool_rounds=mas.max_tool_rounds,
        )

    def output_of(self, mas: MAS, name: str) -> str:
        """Return the variable one agent of a candidate writes.

        An agent that survived the mutation keeps writing whatever it already
        wrote; only an agent the mutation invented needs a name, and it gets the
        conventional one.

        Args:
            mas: The parent, whose agents supply the surviving names.
            name: The agent to look up.

        Returns:
            The output variable name.
        """
        parent = mas.agents.get(name)
        return parent.output if parent is not None else f"{name}_out"

    def inputs_for(self, names: list[str], wiring: Edges, outputs: dict[str, str]) -> dict[str, list[str]]:
        """Work out what each agent reads, from who now feeds it.

        This is the inversion the recon asked for. The seed derived an agent's
        inputs from the placeholders in its prompt, which coupled prompt
        optimization to topology: rewriting a prompt could silently rewire the
        dataflow. Here the topology decides, and the prompt is written to match.

        Args:
            names: The candidate's agent names.
            wiring: The full edge list, sentinels included.
            outputs: The variable each agent writes, keyed by agent name. Passed
                in rather than derived from the name, because an agent that
                survives a mutation keeps the output variable it already had:
                assuming ``f"{name}_out"`` wires every rewired agent to a variable
                nothing writes, and the new edge then changes the schedule and
                nothing else.

        Returns:
            Input variable names per agent, in a stable order.
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

    # --------------------------------------------------------------- helpers

    def edges_of(self, mas: MAS) -> Edges:
        """Return a system's agent-level edges, without the sentinel wiring."""
        return sorted(
            (source, target) for source, target in mas.graph.edges if source in mas.agents and target in mas.agents
        )

    def entry_of(self, mas: MAS) -> str | None:
        """Return the agent a run starts at, used when a candidate must name its entry."""
        fed = sorted(mas.graph.successors(START))
        return fed[0] if fed else None

    def weakest_agent(self, mas: MAS, credit: dict[str, list[float]]) -> str | None:
        """Return the agent with the least credit, or None when there is none to go on."""
        scored = {
            name: [value for value in values if not math.isnan(value)]
            for name, values in credit.items()
            if name in mas.agents
        }
        usable = {name: values for name, values in scored.items() if values}
        if not usable:
            return None
        return min(usable, key=lambda name: sum(usable[name]) / len(usable[name]))

    def touches(self, agent: str | None, before: Edges, after: Edges) -> bool:
        """Whether a candidate changed the edges around one agent.

        Args:
            agent: The agent to check, or None when credit gave no target.
            before: The parent's edges.
            after: The candidate's edges.

        Returns:
            True when the agent's neighbourhood changed, or when there is no
            target to aim at, so that an unguided search is not held back.
        """
        if agent is None:
            return True
        changed = set(before) ^ set(after)
        return any(agent in edge for edge in changed) or agent not in {name for edge in after for name in edge}
