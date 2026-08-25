# Search loop adapted from tritthart/mas-optimization,
# mas_lib/optimization/architecture_evolution/architecture_evolution.py
"""Evolutionary topology search: the loop that drives the graph updater."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from xmas.mas import MAS
from xmas.optimization.base import MASOptimizer
from xmas.optimization.graph_updater import GraphUpdater

if TYPE_CHECKING:  # pragma: no cover
    from xmas.optimization.attribution import Attributor
    from xmas.tasks.base import MASTask


class EvolutionaryTopologyOptimizer(MASOptimizer):
    """Searches topologies by mutating and recombining the ones that scored well.

    The other half of P4's decomposition. Where
    :class:`~xmas.optimization.cantante.CantanteOptimizer` holds the graph fixed
    and improves the prompts, this holds the prompts fixed and improves the graph,
    through the same three-step loop and against the same task interface. Running
    both is a matter of alternating them, not of a third framework.

    Recombination needs a population, so one is kept: the systems that survived
    selection, newest last. A step mutates the incumbent and crosses pairs drawn
    from that population, which is the seed's shape.

    Attributes:
        mas: The current best system.
        updater: Produces the topology candidates.
        population: Recent survivors, the parents recombination draws from.
        history: Aggregate score of the incumbent after each step.
    """

    def __init__(
        self,
        mas: MAS,
        task: MASTask,
        updater: GraphUpdater,
        *,
        mutations: int = 3,
        recombinations: int = 1,
        population_size: int = 6,
        attributor: Attributor | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize the optimizer.

        Args:
            mas: The starting system.
            task: The task to optimize against.
            updater: The topology updater.
            mutations: Mutated candidates proposed per step.
            recombinations: Recombined candidates proposed per step. Ignored
                until the population holds two systems to cross.
            population_size: How many survivors to keep as potential parents.
            attributor: Optional credit assignment, which steers where the
                updater changes the graph.
            seed: Seed for the parent draws.
        """
        super().__init__(mas=mas, task=task, attributor=attributor)
        self.updater = updater
        self.mutations = mutations
        self.recombinations = recombinations
        self.population_size = population_size
        self.population: list[MAS] = [mas]
        self.rng = random.Random(seed)
        self.accepted: list[str] = []

    def propose(self) -> list[MAS]:
        """Mutate the incumbent and recombine the survivors.

        Returns:
            The incumbent first, then the candidates. The incumbent is included
            so that selection compares against a score measured this step, on
            whatever data the task handed out, rather than against a remembered
            one.
        """
        candidates = self.updater.propose(self.mas, self.credit, self.mutations)
        if self.recombinations > 0 and len(self.population) > 1:
            for _ in range(self.recombinations):
                first, second = self.rng.sample(self.population, 2)
                candidates.extend(self.updater.recombine(first, second, 1))
        return [self.mas, *candidates]

    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Keep the best candidate and remember the rest as future parents.

        Args:
            candidates: The evaluated systems, incumbent first.
            scores: Their scores, aligned to ``candidates``.

        Returns:
            The chosen system, which is the incumbent unless something beat it.
        """
        best = max(range(len(candidates)), key=lambda index: scores[index])
        winner = candidates[best] if scores[best] > scores[0] else candidates[0]

        ranked = sorted(range(len(candidates)), key=lambda index: -scores[index])
        self.population = [candidates[index] for index in ranked[: self.population_size]]
        self.accepted.append(self.describe(winner) if winner is not candidates[0] else "no change")
        return winner

    def describe(self, mas: MAS) -> str:
        """Render a system's topology in one line, for the run's readout.

        Args:
            mas: The system to describe.

        Returns:
            Its agent-level edges, or its lone agent when it has no edges.
        """
        edges = self.updater.edges_of(mas)
        if not edges:
            return f"{len(mas.agents)} agents, no edges"
        return " ".join(f"{source}->{target}" for source, target in edges)
