# Search loop adapted from tritthart/mas-optimization,
# mas_lib/optimization/architecture_evolution/architecture_evolution.py
"""Evolutionary topology search: the loop that drives the graph updater."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from xmas.optimization.optimizer import MASOptimizer

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS
    from xmas.optimization.attribution import Attributor
    from xmas.optimization.graph_updater import GraphUpdater
    from xmas.tasks.base import MASTask

__all__ = [
    "EvolutionaryTopologyOptimizer",
]


class EvolutionaryTopologyOptimizer(MASOptimizer):
    """Searches topologies by mutating and recombining the ones that scored well.

    Where :class:`~xmas.optimization.cantante.CantanteOptimizer` holds the
    graph fixed and improves the prompts, this holds the prompts fixed and
    improves the graph, through the same three-step loop and against the same
    task interface. Running both is a matter of alternating them.

    Recombination needs a population, so one is kept: the systems that survived
    selection, best first. A step mutates the incumbent and crosses pairs drawn
    from that population.

    Attributes:
        mas: The current best system.
        updater: Produces the topology candidates.
        population: Recent survivors, the parents recombination draws from.
        history: Aggregate score of the incumbent after each step.
    """

    def __init__(  # noqa: PLR0913  the public constructor, kept for version 1 callers
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

        The incumbent comes first, so selection compares against a score
        measured this step, on whatever data the task handed out, rather than
        against a remembered one.
        """
        candidates = self.updater.propose(self.mas, self.credit, self.mutations)
        if self.recombinations > 0 and len(self.population) > 1:
            for _ in range(self.recombinations):
                first, second = self.rng.sample(self.population, 2)
                candidates.extend(self.updater.recombine(first, second, 1))
        return [self.mas, *candidates]

    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Keep the best candidate, the incumbent unless something beat it, and remember the best as parents."""
        winner = super().select(candidates, scores)
        ranked = sorted(range(len(candidates)), key=lambda index: -scores[index])
        self.population = [candidates[index] for index in ranked[: self.population_size]]
        self.accepted.append(self.describe(winner) if winner is not candidates[0] else "no change")
        return winner

    def describe(self, mas: MAS) -> str:
        """Render a system's topology in one line, its agent-level edges, for the run's readout."""
        edges = self.updater.edges_of(mas)
        if not edges:
            return f"{len(mas.agents)} agents, no edges"
        return " ".join(f"{source}->{target}" for source, target in edges)
