"""A deliberately trivial optimizer, kept because it is trivial.

It exists to exercise the whole loop end to end, propose through evaluate through
select, with no algorithm worth arguing about in the way. When a real method
misbehaves, this is the control: if the reference optimizer improves a score and
yours does not, the loop is fine and the method is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xmas.optimization.optimizer import MASOptimizer
from xmas.optimization.updaters import movable_agents, weakest

if TYPE_CHECKING:
    from xmas.mas import MAS
    from xmas.optimization.attribution import Attributor
    from xmas.tasks.base import MASTask

__all__ = [
    "PREFIXES",
    "PrefixSearchOptimizer",
]


MIN_POPULATION = 2
"""The incumbent plus one candidate: the smallest population that compares anything."""


# Fixed phrases tried against an agent's prompt. Not a method, a fixture.
#
# Fifteen of them, so a population of sixteen is available without repeating one.
# Which matters more than it looks: a probing task's estimator is
# population-relative, and its importance weights are bounded by the population
# size, so the width of the candidate set is a property of the measurement and not
# just of the search.
PREFIXES: tuple[str, ...] = (
    "Think step by step.",
    "Be concise and exact.",
    "Show your reasoning, then the answer.",
    "Work through the problem carefully before answering.",
    "State the final answer on its own line.",
    "Check your arithmetic before you commit to an answer.",
    "Explain each step in one short sentence.",
    "Solve it methodically, then give the result.",
    "Break the problem into parts and handle each one.",
    "Reason it out fully, then state the answer alone.",
    "Do the calculation carefully and report the number.",
    "First restate what is being asked, then solve it.",
    "Take your time and show the intermediate values.",
    "Answer only after verifying each step.",
    "Lay out the working, then the final number.",
)


class PrefixSearchOptimizer(MASOptimizer):
    """Prepends one of a fixed set of phrases to one agent's prompt, keeps the best.

    Every choice here is intentionally dumb: the candidate set is a constant, the
    agent is picked by round-robin unless an attributor says otherwise, and
    selection is argmax. What it is not dumb about is the plumbing, which is the
    machinery every later optimizer reuses.

    Attributes:
        prefixes: The phrases to try.
    """

    def __init__(
        self,
        mas: MAS,
        task: MASTask,
        prefixes: tuple[str, ...] = PREFIXES,
        attributor: Attributor | None = None,
        population: int | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            mas: The starting system.
            task: The task to optimize against.
            prefixes: Phrases to try.
            attributor: Optional credit assignment; when given, the agent with
                the least credit is the one varied.
            population: Candidates to propose per step, incumbent included.
                ``None`` uses every prefix. A knob rather than a constant because
                a probing task scores a population against itself: its weights are
                bounded by the population size, so ``population`` sets the ceiling
                on how far apart two candidates can be scored, and an experiment
                has to be able to state and vary it.

        Raises:
            ValueError: If ``population`` is below two, which leaves the incumbent
                nothing to be compared against.
        """
        if population is not None and population < MIN_POPULATION:
            raise ValueError(f"population must be at least 2 (incumbent plus one candidate), got {population}.")
        super().__init__(mas, task, attributor=attributor)
        self.prefixes = prefixes if population is None else prefixes[: population - 1]
        self.step = 0

    def target_agent(self) -> str:
        """Return the least-credited agent not detached, else the next one round-robin: a deterministic sweep."""
        names = movable_agents(self.mas)
        return weakest(names, self.credit) or names[self.step % len(names)]

    def propose(self) -> list[MAS]:
        """Return one candidate per prefix, plus the incumbent, so a step that finds nothing better keeps it."""
        agent = self.target_agent()
        self.step += 1
        base = self.mas.agents[agent].prompt
        return [self.mas, *(self.mas.copy(prompts={agent: f"{prefix} {base}".strip()}) for prefix in self.prefixes)]
