"""A deliberately trivial optimizer, kept because it is trivial.

It exists to exercise the whole loop end to end, propose through evaluate through
select, with no algorithm worth arguing about in the way. When a real method
misbehaves, this is the control: if the reference optimizer improves a score and
yours does not, the loop is fine and the method is not.
"""

from __future__ import annotations

from xmas.mas import MAS
from xmas.optimization.base import MASOptimizer
from xmas.tasks.base import MASTask

PREFIXES: tuple[str, ...] = (
    "Think step by step.",
    "Be concise and exact.",
    "Show your reasoning, then the answer.",
)
"""Fixed prefixes tried against each agent's prompt. Not a method, a fixture."""


class PrefixSearchOptimizer(MASOptimizer):
    """Prepends one of a fixed set of phrases to one agent's prompt and keeps what scores best.

    Every part of this is intentionally dumb: the candidate set is a constant, the
    agent is chosen by round-robin, and selection is argmax. What it is not dumb
    about is the plumbing, which is the real machinery every later optimizer uses.
    """

    def __init__(self, mas: MAS, task: MASTask, prefixes: tuple[str, ...] = PREFIXES) -> None:
        """Initialize the optimizer.

        Args:
            mas: The starting system.
            task: The task to optimize against.
            prefixes: Phrases to try.
        """
        raise NotImplementedError

    def propose(self) -> list[MAS]:
        """Return one candidate per prefix, all varying the same agent.

        Returns:
            The candidates, plus the incumbent so that selection can keep it.
        """
        raise NotImplementedError

    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Return the highest-scoring candidate.

        Args:
            candidates: The evaluated systems.
            scores: Their scores.

        Returns:
            The best one; ties go to the earliest, which is the incumbent.
        """
        raise NotImplementedError
