"""The MAS optimization loop."""

from __future__ import annotations

from abc import ABC, abstractmethod

from xmas.mas import MAS
from xmas.tasks.base import MASTask


class MASOptimizer(ABC):
    """Improves a MAS against a task by proposing, evaluating and selecting.

    The loop is three steps and the base class owns all three, so a subclass
    decides *what* to propose and *how* to select, never how to evaluate.
    Evaluation goes through the task's narrow interface and nowhere else, which
    is what keeps every optimizer compatible with every task, including tasks
    that do not generate at all.

    Attributes:
        mas: The current best system.
        task: The task being optimized against.
        history: Aggregate score of the incumbent after each step.
    """

    mas: MAS
    task: MASTask
    history: list[float]

    def __init__(self, mas: MAS, task: MASTask) -> None:
        """Initialize the optimizer.

        Args:
            mas: The starting system.
            task: The task to optimize against.
        """
        raise NotImplementedError

    def optimize(self, n_steps: int) -> MAS:
        """Run the loop and return the best system found.

        Args:
            n_steps: Number of propose-evaluate-select rounds.

        Returns:
            The best system found, which may be the one passed in.
        """
        raise NotImplementedError

    @abstractmethod
    def propose(self) -> list[MAS]:
        """Return candidate systems to evaluate this step.

        Returns:
            Candidates, derived from :attr:`mas` via :meth:`xmas.MAS.copy`.
        """

    def evaluate(self, candidates: list[MAS]) -> list[float]:
        """Score candidates through the task.

        Args:
            candidates: Systems to score.

        Returns:
            One aggregate score per candidate, in the order given.
        """
        raise NotImplementedError

    @abstractmethod
    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Choose the system to carry into the next step.

        Args:
            candidates: The evaluated systems, including the incumbent.
            scores: Their scores, aligned to ``candidates``.

        Returns:
            The chosen system.
        """
