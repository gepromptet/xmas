"""The MAS optimization loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from promptolution.utils.prompt import Prompt

from xmas.mas import MAS
from xmas.optimization.attribution import Attributor
from xmas.tasks.base import MASTask
from xmas.trace import Trace

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.tasks.base_task import EvalStrategy


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
        attributor: Optional credit assignment, consulted before proposing.
        history: Aggregate score of the incumbent after each step.
    """

    def __init__(self, mas: MAS, task: MASTask, attributor: Attributor | None = None) -> None:
        """Initialize the optimizer.

        Args:
            mas: The starting system.
            task: The task to optimize against.
            attributor: Assigns per-component credit from the last evaluation. A
                subclass may use it to choose what to change. Optional, because
                an optimizer that changes everything at once needs no credit.
        """
        self.mas = mas
        self.task = task
        self.attributor = attributor
        self.history: list[float] = []
        self.credit: dict[str, list[float]] = {}
        self._rollouts: list[tuple[Trace, float]] = []

    def optimize(self, n_steps: int) -> MAS:
        """Run the loop and return the best system found.

        Args:
            n_steps: Number of propose-evaluate-select rounds.

        Returns:
            The best system found, which may be the one passed in.
        """
        for _ in range(n_steps):
            candidates = self.propose()
            if not candidates:
                break
            scores = self.evaluate(candidates)
            self.mas = self.select(candidates, scores)
            self.history.append(max(scores))
            self._assign_credit(candidates, scores)
        return self.mas

    def evaluate(self, candidates: list[MAS]) -> list[float]:
        """Score candidates through the task.

        Candidates that share a structure differ only in prompts, so they are
        evaluated as one population against a single predictor: that is exactly
        the batching promptolution's tasks already do, including their cache.
        Candidates with a different structure get their own pass, because a
        different graph cannot be expressed as another graph's prompt set.

        With an attributor present the evaluation is forced to the full dataset.
        Contrastive attribution compares candidates on the *same* query, so a
        strategy that hands different candidates different queries would leave it
        nothing to compare.

        Args:
            candidates: Systems to score.

        Returns:
            One aggregate score per candidate, in the order given.
        """
        scores = [0.0] * len(candidates)
        self._rollouts = []
        groups: dict[object, list[tuple[int, MAS]]] = defaultdict(list)
        for index, candidate in enumerate(candidates):
            groups[candidate.structure].append((index, candidate))

        strategy: EvalStrategy | None = "full" if self.attributor is not None else None
        for group in groups.values():
            predictor = group[0][1]
            encoded = [candidate.encode() for _, candidate in group]
            cached = set(self.task.eval_cache)
            result = self.task.evaluate(
                [Prompt(text) for text in encoded],
                predictor=predictor,
                eval_strategy=strategy,
            )
            for (index, _), score in zip(group, result.agg_scores.tolist(), strict=True):
                scores[index] = float(score)
            if strategy == "full":
                self._rollouts.extend(self._align(predictor, encoded, result.scores.tolist(), cached))

        return scores

    def _align(
        self,
        predictor: MAS,
        encoded: list[str],
        scores: list[list[float]],
        cached: set[Any],
    ) -> list[tuple[Trace, float]]:
        """Pair each trace with the score of the rollout that produced it.

        A task answers from its cache where it can, so it runs fewer rollouts
        than the score grid has cells and the traces cannot be zipped onto it.
        Recomputing which cells missed the cache, from a snapshot taken before the
        call, attaches every trace to the exact prompt set and datapoint it came
        from instead of assuming they line up.

        Args:
            predictor: The system that ran, holding the traces.
            encoded: The encoded prompt set per candidate, in evaluation order.
            scores: Per-candidate, per-datapoint scores.
            cached: The task's cache keys as they were before the call.

        Returns:
            One ``(trace, reward)`` pair per rollout actually executed, empty when
            the task answered entirely from its cache.

        Raises:
            ValueError: If the reconstructed positions and the traces disagree in
                length, which means the cache-key shape assumed here no longer
                matches the task's own.
        """
        positions = [
            (index, column)
            for index, text in enumerate(encoded)
            for column, (x, y) in enumerate(zip(self.task.xs, self.task.ys, strict=True))
            if (text, x, str(y)) not in cached
        ]
        # No misses means the task ran nothing, so `last_traces` still holds the
        # previous call's rollouts. Returning early is what makes the zip below
        # strict: from here on, a length mismatch is a reconstruction that has
        # stopped matching the task's own cache keys, and pairing rewards to the
        # wrong rollouts under it would corrupt every attribution silently.
        if not positions:
            return []
        return [
            (trace, float(scores[index][column]))
            for (index, column), trace in zip(positions, predictor.last_traces, strict=True)
        ]

    def _assign_credit(self, candidates: list[MAS], scores: list[float]) -> None:  # noqa: ARG002
        """Ask the attributor to split the last round's rollouts across components.

        Stored rather than returned so that :meth:`propose` can read it on the
        next step without the loop having to thread it through.
        """
        if self.attributor is None or not self._rollouts:
            return
        traces = [trace for trace, _ in self._rollouts]
        rewards = [reward for _, reward in self._rollouts]
        self.credit = self.attributor.attribute(traces, rewards)

    @abstractmethod
    def propose(self) -> list[MAS]:
        """Return candidate systems to evaluate this step.

        Returns:
            Candidates, derived from :attr:`mas` via :meth:`xmas.MAS.copy`.
        """

    @abstractmethod
    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Choose the system to carry into the next step.

        Args:
            candidates: The evaluated systems, including the incumbent.
            scores: Their scores, aligned to ``candidates``.

        Returns:
            The chosen system.
        """
