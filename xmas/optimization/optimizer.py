"""The MAS optimization loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from promptolution.utils.prompt import Prompt

if TYPE_CHECKING:
    from xmas.mas import MAS
    from xmas.optimization.attribution import Attributor
    from xmas.tasks.base import MASTask
    from xmas.trace import Trace

__all__ = [
    "MASOptimizer",
    "traces_by_position",
]


class MASOptimizer(ABC):
    """Improves a MAS against a task by proposing, evaluating and selecting.

    The loop is three steps and the base class owns all three, so a subclass
    decides *what* to propose and may decide *how* to select, never how to evaluate.
    Evaluation goes through the task's narrow interface and nowhere else, which
    is what keeps every optimizer compatible with every task, including tasks
    that do not generate at all.

    Attributes:
        mas: The current best system.
        task: The task being optimized against.
        attributor: Optional credit assignment, consulted before proposing.
        history: Aggregate score of the incumbent after each step.
        credit: Per-component credit from the last round, read by :meth:`propose`.
        rollouts: ``(trace, reward)`` per rollout the last :meth:`evaluate` ran,
            filled only when an attributor is present.
    """

    def __init__(self, mas: MAS, task: MASTask, attributor: Attributor | None = None) -> None:
        """Start from ``mas``; an ``attributor``, which a subclass may consult to choose what to change, is optional."""
        self.mas = mas
        self.task = task
        self.attributor = attributor
        self.history: list[float] = []
        self.credit: dict[str, list[float]] = {}
        self.rollouts: list[tuple[Trace, float]] = []

    def optimize(self, n_steps: int) -> MAS:
        """Run ``n_steps`` propose-evaluate-select rounds and return the best system found, maybe the one passed in."""
        for _ in range(n_steps):
            candidates = self.propose()
            if not candidates:
                break
            scores = self.evaluate(candidates)
            self.mas = self.select(candidates, scores)
            self.history.append(max(scores))
            if self.attributor is not None and self.rollouts:
                traces = [trace for trace, _ in self.rollouts]
                rewards = [reward for _, reward in self.rollouts]
                self.credit = self.attributor.attribute(traces, rewards)
        return self.mas

    @abstractmethod
    def propose(self) -> list[MAS]:
        """Return candidate systems to evaluate this step, the incumbent first, derived via :meth:`xmas.MAS.copy`."""

    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Return the highest-scoring candidate; a tie keeps the incumbent, which comes first."""
        return candidates[scores.index(max(scores))]

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
        nothing to compare. Returns one aggregate score per candidate, in order.
        """
        scores = [0.0] * len(candidates)
        self.rollouts = []
        groups: dict[object, list[tuple[int, MAS]]] = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(candidate.structure, []).append((index, candidate))

        for group in groups.values():
            encoded = [candidate.encode() for _, candidate in group]
            result, traces = self.run_task(group[0][1], encoded, full=self.attributor is not None)
            for (index, _), score in zip(group, result.agg_scores.tolist(), strict=True):
                scores[index] = float(score)
            grid = result.scores.tolist()
            self.rollouts.extend((trace, float(grid[index][column])) for (index, column), trace in traces.items())
        return scores

    def run_task(  # public since before the lint round: a subclass or caller may pass full by position
        self,
        predictor: MAS,
        encoded: list[str],
        full: bool,  # noqa: FBT001
    ) -> tuple[Any, dict[tuple[int, int], Trace]]:
        """Evaluate encoded prompt sets on the task; ``full`` runs every datapoint and keys the traces by cell."""
        cached = set(self.task.eval_cache)
        prompts = [Prompt(text) for text in encoded]
        result = self.task.evaluate(prompts, predictor=predictor, eval_strategy="full" if full else None)
        return result, traces_by_position(self.task, predictor, encoded, cached) if full else {}


def traces_by_position(
    task: MASTask,
    predictor: MAS,
    encoded: list[str],
    cached: set[Any],
) -> dict[tuple[int, int], Trace]:
    """Key each trace by the ``(candidate, datapoint)`` cell of the rollout that produced it.

    A task answers from its cache where it can, so it runs fewer rollouts
    than the score grid has cells and the traces cannot be zipped onto it.
    Recomputing which cells missed the cache, from a snapshot taken before the
    call, attaches every trace to the exact prompt set and datapoint it came
    from instead of assuming they line up. ``cached`` holds the task's cache
    keys as they were before the call; the result is empty when the task
    answered entirely from its cache.

    Raises:
        ValueError: If the reconstructed positions and the traces disagree in
            length, which means the cache-key shape assumed here no longer
            matches the task's own.
    """
    positions = [
        (index, column)
        for index, text in enumerate(encoded)
        for column, (x, y) in enumerate(zip(task.xs, task.ys, strict=True))
        if (text, x, str(y)) not in cached
    ]
    # No misses means the task ran nothing, so `last_traces` still holds the
    # previous call's rollouts. Returning early is what makes the zip below
    # strict: from here on, a length mismatch is a reconstruction that has
    # stopped matching the task's own cache keys, and pairing rewards to the
    # wrong rollouts under it would corrupt every attribution silently.
    if not positions:
        return {}
    return dict(zip(positions, predictor.last_traces, strict=True))
