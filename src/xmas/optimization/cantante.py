# Loop adapted from CANTANTE by Tom Zehle: https://github.com/finitearth/cantante
"""CANTANTE: attribution-guided prompt optimization for a whole system."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import TYPE_CHECKING, Any

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.prompt import Prompt

from xmas.mas import MAS
from xmas.optimization.attribution import Attributor
from xmas.optimization.broker import Board, ProxyTask, Request, rounds
from xmas.optimization.optimizer import MASOptimizer
from xmas.optimization.updaters import AgentPromptView
from xmas.prompts import encode_prompts

if TYPE_CHECKING:  # pragma: no cover
    from xmas.tasks.base import MASTask
    from xmas.trace import Trace


class CantanteOptimizer(MASOptimizer):
    """Optimizes every agent's prompt at once, guided by credit assignment.

    The method's claim is that a multi-agent system does not have to be optimized
    one agent at a time, nor all at once by brute force. Each agent gets its own
    prompt optimizer; those optimizers run together; and the system evaluation
    they share is split back into per-agent credit by an
    :class:`~xmas.optimization.attribution.Attributor`. An agent whose change
    helped sees a better score, one whose change hurt sees a worse one, and
    neither had to be evaluated in isolation.

    The inner optimizers are ordinary promptolution optimizers, unmodified. They
    are suspended on a proxy task while the system evaluation happens; see
    :mod:`xmas.optimization.broker` for why that needs threads.

    Attributes:
        mas: The current best system.
        task: The task being optimized against.
        attributor: Splits each system evaluation into per-agent credit.
        history: Real task score of the incumbent after each step, never credit.
        credit: Mean credit per agent from the last round, for reading afterwards.
        rounds_served: Broker rounds completed, across the whole run.
        unattributed: Rollouts the attributor returned no credit for, across the
            whole run. Above zero means the meta-model's answers were partly
            unreadable, and the run fell back to the measured system score there.
    """

    def __init__(
        self,
        mas: MAS,
        task: MASTask,
        attributor: Attributor,
        optimizer_cls: type[BaseOptimizer],
        *,
        inner_steps: int = 1,
        initial_prompts: dict[str, list[str]] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            mas: The starting system.
            task: The task to optimize against.
            attributor: Credit assignment. Required here, unlike on the base
                class: without it every agent would see the same score and the
                inner optimizers would be optimizing noise.
            optimizer_cls: The promptolution optimizer to run per agent, for
                example ``CAPO`` or ``EvoPromptGA``.
            inner_steps: Steps each per-agent optimizer runs per outer step.
            initial_prompts: Seed population per agent. Defaults to the agent's
                current prompt alone, which is enough for an optimizer that
                mutates but not for one that recombines: a genetic optimizer
                given one parent has nothing to cross it with.
            optimizer_kwargs: Forwarded when each inner optimizer is constructed.
        """
        super().__init__(mas=mas, task=task, attributor=attributor)
        self.optimizer_cls = optimizer_cls
        self.inner_steps = inner_steps
        self.initial_prompts = dict(initial_prompts or {})
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.rounds_served = 0
        self.unattributed = 0

    # ------------------------------------------------------------- the round

    def propose(self) -> list[MAS]:
        """Run one round of every agent's optimizer and return what they settled on.

        Returns:
            The incumbent and the system built from each agent's best prompt. Both
            are returned so that the base class measures the proposal against the
            incumbent on the real task, rather than trusting attributed credit to
            have been right.
        """
        board = Board()
        optimizers: dict[str, BaseOptimizer] = {}
        for name, agent in self.mas.agents.items():
            proxy = ProxyTask(name, self.task, board)
            board.register(proxy)
            optimizers[name] = self.optimizer_cls(
                predictor=AgentPromptView(self.mas, name),
                task=proxy,
                initial_prompts=self.initial_prompts.get(name) or [agent.prompt],
                **self.optimizer_kwargs,
            )

        with ThreadPoolExecutor(max_workers=max(1, len(optimizers)), thread_name_prefix="xmas-agent") as executor:
            workers = [executor.submit(optimizer.optimize, self.inner_steps) for optimizer in optimizers.values()]
            try:
                for batch in rounds(board, workers):
                    self._serve(batch)
            finally:
                board.cancel_all()

        best = {name: str(optimizer.prompts[0]) for name, optimizer in optimizers.items() if optimizer.prompts}
        if not best or best == self.mas.prompts():
            return [self.mas]
        return [self.mas, self.mas.copy(prompts=best)]

    def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
        """Keep the highest-scoring candidate, preferring the incumbent on a tie.

        Args:
            candidates: The evaluated systems, incumbent first.
            scores: Their scores, aligned to ``candidates``.

        Returns:
            The chosen system.
        """
        best = max(range(len(candidates)), key=lambda index: scores[index])
        return candidates[best] if scores[best] > scores[0] else candidates[0]

    # ------------------------------------------------------------ the broker

    def _serve(self, batch: dict[str, Request]) -> None:
        """Answer one round of pending requests with attributed credit."""
        sets = self._joint_sets(batch)
        rewards, traces = self._measure(sets)
        credit = self._attribute(sets, rewards, traces)

        for agent, request in batch.items():
            rows = credit.get(agent)
            if rows is None:
                request.error = RuntimeError(f"The attributor returned no credit for {agent!r}.")
            else:
                request.scores = rows[: len(request.prompts)]
            request.done.set()
        self.rounds_served += 1

    def _joint_sets(self, batch: dict[str, Request]) -> list[dict[str, str]]:
        """Combine each agent's candidates into whole prompt sets to evaluate.

        The pairing is positional and wraps: agent A's prompt ``k`` meets agent
        B's prompt ``k``, and the shorter list repeats. That is the seed's naive
        selector, and its property that matters is that agent A's prompt ``k``
        always lands in set ``k``, which is what lets credit be handed back
        without a lookup table.

        Args:
            batch: Pending requests by agent.

        Returns:
            One prompt set per evaluation, keyed by agent name.
        """
        current = self.mas.prompts()
        varying = {agent: request.prompts for agent, request in batch.items() if request.prompts}
        if not varying:
            return [current]

        width = max(len(prompts) for prompts in varying.values())
        cycles = {agent: cycle(prompts) for agent, prompts in varying.items()}
        return [{**current, **{agent: next(values) for agent, values in cycles.items()}} for _ in range(width)]

    def _measure(self, sets: list[dict[str, str]]) -> tuple[list[list[float]], dict[tuple[int, int], Trace]]:
        """Evaluate whole prompt sets on the real task and recover their traces.

        Traces come back only for the rollouts the task actually ran; a pair it
        answered from its cache has no trace this round. Rather than assume the
        two line up, the cache is snapshotted first and the misses recomputed, so
        every trace is attached to the exact prompt set and datapoint it came
        from.

        Args:
            sets: The prompt sets to evaluate.

        Returns:
            Per-set, per-datapoint scores, and the traces keyed by that position.
        """
        encoded = [encode_prompts(prompt_set) for prompt_set in sets]
        cached = set(self.task.eval_cache)
        result = self.task.evaluate([Prompt(text) for text in encoded], predictor=self.mas, eval_strategy="full")

        positions = [
            (index, column)
            for index, text in enumerate(encoded)
            for column, (x, y) in enumerate(zip(self.task.xs, self.task.ys, strict=True))
            if (text, x, str(y)) not in cached
        ]
        traces = dict(zip(positions, self.mas.last_traces, strict=False))
        return [[float(value) for value in row] for row in result.scores.tolist()], traces

    def _attribute(
        self,
        sets: list[dict[str, str]],
        rewards: list[list[float]],
        traces: dict[tuple[int, int], Trace],
    ) -> dict[str, list[list[float]]]:
        """Split the measured scores into per-agent credit, shaped for the proxies.

        Args:
            sets: The evaluated prompt sets.
            rewards: Per-set, per-datapoint task scores.
            traces: Traces by ``(set, datapoint)``.

        Returns:
            Per-set, per-datapoint credit for each agent.
        """
        width = len(rewards[0]) if rewards else 0
        names = sorted(self.mas.agents)
        grid = {name: [[math.nan] * width for _ in sets] for name in names}

        positions = sorted(traces)
        if positions and self.attributor is not None:
            flat = self.attributor.attribute(
                [traces[position] for position in positions],
                [rewards[index][column] for index, column in positions],
            )
            for name in names:
                values = flat.get(name)
                if values is None:
                    continue
                for (index, column), value in zip(positions, values, strict=True):
                    grid[name][index][column] = value

        self._record(grid, rewards)
        return grid

    def _record(self, grid: dict[str, list[list[float]]], rewards: list[list[float]]) -> None:
        """Fill unattributed cells with the measured score and note how many there were.

        An unattributed cell is one the judge could not read. Leaving it absent is
        not an option here, because the inner optimizer needs a number to rank on,
        so it falls back to the system's own score for that rollout: the verdict a
        uniform attributor would have given. That is a measurement rather than an
        invention, and the count is kept so a run that leaned on it is visible.
        """
        for name, rows in grid.items():
            for index, row in enumerate(rows):
                for column, value in enumerate(row):
                    if math.isnan(value):
                        self.unattributed += 1
                        row[column] = rewards[index][column]
            self.credit[name] = [sum(row) / len(row) for row in rows if row]
