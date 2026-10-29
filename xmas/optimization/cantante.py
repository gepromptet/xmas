# Loop adapted from CANTANTE by Tom Zehle: https://github.com/finitearth/cantante
"""CANTANTE: attribution-guided prompt optimization for a whole system."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from xmas.optimization.broker import Board, ProxyTask, Request, rounds
from xmas.optimization.optimizer import MASOptimizer
from xmas.optimization.updaters import AgentPromptView, movable_agents
from xmas.prompts import encode_prompts

if TYPE_CHECKING:
    from promptolution.optimizers.base_optimizer import BaseOptimizer

    from xmas.mas import MAS
    from xmas.optimization.attribution import Attributor
    from xmas.tasks.base import MASTask
    from xmas.trace import Trace

__all__ = [
    "CantanteOptimizer",
]


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

    attributor: Attributor

    def __init__(  # noqa: PLR0913  the public constructor, kept for version 1 callers
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
            attributor: Credit assignment, required here, unlike on the base
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
        self.initial_prompts = initial_prompts or {}
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.rounds_served = 0
        self.unattributed = 0

    def propose(self) -> list[MAS]:
        """Run one round of every agent's optimizer and return what they settled on.

        A detached agent gets no optimizer and keeps its prompt. The incumbent
        comes back with the system built from each agent's best prompt, so the
        proposal is measured against it on the real task rather than trusting
        attributed credit to have been right.

        Raises:
            ValueError: If every agent is detached.
        """
        board = Board()
        optimizers: dict[str, BaseOptimizer] = {}
        for name in movable_agents(self.mas):
            optimizers[name] = self.optimizer_cls(
                predictor=AgentPromptView(self.mas, name),
                task=ProxyTask(name, self.task, board),
                initial_prompts=self.initial_prompts.get(name) or [self.mas.agents[name].prompt],
                **self.optimizer_kwargs,
            )

        with ThreadPoolExecutor(max_workers=max(1, len(optimizers)), thread_name_prefix="xmas-agent") as executor:
            workers = [executor.submit(optimizer.optimize, self.inner_steps) for optimizer in optimizers.values()]
            try:
                for batch in rounds(board, workers):
                    self.serve(batch)
            finally:
                board.cancel_all()

        best = {name: str(optimizer.prompts[0]) for name, optimizer in optimizers.items() if optimizer.prompts}
        if not best or all(prompt == self.mas.agents[name].prompt for name, prompt in best.items()):
            return [self.mas]
        return [self.mas, self.mas.copy(prompts=best)]

    def serve(self, batch: dict[str, Request]) -> None:
        """Answer one round of pending requests with attributed credit."""
        sets = self.joint_sets(batch)
        rewards, traces = self.measure(sets)
        credit = self.split_credit(sets, rewards, traces)

        for agent, request in batch.items():
            request.scores = credit[agent][: len(request.prompts)]
            request.done.set()
        self.rounds_served += 1

    def joint_sets(self, batch: dict[str, Request]) -> list[dict[str, str]]:
        """Combine each agent's candidates into whole prompt sets to evaluate.

        The pairing is positional and wraps: agent A's prompt ``k`` meets agent
        B's prompt ``k``, and the shorter list repeats, so agent A's prompt ``k``
        always lands in set ``k``, which lets credit be handed back without a
        lookup table.
        """
        current = self.mas.prompts()
        varying = {agent: request.prompts for agent, request in batch.items() if request.prompts}
        if not varying:
            return [current]

        width = max(len(prompts) for prompts in varying.values())
        return [
            {**current, **self.mas.spread({agent: prompts[k % len(prompts)] for agent, prompts in varying.items()})}
            for k in range(width)
        ]

    def measure(self, sets: list[dict[str, str]]) -> tuple[list[list[float]], dict[tuple[int, int], Trace]]:
        """Evaluate whole prompt sets on the real task: per-set, per-datapoint scores, and the traces by that cell.

        A pair the task answered from its cache has no trace this round.
        """
        result, traces = self.run_task(self.mas, [encode_prompts(prompt_set) for prompt_set in sets], full=True)
        return [[float(value) for value in row] for row in result.scores.tolist()], traces

    def split_credit(
        self,
        sets: list[dict[str, str]],
        rewards: list[list[float]],
        traces: dict[tuple[int, int], Trace],
    ) -> dict[str, list[list[float]]]:
        """Split the measured scores into per-agent credit, shaped for the proxies.

        A cell the judge could not read falls back to the system's own score for
        that rollout, because the inner optimizer needs a number to rank on: the
        verdict a uniform attributor would have given. That is a measurement
        rather than an invention, and :attr:`unattributed` counts it so a run
        that leaned on it is visible. Returns per-set, per-datapoint credit for each agent.
        """
        width = len(rewards[0]) if rewards else 0
        names = sorted(movable_agents(self.mas))
        grid = {name: [[math.nan] * width for _ in sets] for name in names}

        positions = sorted(traces)
        if positions:
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

        for name, rows in grid.items():
            for index, row in enumerate(rows):
                for column, value in enumerate(row):
                    if math.isnan(value):
                        self.unattributed += 1
                        row[column] = rewards[index][column]
            self.credit[name] = [sum(row) / len(row) for row in rows if row]
        return grid
