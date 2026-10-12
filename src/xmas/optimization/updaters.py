"""Prompt updaters: a promptolution optimizer applied to one agent of a MAS."""

from __future__ import annotations

from typing import Any

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors.base_predictor import BasePredictor

from xmas.mas import MAS
from xmas.prompts import encode_prompts
from xmas.tasks.base import MASTask


class PromptUpdater:
    """Wraps any promptolution optimizer into a prompt updater for one agent.

    This is where the founding premise pays off. A promptolution optimizer
    already improves a prompt for a single-prompt predictor, and
    :class:`AgentPromptView` turns one agent into exactly that, so the optimizer
    runs unmodified rather than being ported to a multi-agent setting.

    Attributes:
        optimizer_cls: The promptolution optimizer class to run.
        task: The task the inner optimizer optimizes against.
        steps: How many inner optimization steps to run per proposal.
    """

    def __init__(
        self,
        optimizer_cls: type[BaseOptimizer],
        task: MASTask,
        steps: int = 1,
        **optimizer_kwargs: Any,
    ) -> None:
        """Initialize the updater.

        Args:
            optimizer_cls: A promptolution optimizer class, e.g. ``CAPO``.
            task: The task to optimize the agent's prompt against.
            steps: Inner optimization steps per call to :meth:`propose`.
            **optimizer_kwargs: Forwarded when the optimizer is constructed.
        """
        self.optimizer_cls = optimizer_cls
        self.task = task
        self.steps = steps
        self.optimizer_kwargs = optimizer_kwargs

    def target_agent(self, mas: MAS, credit: dict[str, list[float]]) -> str:
        """Choose which agent to work on.

        The agent with the least credit is the one whose prompt is holding the
        system back, so it is where a prompt change has the most to gain. With no
        credit available, the first agent in graph order is used, which is
        deterministic and therefore reproducible.

        Args:
            mas: The current system.
            credit: Per-agent credit, possibly empty.

        Returns:
            The chosen agent's name.
        """
        scored = {name: values for name, values in credit.items() if name in mas.agents and values}
        if not scored:
            return next(iter(mas.agents))
        return min(scored, key=lambda name: sum(scored[name]) / len(scored[name]))

    def propose(self, mas: MAS, credit: dict[str, list[float]], n: int) -> list[MAS]:
        """Propose candidates that differ in one agent's prompt.

        Args:
            mas: The current system.
            credit: Per-agent credit, used to choose which agent to work on.
            n: How many candidates to return.

        Returns:
            Up to ``n`` candidates.
        """
        agent = self.target_agent(mas, credit)
        view = AgentPromptView(mas, agent)
        optimizer = self.optimizer_cls(
            predictor=view,
            task=self.task,
            initial_prompts=[mas.agents[agent].prompt],
            **self.optimizer_kwargs,
        )
        prompts = optimizer.optimize(n_steps=self.steps)
        return [mas.copy(prompts={agent: prompt.construct_prompt()}) for prompt in prompts[:n]]


class AgentPromptView(BasePredictor):
    """One agent of a MAS, presented as a single-prompt predictor.

    A promptolution optimizer improves a prompt for a predictor, and this view is
    that predictor. It
    accepts one agent's prompt, holds every other agent fixed, and runs the whole
    system. The optimizer never learns it is inside a multi-agent graph.

    Attributes:
        mas: The system the agent belongs to.
        agent: Name of the agent whose prompt varies.
    """

    def __init__(self, mas: MAS, agent: str) -> None:
        """Initialize the view.

        Args:
            mas: The system to run.
            agent: Name of the agent whose prompt this view exposes.

        Raises:
            KeyError: If the MAS has no such agent.
        """
        if agent not in mas.agents:
            raise KeyError(f"MAS has no agent {agent!r}; it has {sorted(mas.agents)}.")
        self.mas = mas
        self.agent = agent
        super().__init__(llm=mas.llm)

    def predict(
        self,
        prompts: str | list[str],
        xs: list[str],
        system_prompts: str | list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Run the system with each prompt substituted into this view's agent.

        Args:
            prompts: Candidate prompts for this agent, one per input.
            xs: The inputs.
            system_prompts: Passed through to the underlying system.

        Returns:
            A ``(predictions, sequences)`` pair, matching ``BasePredictor``.
        """
        candidates = [prompts] * len(xs) if isinstance(prompts, str) else list(prompts)
        base = self.mas.prompts()
        encoded = [encode_prompts({**base, self.agent: prompt}) for prompt in candidates]
        return self.mas.predict(encoded, xs, system_prompts=system_prompts)

    # promptolution's BasePredictor calls _extract_preds on every prediction batch.
    def _extract_preds(self, preds: list[str]) -> list[str]:
        """Return predictions unchanged; the terminal agent already extracts."""
        return list(preds)
