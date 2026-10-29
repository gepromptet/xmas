"""Prompt updaters: a promptolution optimizer applied to one agent of a MAS."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from promptolution.predictors.base_predictor import BasePredictor

from xmas.prompts import encode_prompts

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from promptolution.optimizers.base_optimizer import BaseOptimizer

    from xmas.mas import MAS
    from xmas.tasks.base import MASTask

__all__ = [
    "AgentPromptView",
    "PromptUpdater",
    "movable_agents",
    "weakest",
]


def movable_agents(mas: MAS) -> list[str]:
    """Return the agents not detached, in the system's order; ``ValueError`` when every agent is detached.

    Agents sharing a prompt are one prompt to optimize: the first of them
    stands for all, and none moves when one of them is detached.
    """
    seen = {mas.prompt_names.get(name, name) for name in mas.detached}
    names = []
    for name in mas.agents:
        key = mas.prompt_names.get(name, name)
        if key not in seen:
            names.append(name)
        seen.add(key)
    if not names:
        raise ValueError(f"Every agent is detached ({sorted(mas.detached)}); there is no prompt to optimize.")
    return names


def weakest(names: Sequence[str], credit: Mapping[str, Sequence[float]]) -> str | None:
    """Return the one of ``names`` with the least mean credit, ``NaN`` ignored, or ``None`` with no credit for any."""
    allowed = set(names)
    scored = {name: [v for v in values if not math.isnan(v)] for name, values in credit.items() if name in allowed}
    usable = {name: values for name, values in scored.items() if values}
    return min(usable, key=lambda name: sum(usable[name]) / len(usable[name])) if usable else None


class PromptUpdater:
    """Wraps any promptolution optimizer into a prompt updater for one agent.

    A promptolution optimizer improves a prompt for a single-prompt predictor,
    and :class:`AgentPromptView` turns one agent into exactly that, so the
    optimizer runs unmodified.

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
        """Initialize the updater; ``optimizer_kwargs`` go to the optimizer, e.g. ``CAPO``."""
        self.optimizer_cls = optimizer_cls
        self.task = task
        self.steps = steps
        self.optimizer_kwargs = optimizer_kwargs

    def target_agent(self, mas: MAS, credit: dict[str, list[float]]) -> str:
        """Choose which agent to work on.

        The agent with the least credit is the one whose prompt is holding the
        system back, so it is where a prompt change has the most to gain. With no
        credit available, the first agent in graph order is used, which is
        deterministic and therefore reproducible. A detached agent is never chosen.
        """
        names = movable_agents(mas)
        return weakest(names, credit) or names[0]

    def propose(self, mas: MAS, credit: dict[str, list[float]], n: int) -> list[MAS]:
        """Propose up to ``n`` candidates that differ in the prompt of :meth:`target_agent`."""
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
    that predictor. It accepts one agent's prompt, holds every other agent fixed,
    and runs the whole system. The optimizer never learns it is inside a
    multi-agent graph.

    Attributes:
        mas: The system the agent belongs to.
        agent: Name of the agent whose prompt varies.
    """

    def __init__(self, mas: MAS, agent: str) -> None:
        """Expose ``agent``'s prompt; ``KeyError`` if the MAS has no such agent."""
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
        """Run the system on ``xs`` with each prompt in this view's agent; ``BasePredictor``'s pair."""
        candidates = [prompts] * len(xs) if isinstance(prompts, str) else list(prompts)
        base = self.mas.prompts()
        encoded = [encode_prompts({**base, **self.mas.spread({self.agent: prompt})}) for prompt in candidates]
        return self.mas.predict(encoded, xs, system_prompts=system_prompts)

    # promptolution's BasePredictor calls _extract_preds on every prediction batch.
    def _extract_preds(self, preds: list[str]) -> list[str]:
        """Return predictions unchanged; the terminal agent already extracts."""
        return list(preds)
