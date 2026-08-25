"""Updaters: per-component-type proposal machinery."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from promptolution.optimizers.base_optimizer import BaseOptimizer

from xmas.mas import MAS


@runtime_checkable
class Updater(Protocol):
    """Proposes new values for one kind of component.

    One updater exists per component type: prompts, topology, model bindings.
    Each sees the current MAS and the credit its component earned, and returns
    candidate replacements. Adding a research method means adding an updater, not
    a framework.
    """

    def propose(self, mas: MAS, credit: dict[str, list[float]], n: int) -> list[MAS]:
        """Propose candidate systems differing from ``mas`` in this component.

        Args:
            mas: The current system.
            credit: Per-component credit from an
                :class:`~xmas.optimization.attribution.Attributor`.
            n: How many candidates to return.

        Returns:
            Up to ``n`` candidates, derived via :meth:`xmas.MAS.copy`.
        """
        ...


class PromptUpdater:
    """Wraps any promptolution optimizer into a prompt updater for one agent.

    This is where the founding premise pays off. A promptolution optimizer
    already improves a prompt for a single-prompt predictor; a single agent
    inside a MAS *is* such a predictor. So the wrapper hands the optimizer the
    one node it is optimizing and lets it run unmodified, rather than porting its
    algorithm to a multi-agent setting.
    """

    def __init__(self, optimizer_cls: type[BaseOptimizer], **optimizer_kwargs: object) -> None:
        """Initialize the updater.

        Args:
            optimizer_cls: A promptolution optimizer class, e.g. ``CAPO``.
            **optimizer_kwargs: Forwarded when the optimizer is constructed.
        """
        raise NotImplementedError

    def propose(self, mas: MAS, credit: dict[str, list[float]], n: int) -> list[MAS]:
        """Propose candidates that differ in one agent's prompt.

        Args:
            mas: The current system.
            credit: Per-agent credit, used to choose which agent to work on.
            n: How many candidates to return.

        Returns:
            Up to ``n`` candidates.
        """
        raise NotImplementedError


@runtime_checkable
class TopologyUpdater(Protocol):
    """Proposes new topologies. Implementations are a later package.

    Declared now so that the seam exists and the reference optimizer can be
    written against the general shape rather than against prompts specifically.
    """

    def propose(self, mas: MAS, credit: dict[str, list[float]], n: int) -> list[MAS]:
        """Propose candidate systems differing in topology.

        Args:
            mas: The current system.
            credit: Per-component credit.
            n: How many candidates to return.

        Returns:
            Up to ``n`` candidates.
        """
        ...
