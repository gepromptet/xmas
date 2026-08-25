"""Credit assignment: turning one system-level reward into per-component credit."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xmas.trace import Trace


@runtime_checkable
class Attributor(Protocol):
    """Splits a system-level score across the components that produced it.

    An attributor sees rollouts and their rewards and says which component
    deserves the credit. It never proposes a change; that is an
    :class:`~xmas.optimization.updaters.Updater`'s job. Separating the two is
    what lets a new attribution method ship without a new optimizer.

    The contract is deliberately batch-level and does not aggregate: given
    ``n`` rollouts, every returned list has ``n`` entries. Averaging is the
    caller's decision, and an attributor that averaged early would destroy the
    per-rollout signal a later method might want.
    """

    def attribute(
        self,
        traces: list[Trace],
        rewards: list[float],
    ) -> dict[str, list[float]]:
        """Assign credit per component per rollout.

        Args:
            traces: One trace per rollout.
            rewards: The system-level score of each rollout, aligned to ``traces``.

        Returns:
            Credit per rollout, keyed by component name. Every value has the same
            length as ``rewards``.
        """
        ...
