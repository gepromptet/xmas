"""Private execution engine.

The topology compiles to a LangGraph state graph here and nowhere else. No type
from langgraph crosses back out: callers receive a :class:`~xmas.trace.Trace`,
which is xmas's own data. Keeping the dependency behind this wall is what makes
replacing it later a contained change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from xmas.trace import Trace

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS


class CompiledGraph(Protocol):
    """The executable produced by :func:`compile_mas`.

    Declared as a protocol so that nothing outside this module needs to name a
    langgraph type, including the tests.
    """

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = ...) -> dict[str, Any]:
        """Run one state to completion."""
        ...


def compile_mas(mas: MAS) -> CompiledGraph:
    """Compile a MAS topology into an executable graph.

    Agents identified by :func:`xmas.graph.join_nodes` are marked deferred, so a
    node fed by predecessors at unequal depth runs exactly once per query rather
    than once per wave of arriving inputs.

    Args:
        mas: The system to compile.

    Returns:
        The compiled graph.
    """
    raise NotImplementedError


def run_one(mas: MAS, query: str, prompts: dict[str, str]) -> Trace:
    """Execute a compiled MAS against one input with a given prompt set.

    Args:
        mas: The system to run.
        query: The input.
        prompts: Prompt text per agent name, overriding what the agents hold.

    Returns:
        The trace of the run. A failure is recorded in ``Trace.error`` rather
        than raised, so one bad rollout does not lose a whole batch.
    """
    raise NotImplementedError


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Execute a MAS over parallel lists of inputs and prompt sets.

    Args:
        mas: The system to run.
        queries: The inputs.
        prompts: Prompt sets, one per input.

    Returns:
        One trace per input, in the order given.
    """
    raise NotImplementedError
