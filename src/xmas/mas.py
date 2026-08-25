"""The MAS: a multi-agent system that is also a promptolution predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
from promptolution.llms.base_llm import BaseLLM
from promptolution.predictors.base_predictor import BasePredictor

from xmas.agent import Agent
from xmas.spec import MASSpec
from xmas.trace import Trace


class MAS(BasePredictor):
    """A graph of agents that maps an input to an output.

    ``MAS`` subclasses promptolution's ``BasePredictor``, so a one-agent MAS is
    not merely analogous to a promptolution predictor, it is one: hand it to a
    promptolution task and its optimizers work on it unchanged. Everything that
    makes it a multi-agent system, the extra nodes and the edges between them, is
    additional structure on top of that contract.

    The parameters an optimizer may change, per-agent prompts, the topology and
    the per-node model bindings, are all reachable as ordinary attributes, and
    :meth:`copy` is how candidates are derived from them.

    Attributes:
        agents: Agents keyed by name. Mutable, so ``mas.agents["solver"].prompt``
            is a supported edit.
        graph: The topology as a ``networkx.DiGraph`` over agent names plus the
            :data:`~xmas.graph.START` and :data:`~xmas.graph.END` sentinels.
        llm: The default model, used by any agent without its own binding.
        llms: Additional models by name, which agents bind to via ``Agent.model``.
    """

    agents: dict[str, Agent]
    graph: nx.DiGraph
    llms: dict[str, BaseLLM]
    max_steps: int

    def __init__(
        self,
        agents: list[Agent],
        edges: list[tuple[str, str]] | None = None,
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        max_steps: int = 50,
    ) -> None:
        """Initialize a MAS.

        Args:
            agents: The agents, in any order. Names must be unique.
            edges: Directed ``(source, target)`` pairs between agent names.
                Agents left unconnected are wired from the source sentinel and to
                the terminal sentinel automatically.
            llm: Default model for agents without their own binding.
            llms: Models by name, for per-agent bindings via ``Agent.model``.
            max_steps: Cap on executed agent steps per call, so a cyclic topology
                cannot run forever.

        Raises:
            ValueError: If names collide, an edge names an unknown agent, the
                topology is not runnable, or an agent binds to an unknown model.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------- running

    def __call__(self, query: str) -> str:
        """Run the MAS on one input and return the terminal agent's output.

        Args:
            query: The input, bound to the ``query`` state variable.

        Returns:
            The terminal agent's output.
        """
        raise NotImplementedError

    def run(self, query: str) -> Trace:
        """Run the MAS on one input and return the full record of the run.

        This is :meth:`__call__` without the shared-state convenience: the trace
        is returned rather than stashed, which is what concurrent callers want.

        Args:
            query: The input, bound to the ``query`` state variable.

        Returns:
            The trace, whose ``output`` field is what :meth:`__call__` returns.
        """
        raise NotImplementedError

    def run_batch(self, queries: list[str]) -> list[Trace]:
        """Run the MAS over several inputs, concurrently where the LLM allows.

        Args:
            queries: The inputs.

        Returns:
            One trace per query, in the order given.
        """
        raise NotImplementedError

    @property
    def last_trace(self) -> Trace | None:
        """Trace of the most recent :meth:`__call__`, or ``None`` before any.

        This is the single-call ergonomic. Batched execution records into
        :attr:`last_traces` instead; see there for the concurrency contract.
        """
        raise NotImplementedError

    @property
    def last_traces(self) -> tuple[Trace, ...]:
        """Traces of the most recent :meth:`predict` or :meth:`run_batch` call.

        Assigned once, at the end of the batch, rather than appended to as the
        batch progresses, so a reader either sees the previous batch or the whole
        new one. This is the seam an attributor reads after a task evaluation.
        One MAS instance still serves one batch at a time; run concurrent batches
        on separate instances, or use :meth:`run`, which shares no state at all.
        """
        raise NotImplementedError

    # ------------------------------------------------- promptolution predictor

    def predict(
        self,
        prompts: str | list[str],
        xs: list[str],
        system_prompts: str | list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Evaluate prompt sets against inputs, as promptolution's tasks expect.

        ``prompts`` and ``xs`` are parallel: element ``i`` is the prompt set to
        run against input ``i``. Each prompt set is an encoding produced by
        :func:`xmas.prompts.encode_prompts`, which for a single-agent MAS is the
        bare prompt string, so promptolution's optimizers need no adapter.

        The MAS itself is not mutated; each element runs against its own prompts.

        Args:
            prompts: Encoded prompt sets, one per input.
            xs: The inputs.
            system_prompts: Passed through to the underlying models.

        Returns:
            A ``(predictions, sequences)`` pair, matching ``BasePredictor``.
        """
        raise NotImplementedError

    def _extract_preds(self, preds: list[str]) -> list[str]:
        """Return predictions unchanged; a MAS's terminal agent already extracts.

        Args:
            preds: Raw terminal-agent outputs.

        Returns:
            The same list.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- components

    def copy(
        self,
        prompts: dict[str, str] | None = None,
        edges: list[tuple[str, str]] | None = None,
        models: dict[str, str | None] | None = None,
    ) -> MAS:
        """Derive a new MAS with some components replaced.

        This is how an optimizer proposes a candidate. Models are shared with the
        original by reference; everything else is copied, so the original is never
        touched.

        Args:
            prompts: Replacement prompts, keyed by agent name.
            edges: A replacement topology. Replaces the edge set entirely.
            models: Replacement model bindings, keyed by agent name. A ``None``
                value unbinds an agent back to the MAS default.

        Returns:
            The derived MAS.

        Raises:
            ValueError: If a key names an unknown agent, or the result is not
                runnable.
        """
        raise NotImplementedError

    # ---------------------------------------------------------- serialization

    def to_dict(self) -> MASSpec:
        """Return the plain-data spec for this MAS.

        Returns:
            Prompts, topology and model bindings, and nothing else. Models are
            recorded by name; the LLM objects are runtime state, not spec.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
    ) -> MAS:
        """Rebuild a MAS from a spec.

        Args:
            data: A spec produced by :meth:`to_dict`.
            llm: Default model to attach.
            llms: Models by name, resolving the spec's bindings.

        Returns:
            The reconstructed MAS.

        Raises:
            ValueError: If the spec is malformed or binds to an unsupplied model.
        """
        raise NotImplementedError

    def to_yaml(self, path: str | Path) -> None:
        """Write the spec to a YAML file.

        Args:
            path: Destination file.
        """
        raise NotImplementedError

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
    ) -> MAS:
        """Read a MAS spec from a YAML file.

        Args:
            path: Source file.
            llm: Default model to attach.
            llms: Models by name, resolving the spec's bindings.

        Returns:
            The reconstructed MAS.
        """
        raise NotImplementedError

    # -------------------------------------------------------------- internals

    def _compiled(self) -> Any:
        """Return the compiled executable graph, building it on first use.

        The compilation is cached and invalidated whenever the topology, the
        agent set or an agent's wiring changes. The compiled object's type is an
        implementation detail and never appears in a public signature.
        """
        raise NotImplementedError

    def _invalidate(self) -> None:
        """Drop the cached compilation after a structural change."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return an unambiguous representation naming agents and edge count."""
        raise NotImplementedError
