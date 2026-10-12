"""The MAS: a multi-agent system that is also a promptolution predictor."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from promptolution.llms.base_llm import BaseLLM
from promptolution.predictors.base_predictor import BasePredictor

from xmas import runtime
from xmas.agent import Agent
from xmas.graph import SENTINELS, agent_names, build_graph, terminal_agent, validate_graph
from xmas.prompts import decode_prompts, encode_prompts
from xmas.spec import validate_spec
from xmas.tools import Tool, ToolIndex, ToolSource, index_tools, select_tools
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
        default_model: Name recorded for the default model in a serialized spec.
        last_trace: Trace of the most recent :meth:`__call__`, or ``None`` before any.
        last_traces: Traces of the most recent :meth:`predict` or :meth:`run_batch`
            call. Assigned once, at the end of the batch, so a reader either sees
            the previous batch or the whole new one; this is the seam an attributor
            reads after a task evaluation. One MAS instance serves one batch at a
            time; run concurrent batches on separate instances, or use :meth:`run`,
            which shares no state at all.
    """

    def __init__(
        self,
        agents: list[Agent],
        edges: list[tuple[str, str]] | None = None,
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        max_steps: int = 50,
        default_model: str | None = None,
        tools: Sequence[ToolSource | Tool] = (),
        max_tool_rounds: int = 5,
        max_visits: int = 3,
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
            default_model: Name to record for the default model when serializing.
                Defaults to the model id ``llm`` reports.
            tools: Tool objects and tool sources (e.g. ``MCPTools``), mixed, that
                ``Agent.tools`` names resolve against. Runtime state like ``llm``;
                never part of the spec.
            max_tool_rounds: Model turns per agent visit that may request tools,
                before one last turn with the tools withheld.
            max_visits: How often one agent may run in a single query before its
                back edges close, so a prompt that always loops still ends.

        Raises:
            ValueError: If names collide, an edge names an unknown agent, the
                topology is not runnable, an agent binds to an unknown model or
                tool, two sources offer the same tool name, or a tool-using
                agent's model cannot call tools.
        """
        if not agents:
            raise ValueError("A MAS needs at least one agent.")
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be >= 0.")
        super().__init__(llm=llm)  # type: ignore[arg-type]

        self.agents = {agent.name: agent for agent in agents}
        if len(self.agents) != len(agents):
            raise ValueError("Agent names must be unique.")

        self.graph = build_graph([agent.name for agent in agents], list(edges or []))
        validate_graph(self.graph)
        # Force the terminal check now. A MAS with two terminal agents is only
        # broken at the moment it produces an output, and discovering that after
        # a batch of model calls is an expensive way to learn it.
        terminal_agent(self.graph)

        self.llms = dict(llms or {})
        self.max_steps = max_steps
        self.max_visits = max_visits
        self.default_model = default_model if default_model is not None else getattr(llm, "model_id", None)

        for agent in agents:
            if agent.model is not None and agent.model not in self.llms:
                raise ValueError(
                    f"Agent {agent.name!r} binds to model {agent.model!r}, which is not in llms={sorted(self.llms)}."
                )

        self.compilation: tuple[str, Any] | None = None
        self.last_trace: Trace | None = None
        self.last_traces: tuple[Trace, ...] = ()

        self.tools = tuple(tools)
        self.tool_index: ToolIndex | None = None  # built on first use, so tool-free systems never touch their sources
        self.max_tool_rounds = max_tool_rounds
        for agent in agents:  # fail at construction, not mid-run
            if agent.tools:
                self.resolve_tools(agent)

    def __call__(self, query: str) -> str:
        """Run the MAS on one input and return the terminal agent's output.

        Args:
            query: The input, bound to the ``query`` state variable.

        Returns:
            The terminal agent's output.

        Raises:
            RuntimeError: If the run failed.
        """
        trace = self.run(query)
        self.last_trace = trace
        if trace.error is not None:
            raise RuntimeError(f"MAS run failed: {trace.error}")
        return trace.output

    def run(self, query: str) -> Trace:
        """Run the MAS on one input and return the full record of the run.

        This is :meth:`__call__` without the shared-state convenience: the trace
        is returned rather than stashed, which is what concurrent callers want.

        Args:
            query: The input, bound to the ``query`` state variable.

        Returns:
            The trace, whose ``output`` field is what :meth:`__call__` returns.
        """
        return runtime.run_one(self, query, self.prompts())

    def run_batch(self, queries: list[str]) -> list[Trace]:
        """Run the MAS over several inputs, concurrently where the LLM allows.

        Args:
            queries: The inputs.

        Returns:
            One trace per query, in the order given.
        """
        prompts = self.prompts()
        traces = runtime.run_many(self, queries, [prompts] * len(queries))
        self.last_traces = tuple(traces)
        return traces

    def prompts(self) -> dict[str, str]:
        """Return the current prompt of every agent, keyed by name."""
        return {name: agent.prompt for name, agent in self.agents.items()}

    def predict(
        self,
        prompts: str | list[str],
        xs: list[str],
        system_prompts: str | list[str] | None = None,  # noqa: ARG002
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
            system_prompts: Accepted for interface compatibility. Agent prompts
                already carry their instructions, so this is not applied.

        Returns:
            A ``(predictions, sequences)`` pair, matching ``BasePredictor``.
        """
        if not xs:
            return [], []

        encoded = [prompts] * len(xs) if isinstance(prompts, str) else list(prompts)
        if len(encoded) != len(xs):
            raise ValueError(f"Got {len(encoded)} prompt sets for {len(xs)} inputs; they must be parallel.")

        names = list(self.agents)
        decoded = [decode_prompts(item, names) for item in encoded]
        traces = runtime.run_many(self, list(xs), decoded)
        self.last_traces = tuple(traces)

        preds = [trace.output for trace in traces]
        seqs = ["\n".join(f"{r.rendered_prompt}\n{r.raw_response}" for r in trace.records) for trace in traces]
        return preds, seqs

    def _extract_preds(self, preds: list[str]) -> list[str]:
        """Return predictions unchanged; promptolution declares this abstract and forces the name."""
        return list(preds)

    def encode(self) -> str:
        """Return this MAS's prompts as a single promptolution prompt string."""
        return encode_prompts(self.prompts())

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
        prompts = prompts or {}
        models = models or {}
        unknown = (set(prompts) | set(models)) - set(self.agents)
        if unknown:
            raise ValueError(f"copy() names unknown agents: {sorted(unknown)}.")
        new_agents = [
            agent.copy(prompt=prompts.get(name, agent.prompt), model=models.get(name, agent.model))
            for name, agent in self.agents.items()
        ]

        return MAS(
            agents=new_agents,
            edges=list(self.graph.edges) if edges is None else edges,
            llm=self.llm,
            llms=self.llms,
            max_steps=self.max_steps,
            default_model=self.default_model,
            tools=self.tools,
            max_tool_rounds=self.max_tool_rounds,
            max_visits=self.max_visits,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data spec for this MAS.

        Returns:
            Prompts, topology and model bindings, and nothing else. Models are
            recorded by name; the LLM objects are runtime state, not spec.
        """
        ordered = agent_names(self.graph)
        return {
            "version": 1,
            "agents": [self.agents[name].to_dict() for name in ordered],
            "edges": [[source, target] for source, target in self.graph.edges],
            "default_model": self.default_model,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        tools: Sequence[ToolSource | Tool] = (),
    ) -> MAS:
        """Rebuild a MAS from a spec.

        Args:
            data: A spec produced by :meth:`to_dict`.
            llm: Default model to attach.
            llms: Models by name, resolving the spec's bindings.
            tools: Tools and tool sources, resolving the agents' tool names.

        Returns:
            The reconstructed MAS.

        Raises:
            ValueError: If the spec is malformed or binds to an unsupplied model.
        """
        spec = validate_spec(data)
        agents = [Agent.from_dict(dict(item)) for item in spec["agents"]]
        edges = [(source, target) for source, target in spec.get("edges", [])]
        return cls(
            agents=agents,
            edges=edges,
            llm=llm,
            llms=llms,
            default_model=spec.get("default_model"),
            tools=tools,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Write the spec to a YAML file.

        Args:
            path: Destination file.
        """
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        tools: Sequence[ToolSource | Tool] = (),
    ) -> MAS:
        """Read a MAS spec from a YAML file.

        Args:
            path: Source file.
            llm: Default model to attach.
            llms: Models by name, resolving the spec's bindings.
            tools: Tools and tool sources, resolving the agents' tool names.

        Returns:
            The reconstructed MAS.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, llm=llm, llms=llms, tools=tools)

    @property
    def terminal(self) -> str:
        """Name of the agent whose output is the MAS output."""
        return terminal_agent(self.graph)

    def __repr__(self) -> str:
        """Return an unambiguous representation naming agents and edge count."""
        edges = sum(1 for source, target in self.graph.edges if source not in SENTINELS and target not in SENTINELS)
        return f"MAS(agents={agent_names(self.graph)!r}, edges={edges}, terminal={self.terminal!r})"

    def resolve_tools(self, agent: Agent) -> ToolIndex:
        """Return the tools an agent may call, keyed by name.

        Args:
            agent: The agent whose tools are wanted.

        Returns:
            Each of its tools with the source that runs it.

        Raises:
            ValueError: If a name is unknown, or the agent's model cannot call tools.
        """
        if self.tool_index is None:
            self.tool_index = index_tools(self.tools)
        return select_tools(self.tool_index, agent, self.resolve_llm(agent))

    def resolve_llm(self, agent: Agent) -> BaseLLM:
        """Return the model that serves an agent.

        Args:
            agent: The agent whose model is wanted.

        Returns:
            The agent's bound model, or the MAS default when it has none.

        Raises:
            ValueError: If neither a binding nor a default is available.
        """
        if agent.model is not None:
            return self.llms[agent.model]
        if self.llm is None:
            raise ValueError(f"Agent {agent.name!r} has no model binding and the MAS has no default llm.")
        return self.llm

    @property
    def structure(self) -> str:
        """What the compiled graph depends on, and what makes two MASs swappable.

        Two systems with the same structure differ only in prompts, so one can
        evaluate the other's prompt sets. An optimizer uses this to batch a whole
        population of prompt candidates through a single predictor.

        Prompts are deliberately absent: they travel in the run state, so editing
        one must not throw away a compilation. Anything else about an agent, its
        wiring or its model, changes the graph itself.
        """
        lines = [
            f"{agent.name} {','.join(agent.inputs)} -> {agent.output} model={agent.model} tools={','.join(agent.tools)}"
            for agent in sorted(self.agents.values(), key=lambda a: a.name)
        ]
        lines += [f"{source} -> {target}" for source, target in sorted(self.graph.edges)]
        return "\n".join(lines)

    @property
    def compiled(self) -> Any:
        """The compiled executable graph, rebuilt whenever the structure changes.

        The compiled object's type is an implementation detail and never appears
        in a public signature.
        """
        if self.compilation is None or self.compilation[0] != self.structure:
            self.compilation = (self.structure, runtime.compile_mas(self))
        return self.compilation[1]
