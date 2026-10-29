"""The MAS: a multi-agent system that is also a promptolution predictor."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from promptolution.predictors.base_predictor import BasePredictor

from xmas import bundle, examples, runtime
from xmas import verify as verification
from xmas.agent import Agent
from xmas.chains import ChainShape, ForEach
from xmas.check import MERGE_LAST, SystemInfo, check_system, code_maps
from xmas.config import Config, ModelEntry, build_llms, resolve_names, tool_sources
from xmas.edges import Edge, EdgeKind, dump_edges, parse_edges
from xmas.graph import (
    END,
    SENTINELS,
    START,
    agent_names,
    back_edges,
    build_graph,
    forward_graph,
    join_nodes,
    terminal_agent,
    validate_graph,
)
from xmas.health import HealthSettings
from xmas.human import Asker, HumanAction
from xmas.media import TEXT_ONLY, Capabilities
from xmas.nodes import Code, MASModule
from xmas.predicates import parse_predicate
from xmas.prompts import decode_prompts, encode_prompts
from xmas.spec import VERSION_2, validate_spec, validate_spec_v2
from xmas.store import Store, open_store
from xmas.tools import Tool, ToolIndex, ToolSource, index_tools, select_tools, tool
from xmas.values import TEXT, ValueType, parse_type

if TYPE_CHECKING:  # pragma: no cover
    import os
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    import networkx as nx
    from promptolution.llms.base_llm import BaseLLM

    from xmas.budget import Budget
    from xmas.trace import Event, Trace

__all__ = [
    "MAS",
]


QUERY_ONLY = {"query": TEXT}


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

    A system is version 1 or version 2. Version 2 adds code nodes (nodes that run a
    callable without a model), modules (nodes that run a whole system), human
    nodes (which ask a person), typed values, answer forms, declared inputs and
    conditional edges, and is checked by :func:`xmas.check.check_system` when
    it is built.

    Attributes:
        agents: Agents keyed by name. Mutable, so ``mas.agents["solver"].prompt``
            is a supported edit.
        code_nodes: Code nodes keyed by name; empty in version 1.
        modules: Modules keyed by name; empty in version 1.
        human_actions: Human actions keyed by name; empty in version 1.
        detached: Names of the detached nodes (each node's ``detached``),
            which no optimizer may change.
        health: The thresholds of the run's health signals, or ``None`` for
            the defaults.
        edges: The declared edges, in order, without the implicit wiring from
            :data:`~xmas.graph.START` and to :data:`~xmas.graph.END`.
        inputs: The values a run starts with, with their types; ``{query: text}``
            unless a version 2 system declares others.
        merges: ``{value: "last"}`` for values several nodes may write.
        each: ``{node: ForEach}`` for the nodes of ``for each`` chains.
        outputs: The declared outputs with their types; a run's
            ``Trace.outputs`` holds their final values.
        retry: ``{node: {error kind: attempts}}``: how often a failed visit
            runs again on each kind of :data:`~xmas.check.RETRYABLE`.
        prompt_names: ``{agent: prompt name}`` for agents that share one prompt.
        models: ``{model name: Capabilities}``: what each bound model takes and
            gives besides text, as declared, a :class:`~xmas.config.ModelEntry`
            where the name stands for another catalogue id or sets parameters;
            empty when none is (and in version 1).
        version: 1 or 2.
        info: What the version 2 check learned (types, writers, joins); ``None``
            in version 1.
        graph: The topology as a ``networkx.DiGraph`` over node names plus the
            :data:`~xmas.graph.START` and :data:`~xmas.graph.END` sentinels.
        llm: The default model, used by any agent without its own binding.
        llms: Additional models by name, which agents bind to via ``Agent.model``.
        default_model: Name recorded for the default model in a serialized spec.
        budget: Limits for each run, or ``None`` for none; never part of the spec.
        asker: How a run asks a person, or ``None``; never part of the spec.
        store: Where runs are logged and paused runs kept: a store, or
            ``None`` for the one :func:`xmas.store.open_store` opens on first
            use; never part of the spec.
        max_visits: How often one agent of a version 1 cycle may run in a single
            query before its back edges close.
        last_trace: Trace of the most recent :meth:`__call__`, or ``None`` before any.
        last_traces: Traces of the most recent :meth:`predict` or :meth:`run_batch`
            call, assigned once at the end of the batch; this is the seam an
            attributor reads after a task evaluation. One MAS instance serves one
            batch at a time; :meth:`run` shares no state at all.
        bundle: The ``.mas`` bundle this system was loaded from or last saved
            to, or ``None``; never part of the spec, never copied.
        unresolved: The model ids :meth:`load` found missing from the
            catalogue, each with the closest ids; while any is, every run
            refuses. Never part of the spec, never copied.
    """

    agents: dict[str, Agent]
    code_nodes: dict[str, Code]
    modules: dict[str, MASModule]
    human_actions: dict[str, HumanAction]
    health: HealthSettings | None
    edges: tuple[Edge, ...]
    inputs: dict[str, ValueType]
    merges: dict[str, str]
    each: dict[str, ForEach]
    outputs: dict[str, ValueType]
    retry: dict[str, dict[str, int]]
    prompt_names: dict[str, str]
    models: dict[str, Capabilities]
    version: int
    info: SystemInfo | None
    graph: nx.DiGraph[str]
    budget: Budget | None
    asker: Asker | None
    store: Store | str | Path | None
    bundle: Path | None
    unresolved: dict[str, tuple[str, ...]]

    def __init__(  # noqa: PLR0913, PLR0917  the public signature of versions 1 and 2, kept for their callers
        self,
        agents: list[Agent],
        edges: Sequence[tuple[str, str] | Edge] | None = None,
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        max_steps: int = 50,
        default_model: str | None = None,
        tools: Sequence[ToolSource | Tool] = (),
        max_tool_rounds: int = 5,
        max_visits: int = 3,
        budget: Budget | None = None,
        *,
        code_nodes: Sequence[Code] = (),
        inputs: Mapping[str, str | ValueType] | None = None,
        state: Mapping[str, Mapping[str, str]] | None = None,
        version: int | None = None,
        modules: Sequence[MASModule] = (),
        health: HealthSettings | None = None,
        each: Mapping[str, str | Mapping[str, Any] | ForEach] | None = None,
        outputs: Mapping[str, str | ValueType] | None = None,
        retry: Mapping[str, Mapping[str, int]] | None = None,
        prompt_names: Mapping[str, str] | None = None,
        models: Mapping[str, Mapping[str, Any] | Capabilities] | None = None,
        human_actions: Sequence[HumanAction] = (),
        asker: Asker | None = None,
        store: Store | str | Path | None = None,
    ) -> None:
        """Initialize a MAS.

        Args:
            agents: The agents, in any order.
            edges: ``(source, target)`` pairs between node names, or
                :class:`~xmas.edges.Edge` objects. Nodes left unconnected are
                wired from the source sentinel and to the terminal sentinel.
            llm: Default model for agents without their own binding.
            llms: Models by name, for per-agent bindings via ``Agent.model``.
            max_steps: Cap on executed steps per call, so a cyclic topology
                cannot run forever.
            default_model: Name to record for the default model when serializing;
                defaults to the model id ``llm`` reports.
            tools: Tools and tool sources (e.g. ``MCPTools``), mixed: the
                callables registry ``Agent.tools`` and ``Code.uses`` resolve in.
                Runtime state like ``llm``; never part of the spec.
            max_tool_rounds: Model turns per agent visit that may request tools,
                before one last turn with the tools withheld.
            max_visits: How often one agent may run in a single query before its
                version 1 back edges close; a ``while`` edge carries its own.
            budget: Limits for each run, and for each item of a batch; a run
                that reaches one ends with status ``over_budget``.
            code_nodes: Nodes that run a registered callable.
            inputs: The values a run starts with, ``{name: type}``; defaults to
                ``{"query": "text"}``.
            state: ``{value: {"merge": "last"}}`` for each value several nodes
                may write.
            version: 1 or 2. Defaults to 2 when code nodes, modules, human actions,
                conditional edges, a non-text agent, an agent reading a value under
                another name, ``inputs``, ``state``, a detached node, ``each``,
                ``outputs``, ``retry``, ``prompt_names`` or ``models`` are
                used, else 1.
            modules: Nodes that run a whole version 2 system.
            health: Thresholds of the health signals; ``None`` keeps the
                defaults.
            each: ``{node: form}`` for every node of a ``for each`` chain, the
                form a :class:`~xmas.chains.ForEach`, its text such as
                ``"page in plan.slides"``, or its spec mapping.
            outputs: ``{name: type}`` of the values a run returns; several
                nodes may then feed the end.
            retry: ``{node: {error kind: attempts}}`` for agents and code nodes.
            prompt_names: ``{agent: prompt name}``: the agents that share one
                prompt, which must hold the same text.
            models: ``{model name: {"accepts": [...], "produces": [...]}}``
                or :class:`~xmas.media.Capabilities`, keyed by ``llms`` names
                or ``default_model``: the kinds beyond text a model reads,
                such as ``image``, and answers with. A model not named here
                states its own through ``accepts`` and ``produces``
                attributes, else reads and writes text only.
            human_actions: Nodes that ask a person (:class:`~xmas.human.HumanAction`).
            asker: How a run asks a person (:class:`~xmas.human.Asker`);
                runtime state like ``llm``, never part of the spec.
            store: Where runs are logged and paused runs kept: a
                :class:`~xmas.store.Store`, a file path, or ``":memory:"``
                for a store in memory that logs nothing. ``None`` opens the
                configured file (:func:`~xmas.store.open_store`) when a run
                first needs it. Never part of the spec.

        Raises:
            ValueError: If the system is not runnable. A version 2 system
                raises once, listing every problem
                :func:`~xmas.check.check_system` finds.
        """
        self.edges = tuple(edge if isinstance(edge, Edge) else Edge(*edge) for edge in edges or ())
        self.health = health
        self.each = plain_each(each)
        self.outputs = typed(outputs or {})
        self.retry = {name: dict(rules) for name, rules in (retry or {}).items()}
        self.prompt_names = dict(prompt_names or {})
        self.models = plain_models(models)
        others: list[Code | MASModule | HumanAction] = [*code_nodes, *modules, *human_actions]
        self.version = self.infer_version(version, agents, others, inputs, state)
        refuse_unrunnable(self.version, agents, others, max_tool_rounds)
        super().__init__(llm=llm)  # type: ignore[arg-type]

        self.index_nodes(agents, code_nodes, modules, human_actions)
        self.llms = dict(llms or {})
        check_bindings(agents, self.llms)
        self.max_steps = max_steps
        self.max_visits = max_visits
        self.budget = budget
        self.asker = asker
        self.store = open_store(store) if isinstance(store, str | Path) else store
        self.default_model = default_model if default_model is not None else getattr(llm, "model_id", None)
        self.tools = tuple(tools)
        self.tool_index: ToolIndex | None = None  # built on first use, so tool-free systems never touch their sources
        self.max_tool_rounds = max_tool_rounds
        self.compilation: tuple[str, Any] | None = None
        self.last_trace: Trace | None = None
        self.last_traces: tuple[Trace, ...] = ()
        self.bundle = None
        self.unresolved = {}
        if self.version == 1:
            self.build_v1()
        else:
            self.build_v2(inputs, state)

    def index_nodes(
        self,
        agents: list[Agent],
        code_nodes: Sequence[Code],
        modules: Sequence[MASModule],
        human_actions: Sequence[HumanAction],
    ) -> None:
        """Key the nodes of each kind by name; one name for two nodes is refused."""
        nodes: list[Agent | Code | MASModule | HumanAction] = [*agents, *code_nodes, *modules, *human_actions]
        names = [node.name for node in nodes]
        if len(set(names)) != len(names):
            raise ValueError(f"Node names must be unique: {sorted({n for n in names if names.count(n) > 1})}.")
        self.agents = {agent.name: agent for agent in agents}
        self.code_nodes = {step.name: step for step in code_nodes}
        self.modules = {box.name: box for box in modules}
        self.human_actions = {human.name: human for human in human_actions}

    def infer_version(
        self,
        version: int | None,
        agents: list[Agent],
        nodes: Sequence[Code | MASModule | HumanAction],
        inputs: Mapping[str, str | ValueType] | None,
        state: Mapping[str, Mapping[str, str]] | None,
    ) -> int:
        """Return the version asked for, else 2 when a version 2 feature (``nodes`` among them) is used, else 1."""
        uses_v2 = (
            bool(nodes or self.each or self.outputs or self.retry or self.prompt_names)
            or any(node.detached for node in [*agents, *nodes])
            or bool(self.models)
            or inputs is not None
            or state is not None
            or any(edge.kind is not EdgeKind.PLAIN for edge in self.edges)
            or any(agent.output_type != "text" or agent.renamed for agent in agents)
        )
        inferred = version if version is not None else 2 if uses_v2 else 1
        if inferred not in (1, 2):
            raise ValueError(f"version must be 1 or 2, not {inferred!r}.")
        if inferred == 1 and uses_v2:
            raise ValueError(
                "Code nodes, modules, human actions, inputs, outputs, state, detached nodes, models, for each, retry, "
                "shared prompts, read maps, conditional edges and typed writes need version 2."
            )
        return inferred

    def build_v1(self) -> None:
        """Build and check a version 1 topology, and resolve every agent's tools, so it fails now, not mid-run."""
        self.inputs, self.merges, self.info = dict(QUERY_ONLY), {}, None
        self.graph = build_graph(list(self.agents), [(edge.source, edge.target) for edge in self.edges])
        validate_graph(self.graph)
        # Two terminal agents only break the run once it produces an output, after a batch of model calls.
        terminal_agent(self.graph)
        for agent in self.agents.values():
            if agent.tools:
                self.resolve_tools(agent)

    def build_v2(
        self, inputs: Mapping[str, str | ValueType] | None, state: Mapping[str, Mapping[str, str]] | None
    ) -> None:
        """Keep a version 2 system's inputs and merges, then check it whole, so it fails now, not mid-run."""
        self.inputs = typed(inputs if inputs is not None else QUERY_ONLY)
        for value, entry in (state or {}).items():
            if entry != {"merge": MERGE_LAST}:
                raise ValueError(f"state[{value!r}] must be {{'merge': 'last'}}, got {entry!r}.")
        self.merges = dict.fromkeys(state or {}, MERGE_LAST)
        self.info = check_system(self)  # it checks the topology itself, so a bad edge is one problem of the list
        self.graph = self.info.graph

    def chain_system(self, shape: ChainShape, inputs: Mapping[str, ValueType], written: Mapping[str, ValueType]) -> MAS:
        """Build the system one item of a chain runs; :func:`~xmas.check.check_system` asks for it."""
        return ChainSystem(self, shape, inputs, written)

    def checked(self, refusal: str = "A version 1 system is not checked.") -> SystemInfo:
        """Return what the check learned, which only a version 2 system has; ``refusal`` is the ValueError's text."""
        if self.info is None:
            raise ValueError(refusal)
        return self.info

    def runs(self) -> Store:
        """Return the run store, opening the configured one (:func:`~xmas.store.open_store`) on first use.

        A path set as ``store`` after construction opens there.
        """
        if self.store is None or isinstance(self.store, str | Path):
            self.store = open_store(None if self.store is None else str(self.store))
        return self.store

    def callables(self) -> ToolIndex:
        """Return the callables registry: every tool and tool source, indexed on first use."""
        if self.tool_index is None:
            self.tool_index = index_tools(self.tools)
        return self.tool_index

    def resolve_tools(self, agent: Agent) -> ToolIndex:
        """Return the tools an agent may call, each with the source that runs it.

        Raises:
            ValueError: If a name is unknown, or the agent's model cannot call tools.
        """
        llm = self.llms.get(agent.model) if agent.model is not None else self.llm
        return select_tools(self.callables(), agent, llm)

    def resolve_callable(self, step: Code) -> tuple[Tool, ToolSource]:
        """Return the callable a code node uses, with the source that runs it, from the registry ``Agent.tools`` use."""
        index = self.callables()
        if step.uses not in index:
            raise ValueError(f"Code {step.name!r} uses unknown callable {step.uses!r}; available: {sorted(index)}.")
        return index[step.uses]

    @property
    def nodes(self) -> dict[str, Agent | Code | MASModule | HumanAction]:
        """Every node (agents, code nodes, modules, human actions), keyed by name in the topological order of a pass."""
        everything: dict[str, Agent | Code | MASModule | HumanAction] = {
            **self.agents,
            **self.code_nodes,
            **self.modules,
            **self.human_actions,
        }
        return {name: everything[name] for name in agent_names(forward_graph(self.graph, self.while_edges))}

    @property
    def detached(self) -> frozenset[str]:
        """Names of the detached nodes, which no optimizer may change."""
        everything: dict[str, Agent | Code | MASModule | HumanAction] = {
            **self.agents,
            **self.code_nodes,
            **self.modules,
            **self.human_actions,
        }
        return frozenset(name for name, node in everything.items() if node.detached)

    @property
    def while_edges(self) -> tuple[tuple[str, str], ...]:
        """The ``(source, target)`` of every ``while`` edge, in declared order: the loops of a version 2 run."""
        return tuple((edge.source, edge.target) for edge in self.edges if edge.kind is EdgeKind.WHILE)

    def capabilities(self, agent: Agent) -> Capabilities:
        """Return what an agent's model takes and gives besides text.

        The first found wins: the ``models`` entry of the model's name (the
        agent's binding, else ``default_model``), the model object's own
        ``accepts`` and ``produces``, else text only.
        """
        name = agent.model if agent.model is not None else self.default_model
        if name is not None and name in self.models:
            return self.models[name]
        llm = self.llms.get(agent.model) if agent.model is not None else self.llm
        stated = [getattr(llm, key, ()) for key in ("accepts", "produces")]
        if all(isinstance(words, list | tuple) and all(isinstance(w, str) for w in words) for words in stated):
            return Capabilities(*(tuple(words) for words in stated))
        return TEXT_ONLY

    def resolve_llm(self, agent: Agent) -> BaseLLM:
        """Return the agent's bound model, or the MAS default; ``ValueError`` when there is neither."""
        if agent.model is not None:
            return self.llms[agent.model]
        if self.llm is None:
            raise ValueError(f"Agent {agent.name!r} has no model binding and the MAS has no default llm.")
        return self.llm

    def __call__(self, query: str) -> str:
        """Run the MAS on one input and return its output; a failed run raises ``RuntimeError``."""
        trace = self.run(query)
        self.last_trace = trace
        if trace.error is not None:
            raise RuntimeError(f"MAS run failed: {trace.error}")
        return trace.output

    def check_resolved(self) -> None:
        """Refuse to run while a bound model name is not in the catalogue, before any call."""
        if self.unresolved:
            named = "; ".join(
                f"{name!r} (closest: {', '.join(near) or 'none'})" for name, near in self.unresolved.items()
            )
            raise ValueError(
                f"Model names not in the catalogue: {named}. Set the name in models: or fix the catalogue; "
                "verify(1) rechecks."
            )

    def check_runs_on_query(self) -> None:
        """Refuse to run a system whose inputs are not the one ``query`` text."""
        if self.inputs != QUERY_ONLY:
            declared = {name: str(vtype) for name, vtype in self.inputs.items()}
            raise ValueError(
                f"This system's inputs are {declared}, not {{'query': 'text'}}, and run(query) only runs on one "
                "text query; use it as a module. run_state(values) runs it on its inputs."
            )

    def run(self, query: str) -> Trace:
        """Run the MAS on one input and return the trace of the run.

        This is :meth:`__call__` without the shared state: the trace is returned
        rather than stashed, which is what concurrent callers want. A system
        whose inputs are not ``{query: text}`` raises ``ValueError``, and so
        does a system with :attr:`unresolved` model names.
        """
        self.check_resolved()
        self.check_runs_on_query()
        return runtime.run_many(self, [query], [self.prompts()])[0]

    def run_state(
        self,
        values: Mapping[str, Any],
        *,
        on_event: Callable[[Event], None] | None = None,
        run_id: str | None = None,
    ) -> Trace:
        """Run the system on its inputs, given by name, and return the trace of the run.

        A version 2 system takes exactly its inputs, each a value of its type,
        and its trace's ``outputs`` hold the declared outputs. A version 1
        system takes text values, ``query`` among them, and runs as :meth:`run`
        runs. ``on_event`` hears every visit's and item's start and end; it is
        called on the run's threads, so it must be quick and thread-safe. A
        failure inside the run is a failed trace. ``run_id`` names the run, for
        :meth:`stop`, :meth:`answer` and :meth:`resume`; a new random id when
        ``None``. A system with people in it returns a ``paused`` trace when a
        question is still open once every other branch has ended.

        Raises:
            ValueError: Listing every missing, extra or mistyped input, before
                any call; when ``run_id`` names a run running or kept; or for
                :attr:`unresolved` model names.
        """
        self.check_resolved()
        return runtime.run_state(self, values, self.prompts(), on_event, run_id)

    def resume(
        self,
        run_id: str,
        answers: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        on_event: Callable[[Event], None] | None = None,
    ) -> Trace:
        """Continue a paused run from its store, after keeping ``answers`` (``{question id: answer}``).

        It runs on in this process or a new one, with the prompts it paused
        with and its budget counting on; a question past its deadline takes
        its default, and the run pauses again on a question still open.

        Raises:
            ValueError: If the store has no such run, the run is not paused,
                this system is not the one that paused, an answer does not fit,
                no open question is answered or due, or a model name is
                :attr:`unresolved`.
        """
        self.check_resolved()
        return runtime.resume(self, run_id, answers, on_event)

    def answer(self, run_id: str, answers: Mapping[str, Mapping[str, Any]]) -> None:
        """Keep answers to a run's questions, ``{question id: answer}``, while it runs or waits.

        Raises:
            ValueError: If the run is stopped, or listing every answer that is
                not for an open question of the run or does not fit; none is kept then.
        """
        runtime.answer(self, run_id, answers)

    def stop(self, run_id: str, reason: str = "stopped") -> bool:
        """Stop a run by its id; ``False`` when no such run is running or paused.

        A running run starts no call, code node or item after this, and its calls in
        flight are cut; it returns a ``stopped`` trace. A paused run is marked
        stopped in the store, its open questions withdrawn through the asker.
        """
        return runtime.stop(self, run_id, reason)

    def run_batch(self, queries: list[str]) -> list[Trace]:
        """Run the MAS over several inputs, concurrently where the LLM allows; one trace per query, in order."""
        self.check_resolved()
        self.check_runs_on_query()
        prompts = self.prompts()
        traces = runtime.run_many(self, queries, [prompts] * len(queries))
        self.last_traces = tuple(traces)
        return traces

    def prompts(self) -> dict[str, str]:
        """Return the current prompt of every agent, keyed by name."""
        return {name: agent.prompt for name, agent in self.agents.items()}

    def spread(self, prompts: Mapping[str, str]) -> dict[str, str]:
        """Return ``prompts``, keyed by agent, with each text also set on the agents sharing its prompt.

        An agent named in ``prompts`` keeps its own text, so a whole prompt set
        passes unchanged.
        """
        spread = dict(prompts)
        for name, text in prompts.items():
            shared = self.prompt_names.get(name)
            spread.update(
                {other: text for other, key in self.prompt_names.items() if key == shared and other not in prompts}
            )
        return spread

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

        ``system_prompts`` is accepted for the interface and not applied: agent
        prompts carry their instructions. Returns ``(predictions, sequences)``,
        matching ``BasePredictor``; a failed rollout's sequence is empty.
        """
        self.check_resolved()
        if not xs:
            return [], []
        self.check_runs_on_query()

        encoded = [prompts] * len(xs) if isinstance(prompts, str) else list(prompts)
        if len(encoded) != len(xs):
            raise ValueError(f"Got {len(encoded)} prompt sets for {len(xs)} inputs; they must be parallel.")

        names = list(self.agents)
        decoded = [decode_prompts(item, names) for item in encoded]
        traces = runtime.run_many(self, list(xs), decoded)
        self.last_traces = tuple(traces)

        preds = [trace.output for trace in traces]
        # A failed rollout keeps its finished records, but promptolution gets an empty sequence for it.
        seqs = [
            "" if trace.failed else "\n".join(f"{r.rendered_prompt}\n{r.raw_response}" for r in trace.records)
            for trace in traces
        ]
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
        edges: Sequence[tuple[str, str] | Edge] | None = None,
        models: dict[str, str | None] | None = None,
    ) -> MAS:
        """Derive a new MAS with some components replaced.

        This is how an optimizer proposes a candidate. Models, the asker and
        the store are shared with the original by reference, and the budget is
        carried over; everything else is copied, so the original is never
        touched.

        ``prompts`` and ``models`` are keyed by agent name; a prompt given to
        one agent of a shared prompt reaches every agent sharing it. A
        ``None`` model unbinds an agent back to the MAS default. ``edges``
        replaces the edge set entirely. A key naming an unknown agent, or a
        result that is not runnable, raises ``ValueError``.
        """
        prompts = dict(prompts or {})
        models = models or {}
        unknown = (set(prompts) | set(models)) - set(self.agents)
        if unknown:
            raise ValueError(f"copy() names unknown agents: {sorted(unknown)}.")
        prompts = self.spread(prompts)
        new_agents = [
            agent.copy(prompt=prompts.get(name, agent.prompt), model=models.get(name, agent.model))
            for name, agent in self.agents.items()
        ]

        if edges is None:
            edges = list(self.graph.edges) if self.version == 1 else list(self.edges)
        return MAS(
            agents=new_agents,
            edges=edges,
            llm=self.llm,
            llms=self.llms,
            max_steps=self.max_steps,
            default_model=self.default_model,
            tools=self.tools,
            max_tool_rounds=self.max_tool_rounds,
            max_visits=self.max_visits,
            budget=self.budget,
            asker=self.asker,
            store=self.store,
            **self.v2_kwargs(),
        )

    def v2_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments that rebuild this system's version 2 fields.

        :meth:`copy` and every optimizer that constructs a ``MAS(...)`` from an
        existing one pass ``**mas.v2_kwargs()``, so a field added here reaches
        all of them.
        """
        v2 = self.version == VERSION_2
        return {
            "code_nodes": tuple(self.code_nodes.values()),
            "inputs": dict(self.inputs) if v2 else None,
            "state": {value: {"merge": rule} for value, rule in self.merges.items()} if v2 else None,
            "version": self.version,
            "modules": tuple(self.modules.values()),
            "health": self.health,
            "each": dict(self.each),
            "outputs": dict(self.outputs),
            "retry": {name: dict(rules) for name, rules in self.retry.items()},
            "prompt_names": dict(self.prompt_names),
            "models": dict(self.models),
            "human_actions": tuple(self.human_actions.values()),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data spec for this MAS: prompts, topology and model bindings by name.

        The LLM objects, tools and budget are runtime state, not spec. A
        version 2 spec is canonical: sections in :data:`~xmas.spec.V2_KEYS`
        order, empty or default sections left out, edges as
        :func:`~xmas.edges.dump_edges` writes them, prompts in a top-level
        ``prompts`` section, and a ``callables`` entry for every callable a node
        uses.
        """
        if self.version == 1:
            return {
                "version": 1,
                "agents": [self.agents[name].to_dict() for name in agent_names(self.graph)],
                "edges": [[source, target] for source, target in self.graph.edges],
                "default_model": self.default_model,
            }
        sections: dict[str, Any] = {
            "agents": [
                self.spec_entry(name, agent_spec(agent, self.prompt_names.get(name)))
                for name, agent in self.agents.items()
            ],
            "code_nodes": [self.spec_entry(name, step.to_dict()) for name, step in self.code_nodes.items()],
            "modules": [self.spec_entry(name, box.to_dict()) for name, box in self.modules.items()],
            "human_actions": [self.spec_entry(name, human.to_dict()) for name, human in self.human_actions.items()],
            "edges": dump_edges(self.edges),
            "state": {value: {"merge": rule} for value, rule in self.merges.items()},
            "health": health_spec(self.health),
            "models": {name: capabilities.to_spec() for name, capabilities in self.models.items()},
            "callables": self.callable_specs(),
            "default_model": self.default_model,
            "prompts": self.prompt_section(),
        }
        inputs = {} if self.inputs == QUERY_ONLY else {"inputs": type_words(self.inputs)}
        outputs = {"outputs": type_words(self.outputs)} if self.outputs else {}
        return {
            "version": 2,
            **inputs,
            **outputs,
            **{key: value for key, value in sections.items() if value not in (None, [], {})},
        }

    def spec_entry(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Return a node's spec entry with its ``for each`` after the keys that name it and its ``retry`` last."""
        items = list(data.items())
        named = sum(key in ("name", "model", "prompt", "uses", "form") for key in data)
        if name in self.each:
            items.insert(named, ("for each", self.each[name].to_spec()))
        if name in self.retry:
            items.append(("retry", dict(self.retry[name])))
        return dict(items)

    def prompt_section(self) -> dict[str, str]:
        """Return the ``prompts`` section: one text per agent, or per shared prompt name.

        Raises:
            ValueError: If agents that share a prompt hold different texts.
        """
        prompts: dict[str, str] = {}
        for name, agent in self.agents.items():
            key = self.prompt_names.get(name, name)
            if prompts.setdefault(key, agent.prompt) != agent.prompt:
                raise ValueError(
                    f"The agents sharing the prompt {key!r} hold different texts; copy(prompts=...) sets one on all."
                )
        return prompts

    def callable_specs(self) -> dict[str, dict[str, Any]]:
        """Return the ``callables`` section: every callable a node or an edge condition uses, code nodes' first."""
        index = self.callables()
        used = [step.uses for step in self.code_nodes.values()] + [
            t for agent in self.agents.values() for t in agent.tools
        ]
        used += sorted({name for edge in self.edges if edge.when for name, _ in parse_predicate(edge.when).calls()})
        specs: dict[str, dict[str, Any]] = {}
        for name in dict.fromkeys(used):
            item, source = index[name]
            if item.fn is not None:
                module = getattr(item.fn, "__module__", "?")
                entry: dict[str, Any] = {
                    "kind": "python",
                    "import": f"{module}:{getattr(item.fn, '__qualname__', name)}",
                }
            else:
                entry = {"kind": "mcp"}
                server = getattr(source, "name", None)
                if isinstance(server, str):
                    entry["server"] = server
            if item.params:
                entry["params"] = dict(item.params)
            reads = type_words(item.reads)
            if reads:
                entry["reads"] = reads
            entry["writes"] = type_words(item.writes)
            specs[name] = entry
        return specs

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        tools: Sequence[ToolSource | Tool] = (),
    ) -> MAS:
        """Rebuild a MAS from a spec.

        A spec with ``version: 2`` is read as version 2; any other goes through
        :func:`~xmas.spec.validate_spec` as version 1.

        ``llm``, ``llms`` and ``tools`` resolve the spec's model bindings, the
        agents' tool names and the code nodes' callables.

        Raises:
            ValueError: If the spec is malformed, binds to an unsupplied model,
                or (version 2) its ``callables`` do not match those supplied.
        """
        if data.get("version") == VERSION_2:
            return cls.from_v2(data, llm=llm, llms=llms, tools=tools)
        spec = validate_spec(data)
        return cls(
            agents=[Agent.from_dict(dict(item)) for item in spec["agents"]],
            edges=[(source, target) for source, target in spec.get("edges", [])],
            llm=llm,
            llms=llms,
            default_model=spec.get("default_model"),
            tools=tools,
        )

    @classmethod
    def from_v2(
        cls,
        data: dict[str, Any],
        llm: BaseLLM | None,
        llms: dict[str, BaseLLM] | None,
        tools: Sequence[ToolSource | Tool],
    ) -> MAS:
        """Rebuild a version 2 MAS and check its ``callables`` against those supplied."""
        spec: dict[str, Any] = dict(validate_spec_v2(data))
        prompts = spec.get("prompts", {})
        entries = [
            *spec.get("agents", []),
            *spec.get("code_nodes", []),
            *spec.get("modules", []),
            *spec.get("human_actions", []),
        ]
        agents = []
        for item in spec.get("agents", []):
            ((output, output_type),) = item["writes"].items()
            settings = {key: item.get(key) for key in ("tools", "model", "schema")}
            text = prompts[item.get("prompt", item["name"])]
            detached = item.get("detached", False)
            agents.append(
                Agent(
                    item["name"],
                    text,
                    output,
                    item.get("reads"),
                    output_type=output_type,
                    detached=detached,
                    **settings,
                )
            )
        mas = cls(
            agents=agents,
            edges=parse_edges(spec.get("edges", [])),
            llm=llm,
            llms=llms,
            default_model=spec.get("default_model"),
            tools=tools,
            code_nodes=[
                Code(
                    item["name"],
                    item["uses"],
                    item.get("reads", {}),
                    item.get("writes", {}),
                    item.get("detached", False),
                )
                for item in spec.get("code_nodes", [])
            ],
            inputs=spec.get("inputs"),
            state=spec.get("state"),
            version=2,
            modules=[
                MASModule(
                    item["name"],
                    cls.from_v2(
                        item["system"], (llms or {}).get(item["system"].get("default_model"), llm), llms, tools
                    ),
                    item["reads"],
                    item["writes"],
                    item.get("detached", False),
                )
                for item in spec.get("modules", [])
            ],
            health=health_settings(spec["health"]) if "health" in spec else None,
            each={item["name"]: item["for each"] for item in entries if "for each" in item},
            outputs=spec.get("outputs"),
            retry={item["name"]: item["retry"] for item in entries if "retry" in item},
            prompt_names={item["name"]: item["prompt"] for item in spec.get("agents", []) if "prompt" in item},
            models=spec.get("models"),
            human_actions=[HumanAction.from_dict(item) for item in spec.get("human_actions", [])],
        )
        mas.check_callables(spec.get("callables", {}))
        return mas

    def check_callables(self, declared: Mapping[str, Mapping[str, Any]]) -> None:
        """Check a spec's ``callables`` section against the callables this system resolved.

        Raises:
            ValueError: Listing each callable the spec names that no node uses,
                or whose reads or writes differ from the supplied callable's.
        """
        actual = self.callable_specs()
        problems = []
        for name, entry in declared.items():
            if name not in actual:
                problems.append(f"callables lists {name!r}, which no node uses")
                continue
            for key in ("reads", "writes"):
                want = {value: str(parse_type(word)) for value, word in entry.get(key, {}).items()}
                got = actual[name].get(key, {})
                if want != got:
                    problems.append(f"callable {name!r} {key} {want} in the spec, but the one supplied {key} {got}")
        if problems:
            lines = "\n".join(f"- {problem}" for problem in problems)
            raise ValueError(f"The spec's callables do not match the callables supplied:\n{lines}")

    def to_yaml(self, path: str | Path) -> None:
        """Write the spec to a YAML file.

        Raises:
            ValueError: For a version 2 system, which :meth:`save` writes as a
                ``.mas`` bundle; :meth:`to_dict` gives its spec.
        """
        if self.version == VERSION_2:
            raise ValueError(
                "A version 2 system is not saved as one YAML file: it is saved as the .mas bundle "
                "(name.mas/): save(path) writes it; to_dict() gives its spec."
            )
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
        """Read a MAS spec from a YAML file, as :meth:`from_dict` reads it."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, llm=llm, llms=llms, tools=tools)

    def save(self, path: str | os.PathLike[str], *, zipped: bool = False) -> Path:
        """Save the system as a ``.mas`` bundle, a folder or (``zipped``) one zip file, and return its path.

        A version 1 system is saved as a bundle too; :meth:`to_yaml` keeps its
        one file. Into an existing folder only the files whose parsed content
        changed are written, files the system no longer has are deleted, and
        ``examples/`` and ``optimize.yaml`` are never touched; the lock is
        written last. Sets :attr:`bundle`. See :mod:`xmas.bundle`.

        Raises:
            ValueError: If the name does not end in ``.mas``, the folder is not
                a bundle, or a prompt or module name cannot be a file name.
        """
        self.bundle = bundle.write(path, bundle.files_of(self), bundle.code_hashes(self), zipped=zipped)
        return self.bundle

    def verify(  # noqa: PLR0913  the public signature: one keyword per option of xmas.verify.verify
        self,
        level: int = 1,
        *,
        budget: Budget | float | None = None,
        strict: bool = False,
        only: Iterable[str] = (),
        examples: str | os.PathLike[str] | None = None,
        config: Config | None = None,
    ) -> verification.VerifyReport:
        """Verify the system up to ``level`` and return the report; see :func:`xmas.verify.verify`.

        Level 0 compares the bundle with its lock, level 1 checks every part
        without a model call, level 2 runs the warm-up examples under a small
        budget. A report that is not ok warns once, or raises
        :class:`~xmas.verify.VerifyFailed` when ``strict``.
        """
        where = None if examples is None else Path(examples)
        return verification.verify(self, level, budget=budget, strict=strict, only=only, examples=where, config=config)

    def save_examples(
        self,
        trace: Trace,
        *,
        nodes: Sequence[str] | None = None,
        file: str = "warmup",
        path: str | os.PathLike[str] | None = None,
    ) -> list[str]:
        """Record one warm-up case per node from a run's trace into the bundle; see :func:`xmas.examples.save_examples`.

        Returns:
            The names of the cases written to ``examples/<file>.yaml``.
        """
        return examples.save_examples(self, trace, nodes=nodes, file=file, path=None if path is None else Path(path))

    @classmethod
    def load(  # noqa: PLR0913  the public signature: one keyword per loading option
        cls,
        path: str | os.PathLike[str],
        *,
        llm: BaseLLM | None = None,
        llms: Mapping[str, BaseLLM] | None = None,
        tools: Sequence[ToolSource | Tool] = (),
        config: Config | None = None,
        overlay: Mapping[str, str] | None = None,
        resolve: bool = True,
        verify: int | None = None,
        trust: bool = False,
    ) -> MAS:
        """Load a ``.mas`` bundle, a folder or a zip, and build the system it holds.

        A callable offered by ``tools`` is used as given; otherwise a python
        callable is found by its ``import`` path and bound to its ``params``,
        and a remote one reaches the tool server its ``server`` names in the
        configuration. A bundle may name any code, so without ``trust`` load
        imports nothing: a python callable must be a :class:`~xmas.tools.Tool`
        in a module already imported. ``overlay`` (``{agent: model name}``) rebinds agents
        before anything is built. With ``resolve``, every bound model id is
        checked against the configured catalogue: a miss is a
        :class:`~xmas.config.ModelNameWarning` naming the closest ids, and the
        system keeps it in :attr:`unresolved` and refuses to run. A bound
        name no ``llm`` or ``llms`` argument gives gets a model on the
        configured endpoint (:func:`~xmas.config.build_llms`).

        Args:
            path: The bundle.
            llm: The default model.
            llms: Models by the names agents bind.
            tools: Tools and tool sources, which win over the bundle's callables of their names.
            config: Where tool servers and the catalogue come from; :meth:`Config.load` when ``None``.
            overlay: Model names by agent, replacing the bundle's.
            resolve: Whether model names are checked against the catalogue.
            verify: A level to :meth:`verify` the loaded system at before it is returned.
            trust: Import the modules the callables name and wrap plain functions, for a bundle you trust.

        Raises:
            ValueError: If the bundle is malformed or names what cannot be found,
                or the system it holds is not runnable.
        """
        files = bundle.read(path)
        spec = bundle.spec_of(files)
        bundle.read_optimize(files)
        settings = config

        def configured() -> Config:
            nonlocal settings
            if settings is None:
                settings = Config.load()
            return settings

        systems = list(spec_systems(spec))
        for system in systems:  # the shape first, so a malformed entry is named before any import
            (validate_spec_v2 if system.get("version") == VERSION_2 else validate_spec)(dict(system))
        if overlay:
            apply_overlay(spec, overlay)
        sources = [*tools, *callable_sources(systems, tools, configured, trust=trust)]
        given = dict(llms or {})
        default_model = spec.get("default_model")
        # each system's default, a module's too, is built unless llm= stands for them all
        missing = sorted(bound_models(systems, with_defaults=llm is None) - set(given))
        if missing:
            given.update(build_llms(missing, declared_models(systems), configured()))
        default = given.get(default_model) if llm is None and default_model is not None else llm
        mas = cls.from_dict(spec, llm=default, llms=given, tools=sources)
        mas.bundle = Path(path)
        if resolve:
            mas.resolve(configured())
        if verify is not None:
            mas.verify(verify, config=settings)
        return mas

    def resolve(self, config: Config) -> None:
        """Check every bound model id against the catalogue, warn per miss, and keep the misses in ``unresolved``."""
        bound: dict[str, list[str]] = {}
        for system in built_systems(self):
            for agent in system.agents.values():
                name = agent.model if agent.model is not None else system.default_model
                if name is None:
                    continue
                entry = system.models.get(name)
                ident = entry.model if isinstance(entry, ModelEntry) and entry.model else name
                bound.setdefault(ident, []).append(agent.name)
        labels = {ident: ("agent " if len(who) == 1 else "agents ") + ", ".join(who) for ident, who in bound.items()}
        resolution = resolve_names(labels, config)
        for warning in resolution.warnings:
            warnings.warn(warning, stacklevel=3)
        self.unresolved = dict(resolution.suggestions)

    def describe(self) -> dict[str, Any]:
        """Return the system as plain data, for drawing it and naming it.

        Every node with its kind, model, callable or inner system (a module's own
        description), typed reads and writes, and whether it is detached; every
        edge with its condition, the implicit wiring marked ``"implicit":
        True``; ``"merge"`` or ``"join"`` for each node with several incoming
        edges; and an ``id``, the SHA-256 of the canonical JSON of the rest.
        Prompts are not part of it, so editing one keeps the id. A version 1
        system's ``<continue>`` back edges are ``"loop"`` edges.

        A chain node gains its ``"for each"``, an agent reading a value under
        another name its ``"placeholders"`` map, a node that retries its
        ``"retry"``. A human action has its ``"form"``, and its ``"timeout_s"``
        when it times out. Declared ``"outputs"`` and the chains, as ``"groups"``,
        appear only when there are any; a chain node's reads and writes have
        their types for one item. Declared ``"models"`` appear when there
        are any, outside the ``id``: they change no compiled graph.
        """
        declared = {(edge.source, edge.target): edge for edge in self.edges}
        loops = back_edges(self.graph) if self.info is None else set()
        wiring = [pair for pair in self.graph.edges if pair not in declared]
        # Wiring from START first, then the declared edges in their order, then wiring to END.
        ordered = sorted([*declared, *wiring], key=lambda pair: (pair[0] != START) + (pair[1] == END))
        if self.info is not None:
            joins = {name: join.kind for name, join in self.info.joins.items()}
        else:
            joins = dict.fromkeys(sorted(join_nodes(self.graph)), "join")
        description: dict[str, Any] = {
            "version": self.version,
            "inputs": type_words(self.inputs),
            "nodes": [self.describe_node(name, node) for name, node in self.nodes.items()],
            "edges": [self.describe_edge(pair, declared.get(pair), loop=pair in loops) for pair in ordered],
            "joins": joins,
        }
        if self.outputs:
            description["outputs"] = type_words(self.outputs)
        chains = self.info.chains if self.info is not None else {}
        if chains:
            description["groups"] = [
                {"name": head, "for each": str(chain.each), "nodes": [n for n in self.nodes if n in chain.nodes]}
                | chain.each.options
                for head, chain in chains.items()
            ]
        canonical = json.dumps(description, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if self.models:
            description["models"] = {name: capabilities.to_spec() for name, capabilities in self.models.items()}
        description["id"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return description

    def describe_node(self, name: str, node: Agent | Code | MASModule | HumanAction) -> dict[str, Any]:
        """Return one node of :meth:`describe`."""
        info = self.info
        chain = next((c for c in info.chains.values() if name in c.nodes), None) if info is not None else None
        types = chain.system.checked().types if chain is not None else info.types if info is not None else {}

        def typed(names: Iterable[str]) -> dict[str, str]:
            return {name: str(types.get(name, TEXT)) for name in names}

        if isinstance(node, Agent):
            entry: dict[str, Any] = {
                "name": name,
                "kind": "agent",
                "model": node.model if node.model is not None else self.default_model,
                "reads": typed(node.inputs),
                "writes": {node.output: str(parse_type(node.output_type))},
                "tools": list(node.tools),
                "detached": node.detached,
            }
            if node.schema is not None:
                entry["schema"] = deepcopy(node.schema)
        elif isinstance(node, HumanAction):
            entry = {"name": name, "kind": "human_action", "form": node.form, "reads": typed(node.reads.values())}
            entry["writes"] = typed(node.written)
            if node.timeout_s is not None:
                entry["timeout_s"] = node.timeout_s
        elif isinstance(node, MASModule):
            entry = {"name": name, "kind": "module", "reads": typed(node.reads.values())}
            entry.update(writes=typed(node.writes.values()), system=node.system.describe())
        else:
            reads, writes = code_maps(node, self.resolve_callable(node)[0])
            entry = {"name": name, "kind": "code", "uses": node.uses, "reads": typed(reads.values())}
            entry["writes"] = typed(writes.values())
        entry.update(self.node_settings(name, node))
        return entry

    def node_settings(self, name: str, node: Agent | Code | MASModule | HumanAction) -> dict[str, Any]:
        """Return what :meth:`describe_node` adds for any kind: detached, its chain form, read renames, retries."""
        settings: dict[str, Any] = {}
        if node.detached:
            settings["detached"] = True
        if name in self.each:
            settings["for each"] = str(self.each[name])
        if isinstance(node, Agent) and node.renamed:
            settings["placeholders"] = dict(node.reads)
        if name in self.retry:
            settings["retry"] = dict(self.retry[name])
        return settings

    def describe_edge(self, pair: tuple[str, str], edge: Edge | None, *, loop: bool) -> dict[str, Any]:
        """Return one edge of :meth:`describe`: ``edge`` when declared, ``loop`` for a version 1 back edge."""
        source, target = pair
        item: dict[str, Any] = {"from": source, "to": target, "kind": EdgeKind.PLAIN.value}
        if loop:
            item.update(kind="loop", when=f"{runtime.CONTINUE_MARKER} in output", max_visits=self.max_visits)
        elif edge is not None:
            item["kind"] = edge.kind.value
            fields = {"when": edge.when, "max_visits": edge.max_visits, "switch": edge.switch, "case": edge.case}
            item.update({key: value for key, value in fields.items() if value is not None})
            if edge.errors:
                item["errors"] = list(edge.errors)
        if self.info is not None:
            implicit = edge is None
        else:
            implicit = (source == START and self.graph.in_degree(target) == 1) or (
                target == END and self.graph.out_degree(source) == 1
            )
        if implicit:
            item["implicit"] = True
        return item

    @property
    def structure(self) -> str:
        """What the compiled graph depends on, and what makes two MASs swappable.

        Two systems with the same structure differ only in prompts, so one can
        evaluate the other's prompt sets; an optimizer batches a population of
        prompt candidates through one predictor this way. Prompts travel in the
        run state, so they are absent. A version 2 system adds lines for its
        code nodes, edges, answer types and forms, inputs, merges, outputs, read
        maps, chain forms with ``at once`` and ``key``, retries, human actions
        (every setting but the question text) and modules (each with its inner
        structure, indented); detached nodes, health settings, prompt names,
        item budgets, models and questions change no compiled graph and stay
        out.
        """
        lines = [
            f"{agent.name} {','.join(agent.inputs)} -> {agent.output} model={agent.model} tools={','.join(agent.tools)}"
            for agent in sorted(self.agents.values(), key=lambda a: a.name)
        ]
        lines += [f"{source} -> {target}" for source, target in sorted(self.graph.edges)]
        if self.info is None:
            return "\n".join(lines)
        lines += [
            f"code node {step.name} uses={step.uses} reads={canonical(step.reads)} writes={canonical(step.writes)}"
            for step in sorted(self.code_nodes.values(), key=lambda s: s.name)
        ]
        lines += [f"edge {canonical(dataclasses.astuple(edge))}" for edge in self.edges]
        lines += [
            f"form {agent.name} {parse_type(agent.output_type)} {canonical(agent.schema)}"
            for agent in sorted(self.agents.values(), key=lambda a: a.name)
        ]
        lines += [f"input {name} {vtype}" for name, vtype in sorted(self.inputs.items())]
        lines += [f"merge {value} {rule}" for value, rule in sorted(self.merges.items())]
        lines += [f"output {name} {vtype}" for name, vtype in self.outputs.items()]
        lines += [
            f"reads {a.name} {canonical(a.reads)}"
            for a in sorted(self.agents.values(), key=lambda a: a.name)
            if a.renamed
        ]
        lines += [f"each {name} {f} at_once={f.at_once} key={f.key}" for name, f in sorted(self.each.items())]
        lines += [f"retry {name} {canonical(rules)}" for name, rules in sorted(self.retry.items())]
        lines += [
            f"human action {name} "
            f"{canonical({k: v for k, v in human.to_dict().items() if k not in ('name', 'question', 'detached')})}"
            for name, human in sorted(self.human_actions.items())
        ]
        for box in sorted(self.modules.values(), key=lambda b: b.name):
            lines.append(f"module {box.name} reads={canonical(box.reads)} writes={canonical(box.writes)}")
            lines += [f"  {line}" for line in box.system.structure.splitlines()]
        return "\n".join(lines)

    @property
    def compiled(self) -> Any:
        """The compiled executable graph, rebuilt whenever the structure changes; its type is not API."""
        if self.compilation is None or self.compilation[0] != self.structure:
            self.compilation = (self.structure, runtime.compile_mas(self))
        return self.compilation[1]

    @property
    def terminal(self) -> str:
        """Name of the node whose output is the MAS output."""
        return terminal_agent(self.graph)

    def __repr__(self) -> str:
        """Return an unambiguous representation naming agents and edge count."""
        edges = sum(1 for source, target in self.graph.edges if source not in SENTINELS and target not in SENTINELS)
        terminals = sorted(self.graph.predecessors(END))
        ending = f"terminal={terminals[0]!r}" if len(terminals) == 1 else f"terminals={terminals!r}"
        return f"MAS(agents={agent_names(self.graph)!r}, edges={edges}, {ending})"


def agent_spec(agent: Agent, prompt: str | None) -> dict[str, Any]:
    """Return one agent's version 2 entry, defaults left out; its prompt text goes to ``prompts``.

    ``prompt`` names a shared prompt; ``reads`` is a list unless a placeholder reads another name.
    """
    data: dict[str, Any] = {"name": agent.name}
    if agent.model is not None:
        data["model"] = agent.model
    if prompt is not None:
        data["prompt"] = prompt
    if agent.reads != {"query": "query"}:
        data["reads"] = dict(agent.reads) if agent.renamed else list(agent.reads)
    data["writes"] = {agent.output: str(parse_type(agent.output_type))}
    if agent.tools:
        data["tools"] = list(agent.tools)
    if agent.schema is not None:
        data["schema"] = deepcopy(agent.schema)
    if agent.detached:
        data["detached"] = True
    return data


def spec_systems(spec: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield a spec and each module's inner spec, at every depth."""
    yield spec
    for box in spec.get("modules") or []:
        if isinstance(box, dict) and isinstance(box.get("system"), dict):
            yield from spec_systems(box["system"])


def plain_each(each: Mapping[str, str | Mapping[str, Any] | ForEach] | None) -> dict[str, ForEach]:
    """Return the ``for each`` forms by node, each parsed from its text or spec mapping."""
    return {name: form if isinstance(form, ForEach) else ForEach.parse(form) for name, form in (each or {}).items()}


def plain_models(models: Mapping[str, Mapping[str, Any] | Capabilities] | None) -> dict[str, Capabilities]:
    """Return the declared models by name, each spec mapping parsed as a :class:`~xmas.config.ModelEntry`."""
    return {
        name: entry if isinstance(entry, Capabilities) else ModelEntry.parse(entry, f"models entry {name!r}")
        for name, entry in (models or {}).items()
    }


def refuse_unrunnable(
    version: int, agents: Sequence[Agent], others: Sequence[Code | MASModule | HumanAction], max_tool_rounds: int
) -> None:
    """Refuse a system without a node (an agent, in version 1), or a negative ``max_tool_rounds``."""
    if not agents and (version == 1 or not others):
        raise ValueError(
            "A MAS needs at least one node." if version == VERSION_2 else "A MAS needs at least one agent."
        )
    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds must be >= 0.")


def check_bindings(agents: Iterable[Agent], llms: Mapping[str, BaseLLM]) -> None:
    """Refuse an agent bound to a model name ``llms`` does not hold."""
    for agent in agents:
        if agent.model is not None and agent.model not in llms:
            raise ValueError(
                f"Agent {agent.name!r} binds to model {agent.model!r}, which is not in llms={sorted(llms)}."
            )


def apply_overlay(spec: Mapping[str, Any], overlay: Mapping[str, str]) -> None:
    """Rebind the agents of ``spec`` that ``overlay`` names to its model names; an unknown agent is refused."""
    agents = {item["name"]: item for item in spec["agents"]}
    unknown = sorted(set(overlay) - set(agents))
    if unknown:
        raise ValueError(f"The overlay names unknown agents {unknown}; the agents are {sorted(agents)}.")
    for name, model in overlay.items():
        agents[name]["model"] = model


def bound_models(systems: Sequence[Mapping[str, Any]], *, with_defaults: bool) -> set[str]:
    """Return the model names the agents of ``systems`` bind, and each system's default ``with_defaults``."""
    bound = {agent["model"] for system in systems for agent in system.get("agents", []) if agent.get("model")}
    if with_defaults:
        bound |= {system["default_model"] for system in systems if system.get("default_model")}
    return bound


def declared_models(systems: Sequence[Mapping[str, Any]]) -> dict[str, Capabilities]:
    """Return the ``models:`` entries of ``systems``; an outer entry wins over a module's."""
    models: dict[str, Capabilities] = {}
    for system in reversed(systems):
        entries = system.get("models") or {}
        models.update({name: ModelEntry.parse(entry, f"models entry {name!r}") for name, entry in entries.items()})
    return models


def built_systems(mas: MAS) -> Iterator[MAS]:
    """Yield a system and each module's system, at every depth."""
    yield mas
    for box in mas.modules.values():
        yield from built_systems(box.system)


def callable_sources(
    systems: Sequence[Mapping[str, Any]],
    supplied: Sequence[ToolSource | Tool],
    configured: Callable[[], Config],
    *,
    trust: bool,
) -> list[ToolSource | Tool]:
    """Return the tools and tool sources a bundle's callables need, besides those ``supplied``.

    A python callable is found (a :class:`Tool` as it is, a function made one
    when trusted) and bound to its ``params``; the remote ones of each named server
    share one :class:`~xmas.tools.mcp.MCPTools` on the configured address.
    """
    offered = set(index_tools(supplied)) if supplied else set()
    built: dict[str, Tool] = {}
    servers: dict[str, dict[str, Mapping[str, str]]] = {}
    for system in systems:
        for name, entry in system.get("callables", {}).items():
            if name in offered or name in built:
                continue
            if entry.get("kind") == "python":
                built[name] = imported(name, entry, trust=trust)
            elif entry.get("kind") == "mcp":
                server = entry.get("server")
                if not isinstance(server, str):
                    raise ValueError(
                        f"Callable {name!r} is a remote tool with no server: pass tools=[...] offering it, or name "
                        "its server and configure [tools.<name>] url."
                    )
                servers.setdefault(server, {})[name] = entry.get("writes") or {name: "text"}
    return [*built.values(), *(tool_sources(servers, configured()) if servers else ())]


def imported(name: str, entry: Mapping[str, Any], *, trust: bool) -> Tool:
    """Find a python callable by its ``import`` path, as a tool named ``name`` bound to its ``params``.

    Untrusted, only a :class:`Tool` of a module already imported is taken: nothing is imported or wrapped.
    """
    target = entry.get("import")
    if not isinstance(target, str) or ":" not in target:
        raise ValueError(f"Callable {name!r} needs import: 'package.module:function', not {target!r}.")
    module, _, attribute = target.partition(":")
    refusal = (
        f"Callable {name!r}: {target!r} {{}}; a bundle may name any code, so pass tools=[...] offering it, "
        "or load(..., trust=True) (xmas --trust) for a bundle you trust."
    )
    if not trust and module not in sys.modules:
        raise ValueError(refusal.format("is in a module not imported yet"))
    try:
        found: Any = importlib.import_module(module)
        for part in attribute.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Callable {name!r}: {target!r} cannot be imported ({error}); pass tools=[...].") from None
    if isinstance(found, Tool):
        item = found if found.name == name else found.renamed(name)
    elif not trust:
        raise ValueError(refusal.format("is not a tool made with @tool"))
    elif callable(found):
        item = tool(found, name=name, writes=entry.get("writes"))
    else:
        raise ValueError(f"Callable {name!r}: {target!r} is a {type(found).__name__}, not a function.")
    params = entry.get("params")
    return item.bind(**params) if params else item


def health_spec(settings: HealthSettings | None) -> dict[str, Any]:
    """Return the health settings that differ from the defaults, as a spec writes them."""
    if settings is None:
        return {}
    default = HealthSettings()
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name in ("length_factor", "min_samples", "leak")
        if (value := getattr(settings, name)) != getattr(default, name)
    }


def health_settings(spec: Mapping[str, Any]) -> HealthSettings:
    """Read a spec's ``health`` section."""
    data: dict[str, Any] = {name: tuple(value) if name == "leak" else value for name, value in spec.items()}
    return HealthSettings(**data)


def typed(types: Mapping[str, str | ValueType]) -> dict[str, ValueType]:
    """Read ``{name: type}``, each type a word or a :class:`~xmas.values.ValueType`."""
    return {name: vtype if isinstance(vtype, ValueType) else parse_type(vtype) for name, vtype in types.items()}


def type_words(types: Mapping[str, ValueType]) -> dict[str, str]:
    """Write ``{name: type}`` as type words."""
    return {name: str(vtype) for name, vtype in types.items()}


def canonical(value: Any) -> str:
    """Write a structure part as canonical JSON, so equal parts give equal lines."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class ChainSystem(MAS):
    """The nodes of one ``for each`` chain as a system of their own, which one item runs.

    Its exits end it: ``checker -> assemble: else`` outside is ``checker ->
    END: else`` inside. It declares the chain's writes as its outputs, so
    several of its nodes may end it. Models, tools and callables resolve
    through the outer system, so a subclass's ``resolve_llm`` serves the
    chain's agents too, and the outer ``models`` entries hold for them.
    """

    def __init__(
        self, outer: MAS, shape: ChainShape, inputs: Mapping[str, ValueType], written: Mapping[str, ValueType]
    ) -> None:
        """Build and check the chain's system from the outer one."""
        self.outer = outer
        super().__init__(
            [outer.agents[name] for name in shape.nodes if name in outer.agents],
            [*shape.internal, *(dataclasses.replace(edge, target=END) for edge in shape.exits)],
            llm=outer.llm,
            llms=outer.llms,
            max_steps=outer.max_steps,
            default_model=outer.default_model,
            tools=outer.tools,
            max_tool_rounds=outer.max_tool_rounds,
            max_visits=outer.max_visits,
            code_nodes=[outer.code_nodes[name] for name in shape.nodes if name in outer.code_nodes],
            inputs=inputs,
            state={value: {"merge": rule} for value, rule in outer.merges.items() if value in written},
            version=2,
            modules=[outer.modules[name] for name in shape.nodes if name in outer.modules],
            human_actions=[outer.human_actions[name] for name in shape.nodes if name in outer.human_actions],
            health=outer.health,
            outputs=written,
            retry={name: rules for name, rules in outer.retry.items() if name in shape.nodes},
            models={name: entry for name, entry in outer.models.items() if name in {*outer.llms, outer.default_model}},
        )

    def runs(self) -> Store:
        """Return the outer system's run store."""
        return self.outer.runs()

    def callables(self) -> ToolIndex:
        """Return the outer system's callables registry."""
        return self.outer.callables()

    def resolve_tools(self, agent: Agent) -> ToolIndex:
        """Resolve an agent's tools through the outer system."""
        return self.outer.resolve_tools(agent)

    def resolve_callable(self, step: Code) -> tuple[Tool, ToolSource]:
        """Resolve a code node's callable through the outer system."""
        return self.outer.resolve_callable(step)

    def resolve_llm(self, agent: Agent) -> BaseLLM:
        """Resolve an agent's model through the outer system."""
        return self.outer.resolve_llm(agent)
