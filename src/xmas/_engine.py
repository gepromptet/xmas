"""Private execution engine.

The topology compiles to a LangGraph state graph here and nowhere else. No type
from langgraph crosses back out: callers receive a :class:`~xmas.trace.Trace`,
which is xmas's own data. Keeping the dependency behind this wall is what makes
replacing it later a contained change.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Protocol, TypedDict

from langgraph.graph import StateGraph
from promptolution.utils.token_counter import get_token_counter

from xmas.agent import Agent
from xmas.graph import join_nodes, terminal_agent
from xmas.prompts import safe_format
from xmas.trace import AgentRecord, TokenUsage, Trace

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.mas import MAS


def _merge_values(current: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    """Merge one agent's outputs into the shared value map.

    Agents writing concurrently each contribute their own key, so a plain merge
    is well defined. Keeping every variable in one reduced channel, rather than
    building a channel per variable, means the state schema is fixed and typed
    no matter what variables a topology happens to use.

    Args:
        current: Values written so far.
        incoming: Values from the agent that just finished.

    Returns:
        The merged mapping.
    """
    return {**current, **incoming}


class MASState(TypedDict, total=False):
    """The state a run threads through the graph.

    Attributes:
        values: Every state variable, including ``query``, keyed by name.
        prompts: The prompt set for this run, keyed by agent name. Prompts travel
            in the state rather than being baked into the compiled graph, so one
            compilation serves every candidate an optimizer proposes.
        records: Accumulated per-agent records, in completion order.
    """

    values: Annotated[dict[str, str], _merge_values]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]


class CompiledGraph(Protocol):
    """The executable produced by :func:`compile_mas`.

    Declared as a protocol so that nothing outside this module needs to name a
    langgraph type, including the tests. The parameters are deliberately loose:
    the point is to keep the dependency behind this wall, and pinning the exact
    overloaded signature of whatever backend is in use would defeat that by
    tying the protocol to one backend's spelling.
    """

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Run one state to completion and return the final state."""
        ...

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        """Run several states, concurrently where possible."""
        ...


def _state_variables(mas: MAS) -> set[str]:
    """Return every state variable the agents read or write."""
    variables: set[str] = set()
    for agent in mas.agents.values():
        variables.update(agent.inputs)
        variables.add(agent.output)
    return variables


def _make_node(mas: MAS, agent: Agent, count_tokens: Callable[[str], float]) -> Callable[[MASState], dict[str, Any]]:
    """Build the callable LangGraph runs for one agent.

    Args:
        mas: The system the agent belongs to, used to resolve its model.
        agent: The agent to wrap.
        count_tokens: Token counter for the agent's model.

    Returns:
        A function mapping the incoming state to this agent's state updates.
    """
    llm: BaseLLM = mas.resolve_llm(agent)

    def node(state: MASState) -> dict[str, Any]:
        prompts = state.get("prompts") or {}
        template = prompts.get(agent.name, agent.prompt)

        values = state.get("values") or {}
        inputs = {name: str(values.get(name) or "") for name in agent.inputs}
        rendered = safe_format(template, inputs)
        response = llm.get_response([rendered])[0]

        record = AgentRecord(
            agent=agent.name,
            step=-1,  # renumbered once the run finishes and the order is known
            prompt=template,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: response},
            usage=TokenUsage(
                input_tokens=int(count_tokens(rendered)),
                output_tokens=int(count_tokens(response)),
            ),
            model=agent.model,
            raw_response=response,
        )
        return {"values": {agent.output: response}, "records": [record]}

    return node


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
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    joins = join_nodes(mas.graph)

    counters: dict[int, Callable[[str], float]] = {}
    for name, agent in mas.agents.items():
        llm = mas.resolve_llm(agent)
        if id(llm) not in counters:
            counters[id(llm)] = get_token_counter(llm)
        # langgraph's add_node overloads bind their node type to a protocol that a
        # typing.TypedDict class does not satisfy under mypy, even though passing a
        # TypedDict schema is the documented usage. Runtime behaviour is covered by
        # the join-semantics and quickstart tests.
        builder.add_node(  # type: ignore[call-overload]
            name,
            _make_node(mas, agent, counters[id(llm)]),
            defer=name in joins,
        )

    for source, target in mas.graph.edges:
        builder.add_edge(source, target)

    return builder.compile()


def _to_trace(mas: MAS, query: str, result: dict[str, Any]) -> Trace:
    """Assemble a trace from a finished run's state."""
    records: list[AgentRecord] = list(result.get("records") or [])
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(records))
    values: dict[str, str] = result.get("values") or {}
    output_var = mas.agents[terminal_agent(mas.graph)].output
    return Trace(query=query, output=str(values.get(output_var) or ""), records=numbered)


def _initial_state(query: str, prompts: dict[str, str]) -> MASState:
    """Build the state a run starts from."""
    return MASState(values={"query": query}, prompts=dict(prompts), records=[])


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
    config = {"recursion_limit": mas.max_steps}
    try:
        result: dict[str, Any] = mas.compiled.invoke(_initial_state(query, prompts), config=config)
    except Exception as error:  # a failed rollout is data, not a crash
        return Trace(query=query, output="", records=(), error=f"{type(error).__name__}: {error}")
    return _to_trace(mas, query, result)


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Execute a MAS over parallel lists of inputs and prompt sets.

    Args:
        mas: The system to run.
        queries: The inputs.
        prompts: Prompt sets, one per input.

    Returns:
        One trace per input, in the order given.
    """
    if not queries:
        return []

    states = [_initial_state(query, prompt_set) for query, prompt_set in zip(queries, prompts, strict=True)]
    config = {"recursion_limit": mas.max_steps}
    try:
        results: list[dict[str, Any]] = mas.compiled.batch(states, config=config)
    except Exception:  # fall back per-item so one bad rollout cannot lose the batch
        return [run_one(mas, query, prompt_set) for query, prompt_set in zip(queries, prompts, strict=True)]
    return [_to_trace(mas, query, result) for query, result in zip(queries, results, strict=True)]
