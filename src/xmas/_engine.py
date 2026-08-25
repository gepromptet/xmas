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
from xmas.graph import END, back_edges, join_nodes, terminal_agent
from xmas.prompts import safe_format
from xmas.trace import AgentRecord, TokenUsage, Trace

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.mas import MAS


CONTINUE_MARKER = "<continue>"
"""Marker an agent emits to take a back edge instead of moving on.

A cyclic topology is a retry loop, and something has to decide when to stop. The
agent that closes the loop decides, by emitting this marker in its response, which
mirrors the seed engine's ``<continue_iter>`` convention. An agent that never
emits it turns its cycle into a single pass, so a mutated topology that grows a
cycle stays runnable without anyone rewriting a prompt for it.
"""

MAX_NODE_VISITS = 3
"""How often one agent may run in a single query before its back edges close.

The second stop condition, and the one that holds when a prompt emits the marker
unconditionally. Matches the seed's ``stop_condition_max_retries``. The graph-wide
``max_steps`` still caps total super-steps on top of this.
"""


def _merge_counts(current: dict[str, int], incoming: dict[str, int]) -> dict[str, int]:
    """Add per-agent visit counts, so a looping agent accumulates its own.

    Args:
        current: Counts so far.
        incoming: Counts from the agents that just finished.

    Returns:
        The summed mapping.
    """
    merged = dict(current)
    for name, count in incoming.items():
        merged[name] = merged.get(name, 0) + count
    return merged


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
        visits: How often each agent has run, which is what bounds a retry loop.
    """

    values: Annotated[dict[str, str], _merge_values]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]
    visits: Annotated[dict[str, int], _merge_counts]


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
        return {"values": {agent.output: response}, "records": [record], "visits": {agent.name: 1}}

    return node


def _make_router(
    agent: Agent,
    back_targets: list[str],
    forward_targets: list[str],
) -> Callable[[MASState], list[str]]:
    """Build the branch decision for an agent that closes a cycle.

    The agent decides by what it wrote: emitting :data:`CONTINUE_MARKER` takes the
    back edge, anything else moves on. Two guards make the loop finite regardless
    of what the model says, because a prompt that always emits the marker is a
    thing an optimizer will eventually propose: the agent's own visit count, and
    the graph-wide recursion limit behind it.

    Args:
        agent: The agent whose out-edges are being routed.
        back_targets: Targets reached by a back edge, taken when looping.
        forward_targets: Targets reached otherwise, taken when moving on.

    Returns:
        A function mapping state to the node names to run next.
    """
    onward = forward_targets or [END]

    def route(state: MASState) -> list[str]:
        values = state.get("values") or {}
        visits = (state.get("visits") or {}).get(agent.name, 0)
        looping = CONTINUE_MARKER in str(values.get(agent.output) or "")
        return back_targets if looping and visits < MAX_NODE_VISITS else onward

    return route


def compile_mas(mas: MAS) -> CompiledGraph:
    """Compile a MAS topology into an executable graph.

    Agents identified by :func:`xmas.graph.join_nodes` are marked deferred, so a
    node fed by predecessors at unequal depth runs exactly once per query rather
    than once per wave of arriving inputs.

    Cycle-closing edges compile to conditional branches rather than to plain
    edges. A static edge set cannot express a loop, because the edge back and the
    edge onward would both fire on every pass and the run would spin until it hit
    the recursion limit. Routing the two exclusively, on the looping agent's own
    output, is what makes a cyclic topology terminate with an answer.

    Args:
        mas: The system to compile.

    Returns:
        The compiled graph.
    """
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    joins = join_nodes(mas.graph)
    back = back_edges(mas.graph)

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

    for source in mas.graph.nodes:
        targets = sorted(mas.graph.successors(source))
        back_targets = [target for target in targets if (source, target) in back]
        if not back_targets:
            for target in targets:
                builder.add_edge(source, target)
            continue
        forward_targets = [target for target in targets if (source, target) not in back]
        router = _make_router(mas.agents[source], back_targets, forward_targets)
        builder.add_conditional_edges(source, router, sorted({*back_targets, *forward_targets, END}))

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
    return MASState(values={"query": query}, prompts=dict(prompts), records=[], visits={})


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
