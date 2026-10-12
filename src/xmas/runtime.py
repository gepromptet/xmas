"""Compile a MAS and run it.

The topology compiles to a LangGraph state graph here and nowhere else. No type
from langgraph crosses back out: callers receive a :class:`~xmas.trace.Trace`,
which is xmas's own data. Keeping the dependency behind this wall is what makes
replacing it later a contained change.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langgraph.graph import StateGraph
from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from promptolution.utils.token_counter import get_token_counter

from xmas.agent import Agent
from xmas.graph import END, back_edges, join_nodes, terminal_agent
from xmas.prompts import safe_format
from xmas.tools import run_tool_loop
from xmas.trace import AgentRecord, TokenUsage, Trace

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.mas import MAS


CONTINUE_MARKER = "<continue>"
"""Marker an agent emits to take a back edge instead of moving on."""


class MASState(TypedDict):
    """The state a run threads through the graph.

    Attributes:
        values: Every state variable, including ``query``, keyed by name. Agents
            writing concurrently each contribute their own key, so the reducer is
            a plain dict union.
        prompts: The prompt set for this run, keyed by agent name. Prompts travel
            in the state rather than being baked into the compiled graph, so one
            compilation serves every candidate an optimizer proposes.
        records: Accumulated per-agent records, in completion order.
    """

    values: Annotated[dict[str, str], operator.or_]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]


def compile_mas(mas: MAS) -> Any:
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
    builder = StateGraph(MASState)
    joins = join_nodes(mas.graph)
    back = back_edges(mas.graph)

    for name, agent in mas.agents.items():
        llm = mas.resolve_llm(agent)
        # mypy rejects a TypedDict schema here; langgraph documents it as the usage.
        builder.add_node(  # type: ignore[call-overload]
            name,
            agent_node(mas, agent, llm, get_token_counter(llm)),
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
        router = cycle_router(mas, mas.agents[source], back_targets, forward_targets)
        builder.add_conditional_edges(source, router, back_targets + forward_targets + [END])

    return builder.compile()


def run_one(mas: MAS, query: str, prompts: dict[str, str]) -> Trace:
    """Execute a compiled MAS against one input with a given prompt set.

    Args:
        mas: The system to run.
        query: The input.
        prompts: Prompt text per agent name, overriding what the agents hold.

    Returns:
        The trace of the run. A failure is recorded in ``Trace.error`` rather
        than raised.
    """
    return run_many(mas, [query], [prompts])[0]


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Execute a MAS over parallel lists of inputs and prompt sets.

    Args:
        mas: The system to run.
        queries: The inputs.
        prompts: Prompt sets, one per input.

    Returns:
        One trace per input, in the order given. A failed rollout is data, not a
        crash: its trace carries the error and the rest of the batch is kept.
    """
    if not queries:
        return []
    states = [
        MASState(values={"query": query}, prompts=dict(prompt_set), records=[])
        for query, prompt_set in zip(queries, prompts, strict=True)
    ]
    config = {"recursion_limit": mas.max_steps}
    results = mas.compiled.batch(states, config=config, return_exceptions=True)
    return [trace_from_state(mas, query, result) for query, result in zip(queries, results, strict=True)]


def agent_node(
    mas: MAS, agent: Agent, llm: BaseLLM, count_tokens: Callable[[str], float]
) -> Callable[[MASState], dict[str, Any]]:
    """Build the callable LangGraph runs for one agent.

    An agent with tools answers through :func:`xmas.tools.run_tool_loop`; every
    other agent makes one model call. Each visit starts fresh.

    Args:
        mas: The system the agent belongs to, which resolves its tools.
        agent: The agent to wrap.
        llm: Its model.
        count_tokens: Token counter for the model.

    Returns:
        A function mapping the incoming state to this agent's state updates.
    """
    tools = mas.resolve_tools(agent) if agent.tools else {}

    def node(state: MASState) -> dict[str, Any]:
        template = state["prompts"].get(agent.name, agent.prompt)
        inputs = {name: state["values"].get(name, "") for name in agent.inputs}
        rendered = safe_format(template, inputs)
        calls: tuple[Any, ...] = ()
        if agent.tools and hasattr(llm, "get_tool_response"):  # resolve_tools checked the model can call tools
            response, calls, usage = run_tool_loop(
                llm, rendered, tools, mas.max_tool_rounds, DEFAULT_SYS_PROMPT, count_tokens
            )
        else:
            response = llm.get_response([rendered])[0]
            usage = TokenUsage(input_tokens=int(count_tokens(rendered)), output_tokens=int(count_tokens(response)))
        record = AgentRecord(
            agent=agent.name,
            step=-1,  # renumbered once the run finishes and the order is known
            prompt=template,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: response},
            tool_calls=calls,
            usage=usage,
            model=agent.model,
            raw_response=response,
        )
        return {"values": {agent.output: response}, "records": [record]}

    return node


def cycle_router(
    mas: MAS, agent: Agent, back_targets: list[str], forward_targets: list[str]
) -> Callable[[MASState], list[str]]:
    """Build the branch decision for an agent that closes a cycle.

    The agent decides by what it wrote: emitting :data:`CONTINUE_MARKER` takes the
    back edge, anything else moves on. Two guards make the loop finite regardless
    of what the model says: the agent's own visit count against ``mas.max_visits``,
    and the graph-wide recursion limit behind it.

    Args:
        mas: The system, which bounds the visits.
        agent: The agent whose out-edges are being routed.
        back_targets: Targets reached by a back edge, taken when looping.
        forward_targets: Targets reached otherwise, taken when moving on.

    Returns:
        A function mapping state to the node names to run next.
    """
    onward = forward_targets or [END]

    def route(state: MASState) -> list[str]:
        visits = sum(1 for record in state["records"] if record.agent == agent.name)
        looping = CONTINUE_MARKER in state["values"].get(agent.output, "")
        return back_targets if looping and visits < mas.max_visits else onward

    return route


def trace_from_state(mas: MAS, query: str, result: Any) -> Trace:
    """Assemble a trace from a finished run's state, or from the error that ended it."""
    if isinstance(result, Exception):
        return Trace(query=query, output="", records=(), error=f"{type(result).__name__}: {result}")
    records = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(result["records"]))
    output_var = mas.agents[terminal_agent(mas.graph)].output
    return Trace(query=query, output=result["values"].get(output_var, ""), records=records)
