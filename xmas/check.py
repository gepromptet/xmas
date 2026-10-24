"""The dataflow check of a version 2 system, run when the MAS is built.

:func:`check_system` asks every question a run would otherwise answer late and
expensively: does each value a node reads exist before the node runs, is each
value written once (or by alternatives), does each prompt placeholder name a
read, do the names and types of steps and boxes fit, do the edges decide on
values that exist, is every node with several incoming edges a merge or a
join, can each agent's model read and write the kinds the agent does, does each
human node fit its form and its reads, and is each ``first of`` race sound. It
collects every problem and raises them together, so one edit cycle fixes them
all.

A ``for each`` chain is checked twice: inside, as a system of its own that one
item runs, and outside, collapsed into its head, which reads what the chain
reads from outside and writes each value the chain writes as a list.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xmas.agent import Agent
from xmas.chains import Chain, ChainShape, collapse, find_chains
from xmas.edges import ANY_ERROR, Edge, EdgeKind
from xmas.errors import RUN_ENDS, ErrorKind
from xmas.graph import (
    END,
    SENTINELS,
    START,
    agent_names,
    ancestors,
    back_edges,
    build_graph,
    forward_graph,
    loop_problems,
    validate_graph,
)
from xmas.human import ANSWER_FIELDS, COMMON_FIELDS, Human, template_paths
from xmas.llms.meta import supports_images, supports_parts
from xmas.nodes import Box, Step
from xmas.predicates import BUILTINS, KEYWORDS, Predicate, PredicateError, parse_predicate
from xmas.prompts import extract_placeholders
from xmas.routes import Join, RouteGraph
from xmas.schemas import compile_form
from xmas.values import JSON, KINDS, LIST_PREFIX, TEXT, ValueType, fits, media_kind, parse_type

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import networkx as nx

    from xmas.mas import MAS
    from xmas.media import Capabilities
    from xmas.tools import Tool, ToolIndex

__all__ = [
    "MERGE_LAST",
    "RESERVED",
    "RESERVED_READS",
    "RETRYABLE",
    "ROUTED_ERRORS",
    "SENT",
    "SystemInfo",
    "check_system",
    "step_maps",
]


MERGE_LAST = "last"
"""The one merge rule: the latest write wins."""

ROUTED_ERRORS = frozenset(kind.value for kind in ErrorKind if kind not in RUN_ENDS)
"""The error kinds an ``on error`` edge may route; a run always stops on the others."""

SENT = ("image", "audio")
"""The kinds a model reads as content parts, when it accepts them; a ``file`` it reads as its ref's JSON line."""

CONVERSIONS = {"document": "document_text", "table": "table_rows"}
"""The built-in :mod:`xmas.convert` step that turns each kind a model never reads into text or rows."""

CANNOT = {"accepts": ("send content parts", "parts"), "produces": ("answer with images", "image")}
"""What a model object without ``generate_<word>_with_meta`` cannot do, per capability: ``(what, word)``."""

RESERVED_READS = {"error": JSON, "run": JSON}
"""Values xmas fills and no node writes. ``error``, ``{kind, message, node}`` of
the failure routed to a node, is readable by a node an ``on error`` edge reaches
and on that node's edges; ``run``, ``{cost_usd, elapsed_s, calls, records}`` so
far, is readable anywhere but in a box."""

ERRED_ONLY = "which only a node an on error edge reaches can read"

RESERVED = frozenset({*KINDS, *LIST_PREFIX.split(), *BUILTINS, *KEYWORDS, *RESERVED_READS})
"""Names no value may take: the type words, the words of predicates and the
reserved reads. Names starting with ``__`` belong to xmas as well."""

RETRYABLE = frozenset({"outage", "timeout", "rate_limited"})
"""The error kinds a node's ``retry`` may run it again on."""

SHARED_WRITERS = 2  # writers of one value (the input counts as one) from which it needs a merge rule or alternatives
MIN_RACE_CONTESTANTS = 2  # a first of race needs two or more contestants


@dataclass(frozen=True)
class SystemInfo:
    """What the check learned about a version 2 system, for the engine and ``describe``.

    Attributes:
        types: The type of every value, inputs and writes alike.
        writers: The nodes writing each written value, in node order.
        joins: Each node with several incoming edges, :data:`~xmas.graph.END`
            included, as a merge or a join.
        graph: The topology, with the implicit wiring from
            :data:`~xmas.graph.START` and to :data:`~xmas.graph.END`.
        predicates: The parsed predicate of every ``when``, ``while`` and
            ``switch`` edge the engine wires (a switch's is its path).
        chains: Each ``for each`` chain, keyed by its head.
        edges: The edges the engine wires: the declared ones, each chain
            collapsed into its head.
        flow: The topology of ``edges``, what the engine wires; ``graph``
            when there is no chain.
        humans: The nodes of the flow that may pause the run: every human
            node, and each chain head or box whose system holds a human at
            any depth (a pausing container).
        races: Each ``first of`` race of the flow: its source and its
            contestants, in declared order.

    A value a chain writes has its gathered type here, a list; the reserved
    reads (:data:`RESERVED_READS`) are among ``types`` when a node reads them.
    """

    types: dict[str, ValueType]
    writers: dict[str, tuple[str, ...]]
    joins: dict[str, Join]
    graph: nx.DiGraph[str]
    predicates: dict[Edge, Predicate]
    chains: dict[str, Chain]
    edges: tuple[Edge, ...]
    flow: nx.DiGraph[str]
    humans: tuple[str, ...] = ()
    races: dict[str, tuple[str, ...]] = field(default_factory=dict)


class CheckError(ValueError):
    """A system that does not check; ``problems`` holds its lines."""

    def __init__(self, problems: list[str]) -> None:
        lines = "\n".join(f"- {problem}" for problem in problems)
        super().__init__(f"The system does not check ({len(problems)} problem{'s' * (len(problems) > 1)}):\n{lines}")
        self.problems = problems


def step_maps(step: Step, item: Tool) -> tuple[dict[str, str], dict[str, str]]:
    """Return a step's ``(reads, writes)``, each ``{name inside: name in the graph}``.

    Every parameter of the callable is a read; a parameter not renamed reads the
    value of its own name, and a write not renamed keeps its own name.
    """
    reads = {name: step.reads.get(name, name) for name in item.reads}
    writes = {name: step.writes.get(name, name) for name in item.writes}
    return reads, writes


def schema_type(item: Tool, parameter: str) -> str:
    """Name a parameter's type for a message: its media type, else its JSON-schema type."""
    vtype = item.reads[parameter]
    if media_kind(vtype) is not None:
        return str(vtype)
    schema = item.parameters.get("properties", {}).get(parameter, {})
    return str(schema.get("type", "json")) if isinstance(schema, dict) else "json"


def value_problems(name: str) -> Iterator[str]:
    """Say why a value's name is not allowed."""
    if name in RESERVED:
        yield "is a reserved name; pick another"
    elif name.startswith("__"):
        yield "starts with '__'; such names belong to xmas"


def deep(system: MAS, holds: Callable[[MAS], bool], seen: set[int] | None = None) -> bool:
    """Return whether ``holds`` is true of ``system`` or of the system of a box in it, at any depth."""
    if holds(system):
        return True
    seen = set() if seen is None else seen
    if id(system) in seen:
        return False
    seen.add(id(system))
    return any(deep(box.system, holds, seen) for box in system.boxes.values())


def pauses(mas: MAS, name: str) -> bool:
    """Return whether a node is a human, or a box whose system holds one at any depth."""
    return name in mas.humans or (name in mas.boxes and deep(mas.boxes[name].system, lambda s: bool(s.humans)))


def pairs(edges: Sequence[Edge], kind: EdgeKind) -> list[tuple[str, str]]:
    return [(edge.source, edge.target) for edge in edges if edge.kind is kind]


def check_edges(edges: tuple[Edge, ...], endpoints: set[str]) -> tuple[list[str], dict[Edge, Predicate], bool]:
    """Check each edge on its own and each node's set of deciding edges.

    Returns the problems, the parsed predicate (or switch path) of every edge
    that has one, and whether the edges make a graph (known endpoints, one edge
    per pair).
    """
    problems: list[str] = []
    parsed: dict[Edge, Predicate] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        bad = [n for n in (edge.source, edge.target) if n not in endpoints]
        if bad or edge.source == END or edge.target == START:
            problems.append(f"edge {edge.label}: {bad[0]!r} is no node" if bad else f"edge {edge.label} runs backwards")
            continue
        if (edge.source, edge.target) in seen_pairs:
            problems.append(f"edge {edge.label} is listed twice; one edge per pair of nodes")
        seen_pairs.add((edge.source, edge.target))
        problems += error_kind_problems(edge)
        predicate, problem = edge_predicate(edge)
        if problem is not None:
            problems.append(problem)
        elif predicate is not None:
            parsed[edge] = predicate
    problems += deciding_problems(edges)
    problems += switch_case_problems(edges)
    return problems, parsed, len(seen_pairs) == len(edges)


def error_kind_problems(edge: Edge) -> Iterator[str]:
    """Say which error kinds of an ``on error`` edge are unknown, never routed, or ``any`` among others."""
    if edge.kind is not EdgeKind.ERROR:
        return
    for kind in edge.errors:
        if kind == ANY_ERROR and len(edge.errors) > 1:
            yield f"edge {edge.label}: on error 'any' stands alone"
        elif kind in RUN_ENDS:
            yield f"edge {edge.label}: on error {kind!r} is never routed; the run stops on it"
        elif kind != ANY_ERROR and kind not in ROUTED_ERRORS:
            yield f"edge {edge.label}: on error {kind!r} is no error kind; known: {sorted(ROUTED_ERRORS)}"


def edge_predicate(edge: Edge) -> tuple[Predicate | None, str | None]:
    """Parse an edge's ``when``, ``while`` or ``switch`` text into ``(predicate, None)`` or ``(None, problem)``.

    An edge with no such text gives ``(None, None)``.
    """
    text = edge.switch if edge.kind is EdgeKind.SWITCH else edge.when
    if text is None:
        return None, None
    try:
        predicate = parse_predicate(text)
    except PredicateError as exc:
        return None, f"edge {edge.label}: {exc}"
    if edge.kind is EdgeKind.SWITCH and predicate.tree[0] != "path":
        return None, f"edge {edge.label}: switch takes a path such as verdict.action, not {text!r}"
    return predicate, None


def deciding_problems(edges: tuple[Edge, ...]) -> Iterator[str]:
    """Say which deciding nodes lack a when, while or switch edge, or have other than exactly one else edge."""
    deciding = Counter(e.source for e in edges if e.kind in (EdgeKind.WHEN, EdgeKind.WHILE, EdgeKind.SWITCH))
    elses = Counter(e.source for e in edges if e.kind is EdgeKind.ELSE)
    for source in dict.fromkeys(e.source for e in edges):
        if (deciding[source] or elses[source]) and (elses[source] != 1 or not deciding[source]):
            yield (
                f"node {source!r} has {deciding[source]} when, while or switch edges and {elses[source]} else edges; "
                "a deciding node needs at least one of the first and exactly one else"
            )


def switch_case_problems(edges: tuple[Edge, ...]) -> Iterator[str]:
    """Say which nodes switch to one case twice."""
    # a bool never equals a number, as predicates compare: case false and case 0 are two cases
    cases = Counter((e.source, isinstance(e.case, bool), e.case) for e in edges if e.kind is EdgeKind.SWITCH)
    return (f"node {source!r} switches to case {case!r} twice" for (source, _, case), n in cases.items() if n > 1)


NodeReads = dict[str, tuple[str, ValueType, str] | None]
"""Each value one node reads: ``(name inside, type, type word)`` for a step or a box, ``None`` for an agent or a
human."""
Reads = dict[str, NodeReads]
"""Per node, each value it reads (:data:`NodeReads`)."""
Writes = dict[str, dict[str, ValueType]]
Node = Agent | Step | Box | Human


def node_io(mas: MAS, nodes: dict[str, Node]) -> tuple[Reads, Writes, dict[str, list[str]]]:
    """Return what each node reads and writes, and per node the problems of its tools, prompt, callable or box.

    A human's own problems (:meth:`~xmas.human.Human.problems`) are its node's
    problems too.
    """
    problems: dict[str, list[str]] = {}
    reads: Reads = {}
    writes: Writes = {}
    for name, node in nodes.items():
        reads[name], writes[name], problems[name] = io_of(mas, name, node)
    for name, node in nodes.items():
        if isinstance(node, Human) and "value" in node.writes and node.edits in node.reads:
            writes[name][node.writes["value"]] = edited_type(mas, name, node.reads[node.edits], writes)
    return reads, writes, problems


NodeIO = tuple[NodeReads, dict[str, ValueType], list[str]]
"""What one node reads and writes, and its own problems."""


def io_of(mas: MAS, name: str, node: Node) -> NodeIO:
    """Return what one node reads and writes and its problems, by the kind of node."""
    if isinstance(node, Agent):
        return agent_io(mas, name, node)
    if isinstance(node, Human):
        return human_io(name, node)
    if isinstance(node, Box):
        return box_io(mas, name, node)
    return step_io(mas, name, node)


def agent_io(mas: MAS, name: str, agent: Agent) -> NodeIO:
    """Return an agent's reads and its one write, with the problems of its tools and prompt placeholders."""
    problems: list[str] = []
    if agent.tools:
        try:
            mas.resolve_tools(agent)
        except ValueError as exc:
            problems.append(str(exc))
    missing = sorted(extract_placeholders(agent.prompt) - set(agent.reads))
    if missing:
        problems.append(f"agent {name!r}: prompt placeholders {missing} are not among its reads {list(agent.reads)}")
    reads: NodeReads = dict.fromkeys(agent.inputs)
    return reads, {agent.output: parse_type(agent.output_type)}, problems


def human_io(name: str, human: Human) -> NodeIO:
    """Return a human's reads (never a reserved one) and its answer, with the human's own problems."""
    reserved = [value for value in human.reads.values() if value in RESERVED_READS]
    problems = [
        *human.problems(),
        *(
            f"human {name!r} reads {value!r}; a human reads values nodes write, not 'error' or 'run'"
            for value in reserved
        ),
    ]
    reads: NodeReads = dict.fromkeys(value for value in human.reads.values() if value not in reserved)
    return reads, {human.answer: JSON}, problems


def box_io(mas: MAS, name: str, box: Box) -> NodeIO:
    """Return a box's typed reads and its writes; nothing, with a problem, for a box that holds its own system."""
    if deep(box.system, lambda system: system is mas):
        return {}, {}, [f"box {name!r} contains the system it sits in"]
    outer_reads, writes = box.types()
    reads: NodeReads = {
        outer: (inner, outer_reads[outer], str(outer_reads[outer])) for inner, outer in box.reads.items()
    }
    return reads, writes, []


def step_io(mas: MAS, name: str, step: Step) -> NodeIO:
    """Return a step's typed reads and writes, renamed; nothing, with a problem, when its callable does not resolve."""
    try:
        item, _ = mas.resolve_callable(step)
    except ValueError as exc:
        return {}, {}, [str(exc)]
    parameters, produced = item.reads, item.writes
    problems = [
        f"step {name!r}: {label} renames {inside!r}, which is not {what} of {step.uses!r} (it has {sorted(known)})"
        for label, what, renames, known in (
            ("reads", "a parameter", step.reads, parameters),
            ("writes", "a write", step.writes, produced),
        )
        for inside in sorted(set(renames) - set(known))
    ]
    inner_reads, inner_writes = step_maps(step, item)
    reads: NodeReads = {
        outer: (inner, parameters[inner], schema_type(item, inner)) for inner, outer in inner_reads.items()
    }
    return reads, {outer: produced[inner] for inner, outer in inner_writes.items()}, problems


def edited_type(mas: MAS, name: str, value: str, writes: Writes) -> ValueType:
    """Return the type of the value an edit node changes: an input's, a writer's, its chain item's, else json."""
    if value in mas.inputs:
        return mas.inputs[value]
    found = [produced[value] for other, produced in writes.items() if other != name and value in produced]
    if found:
        return found[0]
    each = mas.each.get(name)
    root = mas.inputs.get(each.root) if each is not None and each.item == value and not each.fields else None
    return root.item if root is not None and root.item is not None else JSON


def writer_types(
    mas: MAS, writes: Writes, heads: Sequence[str] = ()
) -> tuple[dict[str, ValueType], dict[str, list[str]], list[str]]:
    """Return the type of every value, the writers of each written value, and the problems of their names and types.

    A chain's head writes the chain's values, whose names its chain's own check judges.
    """
    problems: list[str] = []
    types = dict(mas.inputs)
    writers: dict[str, list[str]] = {}
    for name, produced in writes.items():
        for value, vtype in produced.items():
            if name not in heads:
                problems += [f"{name!r} writes {value!r}, which {problem}" for problem in value_problems(value)]
            if value in types and types[value] != vtype:
                problems.append(f"value {value!r} is {types[value]} elsewhere but {name!r} writes it as {vtype}")
            types.setdefault(value, vtype)
            writers.setdefault(value, []).append(name)
    problems += [
        f"state names {v!r}, which no node writes and no input declares" for v in sorted({*mas.merges} - {*types})
    ]
    return types, writers, problems


def topology_graph(nodes: list[str], edges: Sequence[Edge]) -> nx.DiGraph[str]:
    """Build the topology of nodes and edges, ``while`` edges closing loops and error edges taken on failure."""
    plain = [(edge.source, edge.target) for edge in edges]
    return build_graph(nodes, plain, back=pairs(edges, EdgeKind.WHILE), errors=pairs(edges, EdgeKind.ERROR))


def topology(
    nodes: list[str], edges: tuple[Edge, ...], *, outputs: bool
) -> tuple[nx.DiGraph[str], RouteGraph | None, dict[str, Join], list[str]]:
    """Build the graph and, when its forward part has no cycle, its routes and joins; with the problems found.

    Several nodes may feed the end only of a system that declares its outputs.
    """
    loops = pairs(edges, EdgeKind.WHILE)
    graph = topology_graph(nodes, edges)
    problems: list[str] = []
    try:
        validate_graph(graph)
    except ValueError as exc:
        problems.append(str(exc))
    problems.extend(loop_problems(graph, loops))
    forward = forward_graph(graph, loops)
    if back_edges(forward):
        return graph, None, {}, problems
    routes = RouteGraph(agent_names(forward), edges)
    joins, refusals = routes.classify()
    problems.extend(refusals.values())
    terminals = sorted(graph.predecessors(END))
    end = joins.get(END)
    end_merges = end is not None and end.kind == "merge"
    if len(terminals) > 1 and not outputs and END not in refusals and not end_merges:
        problems.append(
            f"several nodes feed {END!r} ({terminals}) on one pass, so the output is ambiguous; add one final node"
        )
    return graph, routes, joins, problems


@dataclass(frozen=True)
class FlowFacts:
    """What the read and predicate checks know of a flow that has routes.

    Attributes:
        types: The type of every value, the reserved reads included.
        writers: The nodes writing each written value, in node order.
        routes: The routes of the flow.
        before: Per node, the nodes that may run before it.
        erred: The nodes an ``on error`` edge reaches, the ones that may read ``error``.
    """

    types: dict[str, ValueType]
    writers: dict[str, list[str]]
    routes: RouteGraph
    before: dict[str, set[str]]
    erred: set[str]


def read_problems(mas: MAS, reads: Reads, facts: FlowFacts) -> Iterator[str]:
    """Say which reads have no writer that runs first, or have the wrong type for a step or a box."""
    for name, needed in reads.items():
        failed = facts.routes.failed_before(name)
        for value, parameter in needed.items():
            if (box := name in mas.boxes and value in RESERVED_READS) or (value == "error" and name not in facts.erred):
                yield (
                    f"box {name!r} reads {value!r}; a box reads neither 'error' nor 'run', the system inside it does"
                    if box
                    else f"{name!r} reads 'error', {ERRED_ONLY}"
                )
                continue
            if value not in RESERVED_READS and value not in mas.inputs:
                if value not in facts.writers:
                    yield f"{name!r} reads {value!r}, which is neither an input nor written by any node"
                    continue
                writers = facts.writers[value]
                if not facts.before[name].intersection(writers):
                    yield f"{name!r} reads {value!r}, but its writers {writers} do not run before it"
                elif set(writers) <= failed:
                    yield f"{name!r} runs only when {writers[0]!r} fails, so it cannot read {value!r}"
            if parameter is not None and not fits(facts.types[value], parameter[1]):
                yield type_mismatch(mas, name, value, facts.types[value], parameter)


def type_mismatch(mas: MAS, name: str, value: str, vtype: ValueType, parameter: tuple[str, ValueType, str]) -> str:
    """Say that a box or a step passes ``value`` of type ``vtype`` to an input or parameter of another type."""
    inside, wanted, word = parameter
    if name in mas.boxes:
        return f"box {name!r} passes {vtype} {value!r} to its {word} input {inside!r}"
    kind = media_kind(vtype) or media_kind(wanted)
    media = "" if kind is None else f", {kind} into {a(kind)} parameter (xmas.media.{kind.title()}) or a file"
    return (
        f"step {name!r} passes {vtype} {value!r} to parameter {inside!r} ({word}); "
        f"text goes into a string parameter only{media}, json into any other"
    )


def a(kind: str) -> str:
    """Return ``an image`` or ``a table``: the kind with its article."""
    return f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"


def lacking(subject: str, kind: str, attr: str, capabilities: Capabilities, model: str | None) -> str:
    """Say that an agent's model does not take (``attr`` accepts) or give (produces) a kind, and how to fix it."""
    stated: tuple[str, ...] = getattr(capabilities, attr)
    only = " and ".join(dict.fromkeys(("text", *(word for word in stated if word not in ("text", "json")))))
    words = ", ".join(f'"{word}"' for word in dict.fromkeys(("text", *stated, kind)))
    fix = (
        f"give it {attr} = ({words})"
        if model is None
        else f"bind a model that {attr} {kind} (models: {{{model}: {{{attr}: [{kind}]}}}})"
    )
    return f"{subject}, but {model_called(model)} {attr} only {only}; {fix}"


def model_called(model: str | None) -> str:
    """Name a bound model for a message: ``the default model`` when there is no name."""
    return "the default model" if model is None else f"model {model!r}"


def node_problems(
    mas: MAS, nodes: dict[str, Node], types: Mapping[str, ValueType], edges: Sequence[Edge], members: set[str]
) -> Iterator[str]:
    """Say what is wrong with the media agents and steps handle and with the forms of human nodes.

    A model reads an image or audio as a content part, so it must accept the
    kind and be able to send parts; it never reads a document or a table,
    which a conversion step turns into text first. An agent writing an image
    needs a model that produces one, and no tools. A remote tool reads no
    media. A chain's members are its own check's question.
    """
    bindable = {*mas.llms, mas.default_model}
    yield from (f"models names {name!r}, which no agent binds" for name in mas.models if name not in bindable)
    cannot: dict[tuple[str, str], set[str]] = {}
    for name, node in nodes.items():
        if name in members:
            continue
        if isinstance(node, Agent):
            yield from agent_problems(mas, name, node, types, cannot)
        elif isinstance(node, Step):
            yield from step_problems(mas, name, node, types)
        elif isinstance(node, Human) and node.form in ANSWER_FIELDS:
            yield from human_problems(name, node, types, edges)
    for (called, attr), kinds in cannot.items():
        what, word = CANNOT[attr]
        yield (
            f"{called} {attr} {' and '.join(sorted(kinds))} but cannot {what}; "
            f"use APILLM, or give it generate_{word}_with_meta and full_response = True"
        )


def agent_problems(
    mas: MAS, name: str, agent: Agent, types: Mapping[str, ValueType], cannot: dict[tuple[str, str], set[str]]
) -> Iterator[str]:
    """Say which kinds an agent reads or writes that its model does not take or give; ``cannot`` gathers the rest."""
    capabilities = mas.capabilities(agent)
    model = agent.model if agent.model is not None else mas.default_model
    called = model_called(model)
    llm = mas.llms.get(agent.model) if agent.model is not None else mas.llm
    for value in agent.inputs:
        kind = media_kind(types[value]) if value in types else None
        if kind in CONVERSIONS:
            yield (
                f"agent {name!r} reads {kind} {value!r}; a model reads {a(kind)} through a conversion step, "
                f"such as one using xmas.convert.{CONVERSIONS[kind]}"
            )
        elif kind in SENT and not capabilities.takes(kind):
            subject = f"agent {name!r} reads {kind} {value!r}"
            yield lacking(subject, kind, "accepts", capabilities, model) + " or convert it first with a step"
        elif kind in SENT and llm is not None and not supports_parts(llm):
            cannot.setdefault((called, "accepts"), set()).add(kind)
    written = media_kind(parse_type(agent.output_type))
    if written is None:
        return
    if written != "image":
        yield f"agent {name!r} writes {written} {agent.output!r}; a model answers with text, json or image"
    elif agent.tools:
        yield f"agent {name!r} writes image and has tools; an image answer takes none"
    elif not capabilities.gives("image"):
        yield lacking(f"agent {name!r} writes image {agent.output!r}", "image", "produces", capabilities, model)
    elif llm is not None and not supports_images(llm):
        cannot.setdefault((called, "produces"), set()).add("image")


def step_problems(mas: MAS, name: str, step: Step, types: Mapping[str, ValueType]) -> Iterator[str]:
    """Say which media values a step passes to a remote tool, which reads text and json only."""
    try:
        item, _ = mas.resolve_callable(step)
    except ValueError:  # named by the node's own problems
        return
    for value in step_maps(step, item)[0].values() if item.fn is None else ():
        kind = media_kind(types[value]) if value in types else None
        if kind is not None:
            yield f"step {name!r} passes {kind} {value!r} to remote tool {step.uses!r}; remote tools read no media"


def human_problems(name: str, human: Human, types: Mapping[str, ValueType], edges: Sequence[Edge]) -> Iterator[str]:
    """Say which placeholders of a human node read fields of text, its options' wrong type, or a timeout's gap."""
    typed = {placeholder: types.get(value) for placeholder, value in human.reads.items()}
    paths = {path for text in human.templates() for path in template_paths(text) if len(path) > 1}
    for path in sorted(paths):
        if path[0] in typed and typed[path[0]] not in (None, JSON):
            yield f"human {name!r}: placeholder {'.'.join(path)!r} reads a field of {typed[path[0]]} {path[0]!r}"
    if isinstance(human.options, str):
        root, *fields = human.options.split(".")
        vtype = typed.get(root)
        if (vtype is not None and vtype not in (JSON, ValueType("list", TEXT))) or (
            fields and vtype not in (None, JSON)
        ):
            yield (
                f"human {name!r}: options {human.options!r} reads {vtype} {root!r}; options read from state "
                "are a list of text, or json holding one"
            )
    otherwise = any(edge.source == name and edge.kind is EdgeKind.ELSE for edge in edges)
    if human.timeout_s is not None and human.default is None and not otherwise:
        yield f"human {name!r} times out after {human.timeout_s:g} s with no default; give a default or else edge"


def predicate_problems(mas: MAS, predicates: dict[Edge, Predicate], facts: FlowFacts) -> Iterator[str]:
    """Say which predicates read values, fields or cases that do not exist, or call callables wrongly."""
    registered = mas.callables()
    for edge, predicate in predicates.items():
        for path in sorted(predicate.paths()):
            yield from path_problems(mas, edge, path, facts)
        yield from call_problems(f"edge {edge.label}", predicate, registered)


def path_problems(mas: MAS, edge: Edge, path: tuple[str, ...], facts: FlowFacts) -> Iterator[str]:
    """Say whether an edge's predicate can read the value at ``path``: readable at its source, and json for a field."""
    where = f"edge {edge.label}"
    root, fields = path[0], ".".join(path[1:])
    if root in RESERVED_READS:
        if root == "error" and edge.source not in facts.erred:
            yield f"{where} decides on 'error', {ERRED_ONLY}"
        return
    readable = root in mas.inputs or {edge.source, *facts.before[edge.source]}.intersection(facts.writers.get(root, ()))
    if not readable:
        yield f"{where} decides on {root!r}, which is not readable at {edge.source!r}"
    elif fields and facts.types.get(root) != JSON:
        yield f"{where} reads the field {fields!r} of {root!r}, which is not json"
    elif fields:
        for writer in facts.writers.get(root, ()):
            yield from human_field_problems(mas, where, writer, path)
            yield from form_field_problems(mas, edge, writer, path)


def human_field_problems(mas: MAS, where: str, writer: str, path: tuple[str, ...]) -> Iterator[str]:
    """Say when a predicate reads a field a human writer's answer does not have."""
    human = mas.humans.get(writer)
    if human is None or path[0] != human.answer:
        return
    known = (*ANSWER_FIELDS.get(human.form, ()), *COMMON_FIELDS)
    if path[1] not in known:
        yield f"{where} reads {'.'.join(path)}, but {writer!r}'s answer has the fields {list(known)}"


def form_field_problems(mas: MAS, edge: Edge, writer: str, path: tuple[str, ...]) -> Iterator[str]:
    """Say when a predicate reads a field an agent writer's answer form lacks, or switches to a case it never gives."""
    agent = mas.agents.get(writer)
    if agent is None or agent.schema is None:
        return
    where = f"edge {edge.label}"
    root, fields = path[0], ".".join(path[1:])
    form = compile_form(agent.schema)
    if fields not in form.paths():
        yield (
            f"{where} reads {root}.{fields}, but {writer!r}'s answer form has no field {fields!r} "
            f"(it has {list(form.paths())})"
        )
        return
    choices = form.choices(fields) if edge.kind is EdgeKind.SWITCH else None
    if choices is not None and edge.case not in choices:
        yield f"{where}: case {edge.case!r} is not one of {list(choices)}, the choices of {root}.{fields}"


def call_problems(where: str, predicate: Predicate, registered: ToolIndex) -> Iterator[str]:
    """Say which callables a predicate calls that are not registered, or with the wrong number of arguments."""
    for function in sorted({name for name, _ in predicate.calls()} - set(registered)):
        yield f"{where} calls {function!r}, which is no registered callable"
    for function, count in sorted(predicate.calls()):
        if function not in registered:
            continue
        takes = list(registered[function][0].reads)
        if count != len(takes):
            yield (
                f"{where} calls {function!r} with {count} argument{'s' * (count != 1)}, "
                f"but it takes {len(takes)} ({', '.join(takes) or 'none'})"
            )


def writer_count_problems(mas: MAS, writers: dict[str, list[str]], routes: RouteGraph | None) -> Iterator[str]:
    """Say which values have several writers that are neither alternatives nor merged."""
    for value, names in writers.items():
        count = len(names) + (value in mas.inputs)
        if count < SHARED_WRITERS or mas.merges.get(value) == MERGE_LAST:
            continue
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        if routes is not None and value not in mas.inputs and all(routes.exclusive(a, b) for a, b in pairs):
            continue
        sources = (["the input"] if value in mas.inputs else []) + [repr(n) for n in names]
        yield (
            f"value {value!r} has {count} writers ({', '.join(sources)}); give it one writer, make the writers "
            f"alternatives, or declare state: {{{value}: {{merge: last}}}}"
        )


def setting_problems(mas: MAS, nodes: dict[str, Node], members: set[str]) -> Iterator[str]:
    """Say what is wrong with the ``for each``, ``retry`` and shared prompt settings.

    A chain node's retry is its chain's own check's question.
    """
    yield from (f"for each names {name!r}, which is no node" for name in mas.each if name not in nodes)
    yield from retry_problems(mas, nodes, members)
    yield from shared_prompt_problems(mas)


def retry_problems(mas: MAS, nodes: dict[str, Node], members: set[str]) -> Iterator[str]:
    """Say which ``retry`` settings name no node, a box or a human, or give a wrong kind or count."""
    for name, rules in mas.retry.items():
        node = nodes.get(name)
        if node is None:
            yield f"retry names {name!r}, which is no node"
        elif isinstance(node, Box):
            yield f"box {name!r} takes no retry; retry inside its system"
        elif isinstance(node, Human):
            yield f"human {name!r} takes no retry; a person answers once"
        elif name not in members:
            yield from retry_rule_problems(name, rules)


def retry_rule_problems(name: str, rules: Mapping[str, Any]) -> Iterator[str]:
    """Say which of a node's retry rules name a kind it may not retry on, or a count that is no positive integer."""
    for kind, attempts in rules.items():
        count: Any = attempts
        if kind not in RETRYABLE:
            yield f"{name!r} retries on {kind!r}; a node retries on {sorted(RETRYABLE)} only"
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            yield f"{name!r} retries on {kind!r} {count!r} times; give a positive integer"


def shared_prompt_problems(mas: MAS) -> Iterator[str]:
    """Say which shared prompt names name no agent, take another agent's name, or join different texts."""
    sharers: dict[str, list[str]] = {}
    for name, key in mas.prompt_names.items():
        if name not in mas.agents:
            yield f"prompt names {name!r}, which is no agent"
        elif key in mas.agents and key != name:
            yield f"agent {name!r} takes the prompt {key!r}, another agent's name; name the shared prompt apart"
        else:
            sharers.setdefault(key, []).append(name)
    for key, names in sharers.items():
        if len({mas.agents[name].prompt for name in names}) > 1:
            yield f"agents {names} share the prompt {key!r} but hold different texts; give them one text"


def check_races(
    mas: MAS, edges: Sequence[Edge], writes: Writes, heads: Mapping[str, ChainShape]
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    """Check each ``first of`` race of the flow and return its problems and the races, ``{source: contestants}``."""
    races: dict[str, tuple[str, ...]] = {}
    for source, target in pairs(edges, EdgeKind.FIRST):
        races[source] = (*races.get(source, ()), target)
    entered = Counter(name for contestants in races.values() for name in contestants)
    problems = [
        f"{name!r} runs in {count} first of races; a node races in one" for name, count in entered.items() if count > 1
    ]
    for source, contestants in races.items():
        where = f"the first of race from {source!r}"
        problems += race_start_problems(where, source, contestants, edges)
        after, found = contestant_problems(mas, where, contestants, edges, heads)
        problems += found
        problems += race_end_problems(where, contestants, after, writes)
    return problems, races


def race_start_problems(where: str, source: str, contestants: tuple[str, ...], edges: Sequence[Edge]) -> Iterator[str]:
    """Say when a race has one contestant, or its source also decides with when, switch or else edges."""
    if len(contestants) < MIN_RACE_CONTESTANTS:
        yield f"{where} has one contestant, {contestants[0]!r}; a race needs two or more"
    if any(e.source == source and e.kind in (EdgeKind.WHEN, EdgeKind.SWITCH, EdgeKind.ELSE) for e in edges):
        yield f"{where}: {source!r} also decides with when, switch or else edges; race from a node of its own"


def contestant_problems(
    mas: MAS, where: str, contestants: tuple[str, ...], edges: Sequence[Edge], heads: Mapping[str, ChainShape]
) -> tuple[dict[str, str], list[str]]:
    """Return where each well-formed contestant goes on to, and the problems of the contestants that cannot race."""
    after: dict[str, str] = {}
    problems: list[str] = []
    for name in contestants:
        kind = "a chain head" if name in heads else "a human node" if name in mas.humans else None
        kind = "the end" if name == END else kind
        out = [e for e in edges if e.source == name and e.kind is not EdgeKind.WHILE]
        if kind is not None:
            problems.append(f"{where}: {name!r} is {kind}; agents, steps and boxes race")
        elif pauses(mas, name):
            problems.append(f"{where}: box {name!r} holds a human node; a contestant never waits for a person")
        if any(e.kind is EdgeKind.ERROR for e in out):
            problems.append(f"{where}: {name!r} has an on error edge; a contestant's failure is the race's")
        elif len(out) > 1 or any(e.kind is not EdgeKind.PLAIN for e in out):
            problems.append(f"{where}: {name!r} has {len(out)} out-edges; a contestant has one plain edge or none")
        else:
            after[name] = out[0].target if out else END
    return after, problems


def race_end_problems(where: str, contestants: tuple[str, ...], after: dict[str, str], writes: Writes) -> Iterator[str]:
    """Say when a race's contestants go on to different nodes or write different values."""
    if len(set(after.values())) > 1:
        yield f"{where}: its contestants go on to different nodes {after}; they go on to one"
    written = {name: {v: str(t) for v, t in writes.get(name, {}).items()} for name in contestants}
    if len({tuple(sorted(w.items())) for w in written.values()}) > 1:
        yield f"{where}: its contestants write differently {written}; they write the same values"


def outside_reads(shape: ChainShape, reads: Reads, writes: Writes) -> list[str]:
    """Return the values a chain's nodes read from outside it, in order; the item, its writes and reserved reads not."""
    written = {value for name in shape.nodes for value in writes[name]}
    skip = {*written, shape.each.item, *RESERVED_READS}
    return list(dict.fromkeys(value for name in shape.nodes for value in reads[name] if value not in skip))


def check_chains(
    mas: MAS,
    shapes: list[ChainShape],
    nodes: dict[str, Node],
    io: tuple[Reads, Writes],
    types: dict[str, ValueType],
) -> tuple[dict[str, Chain], list[str]]:
    """Check each chain at its edges and build the system one item runs, whose own check judges the inside."""
    reads, writes = io
    chains: dict[str, Chain] = {}
    problems: list[str] = []
    for shape in shapes:
        each = shape.each
        root = types.get(each.root)
        listed = root.item if root is not None and not each.fields else None
        item = JSON if listed is None else listed
        found = []
        if root is not None and listed is None and root != JSON:
            found.append(f"for each {each} runs over {root} {each.root!r}; a chain runs over json or a list")
        if each.key is not None and item != JSON:
            found.append(f"key {each.key!r} needs json items, not {item}")
        found += [f"the item {each.item!r} {problem}" for problem in value_problems(each.item)]
        if each.item in types:
            found.append(f"the item {each.item!r} is an input or a written value too; give it a name of its own")
        found += [
            f"{other!r} writes {value!r}, which the chain writes; a value written in a chain has no writer outside it"
            for name in shape.nodes
            for value in writes[name]
            for other in nodes
            if other not in shape.nodes and value in writes[other]
        ]
        found += [
            f"box {name!r} holds a for each chain; chains do not nest"
            for name in shape.nodes
            if isinstance(box := nodes[name], Box) and deep(box.system, lambda system: bool(system.each))
        ]
        if not shape.exits and not mas.outputs:
            found.append("it feeds the end, which needs declared outputs")
        inputs = {each.item: item, **{v: types[v] for v in outside_reads(shape, reads, writes) if v in types}}
        try:
            system = mas.chain_system(shape, inputs, {v: t for name in shape.nodes for v, t in writes[name].items()})
        except CheckError as exc:
            found += exc.problems
        except ValueError as exc:
            found.append(str(exc))
        problems += [f"chain {shape.head!r}: {problem}" for problem in found]
        if not found:
            info = system.checked()
            written = {value: info.types[value] for value in info.writers}
            chains[shape.head] = Chain(shape.head, shape.nodes, each, inputs, written, system)
    return chains, problems


def check_system(mas: MAS) -> SystemInfo:
    """Check a version 2 system and return what the engine needs to know.

    Raises:
        ValueError: Listing every problem found, one per line.
    """
    nodes: dict[str, Node] = {**mas.agents, **mas.steps, **mas.boxes, **mas.humans}
    problems = name_problems(mas, nodes)
    reads, writes, io_problems = node_io(mas, nodes)
    shapes, shape_problems = find_chains(list(nodes), mas.edges, {n: f for n, f in mas.each.items() if n in nodes})
    members = {name for shape in shapes for name in shape.nodes}
    problems += [problem for name, found in io_problems.items() if name not in members for problem in found]
    problems += setting_problems(mas, nodes, members)
    if shape_problems:
        raise CheckError(problems + shape_problems)

    # The flow: every chain collapsed into its head, which writes the chain's values as lists.
    heads = {shape.head: shape for shape in shapes}
    flow_nodes = [name for name in nodes if name not in members or name in heads]
    flow_writes = flow_writes_of(flow_nodes, heads, writes)
    types, writers, type_problems = writer_types(mas, flow_writes, list(heads))
    chains, chain_problems = check_chains(mas, shapes, nodes, (reads, writes), types)
    flow_reads = flow_reads_of(flow_nodes, heads, (reads, writes), types)
    edges = collapse(shapes, mas.edges)
    edge_problems, predicates, buildable = check_edges(edges, {*flow_nodes, *SENTINELS})
    race_problems, races = check_races(mas, edges, flow_writes, heads)
    problems += type_problems + chain_problems + edge_problems + race_problems

    flow: nx.DiGraph[str] | None = None
    routes: RouteGraph | None = None
    joins: dict[str, Join] = {}
    typed = {**RESERVED_READS, **types}
    if buildable:
        flow, routes, joins, topology_problems = topology(flow_nodes, edges, outputs=bool(mas.outputs))
        problems += topology_problems
        if routes is not None:
            before = {node: ancestors(flow, node) for node in flow}
            erred = {target for _, target in pairs(edges, EdgeKind.ERROR)}
            facts = FlowFacts(typed, writers, routes, before, erred)
            problems += read_problems(mas, flow_reads, facts)
            problems += predicate_problems(mas, predicates, facts)
    problems += writer_count_problems(mas, writers, routes)
    problems += node_problems(mas, nodes, typed, edges, members)
    problems += output_problems(mas, types)

    if problems or flow is None:  # an edge list that makes no graph is always a problem of the list
        raise CheckError(problems)
    types.update(reserved_reads_used(flow_reads, predicates))
    graph = topology_graph(list(nodes), mas.edges) if shapes else flow
    writers_of = {value: tuple(names) for value, names in writers.items()}
    pausing = pausing_nodes(mas, flow_nodes, heads)
    return SystemInfo(types, writers_of, joins, graph, predicates, chains, edges, flow, pausing, races)


def name_problems(mas: MAS, nodes: dict[str, Node]) -> list[str]:
    """Say which node and input names are not allowed, and which frozen names name no node."""
    problems = [f"node {name!r}: names starting with '__' belong to xmas" for name in nodes if name.startswith("__")]
    problems += [f"frozen names {name!r}, which is no node" for name in sorted(mas.frozen - set(nodes))]
    for name in mas.inputs:
        problems += [f"input {name!r} {problem}" for problem in value_problems(name)]
    return problems


def flow_writes_of(flow_nodes: list[str], heads: dict[str, ChainShape], writes: Writes) -> Writes:
    """Return what each node of the flow writes; a chain's head writes its chain's values, each as a list."""
    return {
        name: {v: ValueType("list", t) for m in heads[name].nodes for v, t in writes[m].items()}
        if name in heads
        else writes[name]
        for name in flow_nodes
    }


def flow_reads_of(
    flow_nodes: list[str], heads: dict[str, ChainShape], io: tuple[Reads, Writes], types: dict[str, ValueType]
) -> Reads:
    """Return what each node of the flow reads; a chain's head reads what its chain reads from outside, and its root."""
    reads, writes = io
    flow_reads = {name: reads[name] for name in flow_nodes}
    for head, shape in heads.items():
        outside = [v for v in outside_reads(shape, reads, writes) if v in types]
        flow_reads[head] = dict.fromkeys([*outside, shape.each.root])
    return flow_reads


def output_problems(mas: MAS, types: dict[str, ValueType]) -> Iterator[str]:
    """Say which declared outputs no node writes, or are written with another type."""
    for name, vtype in mas.outputs.items():
        if name not in types:
            yield f"output {name!r} is neither an input nor written by any node"
        elif types[name] != vtype:
            yield f"output {name!r} is declared {vtype} but is {types[name]}"


def reserved_reads_used(flow_reads: Reads, predicates: dict[Edge, Predicate]) -> dict[str, ValueType]:
    """Return the reserved reads some node or predicate of the flow reads, with their types."""
    read = {v for needed in flow_reads.values() for v in needed} | {
        path[0] for predicate in predicates.values() for path in predicate.paths()
    }
    return {value: vtype for value, vtype in RESERVED_READS.items() if value in read}


def pausing_nodes(mas: MAS, flow_nodes: list[str], heads: dict[str, ChainShape]) -> tuple[str, ...]:
    """Return the nodes of the flow that may pause the run: a human, or a chain head or box that holds one."""
    return tuple(
        name
        for name in flow_nodes
        if any(pauses(mas, node) for node in (heads[name].nodes if name in heads else (name,)))
    )
