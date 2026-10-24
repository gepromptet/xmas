"""Compile a MAS and run it.

The topology compiles to a LangGraph state graph here and nowhere else. No type
from langgraph crosses back out: callers receive a :class:`~xmas.trace.Trace`,
which is xmas's own data.

Each run gets its own :class:`~xmas.budget.RunMeter`, handed to the nodes in the
run config under :data:`RUN_KEY`. Nodes check the budget on it before each call
and keep every finished visit's record there, so a failed run still returns
what it did. When the system's store is a file, the meter also writes each
record, event and model call to the run's :class:`~xmas.store.RunLog` as it
happens, and the run's start and end are written beside them.

In a version 2 system values are typed and render by
:func:`~xmas.values.render_value`; steps run callables; a json agent answers in
its answer form, with one retry; an agent reading an image or audio sends it to
its model as a content part, and an agent writing an image keeps the model's
picture in :func:`~xmas.media.store`; a node with conditional or error edges
routes by a :class:`Decision` after each visit; a join is a barrier that runs
once per pass, after all its groups, where version 1 defers; and a box runs a
whole inner system as one node, on the outer run's meter.

A ``for each`` chain compiles to three nodes around its head ``h``:
``__each__.h.in`` starts a pass and fans the list out with one ``Send`` per
item, ``__each__.h`` runs the chain's own system on one item, and
``__each__.h.out`` gathers each write into a list aligned to the items. Every
record a chain's item makes carries the item, and records keep the order of the
items, however the items finish; an item resumed after a pause adds its records
when it ends, after those of the items that ended before the pause.

A system with people in it (a :class:`~xmas.human.Human` at any depth) can
pause. A human ``h`` compiles to two nodes: ``h`` asks (it fills the question,
keeps it in the run's :class:`~xmas.store.Store` and calls the host's asker
once) and the deferred ``h.wait`` takes the answer, so every other branch ends
or reaches its own question first. A wait without an answer interrupts the
graph; the run returns a trace with status ``paused`` and the open questions in
``Trace.pause``, its checkpoints in the store under the run's id, and
:func:`resume` goes on from there, in this process or after a restart. A chain
or box holding a human runs its inner system as a root run of its own thread,
so a pause inside it returns instead of holding the outer step; the
container's deferred ``wait`` node resumes only the paused threads.
"""

from __future__ import annotations

import contextvars
import dataclasses
import json
import operator
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

from langchain_core.runnables.config import get_executor_for_config, var_child_runnable_config
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp, GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.types import Command, Interrupt, Send, interrupt
from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from promptolution.utils.token_counter import get_token_counter

from xmas.agent import Agent
from xmas.budget import RUNNING, RunMeter, VisitMeter, contestant
from xmas.check import SENT, step_maps
from xmas.edges import ANY_ERROR, Edge, EdgeKind
from xmas.errors import (
    RUN_ENDS,
    BadFormat,
    BudgetExceeded,
    ErrorKind,
    RaceLost,
    RunStatus,
    Stopped,
    Unanswered,
    classify,
)
from xmas.graph import END, START, back_edges, join_nodes
from xmas.health import summarize
from xmas.human import Asker, Human, Pause, Question, answer_value, question_id
from xmas.llms.meta import SupportsImages, SupportsMeta, SupportsParts, supports_meta, supports_parts
from xmas.llms.parts import Part, render_parts
from xmas.media import store
from xmas.nodes import Box, Step
from xmas.predicates import Context, Predicate
from xmas.prompts import safe_format
from xmas.schemas import RETRY_NOTE, AnswerForm, FormError, compile_form
from xmas.spec import VERSION_2
from xmas.store import RunLog, Store
from xmas.tools import run_step, run_tool_loop
from xmas.trace import AgentRecord, Event, Rejected, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType, check_value, coerce_media, media_kind, parse_type, plain, render_value

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.budget import Budget
    from xmas.chains import Chain
    from xmas.llms.meta import CallMeta
    from xmas.mas import MAS
    from xmas.routes import Join

__all__ = [
    "CONTINUE_MARKER",
    "MASState",
    "Prompt",
    "answer",
    "compile_mas",
    "cycle_router",
    "node_owner",
    "node_types",
    "pauses",
    "render_prompt",
    "resume",
    "run_many",
    "run_node",
    "run_state",
    "stop",
    "trace_from_state",
]


CONTINUE_MARKER = "<continue>"
"""Marker an agent emits to take a back edge instead of moving on."""

BRANCH_THREADS = 64
"""Threads one graph run (and each query of a batch) may use at once, as LangGraph's ``max_concurrency``.

Model calls wait on the network, not the CPU, and a chain item waiting for its
``at once`` place holds a thread too, so the pool is sized for waiting branches
rather than LangGraph's default of the host's cores plus four; threads start
only when a branch needs one.
"""

BATCH_QUERIES = 16
"""Queries a batch (:func:`run_many`) runs at once, each on its own pool of :data:`BRANCH_THREADS`.

Smaller than the branch pool because one query already fans out inside: 64
queries at once, each with its branches, could overload a model gateway. 16 is
near the old width on a desktop (cores plus four) and the same on every host.
"""

RUN_KEY = "xmas_run"
"""Key of the run's :class:`~xmas.budget.RunMeter` in ``config["configurable"]``."""

ITEM_KEY = "xmas_item"
"""Key of the chain item a run works on, ``(index, key)``, in ``config["configurable"]``."""

ASKING_KEY = "xmas_asking"
"""Key of a run's :class:`Asking`, in ``config["configurable"]`` of a system that can pause."""

RACES_KEY = "xmas_races"
"""Key of a run's own prefix for its ``first of`` race keys, in ``config["configurable"]`` of a system with races:
a box's or a chain item's inner run races on the outer run's meter, under keys of its own."""

Prompt = str | list[Part]
"""What an agent sends its model: one text, or content parts when it reads an image or audio."""


class MASState(TypedDict):
    """The state a run threads through the graph.

    Attributes:
        values: Every value, ``query`` included, keyed by name: a ``str`` for
            text, plain JSON data for json. Nodes writing concurrently each
            contribute their own key, so the reducer is a plain dict union.
        prompts: The run's prompt set, keyed by agent name. Prompts travel in
            the state, so one compilation serves every candidate an optimizer
            proposes.
        records: One record per visit, in completion order, so they also count
            the visits.
        routes: Where each deciding node's latest visit sent the run.
    """

    values: Annotated[dict[str, Any], operator.or_]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]
    routes: Annotated[dict[str, list[str]], operator.or_]


def merge_items(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge chain updates: an entry with ``n`` starts the chain's pass afresh, ``rows`` add items to it."""
    merged = dict(left)
    for head, part in right.items():
        if "n" in part:
            merged[head] = dict(part)
        else:
            entry = merged.get(head, {})
            merged[head] = {**entry, "rows": {**entry.get("rows", {}), **part["rows"]}}
    return merged


class ChainState(MASState):
    """The state of a system with chains.

    Attributes:
        items: Each chain's current pass by head: ``n`` items, the pass's
            number, and one row per finished item, ``{"values": ...}`` or
            ``{"error": ..., "kind": ...}``.
    """

    items: Annotated[dict[str, Any], merge_items]


class AskState(MASState):
    """The state of a system with human nodes or pausing boxes.

    Attributes:
        asks: By node: a human's latest question (``Question.to_dict()``), or a
            box's paused inner thread (``{"thread", "scope", "query"}``),
            ``None`` once it finished.
    """

    asks: Annotated[dict[str, Any], operator.or_]


class ChainAskState(ChainState, AskState):
    """The state of a system with chains and human nodes or pausing boxes."""


@dataclasses.dataclass(frozen=True)
class Asking:
    """How a run that can pause asks: where it keeps questions, who it asks, and where in the run a node is.

    Attributes:
        store: Questions, answers and checkpoints.
        asker: The host's asker, or ``None``.
        run_id: The run's id, the top-level thread.
        thread: The checkpoint thread of the system running now: the run's
            id, or a chain item's or box's own thread under it.
        scope: What the questions asked here start their ids with.
        unattended: A run no one can answer later (``run``, ``run_batch``):
            an unanswered question takes its default or fails the run.
        resumed: The questions the run was paused on when it was resumed.
    """

    store: Store
    asker: Asker | None
    run_id: str
    thread: str
    scope: str
    unattended: bool = False
    resumed: frozenset[str] = frozenset()


class Paused(GraphBubbleUp):
    """A box's inner run paused: the box's wait node finishes the visit once its questions are answered."""

    def __init__(self, marker: dict[str, Any]) -> None:
        """Keep what the wait node needs: the inner thread, its question scope and the box's query."""
        super().__init__(marker)
        self.marker = marker


NodeBody = Callable[[MASState, VisitMeter], tuple[AgentRecord, dict[str, Any]]]
"""What a node does in one visit, on its share of the run's meter: its record, and the values it writes."""


def current_meter(mas: MAS) -> RunMeter:
    """Return the run's meter.

    A graph invoked without a run config, or inheriting another system's (a
    nested run inside a node), counts on a meter of its own.
    """
    meter = get_config().get("configurable", {}).get(RUN_KEY)
    return meter if isinstance(meter, RunMeter) and meter.owns(mas) else RunMeter(mas.budget, owner=mas)


def current_item() -> tuple[int, str | None] | None:
    """Return the chain item the running node works on, ``(index, key)``, or ``None`` outside a chain."""
    item: tuple[int, str | None] | None = get_config().get("configurable", {}).get(ITEM_KEY)
    return item


def current_asking() -> Asking:
    """Return the running node's :class:`Asking`."""
    asking: Asking = get_config()["configurable"][ASKING_KEY]
    return asking


def pauses(mas: MAS) -> bool:
    """Return whether a system can pause: it, or a chain or box in it at any depth, holds a human node."""
    inner = [box.system for box in mas.boxes.values()] + [chain.system for chain in system_chains(mas).values()]
    return bool(mas.humans) or any(pauses(system) for system in inner)


def system_chains(mas: MAS) -> dict[str, Chain]:
    """Return the system's ``for each`` chains by head; none in version 1."""
    return mas.info.chains if mas.info is not None else {}


def run_value(meter: RunMeter, mas: MAS) -> dict[str, Any]:
    """Return the reserved value ``run``: what the run has spent so far, and the visits finished in ``mas``.

    ``cost_usd`` is ``None`` while a call's cost is unknown; ``records`` counts
    chain items' visits too.
    """
    return {
        "cost_usd": None if meter.uncosted_calls else meter.cost_usd,
        "elapsed_s": meter.elapsed_s,
        "calls": meter.calls,
        "records": len(meter.records_of(mas)),
    }


def generate(llm: BaseLLM, rendered: Prompt) -> tuple[str, CallMeta | None]:
    """Make a plain agent's one model call; the metadata is ``None`` unless the model opted in to ``full_response``.

    Content parts go through ``generate_parts_with_meta``, which the check
    made sure the model has.
    """
    if not isinstance(rendered, str):
        if not supports_parts(llm):
            raise TypeError(f"{type(llm).__name__} cannot send pictures or sound: it has no parts path turned on")
        return cast("SupportsParts", llm).generate_parts_with_meta(rendered)
    if supports_meta(llm):
        return cast("SupportsMeta", llm).generate_with_meta(rendered)
    return llm.get_response([rendered])[0], None


def prompt_text(prompt: Prompt) -> str:
    """Return a prompt's text parts, joined: what a token estimate counts."""
    return prompt if isinstance(prompt, str) else "".join(part for part in prompt if isinstance(part, str))


def joined(prompt: Prompt, text: str) -> Prompt:
    """Return a prompt with ``text`` added at its end, to its last text part when it ends in one."""
    if isinstance(prompt, str):
        return prompt + text
    if prompt and isinstance(prompt[-1], str):
        return [*prompt[:-1], prompt[-1] + text]
    return [*prompt, text]


def call(
    llm: BaseLLM, rendered: Prompt, meter: VisitMeter, *, batch_error: bool = False
) -> tuple[str, CallMeta | None]:
    """Make one metered model call.

    With ``batch_error`` (a version 1 agent) a failed call raises as
    promptolution's batch call does, ``RuntimeError: LLM batch failed: ...``,
    whichever path the model takes; :func:`~xmas.errors.classify` reads the
    kind from the cause.
    """
    try:
        return meter.metered(lambda: generate(llm, rendered), llm=llm)
    except Exception as error:
        if not batch_error or not supports_meta(llm) or isinstance(error, BudgetExceeded | GraphBubbleUp):
            raise  # promptolution's get_response wrapped it already, the meter refused the call, or LangGraph signals
        raise RuntimeError(f"LLM batch failed: {error}") from error


@dataclasses.dataclass(frozen=True)
class NodeRole:
    """How a wrapped node decides and reports, beyond what its body does.

    Attributes:
        decision: Where the node goes after a visit, ``None`` for a node
            without conditional or error edges.
        writes_error: Whether a visit an error edge routes writes ``error``.
        starts: Whether the visit starts here and emits its ``start``; a wait
            node finishes a visit its asking node started.
        race: The node's ``first of`` race, its source and how many race.
    """

    decision: Decision | None = None
    writes_error: bool = False
    starts: bool = True
    race: tuple[str, int] | None = None


PLAIN_ROLE = NodeRole()
"""The role of a node that starts its visits, decides nothing and races no one."""


def wrap(mas: MAS, name: str, body: NodeBody, role: NodeRole = PLAIN_ROLE) -> Callable[[MASState], dict[str, Any]]:
    """Turn a node's body into the node LangGraph runs.

    The node times the body, keeps its record on the run's meter and writes its
    values; a node with a decision also writes where it goes in ``routes``. It
    emits a ``start`` and an ``end`` event on the meter.

    In version 2 every record carries its visit number (among this node's
    records in the state, so per item inside a chain), the chain item it worked
    on, and the retries it used: a kind the node's ``retry`` names runs the body
    again, on the same visit meter, while attempts are left. A visit that raises
    gets a failed record (:meth:`Visit.failed_record`) holding every paid call,
    unless the error ends the run (:data:`~xmas.errors.RUN_ENDS`); a visit a
    stop cut after it made a call keeps one, with what its calls used and the
    answer received before the cut. When an error edge routes the error's kind,
    the visit takes it and writes only ``error`` (``{kind, message, node}``,
    when ``role.writes_error``); otherwise the error goes on. Version 1 raises.

    A box whose inner run paused keeps no record yet: the node writes the
    paused thread under ``asks`` for the box's wait node, which finishes the
    visit (``role.starts`` false: the visit has started already), as a human's
    wait node finishes the visit its asking node started.

    A contestant of a ``first of`` race (``role.race``: its source and how many
    race) runs on the race's flag, keyed by its run, source, item and visit. The first
    to finish wins and writes; a later one, or one whose call the win cut,
    writes nothing and keeps a ``lost`` record of what it paid. A contestant
    that fails goes on no further, and the race fails with the last one when
    every contestant failed.
    """
    retry = mas.retry.get(name, {})

    def node(state: MASState) -> dict[str, Any]:
        visit = Visit.begin(mas, name, state, role)
        try:
            record, status, update = visit.attempt(body, state, retry)
        except Paused as paused:
            return {"asks": {name: paused.marker}}
        except GraphBubbleUp:  # LangGraph's own signals, such as an interrupt, pass untouched
            raise
        except Exception as error:
            routed = visit.failed(error)
            if routed is None:
                raise
            return routed
        return visit.end(record, status, **update)

    return node


KINDS = {Agent: "agent", Step: "step", Box: "box", Human: "human"}
"""The record kind of each node class."""


@dataclasses.dataclass(frozen=True)
class Visit:
    """One visit of a wrapped node: where it runs, what it used, and its race, if it races.

    Attributes:
        mas: The system the node belongs to.
        name: The node's name.
        role: How the node decides and reports (:class:`NodeRole`).
        meter: The run's meter.
        here: The visit number and the chain item, stamped on its events and record.
        started: When the visit started, by ``time.perf_counter``.
        visit: The visit's share of the run's meter.
        used: The retries the visit used, by error kind.
        contest: The visit's race key, ``None`` outside a race.
    """

    mas: MAS
    name: str
    role: NodeRole
    meter: RunMeter
    here: dict[str, Any]
    started: float
    visit: VisitMeter
    used: Counter[str]
    contest: str | None

    @classmethod
    def begin(cls, mas: MAS, name: str, state: MASState, role: NodeRole) -> Visit:
        """Start a visit on the run's meter: count it, emit its ``start`` and key its race."""
        meter = current_meter(mas)
        meter.start()  # the run's wall clock starts with its first agent, not with the batch
        index, key = current_item() or (None, None)
        visits = 1 + sum(record.agent == name for record in state["records"])
        here: dict[str, Any] = {"visit": visits, "item": index, "item_key": key}
        if role.starts:
            meter.emit("start", name, **here)
        started, visit = time.perf_counter(), VisitMeter(meter, name)
        races = get_config().get("configurable", {}).get(RACES_KEY)
        contest = None if role.race is None else f"{races}:{role.race[0]}:{index}:{visits}"
        return cls(mas, name, role, meter, here, started, visit, Counter[str](), contest)

    @property
    def entrants(self) -> int:
        """How many race in the visit's race, ``0`` outside one."""
        return 0 if self.role.race is None else self.role.race[1]

    def attempt(
        self, body: NodeBody, state: MASState, retry: Mapping[str, int]
    ) -> tuple[AgentRecord, str, dict[str, Any]]:
        """Run the body with its retries, in the race if the node races; return the record, its status and its update.

        A contestant that finishes after the winner writes nothing and keeps a ``lost`` record.
        """
        with nullcontext() if self.contest is None else contestant(self.meter.race(self.contest)):
            record, values = attempts(body, state, self.visit, retry, self.used)
        if self.used and record.kind == "agent":  # its cost is that of every attempt
            paid = self.visit.meta
            record = dataclasses.replace(record, **paid.record_fields() if paid else {"cost_usd": None, "tokens": None})
        if self.contest is not None and not self.meter.win(self.contest, self.name):
            return dataclasses.replace(record, outputs={}, lost=True, route=()), "lost", {"routes": {self.name: []}}
        update: dict[str, Any] = {"values": values}
        decision = self.role.decision
        if decision is not None:
            route, capped = decision.route(context(self.mas, self.name, state, values, self.meter))
            record = dataclasses.replace(record, route=tuple(route), capped=capped)
            update["routes"] = {self.name: route}
        return record, "ok", update

    def end(self, record: AgentRecord, status: str, **update: Any) -> dict[str, Any]:
        """Stamp the visit's record, keep it on the meter, and emit the visit's end."""
        stamps = {**self.here, "retries": self.used.total()} if self.mas.version == VERSION_2 else {}
        record = dataclasses.replace(record, latency_s=time.perf_counter() - self.started, **stamps)
        self.meter.keep(record, self.mas)
        self.meter.emit("end", self.name, status=status, cost_usd=record.cost_usd, **self.here)
        return {**update, "records": [record]}

    def failed(self, error: Exception) -> dict[str, Any] | None:
        """Keep a failed visit's record and return its update; ``None`` when the error goes on.

        A contestant the win cut keeps a ``lost`` record. An error that ends
        the run, or any error in version 1, keeps no record here.
        """
        if self.contest is not None and isinstance(error, RaceLost):
            return self.lost_race(error)
        kind = kind_of(error)
        stopped = kind is ErrorKind.STOPPED
        cut = stopped and bool(self.visit.metas or self.visit.inner)
        if self.mas.version != VERSION_2 or (kind in RUN_ENDS and not cut) or isinstance(error, Unanswered):
            self.meter.emit("end", self.name, status="stopped" if stopped else "failed", **self.here)
            return None
        target = None if self.role.decision is None or cut else self.role.decision.on_error(kind)
        record = self.failed_record(error, kind, target)
        if cut:  # the answer received before the cut
            record = dataclasses.replace(record, raw_response=error.text if isinstance(error, Stopped) else "")
        failed = self.end(record, "stopped" if stopped else "failed" if target is None else "routed")
        return self.onward(failed, kind, target)

    def failed_record(self, error: BaseException, kind: ErrorKind, target: str | None) -> AgentRecord:
        """Return the record of a version 2 visit that failed: no outputs, the route taken, and what it paid.

        The cost and tokens are those of every call the visit made; a step's are
        ``0``, and a box's those of its inner trace, which the record holds.
        """
        node, visit = self.mas.nodes[self.name], self.visit
        meta = visit.meta
        paid: dict[str, Any] = meta.record_fields() if meta is not None else {}
        if isinstance(node, Step):
            paid["cost_usd"] = 0.0
        elif visit.inner is not None:
            paid.update(paid_inside(visit.inner))
        paid["error_kind"] = kind
        return AgentRecord(
            agent=self.name,
            step=-1,
            prompt="",
            rendered_prompt="",
            inputs={},
            outputs={},
            kind=KINDS[type(node)],
            error=f"{type(error).__name__}: {error}",
            route=None if target is None else (target,),
            inner=visit.inner,
            **paid,
        )

    def lost_race(self, error: RaceLost) -> dict[str, Any]:
        """Keep the ``lost`` record of a contestant whose call the win cut: what it paid and the answer so far."""
        record = self.failed_record(error, ErrorKind.STOPPED, None)
        record = dataclasses.replace(record, error=None, error_kind=None, raw_response=error.text, lost=True)
        return self.end(dataclasses.replace(record, route=()), "lost", routes={self.name: []})

    def others_race_on(self, kind: ErrorKind) -> bool:
        """Count a failed contestant out of its race; return whether other contestants still race on."""
        if self.contest is None or kind is ErrorKind.STOPPED:
            return False
        return self.meter.fail(self.contest) < self.entrants

    def onward(self, failed: dict[str, Any], kind: ErrorKind, target: str | None) -> dict[str, Any] | None:
        """Where a failed visit goes: the others race on, its error edge routes it, or (``None``) the error goes on."""
        if self.others_race_on(kind):
            return {**failed, "routes": {self.name: []}}
        if target is None:
            return None
        if self.role.writes_error:
            message = failed["records"][0].error
            failed["values"] = {"error": {"kind": kind.value, "message": message, "node": self.name}}
        return {**failed, "routes": {self.name: [target]}}


def attempts(
    body: NodeBody, state: MASState, visit: VisitMeter, retry: Mapping[str, int], used: Counter[str]
) -> tuple[AgentRecord, dict[str, Any]]:
    """Run a visit's body; run it again on an error whose kind ``retry`` names, while attempts are left (``used``)."""
    while True:
        try:
            return body(state, visit)
        except GraphBubbleUp:
            raise
        except Exception as error:
            kind = kind_of(error).value
            if used[kind] >= retry.get(kind, 0):
                raise
            used[kind] += 1


def estimate_usage(count_tokens: Callable[[str], float], sent: str, received: str) -> TokenUsage:
    """Estimate one call's usage."""
    return TokenUsage(input_tokens=int(count_tokens(sent)), output_tokens=int(count_tokens(received)))


def combine(first: CallMeta | None, second: CallMeta | None) -> CallMeta | None:
    """Combine two calls' metadata; a model that gives none gives none for either."""
    return second if first is None else first if second is None else first.combine(second)


def answer_plain(
    llm: BaseLLM,
    rendered: Prompt,
    form: AnswerForm,
    count_tokens: Callable[[str], float],
    meter: VisitMeter,
) -> tuple[str, Any, tuple[Rejected, ...], TokenUsage, CallMeta | None]:
    """Ask for an answer in its form, once more with :data:`~xmas.schemas.RETRY_NOTE` if it does not fit.

    Returns the accepted text, its value, the rejected answer (if any), and
    usage and metadata summed over both calls. A retry that does not fit
    either raises :class:`~xmas.errors.BadFormat` with both problems.
    """
    response, meta = call(llm, rendered, meter)
    usage = estimate_usage(count_tokens, prompt_text(rendered), response)
    try:
        return response, form.parse(response), (), usage, meta
    except FormError as first:
        retry = joined(rendered, f"\n\n{RETRY_NOTE.format(problem=first.problem)}")
        again, again_meta = call(llm, retry, meter)
        usage, meta = usage + estimate_usage(count_tokens, prompt_text(retry), again), combine(meta, again_meta)
        try:
            value = form.parse(again)
        except FormError as second:
            node = cast("str", meter.node)  # a visit's meter names its node
            raise BadFormat(node, (first.problem, second.problem), meta) from second
        return again, value, (Rejected(response, first.problem),), usage, meta


def paint(
    node: str, llm: BaseLLM, prompt: Prompt, vtype: ValueType, meter: VisitMeter
) -> tuple[str, Any, CallMeta | None]:
    """Ask for an image answer and keep each picture in the store; return the text, the ref (or refs) and the meta.

    An ``image`` takes exactly one picture, a ``list of image`` at least one;
    anything else raises :class:`~xmas.errors.BadFormat` with the count.
    """
    (text, images), meta = meter.metered(lambda: generate_images(llm, prompt), llm=llm)
    wanted = "at least one image" if vtype.item is not None else "one image"
    if not images or (vtype.item is None and len(images) > 1):
        raise BadFormat(node, (f"expected {wanted}, got {len(images)}",), meta)
    refs = [store().put(data, media_type) for data, media_type in images]
    return text, refs if vtype.item is not None else refs[0], meta


def generate_images(llm: BaseLLM, prompt: Prompt) -> tuple[tuple[str, list[tuple[bytes, str]]], CallMeta]:
    """Make one image call through ``generate_image_with_meta``, which the check made sure the model has."""
    text, images, meta = cast("SupportsImages", llm).generate_image_with_meta(prompt)
    return (text, list(images)), meta


def render_prompt(
    agent: Agent, template: str, values: Mapping[str, Any], types: Mapping[str, ValueType]
) -> tuple[Prompt, str, dict[str, str]]:
    """Render an agent's prompt from the values it reads, as a visit sends it.

    Each read renders by its type, text when ``types`` does not name it, and a
    read absent from ``values`` renders as ``""``. An image or audio read goes
    to the model as a content part (``render_parts``), and the record text
    shows a marker in its place. Returns what is sent, the record text, and
    the rendered inputs by placeholder.
    """
    inputs = {
        placeholder: render_value(values.get(name), types.get(name, TEXT), present=name in values)
        for placeholder, name in agent.reads.items()
    }
    shown = {placeholder for placeholder, name in agent.reads.items() if media_kind(types.get(name, TEXT)) in SENT}
    if not shown:
        prompt = safe_format(template, inputs)
        return prompt, prompt, inputs
    text = {placeholder: line for placeholder, line in inputs.items() if placeholder not in shown}
    parts, rendered = render_parts(template, text, {p: values.get(agent.reads[p]) for p in shown})
    return parts, rendered, inputs


def agent_body(
    mas: MAS, agent: Agent, llm: BaseLLM, count_tokens: Callable[[str], float], types: Mapping[str, ValueType]
) -> NodeBody:
    """Build what an agent does in one visit.

    The prompt renders by :func:`render_prompt`. A json agent's prompt ends
    with its form's format block, and its answer is parsed into the value it
    writes: a plain agent retries in a second call (:func:`answer_plain`), an
    agent with tools in one more turn of its conversation, tools withheld. An
    agent writing an image keeps the model's picture (:func:`paint`).
    """
    written = parse_type(agent.output_type)
    form = compile_form(agent.schema) if written == JSON else None
    paints = media_kind(written) == "image"
    tools = mas.resolve_tools(agent) if agent.tools else None
    batch_error = mas.version != VERSION_2
    reads_run = mas.version == VERSION_2 and "run" in agent.reads.values()

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        template = state["prompts"].get(agent.name, agent.prompt)
        values = {**state["values"], "run": run_value(meter.run, mas)} if reads_run else state["values"]
        prompt, rendered, inputs = render_prompt(agent, template, values, types)
        if form is not None:
            block = f"\n\n{form.format_block()}"
            prompt, rendered = joined(prompt, block), rendered + block

        calls: tuple[ToolCall, ...] = ()
        rejected: tuple[Rejected, ...] = ()
        if tools is not None:
            refused: list[Rejected] = []

            def check(text: str) -> str | None:
                try:
                    cast("AnswerForm", form).parse(text)
                except FormError as error:
                    refused.append(Rejected(text, error.problem))
                    return error.problem
                return None

            response, calls, usage, meta = run_tool_loop(
                llm,
                prompt,
                tools,
                mas.max_tool_rounds,
                DEFAULT_SYS_PROMPT,
                count_tokens,
                meter=meter,
                check=None if form is None else check,
            )
            value: Any = response
            rejected = tuple(refused)
            if form is not None:
                try:
                    value = form.parse(response)
                except FormError as error:
                    raise BadFormat(agent.name, (*(r.error for r in refused), error.problem), meta) from error
        elif form is not None:
            response, value, rejected, usage, meta = answer_plain(llm, prompt, form, count_tokens, meter)
        elif paints:
            response, value, meta = paint(agent.name, llm, prompt, written, meter)
            usage = estimate_usage(count_tokens, prompt_text(prompt), response)
        else:
            response, meta = call(llm, prompt, meter, batch_error=batch_error)
            value, usage = response, estimate_usage(count_tokens, prompt_text(prompt), response)

        record = AgentRecord(
            agent=agent.name,
            step=-1,  # renumbered once the run finishes and the order is known
            prompt=template,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: render_value(value, written if paints else TEXT if form is None else JSON)},
            tool_calls=calls,
            usage=usage,
            model=agent.model,
            raw_response=response,
            rejected=rejected,
            **(meta.record_fields() if meta is not None else {}),
        )
        return record, {agent.output: value}

    return body


def node_body(mas: MAS, node: Agent | Step | Box, types: Mapping[str, ValueType]) -> NodeBody:
    """Build what an agent, a step or a box does in one visit."""
    if isinstance(node, Agent):
        llm = mas.resolve_llm(node)
        return agent_body(mas, node, llm, get_token_counter(llm), types)
    return step_body(mas, node, types) if isinstance(node, Step) else box_body(node, types)


def step_body(mas: MAS, step: Step, types: Mapping[str, ValueType]) -> NodeBody:
    """Build what a step does in one visit: read its parameters, run its callable, write.

    The meter's wall and cost limits are checked first; a step makes no model
    call, so it counts none. A parameter whose value is not in the state is
    left out and takes its default.
    """
    spec, source = mas.resolve_callable(step)
    read_map, write_map = step_maps(step, spec)
    reads_run = "run" in read_map.values()

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        meter.run.check_boundary()
        values = {**state["values"], "run": run_value(meter.run, mas)} if reads_run else state["values"]
        arguments: dict[str, Any] = {}
        inputs: dict[str, str] = {}
        for parameter, name in read_map.items():
            inputs[name] = render_value(values.get(name), types[name], present=name in values)
            if name in values:
                arguments[parameter] = values[name]
        try:
            produced = run_step(spec, source, arguments)
        except BadFormat as error:
            raise BadFormat(step.name, error.problems, error.meta) from error
        writes = {write_map[write]: value for write, value in produced.items()}
        outputs = {name: render_value(value, types[name]) for name, value in writes.items()}
        record = AgentRecord(
            agent=step.name,
            step=-1,
            prompt="",
            rendered_prompt="",
            inputs=inputs,
            outputs=outputs,
            raw_response=raw(outputs, writes),
            cost_usd=0.0,  # a step makes no model call
            kind="step",
        )
        return record, writes

    return body


def raw(outputs: dict[str, str], writes: dict[str, Any]) -> str:
    """Return a node's raw response: its one rendered write, or all its writes as one JSON object."""
    return next(iter(outputs.values())) if len(outputs) == 1 else render_value(plain(writes), JSON)


def box_body(box: Box, types: Mapping[str, ValueType], *, resumes: bool = False) -> NodeBody:
    """Build what a box does in one visit: run its inner system from the mapped values.

    The inner system runs on the outer run's meter, which adopts it, so its
    calls count against the outer budget, and on the chain item the box works
    on, if any. Its failure raises out of the box, so the box's own error edges
    route by the inner error's kind; the visit keeps the inner trace so far for
    the box's failed record.

    An inner system that can pause runs as a root run of its own thread,
    ``<thread>/<box>/<visit>``; a pause there raises :class:`Paused`, and the
    box's wait node (``resumes``) resumes the thread once its questions are
    answered and finishes the visit.
    """
    inner: MAS = box.system
    pausing = pauses(inner)

    def body(state: MASState, visit: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        run = visit.run
        run.check_boundary()
        run.adopt(inner)
        values = state["values"]
        given = {inside: values[outside] for inside, outside in box.reads.items() if outside in values}
        marker = cast("AskState", state)["asks"][box.name] if resumes else None
        query = marker["query"] if marker is not None else render_value(plain(given), JSON)
        kept = len(run.records_of(inner))
        item = current_item()
        try:
            if marker is not None:
                here = dataclasses.replace(current_asking(), thread=marker["thread"], scope=marker["scope"])
                result = alone(inner, run, here, item)
                while "__interrupt__" in result:
                    interrupt({"questions": open_questions(result)})
                    result = alone(inner, run, here, item)
            elif pausing:
                asking, number = current_asking(), 1 + sum(record.agent == box.name for record in state["records"])
                thread, scope = f"{asking.thread}/{box.name}/{number}", f"{asking.scope}/{box.name}/{number}"
                here = dataclasses.replace(asking, thread=thread, scope=scope)
                result = alone(inner, run, here, item, initial_state(given, inner.prompts()))
                if "__interrupt__" in result:
                    raise Paused({"thread": thread, "scope": scope, "query": query, "visit": number})
            else:
                config = run_config(inner, run, item=item)
                result = inner.compiled.invoke(initial_state(given, inner.prompts()), config=config)
        except GraphBubbleUp:
            raise
        except Exception as error:  # a box in a chain: its siblings' inner records sit on the same list
            visit.inner = failed_trace(inner, query, error, run, records_since(run, inner, kept, item))
            raise
        trace, produced = trace_from_state(inner, query, result, run), result["values"]
        writes = {outside: produced.get(inside) for inside, outside in box.writes.items()}
        outputs = {
            outside: render_value(writes[outside], types[outside], present=inside in produced)
            for inside, outside in box.writes.items()
        }
        record = AgentRecord(
            agent=box.name,
            step=-1,
            prompt="",
            rendered_prompt="",
            inputs={
                outside: render_value(values.get(outside), types[outside], present=outside in values)
                for outside in box.reads.values()
            },
            outputs=outputs,
            usage=trace.usage,
            raw_response=raw(outputs, writes),
            kind="box",
            inner=trace,
            **paid_inside(trace),
        )
        return record, writes

    return body


def box_wait(name: str, finish: Callable[[MASState], dict[str, Any]]) -> Callable[[MASState], dict[str, Any]]:
    """Build a pausing box's wait node: it finishes a visit whose inner run paused, and passes any other by."""

    def wait(state: MASState) -> dict[str, Any]:
        marker = cast("AskState", state).get("asks", {}).get(name)
        finished = sum(record.agent == name for record in state["records"])
        return finish(state) if marker is not None and marker["visit"] > finished else {}

    return wait


def human_nodes(
    mas: MAS, human: Human, types: Mapping[str, ValueType]
) -> tuple[Callable[[MASState], dict[str, Any]], NodeBody]:
    """Build a human node's asking node, and the body of its deferred wait node, which :func:`wrap` finishes.

    The asking node fills the question from the state, keeps it in the store,
    calls the host's asker once, and keeps an answer given at once. The wait
    only reads before it interrupts, since LangGraph runs it again on resume:
    the answer kept, else the default once the deadline passed, else it
    pauses. Its record costs nothing and holds ``wait_s``, from asking to
    answering.
    """
    name = human.name
    edited = human.reads.get(human.edits or "")

    def ask(state: MASState) -> dict[str, Any]:
        meter, asking = current_meter(mas), current_asking()
        if meter.stopped is not None:  # a stopped run puts no more questions
            raise Stopped(meter.stopped)
        meter.start()
        index, key = current_item() or (None, None)
        visit = 1 + sum(record.agent == name for record in state["records"])
        meter.emit("start", name, item=index, item_key=key, visit=visit)
        now = datetime.now(UTC)
        question = human.ask(
            state["values"], types, run_id=asking.run_id, visit=visit, now=now, item=index, item_key=key
        )
        question = dataclasses.replace(question, id=question_id(asking.scope, name, visit, index, key))
        asking.store.save_question(
            question.id,
            asking.run_id,
            name,
            question.form,
            question.shown,
            question.asked_at,
            deadline=question.deadline,
            payload=question.to_dict(),
        )
        given = None if asking.asker is None else asking.asker.ask(question)
        if given is not None:
            keep(asking.store, question, given)
        return {"asks": {name: question.to_dict()}}

    def wait(state: MASState, visit: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        asking = current_asking()
        question = Question.from_dict(cast("AskState", state)["asks"][name])
        while (answer := question_answer(asking, question)) is None:
            interrupt({"questions": [question.id]})
        wait_s = (parse_time(answer["answered_at"]) - parse_time(question.asked_at)).total_seconds()
        if question.id in asking.resumed:
            here = {"visit": question.visit, "item": question.item, "item_key": question.item_key}
            visit.run.emit("resume", name, question=question.id, wait_s=wait_s, **here)
        writes = human.state_writes(answer, state["values"].get(edited) if edited is not None else None)
        record = AgentRecord(
            agent=name,
            step=-1,
            prompt=human.question,
            rendered_prompt=question.text,
            inputs=dict(question.shown),
            outputs={written: render_value(value, types.get(written, JSON)) for written, value in writes.items()},
            raw_response=json.dumps(answer, ensure_ascii=False),
            cost_usd=0.0,
            kind="human",
            wait_s=wait_s,
        )
        return record, writes

    return ask, wait


def parse_time(stamp: str) -> datetime:
    """Read an ISO time."""
    return datetime.fromisoformat(stamp)


def keep(store: Store, question: Question, given: Mapping[str, Any], *, timed_out: bool = False) -> dict[str, Any]:
    """Check an answer (:func:`~xmas.human.answer_value`), keep it in the store and return it in full."""
    answer = answer_value(question, given, now=datetime.now(UTC), timed_out=timed_out)
    store.answer(question.id, answer, who=answer["who"], answered_at=answer["answered_at"])
    return answer


def question_answer(asking: Asking, question: Question) -> dict[str, Any] | None:
    """Return a question's answer: the one kept, else its default once due, else ``None``.

    An unattended run takes the default at once, and fails without one.
    """
    row = asking.store.question(question.id)
    if row is not None and row.answer is not None:
        return row.answer
    if question.due(datetime.now(UTC)) or (asking.unattended and question.default is not None):
        return keep(asking.store, question, question.default or {}, timed_out=True)
    if asking.unattended:
        raise Unanswered(f"human node {question.node!r} got no answer in an unattended run")
    return None


def answered(store: Store, questions: list[str]) -> bool:
    """Return whether any of the questions has an answer or a deadline that has passed."""
    now = datetime.now(UTC)
    for question in questions:
        row = store.question(question)
        if row is not None and (
            row.answer is not None or (row.deadline is not None and parse_time(row.deadline) <= now)
        ):
            return True
    return False


def open_questions(result: Mapping[str, Any]) -> list[str]:
    """Return the ids of the questions a paused run waits on, from its interrupts."""
    return [question for pending in result["__interrupt__"] for question in pending.value["questions"]]


def resume_map(store: Store, pending: Sequence[Interrupt]) -> dict[str, None]:
    """Return the resume map of the interrupts with an answered or due question; their nodes read the store."""
    return {each.id: None for each in pending if answered(store, each.value["questions"])}


def alone(
    system: MAS,
    run: RunMeter,
    asking: Asking,
    item: tuple[int, str | None] | None,
    start: MASState | None = None,
) -> dict[str, Any]:
    """Run ``system`` from ``start`` as a root run of ``asking.thread``, with checkpoints of its own.

    A pause inside returns the state with ``__interrupt__`` instead of
    interrupting the node that runs it. Without ``start`` the thread resumes
    the interrupts whose questions are answered or due; when there is none,
    its state and interrupts come back as they are.
    """
    graph = system.compiled.copy(update={"checkpointer": asking.store})
    config = run_config(system, run, item=item, asking=asking)

    def go() -> dict[str, Any]:
        var_child_runnable_config.set(None)  # no parent: LangGraph would nest the run in the node that calls it
        if start is not None:
            return cast("dict[str, Any]", graph.invoke(start, config=config, durability="exit"))
        snapshot = graph.get_state(config)
        ready = resume_map(asking.store, snapshot.interrupts)
        if not ready:
            return {**snapshot.values, "__interrupt__": list(snapshot.interrupts)} if snapshot.next else snapshot.values
        return cast("dict[str, Any]", graph.invoke(Command(resume=ready), config=config, durability="exit"))

    return contextvars.copy_context().run(go)


def paid_inside(trace: Trace) -> dict[str, Any]:
    """Return a box record's cost and tokens: its inner records' summed, ``None`` when any of them is unknown."""
    tokens = [record.tokens for record in trace.records]
    known = cast("list[TokenUsage]", tokens)
    return {"cost_usd": trace.cost_usd, "tokens": None if not tokens or None in tokens else sum(known[1:], known[0])}


@dataclasses.dataclass(frozen=True)
class Decision:
    """Where a version 2 node with conditional or error edges goes after a visit.

    Attributes:
        plain: Targets taken after every visit that did not fail.
        choices: The ordered when-group: ``when``, ``while`` and ``switch``
            edges, each with its predicate (a switch compares its path with its
            case). The first that holds is taken.
        otherwise: The ``else`` target, taken when no choice is.
        errors: The error edges, in order.
    """

    plain: tuple[str, ...]
    choices: tuple[tuple[Edge, Predicate], ...]
    otherwise: str | None
    errors: tuple[Edge, ...]

    @classmethod
    def of(cls, edges: list[Edge], predicates: Mapping[Edge, Predicate]) -> Decision:
        """Build the decision of one node from its out-edges, in declared order, and the check's predicates."""
        choices = []
        for edge in edges:
            if edge.kind is EdgeKind.SWITCH:
                compare = ("cmp", "==", predicates[edge].tree, ("lit", edge.case))
                choices.append((edge, Predicate(f"{edge.switch} == {edge.case!r}", compare)))
            elif edge.kind in (EdgeKind.WHEN, EdgeKind.WHILE):
                choices.append((edge, predicates[edge]))
        return cls(
            plain=tuple(edge.target for edge in edges if edge.kind in (EdgeKind.PLAIN, EdgeKind.FIRST)),
            choices=tuple(choices),
            otherwise=next((edge.target for edge in edges if edge.kind is EdgeKind.ELSE), None),
            errors=tuple(edge for edge in edges if edge.kind is EdgeKind.ERROR),
        )

    def route(self, ctx: Context) -> tuple[list[str], bool]:
        """Return the targets after a visit that did not fail, and whether a ``while`` cap was hit.

        A ``while`` edge that holds while its target has run ``max_visits``
        times is skipped and marks the visit capped; the loop's exit is then
        the ``else`` target, since the other choices were written for the case
        where the loop's condition does not hold.
        """
        exit_ = [*self.plain, *([self.otherwise] if self.otherwise else [])]
        for edge, predicate in self.choices:
            if not predicate.evaluate(ctx):
                continue
            if edge.kind is EdgeKind.WHILE and ctx.visits.get(edge.target, 0) >= cast("int", edge.max_visits):
                return exit_, True
            return [*self.plain, edge.target], False
        return exit_, False

    def on_error(self, kind: ErrorKind) -> str | None:
        """Return the target of the first error edge routing ``kind``."""
        for edge in self.errors:
            if ANY_ERROR in edge.errors or kind.value in edge.errors:
                return edge.target
        return None


def context(mas: MAS, name: str, state: MASState, writes: dict[str, Any], meter: RunMeter) -> Context:
    """Return what a node's predicates see: its fresh writes, its visit counted, the spend so far.

    A chain's head has run as often as the chain has fanned out.
    """
    visits = Counter(record.agent for record in state["records"])
    visits[name] += 1
    passes = cast("ChainState", state).get("items", {})
    for head in system_chains(mas):
        visits[head] = passes.get(head, {}).get("pass", 0)

    def call(function: str, args: tuple[Any, ...]) -> bool:
        spec, source = mas.resolve_callable(Step(function, function))
        produced = run_step(spec, source, dict(zip(spec.reads, args, strict=True)))
        return cast("bool", next(iter(produced.values())))

    return Context(
        values={"run": run_value(meter, mas), **state["values"], **writes},
        visits=visits,
        node=name,
        cost_usd=meter.cost_usd,
        elapsed_s=meter.elapsed_s,
        call=call,
    )


def out_edges(mas: MAS) -> dict[str, list[Edge]]:
    """Return each node's out-edges: the declared ones in order, then the implicit start and end wiring.

    A chain's members are left out: the head stands for the chain, its exits
    leaving from it.
    """
    info = mas.checked()
    edges, flow = info.edges, info.flow
    out: dict[str, list[Edge]] = {}
    for edge in edges:
        out.setdefault(edge.source, []).append(edge)
    declared = {(edge.source, edge.target) for edge in edges}
    for source, target in flow.edges:
        if (source, target) not in declared:
            out.setdefault(source, []).append(Edge(source, target))
    return out


def relay_arrival(state: MASState) -> dict[str, Any]:  # noqa: ARG001 (a node's signature)
    """Pass a group's arrival on to a join's barrier; writes nothing."""
    return {}


def landing_node(name: str, chains: Mapping[str, Chain]) -> str:
    """Return the node an edge into ``name`` lands on: a chain head's ``in`` node, else ``name``."""
    return f"__each__.{name}.in" if name in chains else name


def add_joins(
    builder: StateGraph[MASState, Any, MASState, MASState],
    joins: Mapping[str, Join],
    chains: Mapping[str, Chain],
    exits: Mapping[str, str],
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str]]:
    """Add one barrier per join: it waits for each of the join's groups once per pass.

    A group without a relay (one plain edge from a node without error edges)
    is waited on at its node. Any other group gets the no-op relay node its
    :class:`~xmas.routes.Join` names, which its sources route to. A merge needs
    nothing: its edges stay plain, and it runs on whichever arrives. Nor does a
    join at :data:`~xmas.graph.END`: the run ends once every branch has.
    ``exits`` names the node an edge out of a node leaves from, where that is
    another node.

    Returns the (source, join) pairs the barrier waits on directly, and the
    relay each other (source, join) pair goes to.
    """
    direct: set[tuple[str, str]] = set()
    relays: dict[tuple[str, str], str] = {}
    for node, join in joins.items():
        if join.kind != "join" or node == END:
            continue
        starts = []
        for group, relay in zip(join.groups, join.relays, strict=True):
            if relay is None:
                direct.add((group[0], node))
                starts.append(exits.get(group[0], group[0]))
                continue
            builder.add_node(relay, relay_arrival)
            relays.update(((member, node), relay) for member in group)
            starts.append(relay)
        builder.add_edge(starts, landing_node(node, chains))
    return direct, relays


def router(name: str, targets: dict[str, str]) -> Callable[[MASState], list[str]]:
    """Build the router of a deciding node: it reads the route its wrapper wrote, mapped to relays."""

    def route(state: MASState) -> list[str]:
        return [targets[target] for target in state["routes"].get(name, []) if target in targets]

    return route


def cycle_router(
    mas: MAS, agent: Agent, back_targets: list[str], forward_targets: list[str]
) -> Callable[[MASState], list[str]]:
    """Build the branch decision for a version 1 agent that closes a cycle.

    The agent decides by what it wrote: emitting :data:`CONTINUE_MARKER` takes the
    back edge, anything else moves on. Two guards make the loop finite regardless
    of what the model says: the agent's own visit count against ``mas.max_visits``,
    and the graph-wide recursion limit behind it.
    """
    onward = forward_targets or [END]

    def route(state: MASState) -> list[str]:
        visits = sum(1 for record in state["records"] if record.agent == agent.name)
        looping = CONTINUE_MARKER in str(state["values"].get(agent.output) or "")
        return back_targets if looping and visits < mas.max_visits else onward

    return route


def wire_v1(builder: StateGraph[MASState, Any, MASState, MASState], mas: MAS) -> None:
    """Wire a version 1 graph: plain edges, and a :func:`cycle_router` for each agent that closes a cycle.

    A static edge set cannot express a loop: the edge back and the edge onward
    would both fire on every pass. Routing the two exclusively, on the looping
    agent's own output, is what makes a cyclic topology end with an answer.
    """
    back = back_edges(mas.graph)
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


def wire_v2(
    builder: StateGraph[MASState, Any, MASState, MASState],
    mas: MAS,
    out: dict[str, list[Edge]],
    deciding: set[str],
    waits: Mapping[str, str],
) -> None:
    """Wire a version 2 graph: a deciding node routes by what its wrapper wrote, joins are barriers, the rest plain.

    Each chain becomes its three nodes (:func:`add_chain`); edges into its head
    land on the first, edges out of it leave from the last. Edges out of a
    human or a pausing box leave from its wait node (``waits``).
    """
    chains = system_chains(mas)
    for chain in chains.values():
        add_chain(builder, mas, chain)
    exits = {**{head: f"__each__.{head}.out" for head in chains}, **waits}
    direct, relays = add_joins(builder, mas.checked().joins, chains, exits)
    for source, edges in out.items():
        targets = {
            edge.target: relays.get((source, edge.target), landing_node(edge.target, chains))
            for edge in edges
            if (source, edge.target) not in direct
        }
        leaves = exits.get(source, source)
        if source not in deciding:
            for target in targets.values():
                builder.add_edge(leaves, target)
        else:
            builder.add_conditional_edges(leaves, router(source, targets), sorted(set(targets.values())))


def chain_items(chain: Chain, values: Mapping[str, Any]) -> list[tuple[Any, str | None]]:
    """Return a chain's items with their keys, from the list at its path.

    A missing path or ``None`` is no item. Anything else that is not a list,
    and with ``key`` an item without a scalar key or a key given twice, raises
    ``ValueError`` naming the path, before any item runs; so does a step of the
    path through a value that is no mapping.
    """
    each = chain.each
    found, path = values.get(each.root), each.root
    for field in each.fields:
        if found is None:
            break
        if not isinstance(found, dict):
            raise ValueError(f"for each {each}: {path} holds {type(found).__name__}, not a mapping with {field!r}")
        found, path = found.get(field), f"{path}.{field}"
    if found is None:
        return []
    if not isinstance(found, list):
        raise ValueError(f"for each {each}: {each.over} holds {type(found).__name__}, not a list")
    if each.key is None:
        return [(item, None) for item in found]
    keys = []
    for index, item in enumerate(found):
        key = item.get(each.key) if isinstance(item, dict) else None
        if isinstance(key, bool) or not isinstance(key, str | int | float):
            raise ValueError(f"for each {each}: item {index} of {each.over} has no key {each.key!r} (a text or number)")
        keys.append(str(key))
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"for each {each}: {each.over} repeats the key {each.key!r}: {repeated}")
    return list(zip(found, keys, strict=True))


def total_cost(records: list[AgentRecord]) -> float | None:
    """Return the records' costs summed, ``None`` when there is none or any is unknown."""
    return Trace("", "", tuple(records)).cost_usd


def add_chain(builder: StateGraph[MASState, Any, MASState, MASState], mas: MAS, chain: Chain) -> None:
    """Add a chain's three nodes: a pass starts and fans out, each item runs, the writes are gathered.

    An item runs ``chain.system`` on the item and the outside values it reads,
    on the run's meter, which keeps its records with ``mas``'s, at most ``at
    once`` items at a time. An item whose run fails becomes a row with its
    error, and ``None`` in every gathered list, unless the error ends the run
    (:data:`~xmas.errors.RUN_ENDS`) or is ``money``, which every other item
    would meet too.

    A chain whose system can pause runs each item as a root run of its own
    thread, ``<thread>/<head>/<pass>/<index>``. An item that pauses becomes a
    paused row, and a fourth node, the deferred ``__each__.h.wait``, resumes
    the paused items whose questions are answered, then comes back while any
    item is still paused, and asks for all their questions at once when none
    can go on.
    """
    nodes = ChainNodes(mas, chain)
    builder.add_node(nodes.start, nodes.begin)
    builder.add_node(nodes.target, nodes.run_item)  # type: ignore[arg-type]
    builder.add_node(nodes.gather, nodes.finish)
    builder.add_conditional_edges(nodes.start, nodes.fan, [nodes.target, nodes.gather])
    if not nodes.pausing:
        builder.add_edge(nodes.target, nodes.gather)
        return
    builder.add_node(nodes.hold, nodes.wait, defer=True)
    builder.add_edge(nodes.target, nodes.hold)
    builder.add_conditional_edges(nodes.hold, nodes.onward, [nodes.hold, nodes.gather])


class ChainNodes:
    """The nodes of one ``for each`` chain in ``mas`` (:func:`add_chain`), and the names they go by.

    Attributes:
        mas: The system the chain sits in.
        chain: The chain.
        head: The chain's head, which names its nodes.
        start: The node that starts a pass and fans the items out.
        target: The node that runs one item.
        gather: The node that gathers the items' writes into lists.
        hold: The deferred node that resumes paused items.
        pausing: Whether the chain's system can pause.
    """

    def __init__(self, mas: MAS, chain: Chain) -> None:
        """Name the chain's nodes after its head."""
        self.mas, self.chain, self.head = mas, chain, chain.head
        self.start, self.target = f"__each__.{chain.head}.in", f"__each__.{chain.head}"
        self.gather, self.hold = f"__each__.{chain.head}.out", f"__each__.{chain.head}.wait"
        self.pausing = pauses(chain.system)

    def begin(self, state: ChainState) -> dict[str, Any]:
        """Start a pass: count its items and number it."""
        done = state.get("items", {}).get(self.head, {}).get("pass", 0)
        count = len(chain_items(self.chain, state["values"]))
        return {"items": {self.head: {"n": count, "pass": done + 1, "rows": {}}}}

    def fan(self, state: ChainState) -> list[Send] | list[str]:
        """Send each item to the item node with the outside values it reads; no items go straight to the gather."""
        items = chain_items(self.chain, state["values"])
        if not items:
            return [self.gather]
        values = {name: state["values"][name] for name in self.chain.inputs if name in state["values"]}
        passed = state["items"][self.head]["pass"]
        task = {"values": values, "prompts": state["prompts"], "pass": passed}
        return [
            Send(self.target, {**task, "index": index, "item": item, "key": key})
            for index, (item, key) in enumerate(items)
        ]

    def gate(self, run: RunMeter) -> AbstractContextManager[Any]:
        """Return the chain's ``at once`` gate on the run's meter: one per chain, whatever its head is called."""
        return run.gate(f"each:{id(self.chain.system)}", self.chain.each.at_once)

    def run_item(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run one item once its ``at once`` place is free, as a root run of its own thread when the chain can pause."""
        run, system = current_meter(self.mas), self.chain.system
        run.adopt(system, into=self.mas)
        index, key = task["index"], task["key"]
        with self.gate(run):
            run.emit("item_start", self.head, item=index, item_key=key)
            state = initial_state({self.chain.each.item: task["item"], **task["values"]}, task["prompts"])
            if not self.pausing:
                config = run_config(system, run, item=(index, key))
                return self.outcome(run, index, key, lambda: system.compiled.invoke(state, config))
            asking, number = current_asking(), task["pass"]
            scope = asking.scope if number == 1 else f"{asking.scope}/{self.head}/{number}"
            thread = f"{asking.thread}/{self.head}/{number}/{index}"
            here = dataclasses.replace(asking, thread=thread, scope=scope)
            return self.outcome(run, index, key, lambda: alone(system, run, here, (index, key), state), here)

    def outcome(
        self, run: RunMeter, index: int, key: str | None, go: Callable[[], Any], here: Asking | None = None
    ) -> dict[str, Any]:
        """Run one item and turn what it did into its row and records: values, an error, or a pause."""
        kept = len(run.records_of(self.chain.system))
        try:
            result = go()
        except GraphBubbleUp:
            raise
        except Exception as error:
            failed = self.failed_item(run, index, key, error, kept)
            if failed is None:  # the error ends the run
                raise
            return failed
        if here is not None and "__interrupt__" in result:
            row: dict[str, Any] = {
                "paused": here.thread,
                "scope": here.scope,
                "key": key,
                "questions": open_questions(result),
            }
            return {"items": {self.head: {"rows": {index: row}}}}
        run.emit("item_end", self.head, status="ok", cost_usd=total_cost(result["records"]), item=index, item_key=key)
        produced = result["values"]
        row = {"values": {name: produced[name] for name in self.chain.writes if name in produced}}
        return {"items": {self.head: {"rows": {index: row}}}, "records": result["records"]}

    def failed_item(
        self, run: RunMeter, index: int, key: str | None, error: Exception, kept: int
    ) -> dict[str, Any] | None:
        """Turn an item's error into its row, keeping the records it made after the first ``kept``.

        ``None`` for an error that ends the run, ``money``, which every other
        item would meet too, or an unanswered question: the error goes on.
        """
        mine = [
            record
            for record in run.records_of(self.chain.system)[kept:]
            if record.agent in self.chain.nodes and record.item == index
        ]
        run.emit("item_end", self.head, status="failed", cost_usd=total_cost(mine), item=index, item_key=key)
        kind = kind_of(error)
        if kind in RUN_ENDS or kind is ErrorKind.MONEY or isinstance(error, Unanswered):
            return None
        row: dict[str, Any] = {"error": f"{type(error).__name__}: {error}", "kind": kind.value}
        return {"items": {self.head: {"rows": {index: row}}}, "records": mine}

    def wait(self, state: ChainState) -> dict[str, Any]:
        """Resume the paused items whose questions are answered; interrupt with all their questions while none is."""
        rows = state["items"][self.head]["rows"]
        paused = {index: row for index, row in rows.items() if "paused" in row}
        if not paused:
            return {}
        run, asking = current_meter(self.mas), current_asking()
        run.adopt(self.chain.system, into=self.mas)  # a resumed run's meter is new
        while not (ready := [index for index, row in paused.items() if answered(asking.store, row["questions"])]):
            interrupt({"questions": [question for row in paused.values() for question in row["questions"]]})
        with ThreadPoolExecutor(len(ready)) as pool:
            done = list(pool.map(lambda index: self.resume_item(run, asking, index, paused[index]), ready))
        rows = {index: row for part in done for index, row in part["items"][self.head]["rows"].items()}
        return {
            "items": {self.head: {"rows": rows}},
            "records": [record for part in done for record in part.get("records", [])],
        }

    def resume_item(self, run: RunMeter, asking: Asking, index: int, row: dict[str, Any]) -> dict[str, Any]:
        """Resume one paused item on its own thread, once its ``at once`` place is free."""
        here = dataclasses.replace(asking, thread=row["paused"], scope=row["scope"])
        with self.gate(run):
            go = partial(alone, self.chain.system, run, here, (index, row["key"]))
            return self.outcome(run, index, row["key"], go, here)

    def onward(self, state: ChainState) -> str:
        """Go back to the wait while any item is still paused, else to the gather."""
        rows = state["items"][self.head]["rows"]
        return self.hold if any("paused" in row for row in rows.values()) else self.gather

    def finish(self, state: ChainState) -> dict[str, Any]:
        """Gather each write into a list aligned to the items, ``None`` for an item that did not write it."""
        entry = state.get("items", {}).get(self.head, {})
        rows = [entry.get("rows", {}).get(index, {}).get("values", {}) for index in range(entry.get("n", 0))]
        return {"values": {name: [row.get(name) for row in rows] for name in self.chain.writes}}


def compile_mas(mas: MAS) -> Any:
    """Compile a MAS topology into an executable graph.

    In version 1, agents identified by :func:`xmas.graph.join_nodes` are
    deferred, so a node fed by predecessors at unequal depth runs once per
    query rather than once per wave of arriving inputs. In version 2 a node
    with conditional or error edges gets a :class:`Decision`, a join is a
    barrier that runs its node once per pass, and a chain is its three nodes in
    place of its members, on :class:`ChainState`. A human is its asking node
    and its deferred wait node (:func:`human_nodes`), and so is a box that can
    pause (its own node and :func:`box_wait`); either adds the ``asks``
    channel.
    """
    chains = system_chains(mas)
    members = {name for chain in chains.values() for name in chain.nodes}
    humans = {name: human for name, human in mas.humans.items() if name not in members}
    holding = [name for name, box in mas.boxes.items() if name not in members and pauses(box.system)]
    if humans or holding:
        schema: type[MASState] = ChainAskState if chains else AskState
    else:
        schema = ChainState if chains else MASState
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(schema)
    types = mas.info.types if mas.info is not None else {}
    nodes: dict[str, Agent | Step | Box] = {**mas.agents, **mas.steps, **mas.boxes}
    bodies = {name: node_body(mas, node, types) for name, node in nodes.items() if name not in members}

    out = out_edges(mas) if mas.version == VERSION_2 else {}
    races = mas.info.races if mas.info is not None else {}
    racing = {name: (source, len(names)) for source, names in races.items() for name in names}
    decisions = {  # a contestant decides too: only the winner goes on
        source: Decision.of(edges, mas.checked().predicates)
        for source, edges in out.items()
        if source in racing or any(e.kind not in (EdgeKind.PLAIN, EdgeKind.FIRST) for e in edges)
    }
    deferred = join_nodes(mas.graph) if mas.version != VERSION_2 else set()
    writes_error = mas.version == VERSION_2 and reads(mas, "error")
    for name, body in bodies.items():
        role = NodeRole(decisions.get(name), writes_error=writes_error, race=racing.get(name))
        node = wrap(mas, name, body, role)
        # mypy rejects a TypedDict schema here; langgraph documents it as the usage.
        builder.add_node(name, node, defer=name in deferred, input_schema=schema)  # type: ignore[arg-type]
    for name in holding:
        body = box_body(mas.boxes[name], types, resumes=True)
        role = NodeRole(decisions.get(name), writes_error=writes_error, starts=False, race=racing.get(name))
        finish = wrap(mas, name, body, role)
        builder.add_node(f"{name}.wait", box_wait(name, finish), defer=True, input_schema=schema)  # type: ignore[arg-type]
    for name, human in humans.items():
        ask, wait = human_nodes(mas, human, types)
        finish = wrap(mas, name, wait, NodeRole(decisions.get(name), writes_error=writes_error, starts=False))
        builder.add_node(name, ask, input_schema=schema)  # type: ignore[arg-type]
        builder.add_node(f"{name}.wait", finish, defer=True, input_schema=schema)  # type: ignore[arg-type]
    waits = {name: f"{name}.wait" for name in [*humans, *holding]}
    for name, wait_node in waits.items():
        builder.add_edge(name, wait_node)
    if mas.version == VERSION_2:
        wire_v2(builder, mas, out, set(decisions), waits)
    else:
        wire_v1(builder, mas)
    return builder.compile()


def reads(mas: MAS, value: str) -> bool:
    """Return whether any agent, step or predicate of a version 2 system reads ``value``."""
    reads = {path[0] for predicate in mas.checked().predicates.values() for path in predicate.paths()}
    reads.update(*(agent.reads.values() for agent in mas.agents.values()))
    reads.update(*(step_maps(step, mas.resolve_callable(step)[0])[0].values() for step in mas.steps.values()))
    return value in reads


def final_output(mas: MAS, records: list[AgentRecord]) -> str:
    """Return the rendered write of the terminal node that ran last, or ``""`` if none ran.

    A terminal with several writes gives them as one JSON object.
    """
    terminals = set(mas.graph.predecessors(END))
    ran = [record for record in records if record.agent in terminals and record.error is None]
    if not ran:
        return ""
    outputs = ran[-1].outputs
    return next(iter(outputs.values())) if len(outputs) == 1 else ran[-1].raw_response


def kind_of(error: BaseException) -> ErrorKind:
    """Return the kind of the error that ended a run; hitting ``max_steps`` is ``BUDGET``."""
    if isinstance(error, GraphRecursionError):
        return ErrorKind.BUDGET
    return classify(error)


def build_trace(mas: MAS, query: str, records: list[AgentRecord], meter: RunMeter, **ending: Any) -> Trace:
    """Number a run's records and summarize its health; the budget entry is the meter owner's alone.

    The meter owner's version 2 trace carries the run's id and the system's ``describe()`` id.
    """
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(records))
    values = mas.info.types if mas.info is not None else {"query", *(a.output for a in mas.agents.values())}
    chains = {head: (chain.nodes, dict(chain.each.item_budget or {})) for head, chain in system_chains(mas).items()}
    health = summarize(
        numbered,
        meter=meter if mas is meter.owner else None,
        settings=mas.health,
        while_edges=mas.while_edges,
        values=values,
        chains=chains,
    )
    if mas.version == VERSION_2 and mas is meter.owner:
        ending.update(run_id=meter.run_id, graph_id=meter.graph_id)
    return Trace(query=query, records=numbered, health=health, **ending)


def trace_from_state(mas: MAS, query: str, result: Any, meter: RunMeter) -> Trace:
    """Assemble a trace from a finished run's state, or from the error that ended it.

    With declared ``outputs`` the trace holds each one's final value, ``None``
    when nothing wrote it, and its output renders the first.
    """
    if isinstance(result, BaseException):
        return failed_trace(mas, query, result, meter)
    outputs = mas.outputs
    if not outputs:
        return build_trace(mas, query, result["records"], meter, output=final_output(mas, result["records"]))
    values = result["values"]
    first, vtype = next(iter(outputs.items()))
    output = render_value(values.get(first), vtype, present=first in values)
    return build_trace(mas, query, result["records"], meter, output=output, outputs={n: values.get(n) for n in outputs})


def failed_trace(
    mas: MAS, query: str, error: BaseException, meter: RunMeter, records: list[AgentRecord] | None = None
) -> Trace:
    """Assemble a failed run's trace from ``records``, by default the records its meter kept for ``mas``."""
    return build_trace(
        mas,
        query,
        meter.records_of(mas) if records is None else records,
        meter,
        output="",
        error=f"{type(error).__name__}: {error}",
        error_kind=kind_of(error),
    )


def records_since(meter: RunMeter, mas: MAS, start: int, item: tuple[int, str | None] | None) -> list[AgentRecord]:
    """Return the records the meter kept for ``mas`` from ``start`` on, only ``item``'s when it names a chain item.

    ``start`` skips the records of earlier runs on the meter (an adopted box
    system's earlier visits).
    """
    records = meter.records_of(mas)[start:]
    return records if item is None else [record for record in records if record.item == item[0]]


def initial_state(query: str | Mapping[str, Any], prompts: Mapping[str, str]) -> MASState:
    """Build the state a run starts from: the query, or a box's inner values."""
    values = {"query": query} if isinstance(query, str) else dict(query)
    return MASState(values=values, prompts=dict(prompts), records=[], routes={})


def run_config(
    mas: MAS, meter: RunMeter, item: tuple[int, str | None] | None = None, asking: Asking | None = None
) -> dict[str, Any]:
    """Build one run's config: the step cap, the thread pool, the meter the nodes read, the chain item, if any.

    A run that can pause also carries its :class:`Asking` and its checkpoint thread.
    """
    configurable: dict[str, Any] = {RUN_KEY: meter}
    if mas.info is not None and mas.info.races:
        configurable[RACES_KEY] = uuid.uuid4().hex
    if item is not None:
        configurable[ITEM_KEY] = item
    if asking is not None:
        configurable.update({ASKING_KEY: asking, "thread_id": asking.thread})
    return {"recursion_limit": mas.max_steps, "max_concurrency": BRANCH_THREADS, "configurable": configurable}


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Run a MAS over parallel lists of inputs and prompt sets.

    Every item runs once, with its own meter and budget. An item that fails
    becomes a failed trace that keeps its finished records; the others are
    unaffected. No one answers later: a question the asker does not answer at
    once takes its default, else fails the item.
    """
    return run_all(mas, list(queries), prompts)


def run_state(
    mas: MAS,
    values: Mapping[str, Any],
    prompts: dict[str, str],
    on_event: Callable[[Event], None] | None = None,
    run_id: str | None = None,
) -> Trace:
    """Run a MAS on its declared inputs with a given prompt set; a failure is recorded in the trace, not raised.

    A version 2 system takes exactly its inputs, each fitting its type; a
    :class:`~pathlib.Path` given for a media input is referenced where it is
    (:func:`~xmas.values.coerce_media`), its sha256 checked whenever it is
    read. A version 1 system takes text values, ``query`` among them as
    ``run`` gives it. ``on_event`` is called with each event of the run, on the
    run's threads. The trace's query is the ``query`` value, else the values as
    one line of JSON. ``run_id`` names the run, a new random id when ``None``.

    A system that can pause (:func:`pauses`) runs on its store under the run's
    id: a question not answered by the time every other branch has ended
    returns a ``paused`` trace, which :func:`resume` continues.

    Raises:
        ValueError: Before any call, listing every input missing, extra or of
            the wrong type; or, for a system that can pause, when ``run_id``
            holds a ``/`` (the threads of a run's items and boxes) or the
            store holds a run with it.
    """
    if mas.version == VERSION_2:
        values = {
            name: coerce_media(value, mas.inputs[name]) if name in mas.inputs else value
            for name, value in values.items()
        }
    check_inputs(mas, values)
    query = query_text(values)
    if not pauses(mas):
        return run_all(mas, [dict(values)], [prompts], on_event, run_id)[0]
    store = mas.runs()
    run_id = run_id or uuid.uuid4().hex
    if "/" in run_id:
        raise ValueError(f"Run id {run_id!r} holds a '/', which names the threads of a run's items and boxes.")
    graph_id = mas.describe()["id"]
    meter = RunMeter(mas.budget, owner=mas, listener=on_event, graph_id=graph_id, run_id=run_id)
    try:
        store.begin(run_id, graph_id, query)
    except BaseException:
        meter.close()
        raise
    log(meter, store)
    asking = Asking(store, mas.asker, run_id, run_id, run_id)
    return advance(mas, query, initial_state(dict(values), prompts), meter, asking)


def resume(
    mas: MAS,
    run_id: str,
    answers: Mapping[str, Mapping[str, Any]] | None = None,
    on_event: Callable[[Event], None] | None = None,
) -> Trace:
    """Continue a paused run, after keeping ``answers`` (``{question id: answer}``) as :func:`answer` does.

    The run goes on from its checkpoints in the system's store, in this
    process or a new one, with the prompts it paused with. Its budget counts
    on from what it had used (:meth:`~xmas.budget.RunMeter.seed`), the wait
    left out of its wall time. A question past its deadline takes its default;
    a question neither answered nor due stays open, and the run pauses again
    if it reaches nothing else.

    Raises:
        ValueError: If the store has no such run, the run is not paused, the
            system is not the one that paused (its ``describe()`` id), an
            answer does not fit, or no open question is answered or due.
    """
    store = mas.runs()
    row = store.run(run_id)
    if row is None:
        raise ValueError(f"{store!r} has no run {run_id!r}.")
    if row.status != "paused":
        raise ValueError(f"Run {run_id!r} is {row.status}, not paused.")
    graph_id = mas.describe()["id"]
    if row.system_id != graph_id:
        raise ValueError(f"Run {run_id!r} paused in system {row.system_id}, not {graph_id}: resume with that system.")
    if answers:
        answer(mas, run_id, answers)
    if not answered(store, list(row.paused)):
        open_ = ", ".join(repr(question) for question in row.paused)
        raise ValueError(f"Run {run_id!r} waits on {open_}: none is answered or past its deadline.")
    meter = RunMeter(mas.budget, owner=mas, listener=on_event, graph_id=graph_id, run_id=run_id)
    if row.totals is not None:
        meter.seed(row.totals)
    asking = Asking(store, mas.asker, run_id, run_id, run_id, resumed=frozenset(row.paused))
    try:
        snapshot = mas.compiled.copy(update={"checkpointer": store}).get_state(run_config(mas, meter, asking=asking))
        if not snapshot.values:
            raise ValueError(f"{store!r} holds no checkpoint of run {run_id!r}.")
        meter.records.extend(snapshot.values["records"])
        store.save_run(run_id, graph_id, "running", row.paused)
    except BaseException:
        meter.close()
        raise
    log(meter, store)
    query = query_text({name: snapshot.values["values"][name] for name in mas.inputs})
    return advance(mas, query, Command(resume=resume_map(store, snapshot.interrupts)), meter, asking)


def answer(mas: MAS, run_id: str, answers: Mapping[str, Mapping[str, Any]]) -> None:
    """Keep answers to a run's questions, ``{question id: answer}``, whether the run is paused or still running.

    Each answer is checked against its question (:func:`~xmas.human.answer_value`);
    a running run takes it when it reaches the question's wait.

    Raises:
        ValueError: If the run is stopped; else listing every problem, keeping
            none of the answers: an id that is no open question of the run, an
            answer that does not fit.
    """
    store = mas.runs()
    run = store.run(run_id)
    if run is not None and run.status == "stopped":
        raise ValueError(f"Run {run_id!r} is stopped: it takes no answers.")
    checked: list[tuple[Question, Mapping[str, Any]]] = []
    problems: list[str] = []
    for question_id_, given in answers.items():
        row = store.question(question_id_)
        if row is None or row.run_id != run_id or row.payload is None:
            problems.append(f"{question_id_!r} is no question of run {run_id!r}")
        elif row.answer is not None:
            problems.append(f"{question_id_!r} is answered already")
        else:
            question = Question.from_dict(row.payload)
            try:
                answer_value(question, given, now=datetime.now(UTC))
            except ValueError as error:
                problems.append(str(error))
            checked.append((question, given))
    if problems:
        raise ValueError("The answers were not kept:\n" + "\n".join(f"- {problem}" for problem in problems))
    for question, given in checked:
        keep(store, question, given)


def stop(mas: MAS, run_id: str, reason: str = "stopped") -> bool:
    """Stop a run: a running one through its meter, a paused one in the store; ``False`` for neither.

    A paused run's row becomes ``stopped`` and each of its open questions is
    withdrawn through the asker's ``withdraw``, when it has one; a running
    run's open questions are withdrawn once it ends (:func:`advance`).
    """
    meter = RUNNING.get(run_id)
    if meter is not None:
        meter.stop(reason)
        return True
    store = mas.runs()
    row = store.run(run_id)
    if row is None or row.status != "paused":
        return False
    store.save_run(run_id, row.system_id, "stopped")
    withdraw(mas, store, run_id, reason)
    return True


def withdraw(mas: MAS, store: Store, run_id: str, reason: str) -> None:
    """Withdraw each unanswered question of a stopped run through the asker's ``withdraw``, when it has one."""
    withdraw = getattr(mas.asker, "withdraw", None)
    for row in store.questions(run_id):
        if withdraw is not None and row.answer is None and row.payload is not None:
            withdraw(Question.from_dict(row.payload), reason)


def advance(mas: MAS, query: str, start: MASState | Command[Any], meter: RunMeter, asking: Asking) -> Trace:
    """Run or resume a run that can pause on its thread, until it ends or waits on questions nobody answered yet.

    Answers that arrive before the run returns (from the asker, or through
    :func:`answer`) resume it here. The run's row keeps its status, the
    questions it waits on and the meter's totals. A run stopped while it ran
    ends ``stopped`` even when it reached a pause, and its open questions are
    withdrawn.
    """
    store = asking.store
    graph = mas.compiled.copy(update={"checkpointer": store})
    config = run_config(mas, meter, asking=asking)
    waiting: list[str] = []
    try:
        result = graph.invoke(start, config=config, durability="exit")
        while "__interrupt__" in result and (ready := resume_map(store, result["__interrupt__"])):
            result = graph.invoke(Command(resume=ready), config=config, durability="exit")
    except Exception as error:  # noqa: BLE001  any failure of the run is its failed trace, never raised
        trace = failed_trace(mas, query, error, meter)
    else:
        if "__interrupt__" in result:
            waiting = open_questions(result)
            questions = tuple(
                Question.from_dict(cast("dict[str, Any]", row.payload))
                for row in store.questions(asking.run_id)
                if row.question_id in waiting
            )
            for question in questions:
                if question.id not in asking.resumed and meter.stopped is None:
                    where = {"item": question.item, "item_key": question.item_key, "visit": question.visit}
                    meter.emit("pause", question.node, question=question.id, **where)
            trace = build_trace(mas, query, result["records"], meter, output="", pause=Pause(asking.run_id, questions))
        else:
            trace = trace_from_state(mas, query, result, meter)
        store.save_run(asking.run_id, cast("str", meter.graph_id), trace.status.value, waiting, meter.totals())
    finally:
        meter.close()
    reason = meter.stopped
    if reason is not None and trace.status in (RunStatus.PAUSED, RunStatus.STOPPED):  # a stop while it ran or paused
        if trace.pause is not None:
            trace = failed_trace(mas, query, Stopped(reason), meter)
        store.save_run(asking.run_id, cast("str", meter.graph_id), "stopped", (), meter.totals())
        withdraw(mas, store, asking.run_id, reason)
    if meter.log is not None:
        meter.log.end(trace, meter.totals())
    return trace


def log(meter: RunMeter, store: Store) -> None:
    """Give the meter its run's log, when the store is a file: one in memory logs nothing."""
    if store.path is not None:
        meter.log = RunLog(store, meter.run_id)


def query_text(values: Mapping[str, Any]) -> str:
    """Return a run's query: the ``query`` text, else the values as one line of JSON."""
    return values["query"] if isinstance(values.get("query"), str) else render_value(plain(dict(values)), JSON)


def check_inputs(mas: MAS, values: Mapping[str, Any]) -> None:
    """Refuse values that are not the system's inputs, listing every problem."""
    if mas.version != VERSION_2:
        problems = [] if "query" in values else ["input 'query' is missing"]
        problems += [f"{name!r} is not text" for name, value in values.items() if not isinstance(value, str)]
    else:
        problems = [f"input {name!r} is missing" for name in mas.inputs if name not in values]
        problems += [f"{name!r} is no input of {sorted(mas.inputs)}" for name in values if name not in mas.inputs]
        problems += [
            f"input {name!r} is not {vtype}: {problem}"
            for name, vtype in mas.inputs.items()
            if name in values and (problem := check_value(values[name], vtype)) is not None
        ]
    if problems:
        lines = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"The values do not fit the system's inputs:\n{lines}")


def run_all(
    mas: MAS,
    starts: list[str | Mapping[str, Any]],
    prompts: list[dict[str, str]],
    on_event: Callable[[Event], None] | None = None,
    run_id: str | None = None,
) -> list[Trace]:
    """Run each start (a query or the input values) with its prompt set, on a meter of its own.

    A trace's query is the start's query (:func:`query_text`).

    Each run is logged to the system's store when it is a file (:class:`~xmas.store.RunLog`).
    A system that can pause runs unattended here: its questions go to its
    asker, and its chains and boxes keep their threads in that file, else
    in a store of the batch's own, in memory.

    Raises:
        ValueError: When ``run_id`` names a run the file holds already.
    """
    if not starts:
        return []
    queries = [start if isinstance(start, str) else query_text(start) for start in starts]
    store = mas.runs()
    system_id = mas.describe()["id"] if mas.version == VERSION_2 or store.path is not None else None
    graph_id = system_id if mas.version == VERSION_2 else None
    meters = [RunMeter(mas.budget, owner=mas, listener=on_event, graph_id=graph_id, run_id=run_id) for _ in starts]
    states = [initial_state(start, prompt_set) for start, prompt_set in zip(starts, prompts, strict=True)]
    graph, configs, durable = mas.compiled, [run_config(mas, meter) for meter in meters], {}
    if pauses(mas):
        threads, asker = store if store.path is not None else Store(), mas.asker
        askings = [Asking(threads, asker, m.run_id, m.run_id, m.run_id, unattended=True) for m in meters]
        configs = [run_config(mas, meter, asking=asking) for meter, asking in zip(meters, askings, strict=True)]
        graph, durable = graph.copy(update={"checkpointer": threads}), {"durability": "exit"}
    try:
        if store.path is not None:
            for meter, query in zip(meters, queries, strict=True):
                store.begin(meter.run_id, cast("str", system_id), query)
                log(meter, store)
        results = invoke_batch(graph, states, configs, durable)
    finally:
        for meter in meters:
            meter.close()
    traces = [
        trace_from_state(mas, query, result, meter)
        for query, result, meter in zip(queries, results, meters, strict=True)
    ]
    for meter, trace in zip(meters, traces, strict=True):
        if meter.log is not None:
            meter.log.end(trace, meter.totals())
    return traces


def invoke_batch(
    graph: Any, states: Sequence[MASState], configs: Sequence[dict[str, Any]], durable: Mapping[str, Any]
) -> list[Any]:
    """Invoke the graph on each state with its config, :data:`BATCH_QUERIES` at a time, as ``graph.batch`` would.

    ``graph.batch`` takes its width from the first run's ``max_concurrency``,
    which here sizes each run's own branch pool, so the batch runs on a pool of
    its own. A query that fails returns its error, which becomes its failed trace.
    """

    def invoke(state: MASState, config: dict[str, Any]) -> Any:
        try:
            return graph.invoke(state, config, **durable)
        except Exception as error:  # noqa: BLE001  a failed query is its failed trace, as in batch(return_exceptions=True)
            return error

    if len(states) == 1:
        return [invoke(states[0], configs[0])]
    with get_executor_for_config({"max_concurrency": BATCH_QUERIES}) as executor:
        return list(executor.map(invoke, states, configs))


def node_owner(mas: MAS, node: str) -> MAS:
    """Return the system that runs ``node``: ``mas``, or for a member of a ``for each`` chain the system an item runs.

    Raises:
        ValueError: For a node inside a box, whose example belongs to the box's
            system, or a name that is no node at all.
    """
    for chain in system_chains(mas).values():
        if node in chain.nodes:
            return chain.system
    if node in mas.nodes:
        return mas
    for box in mas.boxes.values():
        try:
            node_owner(box.system, node)
        except ValueError:
            continue
        raise ValueError(f"node {node!r} sits inside box {box.name!r}: give the example to the box's system")
    raise ValueError(f"{node!r} is no node of the system; its nodes: {sorted(mas.nodes)}")


def node_types(system: MAS, node: Agent | Step | Box | Human) -> tuple[dict[str, ValueType], dict[str, ValueType]]:
    """Return what a node reads and writes, ``{value: type}``, as one visit in ``system`` sees them.

    A version 1 system's values are text.
    """
    types = system.info.types if system.info is not None else {}
    if isinstance(node, Agent):
        reads, writes = list(node.reads.values()), [node.output]
    elif isinstance(node, Step):
        read_map, write_map = step_maps(node, system.resolve_callable(node)[0])
        reads, writes = list(read_map.values()), list(write_map.values())
    elif isinstance(node, Box):
        return node.types()
    else:
        reads, writes = list(node.reads.values()), [node.answer]
    return {name: types.get(name, TEXT) for name in reads}, {name: types.get(name, TEXT) for name in writes}


def run_node(mas: MAS, node: str, values: Mapping[str, Any], *, budget: Budget | None = None) -> Trace:
    """Run one node on given values, as one visit of a run would, and return the trace of that visit.

    ``values`` are the node's reads by value name, each checked against its
    type. A read left out renders as absent, as at a visit before anything
    wrote it; ``run`` comes from the visit's own meter, and ``error`` is given
    like any read. A member of a ``for each`` chain runs in the system an item
    runs. No edge out of the node is followed, so no predicate runs, and an
    error fails the trace with its kind. The visit runs on a meter of its own
    under ``budget``. The trace's ``outputs`` hold the node's writes, typed.

    Raises:
        ValueError: Before any call, for a node inside a box, a human node, a
            name that is no node, or values that are not its reads or do not
            fit their types.
    """
    system = node_owner(mas, node)
    target = system.nodes[node]
    if isinstance(target, Human):
        raise ValueError(f"node {node!r} is a human node: it asks a person and takes no example")
    reads, writes = node_types(system, target)
    problems = [f"{name!r} is no read of {node!r}, which reads {sorted(reads)}" for name in values if name not in reads]
    problems += [
        f"{name!r} is not {reads[name]}: {problem}"
        for name, value in values.items()
        if name in reads and (problem := check_value(value, reads[name])) is not None
    ]
    if problems:
        raise ValueError(f"The values do not fit node {node!r}:\n" + "\n".join(f"- {line}" for line in problems))
    types = system.info.types if system.info is not None else {}
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    builder.add_node(node, wrap(system, node, node_body(system, target, types)), input_schema=MASState)  # type: ignore[arg-type]
    builder.add_edge(START, node)
    builder.add_edge(node, END)
    meter = RunMeter(budget, owner=system)
    query = query_text(values)
    try:
        state = initial_state(dict(values), system.prompts())
        graph: Any = builder.compile()
        result = graph.invoke(state, config=run_config(system, meter))
    except Exception as error:  # noqa: BLE001  any failure of the visit is its failed trace, never raised
        return failed_trace(system, query, error, meter)
    finally:
        meter.close()
    produced, records = result["values"], result["records"]
    outputs = {name: produced[name] for name in writes if name in produced}
    output = raw(records[-1].outputs, plain(outputs)) if records else ""
    return build_trace(system, query, records, meter, output=output, outputs=outputs)
