"""Budgets: limits on what one run may spend, and the meter that enforces them.

A :class:`Budget` is runtime state, like a MAS's tools, and never part of the
spec. Each run gets its own :class:`RunMeter`, which counts the run's calls, cost
and wall time and refuses a call before it starts once a limit is reached. The
meter also holds the run's caps on calls and items in flight, passes the run's
events to a listener, and writes its records, events and model calls to the
run's :class:`~xmas.store.RunLog`, when it has one.

A run is stopped through its meter (:meth:`RunMeter.stop`, found by run id in
:data:`RUNNING`): no call, code node or item starts after that, waits for a free slot
end, and a call in flight sees the stop on its :func:`cancel_flag` and is cut.
The same flag cuts the calls of a ``first of`` contestant once another one won.
"""

from __future__ import annotations

import threading
import time
import uuid
import warnings
import weakref
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from xmas.errors import BudgetExceeded, EmptyResponse, RaceLost, Stopped
from xmas.trace import AgentRecord, Event, Trace
from xmas.utils import timestamp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from xmas.llms.meta import CallMeta
    from xmas.store import RunLog

__all__ = [
    "RUNNING",
    "SLICE_S",
    "Budget",
    "CancelFlag",
    "RunMeter",
    "VisitMeter",
    "cancel_flag",
    "contestant",
]


SLICE_S = 0.05
"""How often a wait for a slot, a place or a model's answer checks the cancel flag, in seconds."""

RUNNING: weakref.WeakValueDictionary[str, RunMeter] = weakref.WeakValueDictionary()
"""The meters of this process's runs by run id, from the meter's making until :meth:`RunMeter.close`."""

running_lock = threading.Lock()
cancel: ContextVar[CancelFlag | None] = ContextVar("xmas_cancel_flag", default=None)
current_race: ContextVar[threading.Event | None] = ContextVar("xmas_race", default=None)


def cancel_flag() -> CancelFlag | None:
    """Return the flag of the model call in flight on this thread, or ``None`` outside a metered call.

    A model wrapper polls it, or waits on it, and ends its call once it is set,
    raising what :meth:`CancelFlag.check` raises.
    """
    return cancel.get()


@contextmanager
def contestant(race: threading.Event) -> Iterator[None]:
    """Run the block as a ``first of`` contestant of ``race`` (:meth:`RunMeter.race`): its calls end once it is set."""
    token = current_race.set(race)
    try:
        yield
    finally:
        current_race.reset(token)


class CancelFlag:
    """Whether a call in flight should end: its run was stopped, or its race was won by another contestant."""

    def __init__(self, meter: RunMeter, race: threading.Event | None = None) -> None:
        """Watch ``meter``'s stop, and ``race`` when the call is a contestant's."""
        self.meter = meter
        self.event = meter.halted if race is None else race  # stopping a run sets its races too

    @property
    def reason(self) -> str | None:
        """``"stopped"`` once the run is stopped, ``"lost"`` once another contestant won, else ``None``."""
        if self.meter.stopped is not None:
            return "stopped"
        return "lost" if self.event.is_set() else None

    def is_set(self) -> bool:
        """Whether the call should end."""
        return self.reason is not None

    def wait(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the flag, and return whether it is set."""
        self.event.wait(timeout)
        return self.is_set()

    def check(self, meta: CallMeta | None = None, text: str = "") -> None:
        """Raise once the flag is set, with what the cut call used and the answer it received so far.

        Raises:
            Stopped: With the run's stop reason, once the run is stopped.
            RaceLost: Once another contestant of the call's race won.
        """
        reason = self.reason
        if reason == "stopped":
            raise Stopped(str(self.meter.stopped), meta, text)
        if reason == "lost":
            raise RaceLost("another contestant finished first", meta, text)


@dataclass(frozen=True)
class Budget:
    """Limits for one run, in USD, wall seconds and model calls; ``None`` means no limit.

    A call may start only while the run is below every limit, so a limit of ``0``
    starts no call. A cost limit that cannot be checked, because a call reported
    no cost and the model has no price, refuses the next call instead of silently
    not binding. A negative or NaN limit is refused. ``max_parallel_calls`` caps
    the model calls in flight at once, across every branch, module and item of the
    run; a call waits for a free slot. It is an integer of at least ``1``.
    """

    max_cost_usd: float | None = None
    max_wall_s: float | None = None
    max_calls: int | None = None
    max_parallel_calls: int | None = None

    def __post_init__(self) -> None:
        """Refuse negative and NaN limits, and a parallel cap that is not an integer of at least 1."""
        for name in ("max_cost_usd", "max_wall_s", "max_calls"):
            value = getattr(self, name)
            if value is not None and not value >= 0:
                raise ValueError(f"Budget.{name} must be >= 0, got {value!r}")
        cap: object = self.max_parallel_calls
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError(f"Budget.max_parallel_calls must be an integer >= 1, got {cap!r}")


class RunMeter:
    """What one run has used so far, checked against its budget before each call.

    The counters and the record list are shared by the run's parallel branches,
    so every update holds one lock.

    Attributes:
        budget: The run's limits, or ``None`` to count without limits.
        owner: The MAS the meter belongs to, so a nested run of another MAS that
            inherits this meter can tell it is not its own, unless the meter
            adopted that MAS (:meth:`adopt`).
        started_s: ``time.monotonic()`` when the meter was made, then when the
            run's first node started (:meth:`start`).
        calls: Calls started so far.
        cost_usd: Cost reported or priced so far, in USD.
        uncosted_calls: Finished calls whose cost is unknown.
        records: Records of the owner's finished visits, in wall-clock order;
            an adopted system's are kept apart (:meth:`records_of`).
        run_id: The run's id, given by the host or a new random one; the
            meter is in :data:`RUNNING` under it until :meth:`close`.
        graph_id: The ``describe()`` id of the owner, when the run was given it.
        listener: Called with each of the run's events (:meth:`emit`), or ``None``.
        log: Where the run's records, events and model calls are written as
            they happen, or ``None``.
        stopped: Why the run was stopped (:meth:`stop`), or ``None``.
        halted: Set when the run is stopped.
    """

    def __init__(
        self,
        budget: Budget | None = None,
        owner: object | None = None,
        *,
        listener: Callable[[Event], None] | None = None,
        graph_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Start the clock with every counter at zero, and register the meter under its run id.

        Raises:
            ValueError: If ``run_id`` is the id of a run still in :data:`RUNNING`.
        """
        self.budget = budget
        self.owner = owner
        self.run_id = uuid.uuid4().hex if run_id is None else run_id
        self.graph_id = graph_id
        self.listener = listener
        self.log: RunLog | None = None
        self.started_s = time.monotonic()
        self.calls = 0
        self.cost_usd = 0.0
        self.uncosted_calls = 0
        self.records: list[AgentRecord] = []
        self.adopted: dict[int, tuple[object, list[AgentRecord]]] = {}
        self.lock = threading.Lock()
        self.started = False
        cap = None if budget is None else budget.max_parallel_calls
        self.slots = None if cap is None else threading.BoundedSemaphore(cap)
        self.gates: dict[str, threading.BoundedSemaphore] = {}
        self.stopped: str | None = None
        self.halted = threading.Event()
        self.races: dict[str, threading.Event] = {}
        self.winners: dict[str, str] = {}
        self.failed: dict[str, int] = {}
        with running_lock:
            if self.run_id in RUNNING:
                raise ValueError(f"Run id {self.run_id!r} is in use by a running run.")
            RUNNING[self.run_id] = self

    def close(self) -> None:
        """Take the meter out of :data:`RUNNING` once its run ended or paused; a second call does nothing."""
        with running_lock:
            if RUNNING.get(self.run_id) is self:
                del RUNNING[self.run_id]

    def stop(self, reason: str = "stopped") -> None:
        """Stop the run: emit one ``stop`` event, cut its calls in flight, and start nothing more.

        Every later :meth:`before_call`, :meth:`check_boundary`, :meth:`slot`
        and :meth:`gate` raises :class:`~xmas.errors.Stopped` with ``reason``,
        waits included. A second stop does nothing.
        """
        with self.lock:
            if self.stopped is not None:
                return
            self.stopped = reason
            races = list(self.races.values())
        self.halted.set()
        for race in races:
            race.set()
        self.emit("stop", "", reason=reason)

    def race(self, key: str) -> threading.Event:
        """Return the flag of the ``first of`` race ``key``, set once a contestant wins or the run stops."""
        with self.lock:
            race = self.races.setdefault(key, threading.Event())
            if self.stopped is not None:
                race.set()
            return race

    def win(self, key: str, node: str) -> bool:
        """Claim the race ``key`` for ``node``; only the first claim wins, and it sets the race's flag."""
        with self.lock:
            if key in self.winners:
                return False
            self.winners[key] = node
            self.races.setdefault(key, threading.Event()).set()
            return True

    def fail(self, key: str) -> int:
        """Count a contestant of the race ``key`` that failed; return how many have failed so far."""
        with self.lock:
            self.failed[key] = self.failed.get(key, 0) + 1
            return self.failed[key]

    def flag(self) -> CancelFlag:
        """Return the cancel flag of a call starting now: the run's stop, and the race of the running contestant."""
        return CancelFlag(self, current_race.get())

    def totals(self) -> dict[str, Any]:
        """Return what the run used so far: ``calls``, ``cost_usd``, ``uncosted_calls`` and ``active_s``."""
        with self.lock:
            return {
                "calls": self.calls,
                "cost_usd": self.cost_usd,
                "uncosted_calls": self.uncosted_calls,
                "active_s": self.elapsed_s,
            }

    def seed(self, totals: Mapping[str, Any]) -> None:
        """Continue from :meth:`totals` of an earlier meter of the run, such as one that paused.

        The counts carry over, and the wall clock counts on from ``active_s``,
        so a wall limit measures active time, not the wait between the two.
        """
        with self.lock:
            self.calls = int(totals["calls"])
            self.cost_usd = float(totals["cost_usd"])
            self.uncosted_calls = int(totals["uncosted_calls"])
            self.started_s = time.monotonic() - float(totals["active_s"])
            self.started = True

    def adopt(self, system: object, into: object | None = None) -> None:
        """Let a nested system, such as a module's, run on this meter: one budget for both.

        Its records are kept apart, or with those of ``into``, a system this
        meter owns or adopted, as a chain's items keep theirs with the system
        the chain sits in.
        """
        with self.lock:
            if id(system) not in self.adopted:
                records = [] if into is None else self.records_of(into)
                self.adopted[id(system)] = (system, records)

    def owns(self, system: object) -> bool:
        """Whether ``system`` is the owner or an adopted system."""
        return system is self.owner or id(system) in self.adopted

    def records_of(self, system: object) -> list[AgentRecord]:
        """Return the records kept for ``system``: the owner's, or an adopted system's."""
        if system is self.owner:
            return self.records
        return self.adopted[id(system)][1] if id(system) in self.adopted else []

    @property
    def elapsed_s(self) -> float:
        """Wall seconds since the meter was made, or since :meth:`start`."""
        return time.monotonic() - self.started_s

    def start(self) -> None:
        """Restart the clock when the run's first node starts; later calls do nothing.

        A batch makes every item's meter before it runs any, and an item can wait
        for a free worker; that wait is not the run's wall time.
        """
        with self.lock:
            if not self.started:
                self.started = True
                self.started_s = time.monotonic()

    def before_call(self) -> None:
        """Count a call that is about to start, or refuse it with :class:`~xmas.errors.BudgetExceeded`.

        A stopped run, or a contestant whose race was won, refuses it first
        (:meth:`CancelFlag.check`); then the limits are checked in order:
        calls, wall time, a cost limit with a call of unknown cost
        (``"cost_unknown"``), cost.
        """
        with self.lock:
            self.flag().check()
            self.check(count_calls=True)
            self.calls += 1

    def check_boundary(self) -> None:
        """Check the stop and the wall and cost limits before work that makes no model call, such as a code node."""
        with self.lock:
            self.flag().check()
            self.check(count_calls=False)

    def check(self, *, count_calls: bool) -> None:
        """Raise for the first limit reached, the call limit only when ``count_calls``; the caller holds the lock."""
        budget = self.budget
        if budget is None:
            return
        if count_calls and budget.max_calls is not None and self.calls >= budget.max_calls:
            raise BudgetExceeded("calls", self.calls, budget.max_calls, self.calls)
        elapsed = self.elapsed_s
        if budget.max_wall_s is not None and elapsed >= budget.max_wall_s:
            raise BudgetExceeded("wall_s", elapsed, budget.max_wall_s, self.calls)
        if budget.max_cost_usd is not None:
            if self.uncosted_calls > 0:
                raise BudgetExceeded("cost_unknown", self.uncosted_calls, budget.max_cost_usd, self.calls)
            if self.cost_usd >= budget.max_cost_usd:
                raise BudgetExceeded("cost_usd", self.cost_usd, budget.max_cost_usd, self.calls)

    def after_call(self, meta: CallMeta | None) -> None:
        """Add a finished call's cost, or count it as uncosted when it is unknown."""
        cost = None if meta is None else meta.cost_usd
        with self.lock:
            if cost is None:
                self.uncosted_calls += 1
            else:
                self.cost_usd += cost

    def metered[T](
        self,
        call: Callable[[], tuple[T, CallMeta | None]],
        metas: list[CallMeta | None] | None = None,
        *,
        llm: object = None,
        node: str | None = None,
    ) -> tuple[T, CallMeta | None]:
        """Make one model call on this meter: in a free slot, refused at a limit, counted when it ends.

        ``call`` returns its result and the call's metadata, and sees the
        call's :func:`cancel_flag`. An empty answer and a cut call raise, but
        what they used is counted first. ``metas`` collects the metadata of
        each finished call. The run's log gets the call, made by ``node`` on
        ``llm``, whether it ended or failed.
        """
        kept = [] if metas is None else metas
        with self.slot():
            self.before_call()
            token = cancel.set(self.flag())
            meta: CallMeta | None = None
            failed: BaseException | None = None
            ts, started = timestamp(), time.perf_counter()
            try:
                result, meta = call()
            except (EmptyResponse, Stopped) as error:
                failed, meta = error, error.meta
                self.after_call(meta)
                kept.append(meta)
                raise
            except BaseException as error:
                failed = error
                raise
            finally:
                cancel.reset(token)
                if self.log is not None:
                    self.log.call(node, llm, ts, time.perf_counter() - started, meta, failed)
            self.after_call(meta)
            kept.append(meta)
            return result, meta

    def keep(self, record: AgentRecord, system: object | None = None) -> None:
        """Keep a finished visit's record, so a failed run still has it, and log it; ``system`` names an adopted one."""
        with self.lock:
            records = self.records_of(self.owner if system is None else system)
            records.append(record)
        if self.log is not None:
            self.log.record(record, boxed=records is not self.records)

    def slot(self) -> AbstractContextManager[Any]:
        """Hold one of the run's ``max_parallel_calls`` slots for a model call; no cap holds nothing.

        A wait for a slot ends with :meth:`CancelFlag.check`'s error once the flag is set.
        """
        return nullcontext() if self.slots is None else self.held(self.slots)

    def gate(self, key: str, limit: int | None) -> AbstractContextManager[Any]:
        """Hold one of ``limit`` places under ``key``, such as a chain's items at once; ``None`` holds nothing.

        The first ``limit`` given for a key holds for the run. A wait ends as a wait for a slot does.
        """
        if limit is None:
            return nullcontext()
        with self.lock:
            places = self.gates.setdefault(key, threading.BoundedSemaphore(limit))
        return self.held(places)

    @contextmanager
    def held(self, places: threading.BoundedSemaphore) -> Iterator[None]:
        """Hold one of ``places``, checking the cancel flag after each wait of :data:`SLICE_S` and once held."""
        flag = self.flag()
        while not places.acquire(timeout=SLICE_S):
            flag.check()
        try:
            flag.check()
            yield
        finally:
            places.release()

    def emit(self, kind: str, node: str, **fields: Any) -> None:
        """Log an event and pass it to the listener, which is called on the run's threads.

        A listener that raises is warned about once and gets no more events
        this run: a viewer's bug does not end a paid run.
        """
        listener = self.listener
        if listener is None and self.log is None:
            return
        event = Event(kind, node, self.run_id, self.elapsed_s, **fields)
        if self.log is not None:
            self.log.event(event)
        if listener is None:
            return
        try:
            listener(event)
        except Exception as error:  # noqa: BLE001  a host's listener may raise anything; it is warned about once and dropped
            with self.lock:
                first = self.listener is listener
                self.listener = None
            if first:
                warnings.warn(
                    f"The event listener raised {type(error).__name__}: {error}; it gets no more events this run.",
                    RuntimeWarning,
                    stacklevel=2,
                )


class VisitMeter:
    """One node visit's share of a run's meter.

    Every call counts on the run's meter; the visit also keeps the metadata of
    its own finished calls (``metas``, ``None`` for a model that gives none),
    so a visit that fails after paid attempts still records what they cost. A
    module's visit keeps the trace of its inner system when that fails (``inner``).
    ``node`` names the visited node in the run's log of calls.
    """

    def __init__(self, run: RunMeter, node: str | None = None) -> None:
        """Start a visit with no call made."""
        self.run = run
        self.node = node
        self.metas: list[CallMeta | None] = []
        self.inner: Trace | None = None

    def metered[T](
        self, call: Callable[[], tuple[T, CallMeta | None]], llm: object = None
    ) -> tuple[T, CallMeta | None]:
        """Make one model call on ``llm`` on the run's meter and keep its metadata for the visit."""
        return self.run.metered(call, self.metas, llm=llm, node=self.node)

    @property
    def meta(self) -> CallMeta | None:
        """The visit's calls combined, or ``None`` when it made none or a model gave none."""
        if not self.metas or any(meta is None for meta in self.metas):
            return None
        first, *rest = cast("list[CallMeta]", self.metas)
        for meta in rest:
            first = first.combine(meta)
        return first
