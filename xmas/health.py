"""Health: what looked wrong in a finished run, read from its records.

:func:`summarize` runs at the end of every run and never changes it: a flag is
information for the person reading the trace, not a rule. It reads four things:

- per record, a :class:`Flag` for each signal it shows, in this order: ``retried``,
  ``bad_format``, ``error_routed``, ``item_failed``, ``empty``, ``stuck``,
  ``unfilled``, ``leak``, ``long``, ``short``, ``item_over_budget``;
- per ``while`` edge, how often the loop was taken and whether its cap was hit;
- per run, each limit the budget sets, with what the run spent of it, and the
  time human nodes waited for their answers;
- per ``for each`` chain, its items, the failed ones and what each cost.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from xmas.errors import ErrorKind
from xmas.prompts import extract_placeholders

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from xmas.budget import RunMeter
    from xmas.trace import AgentRecord

__all__ = [
    "DEFAULT_LEAKS",
    "Flag",
    "Health",
    "HealthSettings",
    "summarize",
]


DEFAULT_LEAKS = (r"<\|[A-Za-z_]{2,30}\|>", r"</?think>", r"<<<xmas:agent ")
"""Patterns that should never reach an output: chat-template tokens, reasoning tags, xmas's encoding header."""


@dataclass(frozen=True)
class HealthSettings:
    """How sensitive the signals are; a version 2 spec sets them under ``health:``.

    Attributes:
        length_factor: An output is ``long`` beyond this many times its node's
            median length, and ``short`` under the median divided by it.
        min_samples: The records a node needs before its median is used.
        leak: Regular expressions flagged as ``leak``, on top of
            :data:`DEFAULT_LEAKS`.

    Raises:
        ValueError: If ``length_factor`` is under 1, ``min_samples`` under 1, or
            a leak pattern does not compile.
    """

    length_factor: float = 3.0
    min_samples: int = 3
    leak: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse settings that cannot be applied."""
        if not self.length_factor >= 1:
            raise ValueError(f"health length_factor must be >= 1, got {self.length_factor!r}")
        if self.min_samples < 1:
            raise ValueError(f"health min_samples must be >= 1, got {self.min_samples!r}")
        for pattern in self.leak:
            try:
                re.compile(pattern)
            except (re.error, TypeError) as exc:
                raise ValueError(f"health leak pattern {pattern!r} does not compile: {exc}") from exc


@dataclass(frozen=True)
class Flag:
    """One signal on one record.

    Attributes:
        signal: Which signal, such as ``stuck``.
        node: The node the record belongs to.
        step: The record's step in the run.
        detail: What was seen, in a short plain sentence.
    """

    signal: str
    node: str
    step: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        return asdict(self)


@dataclass(frozen=True)
class Health:
    """What :func:`summarize` found in one run.

    Attributes:
        flags: The per-record flags, in record order.
        loops: Per ``while`` edge taken or capped, keyed ``"source -> target"``:
            ``{"taken": n, "cap_hit": bool}``.
        budget: Per limit the budget sets (``"calls"``, ``"cost_usd"``,
            ``"wall_s"``): ``{"spent": x, "limit": y}``; and ``"waited_s"``,
            ``{"spent": x}``, the seconds human nodes waited for answers,
            which no wall limit counts.
        items: Per ``for each`` chain, keyed by its head: ``{"items": n,
            "failed": k, "cost_usd": [...]}``, one cost per item in item order,
            ``None`` where a record's cost is unknown.
    """

    flags: tuple[Flag, ...] = ()
    loops: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget: dict[str, dict[str, float]] = field(default_factory=dict)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form with only the parts that report something; empty for a healthy run."""
        data: dict[str, Any] = {}
        if self.flags:
            data["flags"] = [flag.to_dict() for flag in self.flags]
        if self.loops:
            data["loops"] = {edge: dict(entry) for edge, entry in self.loops.items()}
        if self.budget:
            data["budget"] = {limit: dict(entry) for limit, entry in self.budget.items()}
        if self.items:
            data["items"] = {head: {**entry, "cost_usd": list(entry["cost_usd"])} for head, entry in self.items.items()}
        return data


def summarize(  # noqa: PLR0913  the public keyword-only signature, kept for version 1 callers
    records: Sequence[AgentRecord],
    *,
    meter: RunMeter | None = None,
    settings: HealthSettings | None = None,
    while_edges: Iterable[tuple[str, str]] = (),
    values: Iterable[str] | None = None,
    chains: Mapping[str, tuple[Sequence[str], Mapping[str, float]]] | None = None,
) -> Health:
    """Read a run's records, in run order, for the health signals.

    The budget entry reads ``meter`` against its budget; without a meter nothing
    was spent, so there is none. Its ``waited_s`` sums the records' ``wait_s``,
    when any has one. ``while_edges`` are the system's
    ``(source, target)`` loops. ``values`` are the system's value names: a
    placeholder naming none of them is literal prompt text, so it is not
    ``unfilled``; when ``None``, every placeholder outside a record's inputs is.
    ``chains`` maps each chain's head to its nodes and its item budget
    (``cost_usd``, ``wall_s``); an item over a limit is flagged on its last
    record, and an unknown cost or latency is never over.
    """
    settings = settings or HealthSettings()
    items, over = item_health(records, chains or {})
    return Health(
        flags=flags(records, settings, None if values is None else frozenset(values), over),
        loops=loops(records, while_edges),
        budget=budget_use(meter, records),
        items=items,
    )


def outputs_text(record: AgentRecord) -> str:
    """Return a record's outputs as one text."""
    return "\n".join(record.outputs.values())


@dataclass(frozen=True)
class SignalRules:
    """What every record of one run is judged by: the settings, the leak patterns and the system's value names."""

    settings: HealthSettings
    leaks: tuple[re.Pattern[str], ...]
    values: frozenset[str] | None


def flags(
    records: Sequence[AgentRecord],
    settings: HealthSettings,
    values: frozenset[str] | None,
    over: Mapping[int, list[tuple[str, str]]],
) -> tuple[Flag, ...]:
    """Return every record's flags, in record order; ``over`` holds the item budget flags by record position."""
    rules = SignalRules(settings, tuple(re.compile(pattern) for pattern in (*DEFAULT_LEAKS, *settings.leak)), values)
    lengths: dict[str, list[int]] = {}
    for record in records:
        if record.error is None:
            lengths.setdefault(record.agent, []).append(len(outputs_text(record)))
    medians = {node: statistics.median(sizes) for node, sizes in lengths.items() if len(sizes) >= settings.min_samples}
    previous: dict[tuple[str, int | None], AgentRecord] = {}
    flags: list[Flag] = []
    for position, record in enumerate(records):
        mine = (record.agent, record.item)
        found = record_signals(record, previous.get(mine), medians.get(record.agent), rules)
        found += over.get(position, [])
        flags.extend(Flag(signal, record.agent, record.step, detail) for signal, detail in found)
        previous[mine] = record
    return tuple(flags)


def record_signals(
    record: AgentRecord,
    before: AgentRecord | None,
    median: float | None,
    rules: SignalRules,
) -> list[tuple[str, str]]:
    """Return ``(signal, detail)`` for each signal one record shows, in the order of the table."""
    ok, text = record.error is None, outputs_text(record)
    failure = f"{'unknown' if record.error_kind is None else record.error_kind.value}: {record.error}"
    same = before.step if before is not None and before.error is None and outputs_text(before) == text else None
    placeholders = extract_placeholders(record.prompt)
    missing = sorted((placeholders if rules.values is None else placeholders & rules.values) - set(record.inputs))
    leak = next((match.group() for pattern in rules.leaks if (match := pattern.search(text))), None)
    size, factor, median = len(text), rules.settings.length_factor, median or 0
    lengths = f"{size} characters, the node's median is {median:g}"
    signals: dict[str, object] = {
        "retried": record.rejected and f"{len(record.rejected)} answer(s) rejected: {record.rejected[0].error}",
        "bad_format": record.error_kind == ErrorKind.BAD_FORMAT and "the answer did not match its form after the retry",
        "error_routed": not ok and record.route and failure,
        # elsewhere an unrouted failure ended the run, and the trace's error says so
        "item_failed": not (ok or record.route)
        and record.item is not None
        and f"{item_name(record)} failed: {failure}",
        "empty": ok and not text.strip() and "the output is empty",
        "stuck": ok and bool(text.strip()) and same is not None and f"the same output as step {same}",
        "unfilled": ok and missing and f"placeholders without a value: {', '.join(missing)}",
        "leak": ok and leak and f"the output holds {leak!r}",
        "long": ok and median and size > factor * median and lengths,
        "short": ok and median and size < median / factor and lengths,
    }
    return [(signal, detail) for signal, detail in signals.items() if isinstance(detail, str) and detail]


def loops(records: Sequence[AgentRecord], while_edges: Iterable[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    """Return ``{"source -> target": {"taken", "cap_hit"}}`` for each while edge taken or capped."""
    loops: dict[str, dict[str, Any]] = {}
    for source, target in dict.fromkeys(while_edges):
        mine = [record for record in records if record.agent == source]
        taken = sum(target in (record.route or ()) for record in mine)
        cap_hit = any(record.capped and target not in (record.route or ()) for record in mine)
        if taken or cap_hit:
            loops[f"{source} -> {target}"] = {"taken": taken, "cap_hit": cap_hit}
    return loops


def budget_use(meter: RunMeter | None, records: Sequence[AgentRecord]) -> dict[str, dict[str, float]]:
    """Return ``{limit: {"spent", "limit"}}`` for each limit the meter's budget sets, then the time waited."""
    use: dict[str, dict[str, float]] = {}
    budget = None if meter is None else meter.budget
    if meter is not None and budget is not None:
        for name, limit, spent in (
            ("calls", budget.max_calls, meter.calls),
            ("cost_usd", budget.max_cost_usd, meter.cost_usd),
            ("wall_s", budget.max_wall_s, meter.elapsed_s),
        ):
            if limit is not None:
                use[name] = {"spent": spent, "limit": limit}
    waits = [record.wait_s for record in records if record.wait_s is not None]
    if waits:
        use["waited_s"] = {"spent": sum(waits)}
    return use


def item_name(record: AgentRecord) -> str:
    """Return how a flag names a record's item, such as ``item 2 (page-3)``."""
    return f"item {record.item}" + ("" if record.item_key is None else f" ({record.item_key})")


def total(parts: Iterable[float | None]) -> float | None:
    """Return the sum, or ``None`` when any part is unknown."""
    numbers = list(parts)
    return None if None in numbers else sum(number for number in numbers if number is not None)


def item_health(
    records: Sequence[AgentRecord], chains: Mapping[str, tuple[Sequence[str], Mapping[str, float]]]
) -> tuple[dict[str, dict[str, Any]], dict[int, list[tuple[str, str]]]]:
    """Return ``Health.items`` and the item budget flags keyed by the position of each item's last record."""
    items: dict[str, dict[str, Any]] = {}
    over: dict[int, list[tuple[str, str]]] = {}
    for head, (nodes, budget) in chains.items():
        by_item: dict[int, list[int]] = {}
        for position, record in enumerate(records):
            if record.agent in nodes and record.item is not None:
                by_item.setdefault(record.item, []).append(position)
        costs: list[float | None] = [None] * (max(by_item, default=-1) + 1)
        failed = 0
        for item, positions in by_item.items():
            mine = [records[position] for position in positions]
            costs[item] = cost = total(record.cost_usd for record in mine)
            wall = total(record.latency_s for record in mine)
            failed += any(record.error is not None and not record.route for record in mine)
            for limit, spent, said in (("cost_usd", cost, "cost {:g} USD"), ("wall_s", wall, "took {:g} s")):
                if limit in budget and spent is not None and spent > budget[limit]:
                    detail = f"{item_name(mine[-1])} {said.format(spent)}, over its budget of {budget[limit]:g}"
                    over.setdefault(positions[-1], []).append(("item_over_budget", detail))
        items[head] = {"items": len(costs), "failed": failed, "cost_usd": costs}
    return items, over
