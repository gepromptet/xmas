"""Verify a system: unchanged since saved, every part in place, and its warm-up examples still answered.

Three levels, each including the ones below it:

- **0, checksums:** the bundle's files and its python callables' source
  against its lock, and whether the system in memory differs from what was
  saved.
- **1, dry:** the system check, every agent's prompt rendered by the engine
  and its answer form compiled, every examples file read and its cases fitted
  to their nodes, and every model name looked up in the configured
  endpoint's catalogue. No model call.
- **2, warm-up:** the examples run, a node case through
  :func:`~xmas.runtime.run_node` and a run case through ``run_state``, under a
  small budget. A code node must write what was recorded; an agent's answer must
  pass its case's checks.

A report that is not ok issues one :class:`VerifyWarning`, or raises
:class:`VerifyFailed` when asked to be strict.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xmas.budget import Budget
from xmas.bundle import LOCK, LockDiff, check_lock, code_hashes, files_of, owned, read, same, systems
from xmas.check import check_system
from xmas.config import Catalogue, Config, read_catalogue, resolve_names
from xmas.examples import FOLDER, UNSET, Check, Example, ExampleFile, parse_examples, run_checks, written
from xmas.human import HumanAction
from xmas.media import Ref
from xmas.runtime import node_owner, node_types, render_prompt, run_node
from xmas.schemas import compile_form
from xmas.spec import VERSION_2
from xmas.values import JSON, TEXT, ValueType, check_value, parse_type, ref_kind

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS
    from xmas.trace import Trace

__all__ = [
    "BUDGET_USD",
    "PER_EXAMPLE_USD",
    "STATUSES",
    "ExampleResult",
    "VerifyFailed",
    "VerifyReport",
    "VerifyWarning",
    "verify",
]


BUDGET_USD = 0.05
"""What all examples together may cost when neither the caller nor an examples file sets a budget."""

PER_EXAMPLE_USD = 0.02
"""What one example may cost when neither its file nor the caller sets a cap."""

STATUSES = ("pass", "fail", "skipped", "error")
"""How an example ended: its checks passed or failed, it was not run, or it could not run."""

DRY_LEVEL = 1
WARM_UP_LEVEL = 2
LEVELS = (0, DRY_LEVEL, WARM_UP_LEVEL)
# the levels: 0 checksums, 1 dry, 2 warm-up; each includes the ones below

SHORTEST_SHOWN_S = 0.1
# an example faster than this shows as "<0.1 s"


PLACEHOLDER_TYPES = {"image": "image/png", "audio": "audio/wav", "document": "application/pdf", "table": "text/csv"}


class VerifyWarning(UserWarning):
    """A verify report that is not ok; its message is the report."""


class VerifyFailed(ValueError):  # noqa: N818 (named for what happened, as BudgetExceeded is)
    """A strict verify whose report is not ok; ``report`` holds it."""

    def __init__(self, report: VerifyReport) -> None:
        """Keep the report; the message is the report as text."""
        super().__init__(str(report))
        self.report = report


@dataclass(frozen=True)
class ExampleResult:
    """How one example went.

    Attributes:
        name: The example's name.
        node: The node of a node case, ``None`` for a run case.
        status: One of :data:`STATUSES`.
        failures: Why it failed or could not run, one line each.
        cost_usd: What its calls cost, ``None`` when unknown.
        latency_s: Its wall seconds.
        notes: What else to know, such as answers its form refused or why it was skipped.
    """

    name: str
    node: str | None
    status: str
    failures: tuple[str, ...] = ()
    cost_usd: float | None = 0.0
    latency_s: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Whether the example ran and every check held."""
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        return lists(asdict(self))


@dataclass(frozen=True)
class VerifyReport:
    """What :func:`verify` found.

    Attributes:
        level: The level run, 0, 1 or 2.
        bundle: The bundle the system was saved to or loaded from, or ``None``.
        lock: What differs from the lock, or ``None`` when there is no lock to compare with.
        unsaved: The bundle files the system in memory would write differently.
        problems: What level 1 found wrong, one line each.
        misses: Each model name not in the catalogue, with the closest names.
        catalogue: The catalogue names were looked up in, ``None`` below level 1.
        examples: One result per example, in file order.
        cost_usd: What the examples cost together, ``None`` when any cost is unknown.
        digest: The bundle digest the lock records, its first 16 hex digits.
        files: How many files the lock records.
        notes: Lines that are neither problems nor results, such as why level 2 did not run.
    """

    level: int
    bundle: Path | None = None
    lock: LockDiff | None = None
    unsaved: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    misses: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    catalogue: Catalogue | None = None
    examples: tuple[ExampleResult, ...] = ()
    cost_usd: float | None = 0.0
    digest: str | None = None
    files: int = 0
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the lock matches, nothing is unsaved, level 1 found nothing and no example failed; skips pass."""
        failed = any(result.status in ("fail", "error") for result in self.examples)
        return self.lock is not None and self.lock.clean and not (self.unsaved or self.problems or failed)

    def lines(self) -> list[str]:
        """Return the report as lines: one block per level, then the verdict."""
        lines = self.level0()
        if self.level >= DRY_LEVEL:
            count = len(self.problems)
            head = "no problems" if not count else f"{count} problem{'s' * (count > 1)}"
            lines.append(f"level 1: {head}; {self.names()}")
            lines += [f"  - {problem}" for problem in self.problems]
        if self.level >= WARM_UP_LEVEL:
            width = max((len(result.name) for result in self.examples), default=0)
            status = max((len(result.status) for result in self.examples), default=0)
            for result in self.examples:
                cost = usd(result.cost_usd)
                took = f"<{SHORTEST_SHOWN_S} s" if result.latency_s < SHORTEST_SHOWN_S else f"{result.latency_s:.1f} s"
                lines.append(f"level 2: {result.name:<{width}}  {result.status:<{status}}  {cost}  {took}")
                lines += [f"  - {failure}" for failure in result.failures]
                lines += [f"  note: {note}" for note in result.notes]
            lines += [f"level 2: {note}" for note in self.notes]
        lines.append(self.verdict())
        return lines

    def level0(self) -> list[str]:
        """Return the level 0 lines: whether the bundle changed since its lock was saved."""
        if self.lock is None:
            return ["level 0: not saved: no lock"]
        if self.lock.clean and not self.unsaved:
            return [f"level 0: unchanged since saved (bundle {self.digest}, {self.files} files)"]
        found = [*asdict(self.lock).items(), ("in memory, not saved", self.unsaved)]
        return [f"level 0: {label.replace('_', ' ')}: {', '.join(paths)}" for label, paths in found if paths]

    def names(self) -> str:
        """Return the line on whether the model names were checked, and against what."""
        catalogue = self.catalogue
        if catalogue is None or catalogue.ids is None:
            why = "" if catalogue is None else catalogue.note.removesuffix("; model names not checked")
            return f"model names not checked ({why})"
        tail = f" ({catalogue.note})" if catalogue.note else ""
        return f"model names checked against the {catalogue.source} catalogue{tail}"

    def verdict(self) -> str:
        """Return the one-line verdict: ok or not, the examples passed and the cost."""
        counts = {status: sum(r.status == status for r in self.examples) for status in STATUSES}
        cost = usd(self.cost_usd)
        tally = ", ".join(f"{n} {status}" for status, n in counts.items() if n and status != "pass")
        examples = f"{counts['pass']} example{'s' * (counts['pass'] != 1)} passed" + (f", {tally}" if tally else "")
        if self.ok:
            return f"ok: {examples}, {cost}" if self.level >= WARM_UP_LEVEL else f"ok: level {self.level}"
        reasons = [] if self.lock is not None and self.lock.clean and not self.unsaved else ["level 0 differs"]
        if self.problems:
            reasons.append(f"{len(self.problems)} problem{'s' * (len(self.problems) > 1)}")
        if self.level >= WARM_UP_LEVEL:
            reasons.append(f"{examples}, {cost}")
        return f"not ok: {'; '.join(reasons)}"

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form, ``ok`` included."""
        catalogue = self.catalogue
        return {
            "ok": self.ok,
            "level": self.level,
            "bundle": None if self.bundle is None else str(self.bundle),
            "lock": None if self.lock is None else lists(asdict(self.lock)),
            "digest": self.digest,
            "files": self.files,
            "unsaved": list(self.unsaved),
            "problems": list(self.problems),
            "misses": {name: list(near) for name, near in self.misses.items()},
            "catalogue": None
            if catalogue is None
            else {"source": catalogue.source, "checked": catalogue.ids is not None, "note": catalogue.note},
            "examples": [result.to_dict() for result in self.examples],
            "cost_usd": self.cost_usd,
            "notes": list(self.notes),
        }

    def warn(self) -> None:
        """Issue one :class:`VerifyWarning` holding the report, when it is not ok."""
        if not self.ok:
            warnings.warn(str(self), VerifyWarning, stacklevel=3)

    def __str__(self) -> str:
        """Return :meth:`lines` as one text."""
        return "\n".join(self.lines())


def usd(cost: float | None) -> str:
    return "cost unknown" if cost is None else f"{cost:.4f} USD"


def lists(data: dict[str, Any]) -> dict[str, Any]:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in data.items()}


def verify(  # noqa: PLR0913  the public signature, kept for version 1 callers
    mas: MAS,
    level: int = 1,
    *,
    budget: Budget | float | None = None,
    strict: bool = False,
    only: Iterable[str] = (),
    examples: str | Path | None = None,
    config: Config | None = None,
) -> VerifyReport:
    """Verify a system up to ``level`` and return the report.

    Args:
        mas: The system; its ``bundle`` attribute, when set, is the folder or zip it was saved to or loaded from.
        level: 0 checksums, 1 dry, 2 warm-up; each includes the ones below.
        budget: What level 2 may spend: a total in USD, or a :class:`~xmas.budget.Budget` whose
            ``max_cost_usd`` is the total and whose other limits hold for each example. Without it, the first
            examples file's ``budget``, else :data:`BUDGET_USD`. Each example is capped at its file's
            ``per example``, else at what is left of a given budget, else at :data:`PER_EXAMPLE_USD`; once nothing
            is left, the rest are skipped.
        strict: Raise :class:`VerifyFailed` instead of warning when the report is not ok.
        only: Run only the examples of these names, or of these nodes.
        examples: An examples file or folder to use instead of the bundle's ``examples/``.
        config: Where model names are looked up; :meth:`~xmas.config.Config.load` when ``None``.

    Raises:
        ValueError: For a level that is not 0, 1 or 2.
        VerifyFailed: When ``strict`` and the report is not ok.
    """
    if level not in LEVELS:
        raise ValueError(f"verify levels are 0, 1 and 2, got {level!r}")
    bundle = mas.bundle
    files = {} if bundle is None else dict(read(bundle))
    parsed: list[ExampleFile] = []
    problems: list[str] = []
    misses: dict[str, tuple[str, ...]] = {}
    catalogue = None
    wanted = set(only)
    if level >= DRY_LEVEL:
        parsed, problems, misses, catalogue = level1(mas, sources(examples, bundle, files), config)
        named = {name for file in parsed for example in file.examples for name in (example.name, example.node)}
        problems += [f"only names no example or node: {name!r}" for name in sorted(wanted - named)]
    results: tuple[ExampleResult, ...] = ()
    cost: float | None = 0.0
    notes: tuple[str, ...] = ()
    if level == WARM_UP_LEVEL and problems:
        notes = ("not run: level 1 found problems",)
    elif level == WARM_UP_LEVEL:
        results, cost = level2(mas, parsed, wanted, budget)
        notes = () if results else ("no examples to run",)
    report = VerifyReport(
        level,
        bundle,
        **level0(mas, files),
        problems=tuple(problems),
        misses=misses,
        catalogue=catalogue,
        examples=results,
        cost_usd=cost,
        notes=notes,
    )
    if not report.ok:
        if strict:
            raise VerifyFailed(report)
        report.warn()
    return report


def level0(mas: MAS, files: Mapping[str, bytes]) -> dict[str, Any]:
    """Compare the bundle's files with its lock, and the system in memory with the files."""
    if LOCK not in files:
        return {}
    ours = files_of(mas)
    unsaved = tuple(
        path for path in sorted({*ours, *filter(owned, files)}) if not same(path, ours.get(path), files.get(path))
    )
    saved, diff = check_lock(files, code_hashes(mas))
    found = {"lock": diff, "unsaved": unsaved}
    return found if saved is None else {**found, "digest": saved.bundle[:16], "files": len(saved.files)}


def sources(
    examples: str | Path | None, bundle: Path | None, files: Mapping[str, bytes]
) -> tuple[dict[str, bytes], Path | None]:
    """Return the examples files keyed by their paths in a bundle, and the folder their media paths start from."""
    if examples is None:
        return dict(files), bundle if bundle is not None and bundle.is_dir() else None
    path = Path(examples)
    if path.is_file():
        root = path.parent.parent if path.parent.name == FOLDER else path.parent
        return {f"{FOLDER}/{path.name}": path.read_bytes()}, root
    if not path.is_dir():
        raise ValueError(f"no examples at {path}: give an examples file, or a bundle folder or its examples folder")
    root = path.parent if path.name == FOLDER else path
    inside = sorted(item for item in (root / FOLDER).rglob("*") if item.is_file())
    linked = [item for item in inside if item.resolve() != root.resolve() / item.relative_to(root)]
    if linked:
        raise ValueError(f"{linked[0]} is or sits in a symlink; examples are plain files")
    return {item.relative_to(root).as_posix(): item.read_bytes() for item in inside}, root


def level1(
    mas: MAS, sources: tuple[dict[str, bytes], Path | None], config: Config | None
) -> tuple[list[ExampleFile], list[str], dict[str, tuple[str, ...]], Catalogue]:
    """Check the system, read the examples, render every prompt and look up every model name; no model call."""
    problems: list[str] = []
    for _, system in systems(mas):
        if system.version == VERSION_2:
            try:
                check_system(system)
            except ValueError as error:
                problems += getattr(error, "problems", [str(error)])
    parsed = parse_examples(*sources, problems=problems)
    cases = [example for file in parsed for example in file.examples]
    for example in cases:
        problems += [f"example {example.name!r}: {line}" for line in case_problems(mas, example)]
    found = list(problems)  # a form the check refused already is not named twice
    problems += [line for line in render_problems(mas, cases) if not any(line.partition(": ")[2] in f for f in found)]
    names = model_names(mas)
    if not names:
        return parsed, problems, {}, Catalogue(None, "none", "no model is named")
    config = Config.load() if config is None else config
    resolution = resolve_names(names, config, catalogue=read_catalogue(config))
    problems += [str(warning) for warning in resolution.warnings]
    return parsed, problems, dict(resolution.suggestions), resolution.catalogue


def case_problems(mas: MAS, example: Example) -> list[str]:
    """Return what does not fit in a case: its node, its inputs, its recorded writes, its check paths."""
    if example.node is None:
        lines = [f"input {name!r} is missing" for name in mas.inputs if name not in example.inputs]
        lines += fit(example.inputs, mas.inputs, "input")
        for key, checks in example.expect.items():
            if key in mas.outputs:
                lines += paths(checks, {key: mas.outputs[key]}, f"output {key!r}")
                lines += [
                    f"{check}: form judges a node's answer, and {key!r} is an output"
                    for check in checks
                    if check.kind == "form"
                ]
                continue
            try:
                system = node_owner(mas, key)
            except ValueError:
                lines.append(f"expect names {key!r}, which is no output or node of the system")
                continue
            lines += paths(checks, node_types(system, system.nodes[key])[1], f"node {key!r}")
        return lines
    try:
        system = node_owner(mas, example.node)
    except ValueError as error:
        return [str(error)]
    node = system.nodes[example.node]
    if isinstance(node, HumanAction):
        return [f"node {example.node!r} is a human action: it asks a person and takes no example"]
    reads, writes = node_types(system, node)
    lines = fit(example.inputs, reads, "read")
    if example.recorded is not UNSET:
        recorded = example.recorded if len(writes) != 1 else {next(iter(writes)): example.recorded}
        if not isinstance(recorded, Mapping):
            lines.append(f"recorded holds the writes {sorted(writes)} as a mapping, got {recorded!r}")
        else:
            lines += [f"recorded {line}" for line in fit(recorded, writes, "write")]
    return lines + paths(example.must, writes, f"node {example.node!r}")


def fit(values: Mapping[str, Any], types: Mapping[str, ValueType], word: str) -> list[str]:
    """Return a line per value that is not one of ``types``, or does not fit its type."""
    lines = [f"{name!r} is no {word}; the {word}s are {sorted(types)}" for name in values if name not in types]
    return lines + [
        f"{word} {name!r} is not {types[name]}: {problem}"
        for name, value in values.items()
        if name in types and (problem := check_value(value, types[name])) is not None
    ]


def paths(checks: Iterable[Check], writes: Mapping[str, ValueType], what: str) -> list[str]:
    """Return a line per check whose path starts at no value ``what`` writes."""
    return [
        f"{check}: {what} writes {sorted(writes)}, not {check.path[0]!r}"
        for check in checks
        if check.path and check.path[0] not in writes
    ]


def render_problems(mas: MAS, cases: Iterable[Example]) -> list[str]:
    """Render every agent's prompt on empty values and on its cases' inputs, and compile its answer form."""
    inputs: dict[str, list[Mapping[str, Any]]] = {}
    for example in cases:
        if example.node is not None:
            inputs.setdefault(example.node, []).append(example.inputs)
    lines = []
    for _, system in systems(mas):
        for agent in system.agents.values():
            owner = node_owner(system, agent.name)
            types = owner.info.types if owner.info is not None else {}
            reads, _ = node_types(owner, agent)
            trials: list[Mapping[str, Any]] = [{name: empty(vtype) for name, vtype in reads.items()}]
            trials += inputs.get(agent.name, []) if system is mas else []
            try:
                for values in trials:
                    render_prompt(agent, agent.prompt, values, types)
                if parse_type(agent.output_type) == JSON:
                    compile_form(agent.schema).format_block()
            except (ValueError, TypeError) as error:
                lines.append(f"agent {agent.name!r}: {error}")
    return lines


def empty(vtype: ValueType) -> Any:
    """Return a placeholder value of a type: ``""`` for text, ``{}`` for json, a ref to nothing for media."""
    kind = ref_kind(vtype)
    if kind is not None:
        ref = Ref(f"example.{kind}", PLACEHOLDER_TYPES.get(kind, "application/octet-stream"))
        return [ref] if vtype.item is not None else ref
    return [] if vtype.item is not None else "" if vtype == TEXT else {}


def model_names(mas: MAS) -> dict[str, str]:
    """Return each catalogue id the agents bind, at every depth, with who binds it: ``{id: "agents a, b"}``."""
    bound: dict[str, list[str]] = {}
    for _, system in systems(mas):
        for agent in system.agents.values():
            name = agent.model if agent.model is not None else system.default_model
            if name is not None:
                model_id = getattr(system.models.get(name), "model", None) or name
                bound.setdefault(model_id, []).append(agent.name)
    return {model_id: f"agent{'s' * (len(who) > 1)} {', '.join(who)}" for model_id, who in bound.items()}


def level2(
    mas: MAS, parsed: list[ExampleFile], wanted: set[str], budget: Budget | float | None
) -> tuple[tuple[ExampleResult, ...], float | None]:
    """Run the examples in file order under the budget; return their results and what they cost together."""
    given = budget.max_cost_usd if isinstance(budget, Budget) else None if budget is None else float(budget)
    declared = (file.budget.max_cost_usd for file in parsed if file.budget and file.budget.max_cost_usd is not None)
    total = Budget(max_cost_usd=given if given is not None else next(declared, BUDGET_USD)).max_cost_usd or 0.0
    limits = budget if isinstance(budget, Budget) else Budget()
    spent: float | None = 0.0
    results = []
    for file in parsed:
        per = file.per_example or Budget()
        for example in file.examples:
            if wanted and example.name not in wanted and example.node not in wanted:
                continue
            left = None if spent is None else total - spent
            if left is None or left <= 0:
                why = "an earlier example's cost is unknown" if left is None else f"nothing left of {total:.4f} USD"
                results.append(ExampleResult(example.name, example.node, "skipped", notes=(f"budget: {why}",)))
                continue
            each = per.max_cost_usd if per.max_cost_usd is not None else left if given is not None else PER_EXAMPLE_USD
            cap = Budget(
                max_cost_usd=min(each, left),
                max_wall_s=per.max_wall_s if per.max_wall_s is not None else limits.max_wall_s,
                max_calls=limits.max_calls,
                max_parallel_calls=limits.max_parallel_calls,
            )
            result = run(mas, example, cap)
            results.append(result)
            spent = None if spent is None or result.cost_usd is None else spent + result.cost_usd
    return tuple(results), spent


def run(mas: MAS, example: Example, cap: Budget) -> ExampleResult:
    """Run one example under ``cap`` and judge it."""
    started = time.perf_counter()
    try:
        if example.node is not None:
            trace = run_node(mas, example.node, example.inputs, budget=cap)
        else:
            system = mas.copy()
            system.budget = cap
            trace = system.run_state(example.inputs)
    except ValueError as error:
        return ExampleResult(example.name, example.node, "error", (str(error),), 0.0, time.perf_counter() - started)
    took = time.perf_counter() - started
    cost = trace.cost_usd if trace.records else 0.0
    failures = example_failures(mas, example, trace)
    if cost is None:
        failures.append("cost_unknown: a call reported no cost and its model has no price, so the cap cannot bind")
    notes = tuple(
        f"{record.agent}: its form refused {len(record.rejected)} answer(s) first"
        for record in trace.records
        if record.rejected
    )
    return ExampleResult(example.name, example.node, "fail" if failures else "pass", tuple(failures), cost, took, notes)


def example_failures(mas: MAS, example: Example, trace: Trace) -> list[str]:
    """Return what failed in an example's trace: the run's error, else the checks that did not hold."""
    if trace.pause is not None:
        return ["the run paused on a question: examples run without people"]
    if trace.failed:
        return [f"{trace.error_kind or 'error'}: {trace.error}"]
    if example.node is not None:
        visits = trace[example.node]
        record = visits[-1] if visits else None
        checks = example.must
        if not checks and example.recorded is not UNSET and record is not None and record.kind == "code":
            checks = (Check("equals", value=example.recorded),)
        return run_checks(checks, trace.outputs, record)
    lines = []
    for key, checks in example.expect.items():
        if key in mas.outputs:
            lines += [f"{key}: {line}" for line in run_checks(checks, {key: trace.outputs.get(key)})]
            continue
        last = {record.item: record for record in sorted(trace[key], key=lambda record: record.visit or 0)}
        if not last:
            lines.append(f"{key}: did not run")
        system = node_owner(mas, key) if last else mas
        for item, record in last.items():
            label = key if item is None else f"{key} item {item}"
            found = [record.error] if record.error else run_checks(checks, written(system, key, record), record)
            lines += [f"{label}: {line}" for line in found]
    return lines
