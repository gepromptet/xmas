"""Warm-up examples: small cases a bundle carries, and the checks that judge their answers.

A bundle's ``examples/<file>.yaml`` holds a few cases::

    version: 1
    budget: {cost_usd: 0.03, per example: {cost_usd: 0.015, wall_s: 120}}
    examples:                           # the cases, in order
    - name: checker-sees-the-overflow     # unique across the bundle's files
      node: checker                       # a node case: inputs are the node's reads, by value name
      from: trace 5d1e04c2, record 7      # where it came from, optional
      inputs: {page: {id: intro}, measure: {words: 13, overflow: true}}
      recorded: {fits: false, fix: cut it}
      must: [{form: valid}, {path: verdict.fits, equals: false}]
    - name: the-whole-deck                # a run case: inputs are the system's inputs
      inputs: {plan: {slides: [], figures: []}, house: plain}
      expect: {deck: [{path: deck.fits, len: 0}]}

A check is one mapping of one kind (:data:`CHECK_KINDS`) with an optional
``path``: a value name, then ``.field`` or ``[i]`` steps. Without a path a
check judges the node's one write, or all its writes as one mapping. A step's
case with ``recorded`` and no ``must`` checks that its writes equal it. In a
run case ``expect`` keys an output or a node; a node's checks apply to its
last visit, per item in a chain. A media input is a ref mapping whose
``location`` is relative to the bundle, such as ``examples/media/3f2a.png``.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from xmas.agent import Agent
from xmas.budget import Budget
from xmas.bundle import LOCK, Lock, check_lock, code_hashes, dump_yaml, outside, put, read, read_yaml, relock
from xmas.media import Ref, store
from xmas.nodes import Step
from xmas.runtime import node_owner, node_types
from xmas.schemas import compile_form
from xmas.values import JSON, TEXT, ValueType, media_kind, parse_type, plain

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS
    from xmas.trace import AgentRecord, Trace

__all__ = [
    "CHECK_KINDS",
    "FOLDER",
    "MEDIA",
    "UNSET",
    "VERSION",
    "Check",
    "Example",
    "ExampleFile",
    "cases_from_trace",
    "parse_examples",
    "run_checks",
    "save_examples",
    "written",
]


VERSION = 1
FOLDER = "examples"
MEDIA = "examples/media"
FILE_KEYS = ("version", "budget", "examples")
CASE_KEYS = ("name", "node", "from", "inputs", "recorded", "must", "expect")
BUDGET_KEYS = {"cost_usd": "max_cost_usd", "wall_s": "max_wall_s"}
PER_EXAMPLE = "per example"
REF_KEYS = frozenset(item.name for item in fields(Ref))
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
PATH = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)((?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*)\Z")
STEP = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
# a value shown in a failed check is cut to this many characters, the cut marked with "..."
SHOWN_WIDTH = 80


class Unset:
    pass


UNSET = Unset()
"""What :attr:`Example.recorded` holds when the case records no reference."""

MISSING = object()


@dataclass(frozen=True)
class Check:
    """One thing that must hold of an answer.

    Attributes:
        kind: One of :data:`CHECK_KINDS`.
        path: Where the checked value is: a value name, then fields and list indexes; ``()`` for the node's writes.
        value: What the kind compares with: the text, the length, the pattern; ``True`` for ``not empty`` and
            ``exists``, ``"valid"`` for ``form``.
        within: The tolerance of ``near``: a number's distance, or a text's share of difference from 0 to 1.
    """

    kind: str
    path: tuple[str | int, ...] = ()
    value: Any = None
    within: float | None = None

    @classmethod
    def parse(cls, spec: Any, where: str) -> Check:
        """Read a check from its mapping, such as ``{path: verdict.fits, equals: false}``.

        Raises:
            ValueError: Naming ``where`` and what is wrong: no kind or two, an unknown kind, a bad path or value.
        """
        if not isinstance(spec, Mapping):
            raise ValueError(f"{where}: a check is a mapping such as {{equals: x}}, got {spec!r}")
        kinds = [key for key in spec if key not in ("path", "within")]
        unknown = [key for key in kinds if key not in CHECK_KINDS]
        if unknown:
            raise ValueError(f"{where}: unknown check {unknown[0]!r}; the kinds are {', '.join(CHECK_KINDS)}")
        if len(kinds) != 1:
            raise ValueError(f"{where}: a check holds one kind, got {kinds or 'none'}")
        kind, value = kinds[0], spec[kinds[0]]
        path = parse_path(spec["path"], where) if "path" in spec else ()
        within = spec.get("within")
        if (within is not None) != (kind == "near"):
            raise ValueError(f"{where}: 'within' goes with 'near', and 'near' needs it")
        if kind == "near" and ((number := as_number(within)) is None or number < 0):
            raise ValueError(f"{where}: within is a number of at least 0, got {within!r}")
        if kind == "near" and as_number(value) is None and not isinstance(value, str):
            raise ValueError(f"{where}: near takes a number or a text, got {value!r}")
        problem = {
            "form": value != "valid" and "form takes 'valid'",
            "not empty": value is not True and "not empty takes true",
            "exists": value is not True and "exists takes true",
            "len": (isinstance(value, bool) or not isinstance(value, int) or value < 0) and "len takes a count",
            "line": not isinstance(value, str) and "line takes a text",
            "matches": pattern_problem(value),
        }.get(kind)
        if problem:
            raise ValueError(f"{where}: {problem}, got {value!r}")
        return cls(kind, path, value, None if within is None else float(within))

    def to_spec(self) -> dict[str, Any]:
        """Return the mapping :meth:`parse` reads."""
        spec: dict[str, Any] = {"path": path_text(self.path)} if self.path else {}
        spec[self.kind] = self.value
        if self.within is not None:
            spec["within"] = self.within
        return spec

    def __str__(self) -> str:
        """Return the check as one line, such as ``verdict.fits equals false``."""
        tail = f" within {self.within:g}" if self.within is not None else ""
        bare = self.kind in ("not empty", "exists")
        shown = " valid" if self.kind == "form" else "" if bare else f" {show(self.value)}"
        return f"{path_text(self.path) + ' ' if self.path else ''}{self.kind}{shown}{tail}"


@dataclass(frozen=True)
class Example:
    """One case: a node case when ``node`` is set, else a run case of the whole system.

    Attributes:
        name: Unique across the bundle's examples files.
        inputs: The node's reads (a node case) or the system's inputs (a run case), by value name; media as refs.
        node: The node a node case runs, or ``None``.
        recorded: The node's writes as once answered, or :data:`UNSET`.
        must: A node case's checks.
        expect: A run case's checks, keyed by output or node.
        origin: Where the case came from (its ``from``), for people.
    """

    name: str
    inputs: Mapping[str, Any]
    node: str | None = None
    recorded: Any = UNSET
    must: tuple[Check, ...] = ()
    expect: Mapping[str, tuple[Check, ...]] = field(default_factory=dict)
    origin: str = ""


@dataclass(frozen=True)
class ExampleFile:
    """One examples file: its cases, in order, and the budget it asks for.

    Attributes:
        examples: The cases.
        budget: The cap on all its cases together, or ``None``.
        per_example: The cap on each case, or ``None``.
    """

    examples: tuple[Example, ...]
    budget: Budget | None = None
    per_example: Budget | None = None


def parse_examples(
    files: Mapping[str, bytes], root: Path | None = None, *, problems: list[str] | None = None
) -> list[ExampleFile]:
    """Read every examples file among a bundle's files, keyed by their paths in the bundle.

    A media input's ``location`` is read from ``root`` (the bundle folder),
    where it is referenced, else from ``files``, its bytes kept in
    :func:`~xmas.media.store`. Given ``problems``, each problem is added to
    it and the cases that read well are returned.

    Raises:
        ValueError: Without ``problems``, listing every problem, one per line, each naming its file.
    """
    collect = problems is not None
    problems = [] if problems is None else problems
    parsed: list[ExampleFile] = []
    seen: dict[str, str] = {}
    for path in sorted(path for path in files if re.fullmatch(rf"{FOLDER}/[^/]+\.yaml", path)):
        try:
            examples, budgets = file(read_yaml(files, path), path, files, root, problems)
        except (ValueError, TypeError) as error:  # a file that is no YAML is named by its own message
            problems.append(str(error) if str(error).startswith(path) else f"{path}: {error}")
            continue
        for example in examples:
            if example.name in seen:
                problems.append(f"{path}: example {example.name!r} is named in {seen[example.name]} too")
            seen.setdefault(example.name, path)
        parsed.append(ExampleFile(tuple(examples), *budgets))
    if problems and not collect:
        raise ValueError("\n".join(problems))
    return parsed


def file(
    data: Any, path: str, files: Mapping[str, bytes], root: Path | None, problems: list[str]
) -> tuple[list[Example], tuple[Budget | None, Budget | None]]:
    """Read one file's mapping; a bad case adds its problem and is left out."""
    if not isinstance(data, Mapping):
        raise ValueError(f"an examples file is a mapping with {', '.join(FILE_KEYS)}")
    known(data, FILE_KEYS, "the file")
    if data.get("version") != VERSION:
        raise ValueError(f"version is {VERSION}, got {data.get('version')!r}")
    budgets = read_budgets(data.get("budget"))
    cases = data.get("examples")
    if not isinstance(cases, list):
        raise ValueError("examples is a list of cases")
    examples = []
    for index, case in enumerate(cases):
        try:
            examples.append(read_case(case, path, index, files, root))
        except ValueError as error:
            problems.append(str(error))
    return examples, budgets


def known(data: Mapping[Any, Any], keys: Sequence[str], where: str) -> None:
    unknown = [str(key) for key in data if key not in keys]
    if unknown:
        raise ValueError(f"{where} takes {', '.join(keys)}; unknown: {', '.join(unknown)} (a note goes in a # comment)")


def read_budgets(spec: Any) -> tuple[Budget | None, Budget | None]:
    if spec is None:
        return None, None
    if not isinstance(spec, Mapping):
        raise ValueError(f"budget is a mapping such as {{cost_usd: 0.03}}, got {spec!r}")
    known(spec, [*BUDGET_KEYS, PER_EXAMPLE], "budget")
    per = spec.get(PER_EXAMPLE)
    if per is not None and not isinstance(per, Mapping):
        raise ValueError(f"budget.per example is a mapping such as {{cost_usd: 0.015}}, got {per!r}")
    if per is not None:
        known(per, list(BUDGET_KEYS), "budget.per example")
    total = caps(spec)
    return Budget(**total) if total else None, None if per is None else Budget(**caps(per))


def caps(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {BUDGET_KEYS[key]: value for key, value in spec.items() if key in BUDGET_KEYS}


def read_case(case: Any, path: str, index: int, files: Mapping[str, bytes], root: Path | None) -> Example:
    where = f"{path}: example {index + 1}"
    if not isinstance(case, Mapping):
        raise ValueError(f"{where}: a case is a mapping with {', '.join(CASE_KEYS)}")
    name = case.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError(f"{where}: name is a word of letters, digits, '_', '.' and '-', got {name!r}")
    where = f"{path}: example {name!r}"
    known(case, CASE_KEYS, where)
    node = case.get("node")
    if node is not None and not isinstance(node, str):
        raise ValueError(f"{where}: node is a node's name, got {node!r}")
    inputs = case.get("inputs", {})
    if not isinstance(inputs, Mapping) or not all(isinstance(key, str) for key in inputs):
        raise ValueError(f"{where}: inputs map value names to values, got {inputs!r}")
    origin = case.get("from", "")
    if not isinstance(origin, str):
        raise ValueError(f"{where}: from is a text, got {origin!r}")
    for key, needs in (("recorded", "a node case"), ("must", "a node case"), ("expect", "a run case")):
        if key in case and (node is None) == (needs == "a node case"):
            raise ValueError(f"{where}: {key} belongs to {needs}")
    must = case.get("must", [])
    if not isinstance(must, list):
        raise ValueError(f"{where}: must is a list of checks")
    expect = case.get("expect", {})
    if not isinstance(expect, Mapping) or not all(isinstance(checks, list) for checks in expect.values()):
        raise ValueError(f"{where}: expect maps an output or a node to a list of checks")
    media = Media(files, root, where)
    return Example(
        name=name,
        inputs={key: media.load(value) for key, value in inputs.items()},
        node=node,
        recorded=media.load(case["recorded"]) if "recorded" in case else UNSET,
        must=tuple(Check.parse(spec, f"{where}: must {i + 1}") for i, spec in enumerate(must)),
        expect={
            str(key): tuple(Check.parse(spec, f"{where}: expect {key} {i + 1}") for i, spec in enumerate(checks))
            for key, checks in expect.items()
        },
        origin=origin,
    )


@dataclass(frozen=True)
class Media:
    """Where a case's media come from: the bundle's files, else its folder."""

    files: Mapping[str, bytes]
    root: Path | None
    where: str

    def load(self, value: Any) -> Any:
        """Return ``value`` with each ref mapping in it, at any depth, as a :class:`~xmas.media.Ref`."""
        if isinstance(value, Mapping) and {"location", "media_type"} <= set(value) <= REF_KEYS:
            return self.ref(value)
        return each(value, self.load)

    def ref(self, spec: Mapping[str, Any]) -> Ref:
        try:
            given = Ref.from_dict(spec)
        except ValueError as error:
            raise ValueError(f"{self.where}: {error}") from error
        location = PurePosixPath(given.location)
        if outside(given.location):
            raise ValueError(f"{self.where}: media location {given.location!r} is not a path inside the bundle")
        on_disk = None if self.root is None else self.root / location
        if self.root is not None and self.root.joinpath(location).resolve() != self.root.resolve() / location:
            raise ValueError(f"{self.where}: media {given.location} is or sits in a symlink; media are plain files")
        if on_disk is not None and on_disk.is_file():
            ref = Ref.from_path(on_disk, given.media_type)
        elif given.location in self.files:
            ref = store().put(self.files[given.location], given.media_type, location.name)
        else:
            raise ValueError(f"{self.where}: media {given.location} is not in the bundle")
        if given.sha256 is not None and given.sha256 != ref.sha256:
            raise ValueError(f"{self.where}: media {given.location} changed since the case was written (sha256)")
        return ref


def parse_path(text: Any, where: str) -> tuple[str | int, ...]:
    """Read ``verdict.orders[0].kind`` into ``("verdict", "orders", 0, "kind")``."""
    match = PATH.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"{where}: path is a value name, then .field or [i] steps, got {text!r}")
    steps = [name or int(index) for name, index in STEP.findall(match.group(2))]
    return (match.group(1), *steps)


def path_text(path: Sequence[str | int]) -> str:
    return "".join(f"[{step}]" if isinstance(step, int) else f".{step}" if i else step for i, step in enumerate(path))


def pattern_problem(value: Any) -> str | None:
    if not isinstance(value, str):
        return "matches takes a regular expression"
    try:
        re.compile(value)
    except re.error as error:
        return f"matches takes a regular expression ({error})"
    return None


def as_number(value: Any) -> float | None:
    return None if isinstance(value, bool) or not isinstance(value, int | float) else float(value)


def show(value: Any) -> str:
    text = json.dumps(plain(value), ensure_ascii=False, default=str)
    return text if len(text) <= SHOWN_WIDTH else f"{text[: SHOWN_WIDTH - len('...')]}..."


def same(left: Any, right: Any) -> bool:
    """Whether two values are equal: refs by their bytes' sha256, a bool only to a bool."""
    if isinstance(left, Ref) or isinstance(right, Ref):
        return isinstance(left, Ref) and isinstance(right, Ref) and digest(left) == digest(right)
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def digest(ref: Ref) -> str:
    return ref.sha256 or hashlib.sha256(ref.read_bytes()).hexdigest()


def find(values: Mapping[str, Any], path: tuple[str | int, ...]) -> tuple[Any, str]:
    """Return the value at ``path``, or :data:`MISSING` with why."""
    if not path:
        return one_or_all(values), ""
    head = str(path[0])
    if head not in values:
        return MISSING, f"no value {head!r}; there are {sorted(values)}"
    found = values[head]
    for step in path[1:]:
        listed = isinstance(step, int) and isinstance(found, list) and step < len(found)
        if listed or (isinstance(step, str) and isinstance(found, Mapping) and step in found):
            found = found[step]
        else:
            return MISSING, f"nothing at {path_text(path)}"
    return found, ""


def one_or_all(values: Mapping[str, Any]) -> Any:
    return next(iter(values.values())) if len(values) == 1 else dict(values)


def run_checks(checks: Sequence[Check], values: Mapping[str, Any], record: AgentRecord | None = None) -> list[str]:
    """Judge a node's writes, ``{value name: value}``, and return one line per check that failed.

    ``record`` is the visit that wrote them, which ``form`` reads: its error,
    and the answers its form refused. Without one, ``form`` has nothing to judge.
    """
    failed = []
    for check in checks:
        found, why = find(values, check.path)
        problem = JUDGES[check.kind](found, check, record) if found is not MISSING else why
        if problem:
            failed.append(f"{check}: {problem}")
    return failed


def contains(found: Any, want: Any, *, wanted: bool) -> str | None:
    if isinstance(found, str) and isinstance(want, str):
        held = want in found
    elif isinstance(found, list):
        held = any(same(item, want) for item in found)
    else:
        return f"got {show(found)}, not a text or a list"
    return None if held == wanted else f"got {show(found)}"


def length(found: Any, want: int) -> str | None:
    if not isinstance(found, str | list | Mapping):
        return f"got {show(found)}, which has no length"
    return None if len(found) == want else f"has length {len(found)}"


def in_text(found: Any, held: Callable[[str], bool], miss: str = "got") -> str | None:
    if not isinstance(found, str):
        return f"got {show(found)}, not a text"
    return None if held(found) else f"{miss} {show(found)}"


def near(found: Any, want: Any, within: float) -> str | None:
    """Judge ``near``: numbers by their distance, texts by their share of difference."""
    if as_number(want) is not None:
        number = as_number(found)
        if number is None:
            return f"got {show(found)}, not a number"
        off = abs(number - float(want))
    elif isinstance(found, str):
        off = 1 - difflib.SequenceMatcher(None, found, want).ratio()
    else:
        return f"got {show(found)}, not a text"
    return None if off <= within else f"got {show(found)}, off by {off:.3g}"


JUDGES: dict[str, Callable[[Any, Check, AgentRecord | None], str | None]] = {
    "form": lambda found, check, record: "no visit to judge" if record is None else record.error,  # noqa: ARG005 (one signature for every kind)
    "equals": lambda found, check, *_: None if same(found, check.value) else f"got {show(found)}",
    "contains": lambda found, check, *_: contains(found, check.value, wanted=True),
    "not contains": lambda found, check, *_: contains(found, check.value, wanted=False),
    "matches": lambda found, check, *_: in_text(found, lambda text: bool(re.search(check.value, text, re.MULTILINE))),
    "line": lambda found, check, *_: in_text(found, lambda text: check.value in text.splitlines(), "no such line in"),
    "len": lambda found, check, *_: length(found, check.value),
    "not empty": lambda found, *_: "empty" if found is None or found in ("", [], {}) else None,
    "exists": lambda found, *_: "absent" if found is None else None,
    "near": lambda found, check, *_: near(found, check.value, float(check.within or 0)),
}
CHECK_KINDS = tuple(JUDGES)
"""The kinds of check, one per check mapping."""


def parse_rendered(text: str, vtype: ValueType) -> Any:
    """Return a value from the line a record shows for it (:func:`~xmas.values.render_value`'s inverse)."""
    if vtype == TEXT:
        return text
    value = json.loads(text)
    if media_kind(vtype) is None:
        return value
    if vtype.item is not None:
        return [None if item is None else Ref.from_dict(item) for item in value]
    return Ref.from_dict(value)


def written(system: MAS, name: str, record: AgentRecord) -> dict[str, Any]:
    """Return the writes of a node's visit, typed from the record's rendered outputs."""
    _, writes = node_types(system, system.nodes[name])
    return {
        value: parse_rendered(record.outputs[value], vtype)
        for value, vtype in writes.items()
        if value in record.outputs
    }


def cases_from_trace(mas: MAS, trace: Trace, nodes: Sequence[str] | None, store_dir: Path) -> list[dict[str, Any]]:
    """Return one node case per node, as the mapping an examples file holds, from a run's trace.

    A case comes from the node's first visit (item 0 in a chain) that did not
    fail: its inputs are the record's inputs mapped back from placeholders
    (agents) or parameters (steps) to value names and parsed by type, a read
    absent at that visit left out; ``recorded`` is its typed write. A json
    agent must give a valid answer with every top-level field of its form, a
    text agent a non-empty one; a step must write what it wrote. Media are
    copied into ``store_dir`` as ``<sha256[:16]><ext>``. With ``nodes=None``
    every agent and step of ``mas`` that ran is taken, else exactly those named.

    Raises:
        ValueError: For a named node that is no agent or step of ``mas``, or did not finish a first visit.
    """
    ran = [r.agent for r in trace.records if r.kind in ("agent", "step")]
    chosen = list(dict.fromkeys(ran)) if nodes is None else list(nodes)
    cases = []
    for name in chosen:
        record = next(
            (
                r
                for r in trace.records
                if r.agent == name and r.error is None and not r.lost and r.visit in (None, 1) and r.item in (None, 0)
            ),
            None,
        )
        if record is None and nodes is None:
            continue
        if record is None:
            raise ValueError(f"node {name!r} finished no first visit in the trace")
        system = node_owner(mas, name)
        node = system.nodes[name]
        if not isinstance(node, Agent | Step):
            raise ValueError(f"node {name!r} is no agent or step: only those are recorded")
        reads, _ = node_types(system, node)
        shown = (
            {node.reads[key]: text for key, text in record.inputs.items()} if isinstance(node, Agent) else record.inputs
        )
        inputs = {
            value: spec(parse_rendered(text, reads[value]), store_dir)
            for value, text in shown.items()
            if value in reads and value != "run" and text != ""
        }
        case: dict[str, Any] = {"name": name, "node": name}
        if trace.run_id is not None:
            case["from"] = f"trace {trace.run_id[:8]}, record {record.step}"
        case.update(inputs=inputs, recorded=spec(one_or_all(written(system, name, record)), store_dir))
        must = default_checks(node)
        if must:
            case["must"] = [check.to_spec() for check in must]
        cases.append(case)
    return cases


def default_checks(node: Agent | Step) -> list[Check]:
    """Return a recorded node's checks: a json agent's form and its fields, a text agent's non-empty answer."""
    if isinstance(node, Step):
        return []
    if parse_type(node.output_type) == JSON:
        form = compile_form(node.schema)
        return [
            Check("form", value="valid"),
            *(Check("exists", (node.output, f), value=True) for f in form.fields or {}),
        ]
    if parse_type(node.output_type) == TEXT:
        return [Check("not empty", value=True)]
    return [Check("form", value="valid")]


def spec(value: Any, store_dir: Path) -> Any:
    """Return ``value`` as an examples file writes it, each ref's bytes copied into ``store_dir``."""
    if not isinstance(value, Ref):
        return each(value, lambda item: spec(item, store_dir))
    data = value.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    target = store_dir / f"{digest[:16]}{'' if value.remote else PurePosixPath(value.name).suffix.lower()}"
    put(target, data)
    return {"location": f"{MEDIA}/{target.name}", "media_type": value.media_type, "sha256": digest}


def each(value: Any, do: Callable[[Any], Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: do(item) for key, item in value.items()}
    return [do(item) for item in value] if isinstance(value, list) else value


def write_cases(folder: Path, file: str, cases: Sequence[Mapping[str, Any]]) -> None:
    """Put ``cases`` into ``folder/examples/<file>.yaml``, each replacing its namesake, once the result parses."""
    if not NAME.match(file):
        raise ValueError(f"an examples file is named by a plain word, got {file!r}")
    relative = f"{FOLDER}/{file}.yaml"
    path = folder / relative
    data = read_yaml({relative: path.read_bytes()}, relative) if path.is_file() else None
    data = dict(data) if isinstance(data, Mapping) else {"version": VERSION, "examples": []}
    kept = [case for case in data.get("examples") or [] if isinstance(case, Mapping)]
    new = {case["name"]: case for case in cases}
    data["examples"] = [dict(new.pop(case.get("name"), case)) for case in kept] + [dict(case) for case in new.values()]
    text = dump_yaml(data)
    parse_examples({relative: text}, folder)
    put(path, text)


def save_examples(
    mas: MAS, trace: Trace, *, nodes: Sequence[str] | None = None, file: str = "warmup", path: Path | None = None
) -> list[str]:
    """Record node cases from a run's trace (:func:`cases_from_trace`) into a bundle folder; return their names.

    The cases go into ``examples/<file>.yaml`` of ``path``, else of the
    folder ``mas`` was saved to or loaded from, each replacing the case of its
    name, their media into ``examples/media/``. A bundle unchanged since it
    was saved gets its lock refreshed; one changed since keeps its lock, so
    ``verify(0)`` still shows the change.

    Raises:
        ValueError: Without a folder to write to, for a zipped bundle, or as :func:`cases_from_trace` raises.
    """
    folder = Path(path) if path is not None else mas.bundle
    if folder is None:
        raise ValueError("save_examples writes into a bundle folder: save the system first, or give path=")
    if folder.is_file():
        raise ValueError(f"{folder} is a zipped bundle: save_examples writes into a bundle folder")
    cases = cases_from_trace(mas, trace, nodes, folder / MEDIA)
    lock = clean_lock(mas, folder)
    write_cases(folder, file, cases)
    if lock is not None:
        relock(folder, lock.code)
    return [case["name"] for case in cases]


def clean_lock(mas: MAS, folder: Path) -> Lock | None:
    """Return a bundle folder's lock when the folder still matches it, code included."""
    if not (folder / LOCK).is_file():
        return None
    lock, diff = check_lock(read(folder), code_hashes(mas))
    return lock if diff.clean else None  # a lock edited by hand stays, so verify(0) shows it
