"""People in the graph: a :class:`HumanAction` node puts a question to a person and writes the answer.

A human action has one of four forms (:data:`FORMS`), and its answer is one json
value with the fields of its form (:data:`ANSWER_FIELDS`) plus who answered and
when (:data:`COMMON_FIELDS`):

- ``ask``: pick one of the ``options``, ``{option}``;
- ``approve``: say yes or no, ``{approved, option}``, ``option`` the label shown
  for the choice;
- ``judge``: pick one of a ``pair`` of read values by name, or ``"tie"``,
  ``{pick}``;
- ``edit``: change the value of the read ``edits`` names, ``{changed}``, the
  edited value written under ``writes["value"]``.

The question, the labels and the literal options are templates over the node's
reads; a placeholder may be a path such as ``{plan.title}``. A run asks through
a host's :class:`Asker`; a question not answered at once pauses the run until
the host answers it.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol, cast

from xmas.trace import Trace
from xmas.values import TEXT, ValueType, check_value, parse_type, render_value

__all__ = [
    "ANSWER_FIELDS",
    "COMMON_FIELDS",
    "DEFAULT_LABELS",
    "EDITED",
    "FORMS",
    "TIE",
    "Asker",
    "HumanAction",
    "Pause",
    "Question",
    "ReplayAsker",
    "answer_value",
    "fill",
    "question_id",
    "template_paths",
]


ANSWER_FIELDS: dict[str, tuple[str, ...]] = {
    "ask": ("option",),
    "judge": ("pick",),
    "approve": ("approved", "option"),
    "edit": ("changed",),
}
"""The fields an answer of each form holds besides :data:`COMMON_FIELDS`."""

FORMS = tuple(ANSWER_FIELDS)
"""The forms of a human action."""

EDITED = "value"
"""The key of an edit's edited value in what a host gives and in :func:`answer_value`'s
result; :meth:`HumanAction.state_writes` writes it under ``writes["value"]``, not in the answer."""

COMMON_FIELDS = ("who", "asked_at", "answered_at", "comment", "timed_out")
"""The fields every answer holds: who answered (``None`` for a default taken on
timeout), when it was asked and answered (ISO UTC), a comment or ``None``, and
whether the deadline passed."""

TIE = "tie"
"""A judge's pick when neither of the pair wins."""

DEFAULT_LABELS = {"yes": "Approve", "no": "Reject"}
"""The labels of an approve node that sets none."""

TAKES = {"ask": "options", "approve": "labels", "judge": "pair", "edit": "edits"}
"""The setting each form takes and the others refuse."""

GIVEN = ("who", "comment", "answered_at")
"""What a host may send besides the form's fields; xmas fills the rest."""

YES_NO = {True: "yes", False: "no"}

ROLES = ("answer", "value")
"""The keys of ``writes``: the answer, and an edit's edited value."""

PAIR_SIZE = 2
"""A judge node picks between two reads."""

PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}(?!\})")


def template_paths(template: str) -> list[tuple[str, ...]]:
    """Return the placeholders of a template, each as a path such as ``("plan", "title")``, in order."""
    return [tuple(match.group(1).split(".")) for match in PLACEHOLDER.finditer(template)]


def fill(template: str, values: Mapping[str, Any], types: Mapping[str, ValueType]) -> str:
    """Fill a template's placeholders from ``values`` by placeholder name.

    A whole value renders as an agent's input renders; a path into a json
    value renders its field, text as itself and anything else as JSON. A
    placeholder naming no value is left as written.
    """

    def one(match: re.Match[str]) -> str:
        root, *fields = match.group(1).split(".")
        if root not in values:
            return match.group(0)
        if not fields:
            return render_value(values[root], types.get(root, TEXT))
        return leaf(follow(values[root], fields))

    return PLACEHOLDER.sub(one, template)


def follow(value: Any, fields: Iterable[str]) -> Any:
    """Follow field names through nested mappings; ``None`` where one is missing."""
    for name in fields:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def leaf(value: Any) -> str:
    """Render one field: text as itself, ``None`` as nothing, anything else as JSON."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def is_text(value: Any) -> bool:
    """Whether a value is non-empty text."""
    return isinstance(value, str) and bool(value)


def texts(where: str, label: str, value: Any) -> tuple[str, ...]:
    """Return a list of texts as a tuple, refusing a bare text or anything else."""
    if isinstance(value, str) or not isinstance(value, Sequence) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where}: {label} is a list of texts, got {value!r}.")
    return tuple(value)


def question_id(run_id: str, node: str, visit: int, item: int | None = None, item_key: str | None = None) -> str:
    """Return a question's id, ``<run_id>/<node>[/<item key or index>]/<visit>``."""
    where = item_key if item_key is not None else item
    return "/".join(str(part) for part in (run_id, node, where, visit) if part is not None)


@dataclass(frozen=True, init=False)
class HumanAction:
    """A node that asks a person and writes the answer.

    Attributes:
        name: Unique node name in the MAS.
        form: One of :data:`FORMS`.
        reads: ``{placeholder: value}``; a list of value names reads each under
            its own name. What the person is shown.
        writes: ``{"answer": name}``, given as the name alone or as
            ``{name: json}``; an edit adds ``"value"``, the name its edited
            value is written under.
        question: The question, a template over the reads.
        options: An ask node's choices: a list of label templates, or the path
            of a read holding them (a list of text, or of ``{label: ...}`` maps).
        labels: An approve node's ``{"yes": ..., "no": ...}`` label templates;
            :data:`DEFAULT_LABELS` for the ones not set.
        pair: A judge node's two read names, which its answer picks between.
        edits: An edit node's read whose value the person changes.
        timeout_s: Seconds the question waits for an answer; ``0`` never
            waits. ``None`` waits until answered.
        default: The answer taken when the deadline passes, as a partial
            answer such as ``{"approved": False}``.
        detached: Whether the node is detached: no optimizer changes its prompt, model, wiring or edges.

    Construction checks the shape of each setting; whether the settings fit the
    form and the reads is :meth:`problems`' question, asked by
    :func:`xmas.check.check_system`.
    """

    name: str
    form: str
    reads: dict[str, str]
    writes: dict[str, str]
    question: str
    options: tuple[str, ...] | str | None
    labels: dict[str, str] | None
    pair: tuple[str, ...] | None
    edits: str | None
    timeout_s: float | None
    default: dict[str, Any] | None
    detached: bool

    def __init__(  # noqa: PLR0913  the public signature: one keyword per setting, as a spec writes them
        self,
        name: str,
        form: str,
        reads: Sequence[str] | Mapping[str, str] = (),
        writes: str | Mapping[str, str] = "answer",
        *,
        question: str = "",
        options: Sequence[str] | str | None = None,
        labels: Mapping[str, str] | None = None,
        pair: Sequence[str] | None = None,
        edits: str | None = None,
        timeout_s: float | None = None,
        default: Mapping[str, Any] | None = None,
        detached: bool = False,
    ) -> None:
        """Refuse a setting of the wrong shape and keep plain copies; the fields are described on the class."""
        where = f"HumanAction {name!r}"
        if not is_text(name) or not is_text(form) or not isinstance(question, str):
            raise ValueError(f"{where}: a human action needs a name, a form and a text question.")
        if not isinstance(detached, bool):
            raise ValueError(f"{where}: detached must be true or false, got {detached!r}.")
        given: dict[str, Any] = {
            "name": name,
            "form": form,
            "reads": plain_reads(where, reads),
            "writes": plain_writes(where, writes),
            "question": question,
            "options": options if options is None or isinstance(options, str) else texts(where, "options", options),
            "pair": None if pair is None else texts(where, "pair", pair),
            "labels": yes_no_labels(plain_mapping(where, "labels", labels)),
            "default": plain_mapping(where, "default", default),
            "edits": checked_edits(where, edits),
            "timeout_s": checked_timeout(where, timeout_s),
            "detached": detached,
        }
        for key, value in given.items():
            object.__setattr__(self, key, value)

    @property
    def answer(self) -> str:
        """The value the answer is written under."""
        return self.writes["answer"]

    @property
    def written(self) -> tuple[str, ...]:
        """The values the node writes: the answer, and an edit's edited value."""
        return tuple(self.writes.values())

    @property
    def renamed(self) -> bool:
        """Whether a placeholder reads a value of another name."""
        return any(placeholder != value for placeholder, value in self.reads.items())

    def templates(self) -> list[str]:
        """Return the texts filled from the reads: the question, the labels and literal options."""
        texts = [self.question, *(self.labels or {}).values()]
        return texts + list(self.options) if isinstance(self.options, tuple) else texts

    def problems(self) -> Iterator[str]:
        """Say what is wrong with the node on its own: its form, its settings, its templates and its default."""
        where = f"human action {self.name!r}"
        if self.form not in FORMS:
            yield f"{where}: form {self.form!r} is not one of {list(FORMS)}"
            return
        found = itertools.chain(self.setting_problems(), self.write_problems(), self.template_problems())
        yield from (f"{where}: {problem}" for problem in found)
        if self.timeout_s is not None and self.timeout_s < 0:
            yield f"{where}: timeout_s is {self.timeout_s!r}; give 0 or more seconds"
        if self.default is not None:
            yield from (f"{where}: default {problem}" for problem in self.default_problems())

    def setting_problems(self) -> Iterator[str]:
        """Say which form setting is set on the wrong form, missing, or names no read."""
        reads = self.reads
        for form, setting in TAKES.items():
            if form != self.form and getattr(self, setting) is not None:
                yield f"{setting} belongs to the {form} form, not to {self.form}"
        if self.form == "ask" and not self.options:
            yield "an ask node needs options, a list of labels or the path of a read"
        if isinstance(self.options, str) and self.options.split(".", maxsplit=1)[0] not in reads:
            yield f"options {self.options!r} is not the path of a read ({list(reads)})"
        if self.form == "judge":
            yield from self.pair_problems()
        if self.form == "edit" and self.edits not in reads:
            yield f"edits names one of its reads {list(reads)}, got {self.edits!r}"

    def pair_problems(self) -> Iterator[str]:
        """Say what is wrong with a judge node's pair: it names two different reads."""
        reads = self.reads
        if self.pair is None:
            yield "a judge node needs a pair, the names of two reads"
        elif len(self.pair) != PAIR_SIZE or len(set(self.pair)) != PAIR_SIZE or not set(self.pair) <= set(reads):
            yield f"pair names two different reads of {list(reads)}, got {list(self.pair)}"

    def write_problems(self) -> Iterator[str]:
        """Say what is wrong with the writes for the form, and with an approve node's label keys."""
        wanted = ROLES if self.form == "edit" else ROLES[:1]
        if tuple(sorted(self.writes)) != wanted:
            yield f"the {self.form} form writes {list(wanted)}, got {sorted(self.writes)}"
        yield from (
            f"labels sets {key!r}; an approve node labels 'yes' and 'no'"
            for key in sorted(set(self.labels or {}) - set(DEFAULT_LABELS))
        )

    def template_problems(self) -> Iterator[str]:
        """Say which placeholder of the question, labels or options is not among the reads."""
        for text in self.templates():
            for path in template_paths(text):
                if path[0] not in self.reads:
                    yield f"placeholder {'.'.join(path)!r} is not among its reads {list(self.reads)}"

    def default_problems(self) -> Iterator[str]:
        """Say what is wrong with the default answer's fields."""
        default: Mapping[str, Any] = self.default or {}
        allowed = [*ANSWER_FIELDS[self.form], "comment"]
        foreign = sorted(set(default) - set(allowed))
        if foreign:
            yield f"sets {foreign}; the {self.form} form's answer takes {allowed}"
        options = self.options if isinstance(self.options, tuple) else None
        choices = (*self.pair, TIE) if self.form == "judge" and self.pair else None
        yield from field_problems(self.form, {k: v for k, v in default.items() if k in allowed}, options or choices)

    def ask(  # noqa: PLR0913  the public signature: the run, visit and item a question is asked in, by keyword
        self,
        values: Mapping[str, Any],
        types: Mapping[str, ValueType],
        *,
        run_id: str,
        visit: int,
        now: datetime,
        item: int | None = None,
        item_key: str | None = None,
        node: str | None = None,
    ) -> Question:
        """Return the question one visit puts to the person, filled from the run's values.

        ``values`` and ``types`` are keyed by value name; ``node`` is the name
        the question gives, the node's own by default.

        Raises:
            ValueError: If an ask node's options read from state are not a
                list of text or of ``{label: ...}`` maps.
        """
        mine = {placeholder: values.get(value) for placeholder, value in self.reads.items()}
        typed = {placeholder: types.get(value, TEXT) for placeholder, value in self.reads.items()}
        name = node or self.name
        deadline = None if self.timeout_s is None else (now + timedelta(seconds=self.timeout_s)).isoformat()
        return Question(
            id=question_id(run_id, name, visit, item, item_key),
            run_id=run_id,
            node=name,
            form=self.form,
            text=fill(self.question, mine, typed),
            shown={placeholder: render_value(mine[placeholder], typed[placeholder]) for placeholder in mine},
            options=self.shown_options(mine, typed),
            item=item,
            item_key=item_key,
            visit=visit,
            asked_at=now.isoformat(),
            deadline=deadline,
            default=None if self.default is None else dict(self.default),
            edits=self.edits if self.form == "edit" else None,
            value_type=str(types[self.writes["value"]]) if self.writes.get("value") in types else None,
        )

    def shown_options(self, values: Mapping[str, Any], types: Mapping[str, ValueType]) -> tuple[str, ...]:
        """Return the choices shown: an ask node's options, an approve node's labels, a judge's pair and tie."""
        if self.form == "approve":
            labels = {**DEFAULT_LABELS, **(self.labels or {})}
            return (fill(labels["yes"], values, types), fill(labels["no"], values, types))
        if self.form == "judge":
            return (*(self.pair or ()), TIE)
        if isinstance(self.options, tuple):
            return tuple(fill(option, values, types) for option in self.options)
        if isinstance(self.options, str):
            root, *fields = self.options.split(".")
            found = follow(values.get(root), fields) if fields else values.get(root)
            shown = [item.get("label") if isinstance(item, Mapping) else item for item in found or [None]]
            if not isinstance(found, list) or not all(isinstance(label, str) for label in shown):
                raise ValueError(
                    f"human action {self.name!r}: options {self.options!r} must be a list of text or of {{label: ...}} "
                    f"maps, got {found!r}"
                )
            return tuple(str(label) for label in shown)
        return ()

    def state_writes(self, answer: Mapping[str, Any], original: Any = None) -> dict[str, Any]:
        """Return what the node writes for an answer: the answer, and an edit's value (``original`` when unchanged)."""
        out: dict[str, Any] = {self.answer: {k: v for k, v in answer.items() if k != EDITED}}
        if "value" in self.writes:
            out[self.writes["value"]] = answer.get(EDITED, original)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form, the set keys only, in the order a spec writes them."""
        data: dict[str, Any] = {"name": self.name, "form": self.form}
        if self.reads:
            data["reads"] = dict(self.reads) if self.renamed else list(self.reads)
        data["writes"] = self.answer if list(self.writes) == ["answer"] else dict(self.writes)
        given = {
            "question": self.question or None,
            "options": list(self.options) if isinstance(self.options, tuple) else self.options,
            "labels": None if self.labels is None else dict(self.labels),
            "pair": None if self.pair is None else list(self.pair),
            "edits": self.edits,
            "timeout_s": self.timeout_s,
            "default": None if self.default is None else dict(self.default),
        }
        data.update((key, value) for key, value in given.items() if value is not None)
        if self.detached:
            data["detached"] = True
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HumanAction:
        """Read a spec entry; ``for each`` is the MAS's, so it is skipped here."""
        settings = {key: value for key, value in data.items() if key not in ("name", "form", "reads", "for each")}
        return cls(data["name"], data["form"], data.get("reads", ()), **settings)


def plain_reads(where: str, reads: Any) -> dict[str, str]:
    """Return a human action's reads as ``{placeholder: value}``; a list reads each value under its own name."""
    if isinstance(reads, Mapping):
        pairs = list(reads.items())
    else:
        pairs = [(value, value) for value in texts(where, "reads", reads)]
    if not all(is_text(k) and is_text(v) for k, v in pairs):
        raise ValueError(f"{where}: reads must name values, got {reads!r}.")
    return dict(pairs)


def plain_writes(where: str, writes: Any) -> dict[str, str]:
    """Return a human action's writes as ``{role: value}``; a bare name or ``{name: json}`` is the answer."""
    named = {"answer": writes} if isinstance(writes, str) else writes
    if isinstance(named, Mapping) and len(named) == 1 and not set(named) <= set(ROLES):
        ((value, word),) = named.items()  # {name: type}, as an agent writes
        named = {"answer": value} if word == "json" else named
    if (
        not isinstance(named, Mapping)
        or "answer" not in named
        or not set(named) <= set(ROLES)
        or not all(is_text(v) for v in named.values())
    ):
        raise ValueError(
            f"{where}: writes is a value name, {{name: json}} or {{answer: name, value: name}}, got {writes!r}."
        )
    return dict(named)


def plain_mapping(where: str, label: str, value: Any) -> dict[Any, Any] | None:
    """Return a plain copy of a mapping setting, ``None`` when unset; anything else is refused."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: {label} is a mapping, got {value!r}.")
    return dict(value)


def yes_no_labels(labels: dict[Any, Any] | None) -> dict[Any, Any] | None:
    """Return an approve node's labels keyed ``yes`` and ``no``: YAML reads a bare yes and no as true and false."""
    return None if labels is None else {YES_NO.get(key, key): text for key, text in labels.items()}


def checked_edits(where: str, edits: Any) -> str | None:
    """Return an edit node's ``edits`` once it names a read, or ``None``."""
    if edits is not None and not is_text(edits):
        raise ValueError(f"{where}: edits names a read, got {edits!r}.")
    return cast("str | None", edits)


def checked_timeout(where: str, timeout: Any) -> float | None:
    """Return ``timeout_s`` once it is a number of seconds, or ``None``."""
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int | float)):
        raise ValueError(f"{where}: timeout_s is a number of seconds, got {timeout!r}.")
    return cast("float | None", timeout)


@dataclass(frozen=True)
class Question:
    """One question a visit of a human action puts to a person.

    Attributes:
        id: ``<run_id>/<node>[/<item key or index>]/<visit>``.
        run_id: The run that asks.
        node: The human action.
        form: The node's form.
        text: The question, filled from the reads.
        shown: Each read, by placeholder, rendered as an agent's input renders.
        options: The choices: an ask node's options, an approve node's yes and
            no labels, a judge's pair names and ``"tie"``; empty for an edit.
        item: The chain item's index, inside a chain.
        item_key: The chain item's key, for a chain with ``key``.
        visit: Which visit of the node asks, from 1.
        asked_at: When it was asked, ISO UTC.
        deadline: When its default is taken, ISO UTC; ``None`` without a timeout.
        default: The partial answer taken at the deadline.
        edits: An edit's placeholder in ``shown`` whose value the person changes.
        value_type: The type an edit's value is checked against, such as
            ``text``; ``None`` when not known.
    """

    id: str
    run_id: str
    node: str
    form: str
    text: str
    shown: dict[str, str]
    options: tuple[str, ...]
    item: int | None
    item_key: str | None
    visit: int
    asked_at: str
    deadline: str | None = None
    default: dict[str, Any] | None = None
    edits: str | None = None
    value_type: str | None = None

    def due(self, now: datetime) -> bool:
        """Whether the deadline has passed at ``now``."""
        return self.deadline is not None and now >= datetime.fromisoformat(self.deadline)

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        data = asdict(self)
        data["options"] = list(self.options)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Question:
        """Read the form :meth:`to_dict` writes."""
        return cls(**{**data, "options": tuple(data["options"]), "shown": dict(data["shown"])})


@dataclass(frozen=True)
class Pause:
    """A paused run: the questions it waits on, in the order they were asked."""

    run_id: str
    questions: tuple[Question, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        return {"run_id": self.run_id, "questions": [question.to_dict() for question in self.questions]}


class Asker(Protocol):
    """How a run reaches a person: the host's adapter, such as a card in a chat.

    ``ask`` is called once per question; it returns an answer at once (the
    fields a host gives: the form's, ``who`` and optionally ``comment`` and
    ``answered_at``) or ``None``, and the run pauses until the host answers
    through ``MAS.answer`` or ``MAS.resume``. A host may also define
    ``withdraw(question, reason)``, called for each open question of a run
    that is stopped.
    """

    def ask(self, question: Question) -> Mapping[str, Any] | None:
        """Put a question to a person; return the answer when it is known at once."""


class ReplayAsker:
    """Answers questions from recorded traces, asking nobody: what an optimizer uses.

    A question is answered by the human record of its node, item (key, else
    index) and visit, from the first trace that holds one; a question with no
    record, or whose recorded answer was a default taken on timeout, gets
    ``None``. A human record's ``raw_response`` is its answer's JSON. A
    replayed answer is given when it is asked, so its recorded ``answered_at``
    is left out.
    """

    def __init__(self, traces: Trace | Iterable[Trace]) -> None:
        """Index the human records of ``traces``, modules' inner records included."""
        self.answers: dict[tuple[str, str | int | None, int | None], dict[str, Any]] = {}
        for trace in [traces] if isinstance(traces, Trace) else traces:
            for _, record in trace.walk():
                if record.kind != "human_action" or record.error is not None:
                    continue
                where = record.item_key if record.item_key is not None else record.item
                self.answers.setdefault((record.agent, where, record.visit), json.loads(record.raw_response))

    def ask(self, question: Question) -> dict[str, Any] | None:
        """Return the recorded answer to a question, or ``None``."""
        where = question.item_key if question.item_key is not None else question.item
        recorded = self.answers.get((question.node, where, question.visit))
        if recorded is None or recorded.get("timed_out"):
            return None
        given = (*ANSWER_FIELDS[question.form], "who", "comment", EDITED)
        return {key: value for key, value in recorded.items() if key in given}


def field_problems(form: str, given: Mapping[str, Any], choices: Sequence[str] | None) -> Iterator[str]:
    """Say which of a form's fields that are present have the wrong type or an unknown choice."""
    for name in ("approved", "changed"):
        if name in given and not isinstance(given[name], bool):
            yield f"{name} is true or false, got {given[name]!r}"
    for name in ("option", "pick"):
        if name in given and choices is not None and form != "approve" and given[name] not in choices:
            yield f"{name} {given[name]!r} is not one of {list(choices)}"
        elif name in given and not isinstance(given[name], str):
            yield f"{name} is a text, got {given[name]!r}"


def answer_value(
    question: Question, given: Mapping[str, Any], *, now: datetime, timed_out: bool = False
) -> dict[str, Any]:
    """Return the full answer to a question from what a host gave, or from its default on timeout.

    The answer holds the form's fields (an approve's ``option`` filled with the
    label of the choice), then :data:`COMMON_FIELDS`. On timeout ``given`` is
    the default; the form's fields it leaves out are ``None`` and ``who`` is
    ``None``. An edit given no ``value`` is unchanged.

    Raises:
        ValueError: Listing every problem: a missing, extra or mistyped field,
            an unknown choice, a missing ``who``, an ``answered_at`` that is no
            ISO time with its zone or comes before the question was asked, an
            edited value not of the question's ``value_type``.
    """
    fields = ANSWER_FIELDS[question.form]
    allowed = (*fields, *GIVEN, *([EDITED] if question.form == "edit" else []))
    problems = [f"{key!r} is not a field of the {question.form} form" for key in given if key not in allowed]
    choices = None if question.form in ("approve", "edit") else question.options
    problems += field_problems(question.form, given, choices)
    who = given.get("who")
    if timed_out:
        who = None
    elif not isinstance(who, str) or not who:
        problems.append("who answered is missing")
    comment = given.get("comment")
    if comment is not None and not isinstance(comment, str):
        problems.append(f"comment is a text, got {comment!r}")
    answered = given.get("answered_at") if not timed_out else None
    if answered is not None:
        problems += time_problems(answered, question.asked_at)
    answer = {name: given.get(name) for name in fields}
    if question.form == "approve":
        problems += fill_approval(question, given, answer, timed_out=timed_out)
    elif question.form in ("ask", "judge") and not timed_out and given.get(fields[0]) is None:
        problems.append(f"{fields[0]} is missing")
    if question.form == "edit":
        problems += fill_edit(question, given, answer)
    if problems:
        raise ValueError(f"The answer to {question.id!r} does not fit: " + "; ".join(problems) + ".")
    return {
        **answer,
        "who": who,
        "asked_at": question.asked_at,
        "answered_at": answered or now.isoformat(),
        "comment": comment,
        "timed_out": timed_out,
    }


def fill_approval(
    question: Question, given: Mapping[str, Any], answer: dict[str, Any], *, timed_out: bool
) -> list[str]:
    """Fill an approve answer's choice and its label into ``answer``; return what does not fit."""
    approved = given.get("approved")
    if approved is None and given.get("option") in question.options:
        approved = given["option"] == question.options[0]  # the label alone says the choice
        answer["approved"] = approved
    if isinstance(approved, bool):
        label = question.options[0 if approved else 1]
        answer["option"] = label
        if given.get("option", label) != label:
            return [f"option {given['option']!r} is not the label of the choice, {label!r}"]
    elif not timed_out and approved is None:
        return ["approved is missing"]
    return []


def fill_edit(question: Question, given: Mapping[str, Any], answer: dict[str, Any]) -> list[str]:
    """Fill an edit answer's edited value and whether it changed into ``answer``; return what does not fit."""
    if EDITED not in given:
        if given.get("changed"):
            return ["changed is true, but the edited value is missing"]
        answer["changed"] = False
        return []
    edited = given[EDITED]
    answer[EDITED] = edited
    if "changed" not in given and question.edits is not None:
        before = question.shown.get(question.edits)
        answer["changed"] = before not in (leaf(edited), json.dumps(edited, ensure_ascii=False))
    wrong = None if question.value_type is None else check_value(edited, parse_type(question.value_type))
    return [] if wrong is None else [f"the edited value is not {question.value_type}: {wrong}"]


def time_problems(answered: Any, asked_at: str) -> list[str]:
    """Say what is wrong with a host's ``answered_at``: not an ISO time with its zone, or before ``asked_at``."""
    try:
        moment = datetime.fromisoformat(answered) if isinstance(answered, str) else None
    except ValueError:
        moment = None
    if moment is None or moment.tzinfo is None:
        return [f"answered_at is an ISO time with its zone, such as 2026-10-24T09:30:00+00:00, got {answered!r}"]
    if moment < datetime.fromisoformat(asked_at):
        return [f"answered_at {answered} comes before the question was asked, {asked_at}"]
    return []
