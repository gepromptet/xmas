"""The run log: LangGraph checkpoints plus every run's rows, in one SQLite database.

:class:`Store` is a LangGraph checkpoint saver on the standard library's
``sqlite3``. Its ``checkpoints`` and ``writes`` tables are the ones
``langgraph-checkpoint-sqlite``'s ``SqliteSaver`` creates, column for column,
filled through the same serializer calls, so either saver opens a file the
other wrote. Beside them, a run writes as it goes (:class:`RunLog`): its row
in ``xmas_runs`` when it starts and when it ends, one row in ``xmas_records``
the moment each visit's record is kept, one in ``xmas_events`` per event and
one in ``xmas_calls`` per model call (its completion id and logprobs
included); ``xmas_questions`` holds one row per
question a run put to a person. Records, events and calls are only added, each
in a transaction of its own, so a run that dies keeps what it wrote, and
:meth:`Store.trace` rebuilds its trace.

Where the store lives is configuration, never a spec key: :func:`open_store`
opens the file ``XMAS_RUNS`` names, else ``[store] runs`` in the configuration
file, else :func:`~xmas.config.runs_path`. ``:memory:`` opens one in memory,
which logs nothing and whose pauses resume in the same process only.

Reading a checkpoint back builds LangGraph's safe types and :data:`XMAS_TYPES`
only, xmas classes with their tuples restored; any other class in a checkpoint
comes back as its plain fields. Hand :attr:`Store.serde` to another saver to
read xmas values the same way.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import sqlite3
import threading
import uuid
import warnings
import weakref
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from xmas.errors import ErrorKind, classify
from xmas.health import Flag, Health, summarize
from xmas.human import Pause, Question
from xmas.media import Ref
from xmas.trace import AgentRecord, Event, Rejected, TokenUsage, ToolCall, Trace

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator, Mapping, Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.serde.base import SerializerProtocol

    from xmas.llms.meta import CallMeta

__all__ = [
    "XMAS_TYPES",
    "QuestionRow",
    "RunLog",
    "RunRow",
    "Store",
    "open_store",
]


XMAS_TYPES: tuple[type, ...] = (AgentRecord, TokenUsage, ToolCall, Rejected, Trace, Event, ErrorKind, Health, Flag, Ref)
"""The xmas classes a checkpoint may hold besides LangGraph's own safe types."""

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
CREATE TABLE IF NOT EXISTS xmas_runs (
    run_id TEXT PRIMARY KEY,
    system_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    paused TEXT NOT NULL DEFAULT '[]',
    totals TEXT,
    ended_at TEXT,
    query TEXT,
    output TEXT,
    error TEXT,
    error_kind TEXT,
    steps TEXT,
    result BLOB
);
CREATE TABLE IF NOT EXISTS xmas_questions (
    question_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node TEXT NOT NULL,
    form TEXT NOT NULL,
    shown TEXT NOT NULL DEFAULT '{}',
    asked_at TEXT NOT NULL,
    answered_at TEXT,
    answer TEXT,
    who TEXT,
    deadline TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS xmas_questions_run ON xmas_questions (run_id);
CREATE TABLE IF NOT EXISTS xmas_records (
    run_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    boxed INTEGER NOT NULL,
    kept_at TEXT NOT NULL,
    node TEXT NOT NULL,
    kind TEXT NOT NULL,
    item INTEGER,
    item_key TEXT,
    visit INTEGER,
    model TEXT,
    served_model TEXT,
    cost_usd REAL,
    latency_s REAL,
    tokens TEXT,
    finish_reason TEXT,
    error_kind TEXT,
    error TEXT,
    route TEXT,
    capped INTEGER NOT NULL,
    lost INTEGER NOT NULL,
    inputs TEXT NOT NULL,
    outputs TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    rendered_prompt TEXT NOT NULL,
    record BLOB NOT NULL,
    PRIMARY KEY (run_id, position)
);
CREATE TABLE IF NOT EXISTS xmas_events (
    run_id TEXT NOT NULL,
    at_s REAL NOT NULL,
    kind TEXT NOT NULL,
    node TEXT NOT NULL,
    item INTEGER,
    item_key TEXT,
    visit INTEGER,
    status TEXT,
    cost_usd REAL,
    question TEXT,
    wait_s REAL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS xmas_events_run ON xmas_events (run_id);
CREATE TABLE IF NOT EXISTS xmas_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node TEXT,
    ts TEXT NOT NULL,
    model TEXT,
    served_model TEXT,
    tokens TEXT,
    cost_usd REAL,
    latency_s REAL,
    finish_reason TEXT,
    error_kind TEXT,
    error TEXT,
    params TEXT,
    response_id TEXT,
    logprobs TEXT
);
CREATE INDEX IF NOT EXISTS xmas_calls_run ON xmas_calls (run_id);
"""
ADDED_COLUMNS = (("xmas_calls", "response_id", "TEXT"), ("xmas_calls", "logprobs", "TEXT"))
"""Columns added to a table after its first release, given to a file that predates them."""
LOGGED = ("xmas_records", "xmas_events", "xmas_calls")
RUN_COLUMNS = "run_id, system_id, status, created_at, updated_at, paused, totals, ended_at"
CALL_COLUMNS = (
    "call_id", "run_id", "node", "ts", "model", "served_model", "tokens", "cost_usd", "latency_s", "finish_reason",
    "error_kind", "error", "params", "response_id", "logprobs",
)  # fmt: skip
SECRET = re.compile(r"(^|_)(key|secret|token|password|credentials?)$|auth|header", re.IGNORECASE)
"""The call parameters a call row leaves out: keys, tokens, passwords and headers."""
CHECKPOINT_COLUMNS = "thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata"
FILTER_KEY = re.compile(r"^[a-zA-Z0-9_.-]+$")


class Serializer(JsonPlusSerializer):
    """LangGraph's serializer, giving xmas classes back their tuples, which msgpack stores as lists."""

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Decode as LangGraph does, then restore the tuple fields of xmas classes."""
        return tuples(super().loads_typed(data))


def tuples(value: Any) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = tuples(item)
    elif isinstance(value, list):
        value[:] = [tuples(item) for item in value]
    elif isinstance(value, XMAS_TYPES) and is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for field in fields(value):
            item = getattr(value, field.name)
            fixed = tuples(item)
            if isinstance(fixed, list) and str(field.type).startswith("tuple["):
                nested = str(field.type).startswith("tuple[tuple[")
                fixed = tuple(tuple(entry) if nested and isinstance(entry, list) else entry for entry in fixed)
            if fixed is not item:
                changes[field.name] = fixed
        return replace(value, **changes) if changes else value
    return value


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RunRow:
    """One run as the store knows it.

    Attributes:
        run_id: The run's id, also its checkpoint thread.
        system_id: The ``describe()`` id of the system that ran it.
        status: ``running``, ``paused``, or how the run ended (:class:`~xmas.errors.RunStatus`).
        created_at: When the run started, ISO UTC.
        updated_at: When the row was last saved, ISO UTC.
        paused: The ids of the questions the run waits on.
        totals: What the run had used when it was last saved, as
            :meth:`~xmas.budget.RunMeter.totals` gives it, so a resumed run's
            budget counts on; ``None`` until saved.
        ended_at: When the run ended, ISO UTC; ``None`` while it runs or
            waits, and for a run whose process ended first.
    """

    run_id: str
    system_id: str
    status: str
    created_at: str
    updated_at: str
    paused: tuple[str, ...] = ()
    totals: dict[str, Any] | None = None
    ended_at: str | None = None


@dataclass(frozen=True)
class QuestionRow:
    """One question a run put to a person, and its answer once given.

    Attributes:
        question_id: The question's id.
        run_id: The run that asked it.
        node: The human node that asked it.
        form: The node's form, such as ``approve``.
        shown: What the person was shown, by name.
        asked_at: When it was asked, ISO UTC.
        answered_at: When it was answered, ISO UTC; ``None`` while open.
        answer: The answer; ``None`` while open.
        who: Who answered; ``None`` while open or for a default taken on timeout.
        deadline: When the question's default is taken, ISO UTC; ``None`` without a timeout.
        payload: The whole question as :meth:`~xmas.human.Question.to_dict` gives it.
    """

    question_id: str
    run_id: str
    node: str
    form: str
    shown: dict[str, str]
    asked_at: str
    answered_at: str | None = None
    answer: dict[str, Any] | None = None
    who: str | None = None
    deadline: str | None = None
    payload: dict[str, Any] | None = None


class Store(BaseCheckpointSaver[str]):
    """A checkpoint saver and the run log beside it (runs, records, events, calls, questions), in one SQLite database.

    One connection serves every thread, each statement under one lock. Async
    methods raise, as ``SqliteSaver``'s do.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, serde: SerializerProtocol | None = None) -> None:
        """Open the database file at ``path``, made with its folder if missing; in memory for ``None``, ``:memory:``."""
        super().__init__(serde=serde or Serializer(allowed_msgpack_modules=XMAS_TYPES))
        self.path = Path(path).expanduser() if path is not None and str(path) != ":memory:" else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path or ":memory:", check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.executescript(SCHEMA)
        for table, column, kind in ADDED_COLUMNS:  # a file made before the column was added gets it
            present = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in present:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        """Run one statement in its own transaction and return its rows."""
        with self.lock, self.conn:
            return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        """Close the connection."""
        with self.lock:
            self.conn.close()

    def __repr__(self) -> str:
        """Name the file, or say it is in memory."""
        return f"Store({str(self.path)!r})" if self.path else "Store(in memory)"

    # LangGraph's saver interface

    def checkpoint_tuple(self, row: Sequence[Any]) -> CheckpointTuple:
        """Build LangGraph's checkpoint tuple from a row of ``checkpoints``, with its pending writes."""
        thread_id, checkpoint_ns, checkpoint_id, parent_id, type_, checkpoint, metadata = row
        writes = self.execute(
            "SELECT task_id, channel, type, value, task_path, idx FROM writes"
            " WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
            (thread_id, checkpoint_ns, checkpoint_id),
        )
        writes.sort(key=lambda w: writes_sort_key(w[4], w[0], w[5]))

        def config(checkpoint_id: str) -> RunnableConfig:
            ids = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}
            return {"configurable": ids}

        return CheckpointTuple(
            config(checkpoint_id),
            self.serde.loads_typed((type_, checkpoint)),
            cast("CheckpointMetadata", json.loads(metadata) if metadata is not None else {}),
            config(parent_id) if parent_id else None,
            [(task, channel, self.serde.loads_typed((t, value))) for task, channel, t, value, _, _ in writes],
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return the checkpoint ``config`` names, else its thread's latest."""
        return next(self.list({"configurable": {"checkpoint_ns": "", **config["configurable"]}}, limit=1), None)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002  the keyword LangGraph's BaseCheckpointSaver.list names
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield the matching checkpoints, newest first; ``filter`` matches metadata values."""
        wheres: list[str] = []
        params: list[Any] = []
        if config is not None:
            ids = config["configurable"]
            wheres.append("thread_id = ?")
            params.append(str(ids["thread_id"]))
            if ids.get("checkpoint_ns") is not None:
                wheres.append("checkpoint_ns = ?")
                params.append(ids["checkpoint_ns"])
            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = ?")
                params.append(checkpoint_id)
        for key, value in (filter or {}).items():
            if not FILTER_KEY.match(key):
                raise ValueError(f"A metadata filter key is letters, digits, '_', '.' and '-', not {key!r}")
            if value is None:
                wheres.append(f"json_extract(CAST(metadata AS TEXT), '$.{key}') IS NULL")
                continue
            wheres.append(f"json_extract(CAST(metadata AS TEXT), '$.{key}') = ?")
            plain = isinstance(value, str | int | float)
            params.append(value if plain else json.dumps(value, separators=(",", ":")))
        if before is not None:
            wheres.append("checkpoint_id < ?")
            params.append(get_checkpoint_id(before))
        sql = f"SELECT {CHECKPOINT_COLUMNS} FROM checkpoints"
        sql += (" WHERE " + " AND ".join(wheres) if wheres else "") + " ORDER BY checkpoint_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self.execute(sql, params):
            yield self.checkpoint_tuple(row)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,  # noqa: ARG002
    ) -> RunnableConfig:
        """Save ``checkpoint`` as the child of the one ``config`` names."""
        ids = config["configurable"]
        type_, blob = self.serde.dumps_typed(checkpoint)
        meta = json.dumps(get_checkpoint_metadata(config, metadata), ensure_ascii=False).encode("utf-8", "ignore")
        self.execute(
            f"INSERT OR REPLACE INTO checkpoints ({CHECKPOINT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(ids["thread_id"]),
                ids["checkpoint_ns"],
                checkpoint["id"],
                ids.get("checkpoint_id"),
                type_,
                blob,
                meta,
            ),
        )
        return {
            "configurable": {
                "thread_id": ids["thread_id"],
                "checkpoint_ns": ids["checkpoint_ns"],
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        """Save a task's pending writes against the checkpoint ``config`` names."""
        ids = config["configurable"]
        verb = "REPLACE" if all(channel in WRITES_IDX_MAP for channel, _ in writes) else "IGNORE"
        key = (str(ids["thread_id"]), str(ids["checkpoint_ns"]), str(ids["checkpoint_id"]), task_id, task_path)
        rows = [
            (*key, WRITES_IDX_MAP.get(channel, idx), channel, *self.serde.dumps_typed(value))
            for idx, (channel, value) in enumerate(writes)
        ]
        with self.lock, self.conn:
            self.conn.executemany(
                f"INSERT OR {verb} INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx,"
                " channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread's checkpoints and writes."""
        self.delete_rows("thread_id = ?", (str(thread_id),))

    def get_next_version(self, current: str | int | None, channel: None) -> str:  # noqa: ARG002
        """Return the next channel version, in ``SqliteSaver``'s format: a padded count, a dot, a random tail."""
        count = 0 if current is None else current if isinstance(current, int) else int(current.split(".")[0])
        return f"{count + 1:032}.{random.random():016}"

    # Runs and questions

    def save_run(
        self,
        run_id: str,
        system_id: str,
        status: str,
        paused: Sequence[str] = (),
        totals: Mapping[str, Any] | None = None,
    ) -> RunRow:
        """Create or update a run's row; ``created_at`` stays as first saved, ``totals`` as last given."""
        now = now_iso()
        self.execute(
            "INSERT INTO xmas_runs (run_id, system_id, status, created_at, updated_at, paused, totals)"
            " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (run_id) DO UPDATE SET"
            " system_id = excluded.system_id, status = excluded.status, updated_at = excluded.updated_at,"
            " paused = excluded.paused, totals = COALESCE(excluded.totals, totals)",
            (run_id, system_id, status, now, now, json.dumps(list(paused)), dump_json(totals)),
        )
        return cast("RunRow", self.run(run_id))

    def begin(self, run_id: str, system_id: str, query: str) -> RunRow:
        """Save a starting run's row, ``running``; a ``run_id`` saved already is refused with ``ValueError``."""
        now = now_iso()
        try:
            self.execute(
                "INSERT INTO xmas_runs (run_id, system_id, status, created_at, updated_at, query)"
                " VALUES (?, ?, 'running', ?, ?, ?)",
                (run_id, system_id, now, now, query),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Run id {run_id!r} is taken in {self!r}.") from None
        return cast("RunRow", self.run(run_id))

    def run(self, run_id: str) -> RunRow | None:
        """Return a run's row, or ``None`` for an unknown id."""
        rows = self.execute(f"SELECT {RUN_COLUMNS} FROM xmas_runs WHERE run_id = ?", (run_id,))
        return run_row(rows[0]) if rows else None

    def runs(
        self, status: str | None = None, system_id: str | None = None, limit: int | None = None
    ) -> tuple[RunRow, ...]:
        """Return the runs' rows, oldest first: with ``status`` and ``system_id`` when given; the last ``limit``."""
        wheres = [f"{name} = ?" for name, value in (("status", status), ("system_id", system_id)) if value is not None]
        params: list[Any] = [value for value in (status, system_id) if value is not None]
        sql = f"SELECT {RUN_COLUMNS} FROM xmas_runs WHERE {' AND '.join(wheres) or '1'}"
        sql += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return tuple(run_row(row) for row in reversed(self.execute(sql, params)))

    def trace(self, run_id: str) -> Trace:
        """Rebuild a run's trace from its rows, its health summarized again from its records.

        A run that ended or paused has the trace it returned. A run without an
        end, whose process ended first, has every record it kept, in the
        order kept, and an ``error`` saying so.

        Raises:
            KeyError: For an unknown ``run_id``.
        """
        rows = self.execute(
            "SELECT system_id, status, paused, query, output, error, error_kind, steps, result"
            " FROM xmas_runs WHERE run_id = ?",
            (run_id,),
        )
        if not rows:
            raise KeyError(f"no run {run_id!r} in {self!r}")
        system_id, status, paused, query, output, error, error_kind, steps, result = rows[0]
        kept = self.own_records(run_id)
        if status == "running" or steps is None:
            order, error = list(kept), "the run has no end in the store: it still runs, or its process ended first"
        else:
            order = json.loads(steps)
        records = tuple(replace(kept[position], step=index) for index, position in enumerate(order))
        ending = self.load(result) if result is not None else {"outputs": {}, "graph_id": system_id}
        waits = set(json.loads(paused))
        questions = tuple(
            Question.from_dict(row.payload)
            for row in self.questions(run_id)
            if row.question_id in waits and row.payload is not None
        )
        return Trace(
            query=query or "",
            output=output or "",
            records=records,
            error=error,
            error_kind=None if error_kind is None else ErrorKind(error_kind),
            health=summarize(records),
            outputs=ending["outputs"],
            run_id=run_id,
            graph_id=ending["graph_id"],
            pause=Pause(run_id, questions) if status == "paused" else None,
        )

    def events(self, run_id: str) -> tuple[Event, ...]:
        """Return a run's events in the order they happened."""
        rows = self.execute(
            "SELECT kind, node, run_id, at_s, item, item_key, visit, status, cost_usd, question, wait_s, reason"
            " FROM xmas_events WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        )
        return tuple(Event(*row) for row in rows)

    def calls(self, run_id: str) -> tuple[dict[str, Any], ...]:
        """Return a run's model calls in the order they ended: each row by column, with its JSON columns read.

        ``logprobs`` is a list of ``[token, logprob]`` pairs when the call's
        request asked for logprobs, else ``None``.
        """
        rows = self.execute(
            f"SELECT {', '.join(CALL_COLUMNS)} FROM xmas_calls WHERE run_id = ? ORDER BY rowid", (run_id,)
        )
        calls = [dict(zip(CALL_COLUMNS, row, strict=True)) for row in rows]
        for call in calls:
            for column in ("tokens", "params", "logprobs"):
                call[column] = loads(call[column])
        return tuple(calls)

    def own_records(self, run_id: str, before: int | None = None) -> dict[int, AgentRecord]:
        """Return the run's own records (not a box's inner ones) by position, in the order kept; those ``before``."""
        rows = self.execute(
            "SELECT position, record FROM xmas_records WHERE run_id = ? AND boxed = 0 AND position < ?"
            " ORDER BY position",
            (run_id, before if before is not None else 2**62),
        )
        return {position: self.load(blob) for position, blob in rows}

    def load(self, blob: bytes) -> Any:
        """Read back what :meth:`dump` wrote."""
        return self.serde.loads_typed(("msgpack", blob))

    def dump(self, value: Any) -> bytes:
        """Return the msgpack of a record, or of a trace's ending: dataclasses and plain values."""
        return self.serde.dumps_typed(value)[1]

    def save_question(  # noqa: PLR0913, PLR0917  the public signature: callers pass the question's columns positionally
        self,
        question_id: str,
        run_id: str,
        node: str,
        form: str,
        shown: Mapping[str, str],
        asked_at: str | None = None,
        *,
        deadline: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> QuestionRow:
        """Record a question as asked, ``asked_at`` now unless given; an id already saved is refused."""
        try:
            self.execute(
                "INSERT INTO xmas_questions (question_id, run_id, node, form, shown, asked_at, deadline, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question_id,
                    run_id,
                    node,
                    form,
                    json.dumps(dict(shown)),
                    asked_at or now_iso(),
                    deadline,
                    dump_json(payload),
                ),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"question {question_id!r} is already saved") from None
        return cast("QuestionRow", self.question(question_id))

    def answer(
        self, question_id: str, answer: Mapping[str, Any], who: str | None = None, answered_at: str | None = None
    ) -> QuestionRow:
        """Record a question's answer, ``answered_at`` now unless given; raises ``KeyError`` for an unknown id."""
        rows = self.execute(
            "UPDATE xmas_questions SET answer = ?, who = ?, answered_at = ? WHERE question_id = ? RETURNING *",
            (json.dumps(dict(answer)), who, answered_at or now_iso(), question_id),
        )
        if not rows:
            raise KeyError(f"no question {question_id!r}")
        return question_row(rows[0])

    def question(self, question_id: str) -> QuestionRow | None:
        """Return a question's row, or ``None`` for an unknown id."""
        return next(iter(self.select_questions("question_id = ?", question_id)), None)

    def questions(self, run_id: str) -> tuple[QuestionRow, ...]:
        """Return a run's questions in the order asked."""
        return self.select_questions("run_id = ?", run_id)

    def select_questions(self, where: str, *params: Any) -> tuple[QuestionRow, ...]:
        """Return the questions ``where`` matches, in the order asked."""
        rows = self.execute(f"SELECT * FROM xmas_questions WHERE {where} ORDER BY asked_at, rowid", params)
        return tuple(question_row(row) for row in rows)

    def due(self, now: datetime | None = None) -> tuple[RunRow, ...]:
        """Return the paused runs with an open question whose deadline has passed at ``now``, oldest first.

        A ``now`` without a zone is local time, as ``datetime.now()`` gives it.
        """
        moment = (now or datetime.now(UTC)).astimezone()
        rows = self.execute(
            "SELECT DISTINCT run_id, deadline FROM xmas_questions WHERE answer IS NULL AND deadline IS NOT NULL"
        )
        late = {run_id for run_id, deadline in rows if datetime.fromisoformat(deadline) <= moment}
        return tuple(run for run in self.runs("paused") if run.run_id in late)

    def prune_runs(self, older_than_days: float) -> tuple[str, ...]:
        """Delete the runs that ended (not running or paused) over ``older_than_days`` ago; return their ids."""
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        ended = [
            run.run_id
            for run in self.runs()
            if run.status not in ("running", "paused") and datetime.fromisoformat(run.updated_at) < cutoff
        ]
        for run_id in ended:
            self.delete(run_id)
        return tuple(ended)

    def delete(self, run_id: str) -> None:
        """Delete a run's rows (run, questions, records, events, calls) and threads (``run_id``, ``run_id/...``)."""
        prefix = run_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"
        self.delete_rows("thread_id = ? OR thread_id LIKE ? ESCAPE '\\'", (run_id, prefix), run_id)

    def delete_rows(self, threads: str, params: Sequence[Any], run_id: str | None = None) -> None:
        """Delete the checkpoints and writes of the threads ``threads`` matches, and the rows of ``run_id``."""
        with self.lock, self.conn:
            for table in ("xmas_runs", "xmas_questions", *LOGGED) if run_id is not None else ():
                self.conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            for table in ("checkpoints", "writes"):
                self.conn.execute(f"DELETE FROM {table} WHERE {threads}", params)


class RunLog:
    """One run's writes to a file store as the run goes: each record kept, each event, each model call, its end.

    Each write is one insert in a transaction of its own, so a run that dies
    keeps every row written before. A write that fails is warned about once,
    and the run goes on without its log: a full disk does not end a paid run.
    """

    def __init__(self, store: Store, run_id: str) -> None:
        """Log ``run_id``, whose row the store holds, counting positions on from its last record."""
        self.store = store
        self.run_id = run_id
        (last,) = store.execute("SELECT MAX(position) FROM xmas_records WHERE run_id = ?", (run_id,))[0]
        self.first = 0 if last is None else last + 1
        self.positions = itertools.count(self.first)
        self.kept: dict[Visit, list[tuple[int, AgentRecord]]] = {}
        self.failed = False

    def write(self, sql: str, params: Sequence[Any]) -> None:
        """Run one statement on the log; the first failure warns once and turns the log off."""
        if self.failed:
            return
        try:
            self.store.execute(sql, params)
        except sqlite3.Error as error:
            self.failed = True
            warnings.warn(
                f"The run log of {self.run_id!r} in {self.store!r} failed ({error}); the run goes on unlogged.",
                RuntimeWarning,
                stacklevel=3,
            )

    def record(self, record: AgentRecord, *, boxed: bool = False) -> None:
        """Write a visit's record the moment it is kept; ``boxed`` for a record of a box's inner system."""
        position = next(self.positions)
        if not boxed:
            self.kept.setdefault(visit(record), []).append((position, record))
        tokens = None if record.tokens is None else json.dumps(record.tokens.to_dict())
        self.write(
            f"INSERT INTO xmas_records VALUES ({', '.join('?' * 25)})",
            (
                self.run_id,
                position,
                boxed,
                now_iso(),
                record.agent,
                record.kind,
                record.item,
                record.item_key,
                record.visit,
                record.model,
                record.served_model,
                record.cost_usd,
                record.latency_s,
                tokens,
                record.finish_reason,
                None if record.error_kind is None else record.error_kind.value,
                record.error,
                None if record.route is None else json.dumps(list(record.route)),
                record.capped,
                record.lost,
                json.dumps(record.inputs, ensure_ascii=False),
                json.dumps(record.outputs, ensure_ascii=False),
                record.raw_response,
                record.rendered_prompt,
                self.store.dump(record),
            ),
        )

    def event(self, event: Event) -> None:
        """Write one event of the run."""
        self.write(
            f"INSERT INTO xmas_events VALUES ({', '.join('?' * 12)})",
            (
                self.run_id,
                event.at_s,
                event.kind,
                event.node,
                event.item,
                event.item_key,
                event.visit,
                event.status,
                event.cost_usd,
                event.question,
                event.wait_s,
                event.reason,
            ),
        )

    def call(  # noqa: PLR0913, PLR0917  the public signature: the budget passes one call's facts positionally
        self,
        node: str | None,
        llm: object,
        ts: str,
        latency_s: float,
        meta: CallMeta | None,
        error: BaseException | None = None,
    ) -> None:
        """Write one model call as it ended: who made it, on which model, what it used, how it ended.

        Its parameters are the model's ``call_params()``, when it has them,
        without any key, token, password or header. The completion's id and
        its logprobs are the meta's.
        """
        params = getattr(llm, "call_params", None)
        sent = {key: value for key, value in params().items() if not SECRET.search(key)} if callable(params) else None
        kind = meta.error_kind if meta is not None else None
        if kind is None and error is not None:
            kind = classify(error)
        self.write(
            f"INSERT INTO xmas_calls ({', '.join(CALL_COLUMNS)}) VALUES ({', '.join('?' * len(CALL_COLUMNS))})",
            (
                uuid.uuid4().hex,
                self.run_id,
                node,
                ts,
                getattr(llm, "model", None) or getattr(llm, "model_id", None) or type(llm).__name__,
                None if meta is None else meta.served_model,
                None if meta is None or meta.tokens is None else json.dumps(meta.tokens.to_dict()),
                None if meta is None else meta.cost_usd,
                latency_s if meta is None else meta.latency_s,
                None if meta is None else meta.finish_reason,
                None if kind is None else kind.value,
                None if error is None else f"{type(error).__name__}: {error}",
                None if sent is None else json.dumps(sent, default=str),
                None if meta is None else meta.response_id,
                None if meta is None or meta.logprobs is None else json.dumps([list(pair) for pair in meta.logprobs]),
            ),
        )

    def end(self, trace: Trace, totals: Mapping[str, Any]) -> None:
        """Write how the run ended or paused: its status, totals and outcome, and its records' order in ``trace``."""
        status = trace.status.value
        paused = [] if trace.pause is None else [question.id for question in trace.pause.questions]
        self.write(
            "UPDATE xmas_runs SET status = ?, updated_at = ?, paused = ?, totals = ?, ended_at = ?, query = ?,"
            " output = ?, error = ?, error_kind = ?, steps = ?, result = ? WHERE run_id = ?",
            (
                status,
                now_iso(),
                json.dumps(paused),
                dump_json(totals),
                None if status == "paused" else now_iso(),
                trace.query,
                trace.output,
                trace.error,
                None if trace.error_kind is None else trace.error_kind.value,
                json.dumps(self.steps(trace.records)),
                self.store.dump({"outputs": trace.outputs, "graph_id": trace.graph_id}),
                self.run_id,
            ),
        )

    def steps(self, records: Sequence[AgentRecord]) -> list[int]:
        """Return the position of each of ``records`` in the log: kept by this log, else before it (before a pause)."""
        earlier: dict[Visit, list[tuple[int, AgentRecord]]] | None = None
        steps = []
        for record in records:
            position = take(self.kept.get(visit(record), []), record)
            if position is None:
                if earlier is None:
                    earlier = {}
                    for kept, old in self.store.own_records(self.run_id, before=self.first).items():
                        earlier.setdefault(visit(old), []).append((kept, old))
                position = take(earlier.get(visit(record), []), record)
            if position is None:  # pragma: no cover - every record of a trace was kept
                self.record(record)
                position = cast("int", take(self.kept[visit(record)], record))
            steps.append(position)
        return steps


Visit = tuple[str, int | None, int | None]


def visit(record: AgentRecord) -> Visit:
    """Return what tells a record from most others of its run: its node, item and visit."""
    return record.agent, record.item, record.visit


def take(candidates: list[tuple[int, AgentRecord]], record: AgentRecord) -> int | None:
    """Remove and return the position of the first candidate equal to ``record``, its step aside."""
    for index, (position, kept) in enumerate(candidates):
        if replace(record, step=kept.step) == kept:
            del candidates[index]
            return position
    return None


def dump_json(value: Mapping[str, Any] | None) -> str | None:
    return None if value is None else json.dumps(dict(value))


def loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def run_row(row: Sequence[Any]) -> RunRow:
    run_id, system_id, status, created_at, updated_at, paused, totals, ended_at = row
    return RunRow(run_id, system_id, status, created_at, updated_at, tuple(json.loads(paused)), loads(totals), ended_at)


def question_row(row: Sequence[Any]) -> QuestionRow:
    question_id, run_id, node, form, shown, asked_at, answered_at, answer, who, deadline, payload = row
    return QuestionRow(
        question_id,
        run_id,
        node,
        form,
        json.loads(shown),
        asked_at,
        answered_at,
        json.loads(answer) if answer is not None else None,
        who,
        deadline,
        json.loads(payload) if payload is not None else None,
    )


OPEN: weakref.WeakValueDictionary[Path, Store] = weakref.WeakValueDictionary()
open_lock = threading.Lock()


def open_store(location: str | os.PathLike[str] | None = None) -> Store:
    """Open the store at ``location``, else where the configuration puts it, else at :func:`~xmas.config.runs_path`.

    The configuration is ``XMAS_RUNS``, then ``[store] runs`` in the
    configuration file. ``:memory:`` opens a store in memory, which logs no
    run. A file already open in this process is shared, not opened twice.
    """
    from xmas.config import Config, runs_path  # noqa: PLC0415  here, not at the top: xmas.config imports the models

    if location is None:
        location = Config.load().run_store or runs_path()
    if str(location) == ":memory:":
        return Store()
    path = Path(location).expanduser().resolve()
    with open_lock:
        store = OPEN.get(path)
        if store is None:
            store = OPEN[path] = Store(path)
        return store
