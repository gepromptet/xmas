"""People in the graph: a human action pauses its branch, the run returns paused, and resumes from its store."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.serde.event_hooks import SerdeEvent, register_serde_event_listener

from tests.conftest import MockLLM, SeeingLLM, req
from xmas import MAS, Agent, Budget, Code, MASModule
from xmas.budget import RUNNING
from xmas.edges import Edge, EdgeKind
from xmas.errors import ErrorKind, Stopped
from xmas.graph import END
from xmas.human import HumanAction, Question, ReplayAsker
from xmas.llms.meta import CallMeta
from xmas.optimization.graph_updater import GraphUpdater
from xmas.optimization.synthesis import MechanicalSynthesizer
from xmas.optimization.topology import TopologyBudget
from xmas.store import Store
from xmas.tools import tool
from xmas.trace import Event, Trace

ROOT = Path(__file__).resolve().parents[1]
TOM = {"who": "tom"}


# ---------------------------------------------------------------- building systems


def people(
    agents: list[Agent],
    edges: Sequence[tuple[str, str] | Edge] = (),
    human_actions: Sequence[HumanAction] = (),
    **kwargs: Any,
) -> MAS:
    """A system of agents and people."""
    return MAS(agents, edges, human_actions=human_actions, **kwargs)


def run(mas: MAS, values: Mapping[str, Any], run_id: str, events: list[Event] | None = None) -> Trace:
    return mas.run_state(values, run_id=run_id, on_event=None if events is None else events.append)


def resume(mas: MAS, run_id: str, answers: Mapping[str, Any] | None = None, events: list[Event] | None = None) -> Trace:
    return mas.resume(run_id, answers, on_event=None if events is None else events.append)


class Asker:
    """Records every question; answers from ``reply`` (``None`` pauses)."""

    def __init__(self, reply: Callable[[Question], Mapping[str, Any] | None] = lambda q: None) -> None:
        """Answer from ``reply``."""
        self.reply, self.asked = reply, cast("list[Question]", [])

    def ask(self, question: Question) -> Mapping[str, Any] | None:
        """Record the question and reply."""
        self.asked.append(question)
        return self.reply(question)


def gate(**settings: Any) -> HumanAction:
    """An approve gate on the draft."""
    labels = {"yes": "Ship {draft}", "no": "Redraw"}
    return HumanAction(
        "approve", "approve", ["draft"], "approval", question="Ship this? {draft}", labels=labels, **settings
    )


def gated(llm: MockLLM | None = None, *, human: HumanAction | None = None, **kwargs: Any) -> MAS:
    """The writer drafts, ``approve`` asks; ship when approved, else hold."""
    agents = [
        Agent("writer", "Write {query}", "draft"),
        Agent("ship", "Ship {draft}", "shipped", ["draft"]),
        Agent("hold", "Hold {draft}", "held", ["draft"]),
    ]
    edges: list[tuple[str, str] | Edge] = [
        ("writer", "approve"),
        Edge("approve", "ship", EdgeKind.WHEN, when="approval.approved"),
        Edge("approve", "hold", EdgeKind.ELSE),
    ]
    model = llm or MockLLM(lambda p: "a draft" if p.startswith("Write") else p.split()[0].lower())
    outputs = {"approval": "json", "shipped": "text", "held": "text"}
    return people(agents, edges, [human or gate()], llm=model, outputs=outputs, **kwargs)


def plan(*keys: str) -> dict[str, Any]:
    return {"plan": {"slides": [{"id": key} for key in keys]}}


def chained(asker: Any = None, *, confirm: bool = False, side: bool = True, **kwargs: Any) -> MAS:
    """A chain per slide: drafter -> fix (an approve) [-> confirm (an approve)], beside a noter."""
    calls: list[str] = []

    def write(prompt: str) -> str:
        calls.append(prompt)
        return "notes" if prompt.startswith("Note") else f"draft {json.loads(prompt[6:])['id']}"

    each = {"drafter": {"item": "page", "in": "plan.slides", "key": "id"}, "fix": "page in plan.slides"}
    human_actions = [HumanAction("fix", "approve", ["draft"], "fixed", question="Fix {draft}?")]
    edges: list[tuple[str, str] | Edge] = [("drafter", "fix")]
    if confirm:
        human_actions.append(HumanAction("confirm", "approve", ["draft"], "confirmed", question="Sure about {draft}?"))
        each["confirm"] = "page in plan.slides"
        edges.append(("fix", "confirm"))
    agents = [Agent("drafter", "Draft {page}", "draft", ["page"])]
    outputs = {"fixed": "list of json", **({"confirmed": "list of json"} if confirm else {})}
    if side:
        agents.append(Agent("noter", "Note {plan}", "notes", ["plan"]))
        outputs["notes"] = "text"
    llm = MockLLM(write)
    mas = people(
        agents,
        edges,
        human_actions,
        asker=asker,
        each=each,
        outputs=outputs,
        inputs={"plan": "json"},
        llm=llm,
        **kwargs,
    )
    mas.calls = calls  # type: ignore[attr-defined]
    return mas


def open_ids(trace: Trace) -> list[str]:
    return [question.id for question in cast("Any", trace.pause).questions]


# ------------------------------------------------------------------------- tests


def test_approve_pauses_and_resumes(tmp_path: Path) -> None:
    mas = gated(store=Store(tmp_path / "runs.db"))
    events: list[Event] = []
    paused = run(mas, {"query": "cats"}, "r1", events)
    assert paused.status == "paused"
    assert paused.error is None
    assert paused.outputs == {}
    assert [record.agent for record in paused.records] == ["writer"]
    (question,) = cast("Any", paused.pause).questions
    assert (question.id, question.run_id, question.node, question.form) == ("r1/approve/1", "r1", "approve", "approve")
    assert (question.text, question.options, question.shown) == (
        "Ship this? a draft",
        ("Ship a draft", "Redraw"),
        {"draft": "a draft"},
    )
    store = mas.runs()
    row = cast("Any", store.run("r1"))
    assert (row.status, row.paused, row.totals["calls"]) == ("paused", ("r1/approve/1",), 1)
    with pytest.raises(ValueError, match="is taken"):
        run(mas, {"query": "cats"}, "r1")

    time.sleep(0.02)
    done = resume(mas, "r1", {"r1/approve/1": {"approved": True, **TOM}}, events)
    assert done.status == "ok"
    assert done.pause is None
    approval = done.outputs["approval"]
    assert {k: approval[k] for k in ("approved", "option", "who", "timed_out", "comment")} == {
        "approved": True, "option": "Ship a draft", "who": "tom", "timed_out": False, "comment": None,
    }  # fmt: skip
    assert approval["asked_at"] == question.asked_at
    assert done.outputs["shipped"] == "ship"
    assert done.outputs["held"] is None
    assert [record.agent for record in done.records] == ["writer", "approve", "ship"]
    (human,) = done["approve"]
    assert (human.kind, human.cost_usd, human.route, human.visit) == ("human_action", 0.0, ("ship",), 1)
    assert human.inputs == {"draft": "a draft"}
    assert human.rendered_prompt == "Ship this? a draft"
    assert cast("float", human.wait_s) >= 0.02
    assert cast("float", human.latency_s) < 0.02
    assert json.loads(human.raw_response) == approval
    assert cast("Any", store.run("r1")).status == "ok"
    assert store.question("r1/approve/1").answer == approval  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="is ok, not paused"):
        resume(mas, "r1", {})
    with pytest.raises(ValueError, match="has no run"):
        resume(mas, "nope")

    # The answer routes: rejected goes to hold.
    held = gated()
    run(held, {"query": "cats"}, "r2")
    assert resume(held, "r2", {"r2/approve/1": {"approved": False, **TOM}}).outputs["held"] == "hold"


def test_an_optimizer_keeps_people_where_they_are() -> None:
    mas = gated(asker=Asker(), store=Store())
    updater = GraphUpdater(MechanicalSynthesizer(), TopologyBudget(min_agents=2, max_agents=5, max_edges=8), seed=0)
    fixed = updater.fixed_of(mas)
    assert fixed is not None
    assert "approve" in fixed.nodes
    assert "approve" in fixed.code
    rebuilt = updater.rebuild(mas, updater.names_of(mas), updater.edges_of(mas))
    assert rebuilt.human_actions == mas.human_actions
    assert (rebuilt.asker, rebuilt.store) == (mas.asker, mas.store)
    assert set(rebuilt.agents) == {"writer", "ship", "hold"}
    assert rebuilt.run_state({"query": "cats"}, run_id="o1").status == "paused"


def test_a_pause_lets_other_branches_end() -> None:
    llm = MockLLM(lambda p: p.split()[0])
    agents = [Agent(name, f"{name} {{{read}}}", f"{name}_out", [read]) for name, read in (
        ("n1", "query"), ("n2", "n1_out"), ("n3", "n2_out"))]  # fmt: skip
    agents.append(Agent("join", "join {approval} {n3_out}", "joined", ["approval", "n3_out"]))
    ask = HumanAction("approve", "approve", ["query"], "approval", question="Go on with {query}?")
    edges = [("n1", "n2"), ("n2", "n3"), ("approve", "join"), ("n3", "join")]
    mas = people(agents, edges, [ask], llm=llm)
    paused = run(mas, {"query": "q"}, "u1")
    assert [record.agent for record in paused.records] == ["n1", "n2", "n3"]  # the branch ended before the pause
    assert open_ids(paused) == ["u1/approve/1"]
    done = resume(mas, "u1", {"u1/approve/1": {"approved": True, **TOM}})
    assert [record.agent for record in done.records] == ["n1", "n2", "n3", "approve", "join"]

    # Two human actions on two branches pause together; answering one keeps the other open under its id.
    check = HumanAction("check", "approve", ["n2_out"], "checked", question="Is {n2_out} right?")
    agents[2] = Agent("n3", "n3 {n2_out} {checked}", "n3_out", ["n2_out", "checked"])
    edges = [("n1", "n2"), ("n2", "check"), ("check", "n3"), ("approve", "join"), ("n3", "join")]
    both = people(agents, edges, [ask, check], llm=llm)
    paused = run(both, {"query": "q"}, "u2")
    assert sorted(open_ids(paused)) == ["u2/approve/1", "u2/check/1"]
    still = resume(both, "u2", {"u2/check/1": {"approved": True, **TOM}})
    assert still.status == "paused"
    assert open_ids(still) == ["u2/approve/1"]
    done = resume(both, "u2", {"u2/approve/1": {"approved": True, **TOM}})
    assert done.status == "ok"
    assert [r.agent for r in done.records].count("join") == 1
    assert done["join"][0].inputs["n3_out"] == "n3"


def test_pauses_inside_chain_items() -> None:
    # Items a and c are answered at once; b pauses; the side branch and the other items end.
    asker = Asker(lambda q: None if q.item_key == "b" else {"approved": True, **TOM})
    mas = chained(asker)
    paused = run(mas, plan("a", "b", "c"), "c1")
    assert open_ids(paused) == ["c1/fix/b/1"]
    question = cast("Any", paused.pause).questions[0]
    assert (question.item, question.item_key, question.text) == (1, "b", "Fix draft b?")
    assert sorted((r.agent, r.item_key) for r in paused.records if r.agent != "noter") == [
        ("drafter", "a"), ("drafter", "c"), ("fix", "a"), ("fix", "c"),
    ]  # fmt: skip
    assert "noter" in paused.agents
    done = resume(mas, "c1", {"c1/fix/b/1": {"approved": False, **TOM}})
    assert [fixed["approved"] for fixed in done.outputs["fixed"]] == [True, False, True]
    assert sum(call.startswith("Draft") for call in mas.calls) == 3  # type: ignore[attr-defined]
    assert [len(asker.asked), len(done["drafter"]), len(done["fix"])] == [3, 3, 3]

    # Three paused items, one answered: the other two stay paused under their ids.
    mas = chained()
    paused = run(mas, plan("a", "b", "c"), "c2")
    assert sorted(open_ids(paused)) == ["c2/fix/a/1", "c2/fix/b/1", "c2/fix/c/1"]  # in the order asked
    still = resume(mas, "c2", {"c2/fix/a/1": {"approved": True, **TOM}})
    assert sorted(open_ids(still)) == ["c2/fix/b/1", "c2/fix/c/1"]
    done = resume(mas, "c2", {"c2/fix/b/1": {"approved": True, **TOM}, "c2/fix/c/1": {"approved": False, **TOM}})
    assert [fixed["approved"] for fixed in done.outputs["fixed"]] == [True, True, False]
    assert sum(call.startswith("Draft") for call in mas.calls) == 3  # type: ignore[attr-defined]

    # A second human in an item pauses it again.
    mas = chained(confirm=True, side=False)
    paused = run(mas, plan("a"), "c3")
    assert open_ids(paused) == ["c3/fix/a/1"]
    again = resume(mas, "c3", {"c3/fix/a/1": {"approved": True, **TOM}})
    assert open_ids(again) == ["c3/confirm/a/1"]
    done = resume(mas, "c3", {"c3/confirm/a/1": {"approved": True, **TOM}})
    assert done.outputs["confirmed"][0]["approved"] is True
    assert [(r.agent, r.item_key) for r in done.records] == [("drafter", "a"), ("fix", "a"), ("confirm", "a")]


def boxed_inner(*, confirm: bool = False) -> MAS:
    """The module's system: writer -> approve [-> confirm]."""
    agents = [Agent("writer", "Write {topic}", "draft", ["topic"])]
    human_actions, edges = [gate()], [("writer", "approve")]
    if confirm:
        human_actions.append(HumanAction("confirm", "approve", ["approval"], "confirmed", question="Sure?"))
        edges.append(("approve", "confirm"))
    llm = MockLLM(lambda p: "a draft")
    return people(agents, edges, human_actions, inputs={"topic": "text"}, llm=llm)


def test_pauses_inside_boxes() -> None:
    box = MASModule("review", boxed_inner(), reads={"topic": "query"}, writes={"approval": "approval"})
    side = Agent("side", "Side {query}", "aside")
    mas = people([side], modules=[box], outputs={"approval": "json", "aside": "text"}, llm=MockLLM(lambda p: "s"))
    paused = run(mas, {"query": "q"}, "n1")
    assert open_ids(paused) == ["n1/review/1/approve/1"]
    assert paused.agents == ("side",)
    done = resume(mas, "n1", {"n1/review/1/approve/1": {"approved": True, **TOM}})
    assert done.status == "ok"
    assert done.outputs["approval"]["approved"] is True
    (record,) = done["review"]
    assert record.kind == "module"
    assert [(r.agent, r.kind) for r in cast("Any", record.inner).records] == [
        ("writer", "agent"),
        ("approve", "human_action"),
    ]

    # A second human inside the module pauses it again.
    twice = MASModule("review", boxed_inner(confirm=True), reads={"topic": "query"}, writes={"confirmed": "confirmed"})
    mas = people([], modules=[twice], outputs={"confirmed": "json"})
    run(mas, {"query": "q"}, "b4")
    second = resume(mas, "b4", {"b4/review/1/approve/1": {"approved": True, **TOM}})
    assert open_ids(second) == ["b4/review/1/confirm/1"]
    done = resume(mas, "b4", {"b4/review/1/confirm/1": {"approved": True, **TOM}})
    assert done.outputs["confirmed"]["approved"] is True
    assert len(done["review"]) == 1

    # A module in a chain item.
    each = {"review": "page in pages"}
    box_in = MASModule("review", boxed_inner(), reads={"topic": "page"}, writes={"approval": "approval"})
    chain = people(
        [], modules=[box_in], each=each, inputs={"pages": "list of text"}, outputs={"approval": "list of json"}
    )
    paused = run(chain, {"pages": ["x", "y"]}, "n2")
    assert sorted(open_ids(paused)) == ["n2/review/1/approve/0/1", "n2/review/1/approve/1/1"]  # module visit, item
    done = resume(chain, "n2", {qid: {"approved": True, **TOM} for qid in open_ids(paused)})
    assert [a["approved"] for a in done.outputs["approval"]] == [True, True]

    # A module visited twice in a loop: each visit pauses on its own thread.
    again = Agent("again", "Again {approval}", "verdict", ["approval"], output_type="json")
    loop = Edge("again", "review", EdgeKind.WHILE, when="not approval.approved", max_visits=2)
    edges: list[tuple[str, str] | Edge] = [("review", "again"), loop, Edge("again", END, EdgeKind.ELSE)]
    looped = people([again], edges, modules=[box], llm=MockLLM(lambda p: "{}"), outputs={"verdict": "json"})
    paused = run(looped, {"query": "q"}, "n3")
    assert open_ids(paused) == ["n3/review/1/approve/1"]
    second = resume(looped, "n3", {"n3/review/1/approve/1": {"approved": False, **TOM}})
    assert open_ids(second) == ["n3/review/2/approve/1"]
    done = resume(looped, "n3", {"n3/review/2/approve/1": {"approved": True, **TOM}})
    assert [r.agent for r in done.records] == ["review", "again", "review", "again"]


RESTART = """
import json, sys
from tests.test_human import restartable, run
mas = restartable(sys.argv[1])
trace = run(mas, {"query": "cats"}, "r9")
print(json.dumps({"status": trace.status, "open": [q.id for q in trace.pause.questions]}))
"""


def restartable(path: str, **kwargs: Any) -> MAS:
    """The approve gate on a file store, every call costing 0.01."""
    llm = MockLLM(lambda p: "a draft" if p.startswith("Write") else "done", meta=True)
    return gated(llm, store=Store(path), **kwargs)


def test_resume_after_a_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "runs.db")
    child = subprocess.run(
        [sys.executable, "-c", RESTART, path], cwd=ROOT, capture_output=True, text=True, check=True, timeout=120
    )
    assert json.loads(child.stdout.strip().splitlines()[-1]) == {"status": "paused", "open": ["r9/approve/1"]}

    changed = restartable(path)
    changed.agents["writer"].detached = True  # a detached writer is another system
    with pytest.raises(ValueError, match="resume with that system"):
        resume(changed, "r9", {"r9/approve/1": {"approved": True, **TOM}})
    time.sleep(0.3)  # the wait counts against no wall limit
    fresh = restartable(path, budget=Budget(max_calls=2, max_wall_s=0.25))
    done = resume(fresh, "r9", {"r9/approve/1": {"approved": True, **TOM}})
    assert done.status == "ok"
    assert done.outputs["shipped"] == "done"
    assert [r.agent for r in done.records] == ["writer", "approve", "ship"]
    assert done.cost_usd == pytest.approx(0.02)
    assert cast("Any", fresh.runs().run("r9")).totals["calls"] == 2

    # The calls made before the pause count: one call left of a limit of one is none.
    subprocess.run([sys.executable, "-c", RESTART, str(tmp_path / "b.db")], cwd=ROOT, check=True, timeout=120)
    tight = restartable(str(tmp_path / "b.db"), budget=Budget(max_calls=1))
    over = resume(tight, "r9", {"r9/approve/1": {"approved": True, **TOM}})
    assert over.status == "over_budget"
    assert [r.agent for r in over.records] == ["writer", "approve"]

    # a row whose checkpoints are gone is refused, and the run's id is free again
    later = gated()
    later.store = str(tmp_path / "later.db")  # a path set after construction opens there
    assert run(later, {"query": "q"}, "r10").status == "paused"
    assert later.runs().path == tmp_path / "later.db"
    later.runs().delete_thread("r10")
    with pytest.raises(ValueError, match="holds no checkpoint of run 'r10'"):
        resume(later, "r10", {"r10/approve/1": {"approved": True, **TOM}})
    assert "r10" not in RUNNING

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    later.runs().save_run = broken  # type: ignore[method-assign]
    with pytest.raises(OSError, match="disk full"):
        run(later, {"query": "q"}, "r11")
    assert "r11" not in RUNNING


def test_timeouts_take_the_default() -> None:
    never = gated(human=gate(timeout_s=0, default={"approved": False}))
    events: list[Event] = []
    done = run(never, {"query": "q"}, "t1", events)
    assert done.status == "ok"
    assert "pause" not in [e.kind for e in events]
    approval = done.outputs["approval"]
    assert (approval["approved"], approval["option"], approval["who"], approval["timed_out"]) == (
        False, "Redraw", None, True,
    )  # fmt: skip
    assert done.outputs["held"] == "hold"

    waits = gated(human=gate(timeout_s=0.2, default={"approved": False}))
    paused = run(waits, {"query": "q"}, "t2")
    assert cast("Any", paused.pause).questions[0].deadline is not None
    store = waits.runs()
    assert store.due() == ()
    with pytest.raises(ValueError, match="none is answered or past its deadline"):
        resume(waits, "t2")
    time.sleep(0.25)
    assert [row.run_id for row in store.due()] == ["t2"]
    assert [row.run_id for row in store.due(datetime.now())] == ["t2"]  # noqa: DTZ005  a time without a zone is local: the point of this line
    done = resume(waits, "t2")
    assert done.outputs["approval"]["timed_out"] is True
    assert done.outputs["held"] == "hold"
    assert store.due() == ()


def test_forms_and_answer_checks() -> None:
    # ask: the options come from a read's path
    llm = MockLLM(lambda p: '{"title": "T", "choices": ["short", "long"]}')
    pick = HumanAction(
        "pick", "ask", ["outline"], "picked", question="Which for {outline.title}?", options="outline.choices"
    )
    writer = Agent("writer", "Outline {query}", "outline", output_type="json")
    asked = people([writer], [("writer", "pick")], [pick], llm=llm)
    paused = run(asked, {"query": "q"}, "f1")
    question = cast("Any", paused.pause).questions[0]
    assert (question.text, question.options) == ("Which for T?", ("short", "long"))
    with pytest.raises(ValueError, match="medium") as refused:
        resume(asked, "f1", {"f1/pick/1": {"option": "medium", "extra": 1}})
    assert all(word in str(refused.value) for word in ("medium", "extra", "who"))  # every problem in one error
    store = asked.runs()
    assert cast("Any", store.run("f1")).status == "paused"
    assert store.question("f1/pick/1").answer is None  # type: ignore[union-attr]
    assert resume(asked, "f1", {"f1/pick/1": {"option": "long", **TOM}}).outputs == {}
    assert store.question("f1/pick/1").answer["option"] == "long"  # type: ignore[union-attr,index]

    # judge: a pick names one of the pair, or a tie
    duel = HumanAction("duel", "judge", ["a", "b"], "verdict", question="Which is better?", pair=["a", "b"])
    pair = [Agent("a", "A {query}", "a"), Agent("b", "B {query}", "b")]
    judged = people(pair, [("a", "duel"), ("b", "duel")], [duel], llm=MockLLM(lambda p: p[0]))
    paused = run(judged, {"query": "q"}, "f2")
    assert cast("Any", paused.pause).questions[0].options == ("a", "b", "tie")
    verdict = resume(judged, "f2", {"f2/duel/1": {"pick": "tie", **TOM}}).records[-1].outputs["verdict"]
    assert json.loads(verdict)["pick"] == "tie"

    # edit: the edited value is written with its type
    fix = HumanAction("fix", "edit", ["draft"], {"answer": "fixed", "value": "final"}, question="Fix it", edits="draft")
    edited = people(
        [Agent("writer", "Write {query}", "draft")], [("writer", "fix")], [fix], llm=MockLLM(lambda p: "old")
    )
    run(edited, {"query": "q"}, "f3")
    done = resume(edited, "f3", {"f3/fix/1": {"value": "new", **TOM}})
    (record,) = done["fix"]
    assert record.outputs["final"] == "new"
    assert json.loads(record.outputs["fixed"])["changed"] is True


def test_askers_and_unattended_runs(tmp_path: Path) -> None:
    # The asker is asked once per question, across a resume and a restart.
    path = tmp_path / "runs.db"
    asker = Asker()
    mas = chained(asker, confirm=True, side=False, store=Store(path))
    run(mas, plan("a"), "a1")
    resume(mas, "a1", {"a1/fix/a/1": {"approved": True, **TOM}})
    assert [q.id for q in asker.asked] == ["a1/fix/a/1", "a1/confirm/a/1"]
    later = Asker()
    restarted = chained(later, confirm=True, side=False, store=Store(path))
    assert resume(restarted, "a1", {"a1/confirm/a/1": {"approved": True, **TOM}}).status == "ok"
    assert later.asked == []

    # An asker that answers at once never pauses, in a chain item either.
    assert run(chained(Asker(lambda q: {"approved": True, **TOM})), plan("a", "b"), "a3").status == "ok"
    events: list[Event] = []
    now = gated(asker=Asker(lambda q: {"approved": True, **TOM}))
    done = run(now, {"query": "q"}, "a2", events)
    assert done.status == "ok"
    assert done.outputs["shipped"] == "ship"
    assert [e.kind for e in events if e.node == "approve"] == ["start", "end"]

    # Unattended runs replay recorded answers, take the default, or fail by name.
    replayed = gated(asker=ReplayAsker(done)).run("q")
    assert replayed.status == "ok"
    assert replayed.outputs["approval"]["approved"] is True
    assert replayed.outputs["shipped"] == done.outputs["shipped"]
    assert cast("float", replayed["approve"][0].wait_s) >= 0  # answered when asked again
    defaulted = gated(human=gate(default={"approved": False})).run("q")
    assert defaulted.outputs["approval"]["timed_out"] is True
    assert defaulted.outputs["held"] == "hold"
    failed = gated().run("q")
    assert failed.error == "Unanswered: human action 'approve' got no answer in an unattended run"
    assert [r.agent for r in failed.records] == ["writer"]
    split = Code("split", "pages", reads={"query": "query"}, writes={"pages": "plan"})
    itemized = MAS(
        [Agent("drafter", "Draft {page}", "draft", ["page"])],
        [("split", "drafter"), ("drafter", "fix")],
        code_nodes=[split],
        human_actions=[HumanAction("fix", "approve", ["draft"], "fixed")],
        each={"drafter": {"item": "page", "in": "plan"}, "fix": "page in plan"},
        tools=[pages],
        outputs={"fixed": "list of json"},
        llm=MockLLM(lambda p: p),
    )
    (failed,) = itemized.run_batch(["q"])  # an item's open question fails the run, not just the item
    assert failed.error == "Unanswered: human action 'fix' got no answer in an unattended run"


@tool
def pages(query: str) -> list[str]:
    """Split a query into two pages."""
    return [f"{query} 1", f"{query} 2"]


def test_early_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    # An answer given while the run works on another branch: no pause.
    holder: dict[str, MAS] = {}

    def late(prompt: str) -> str:
        if prompt.startswith("n2"):
            holder["mas"].answer("e1", {"e1/approve/1": {"approved": True, **TOM}})
        return prompt.split(maxsplit=1)[0]

    agents = [Agent("n1", "n1 {query}", "n1_out"), Agent("n2", "n2 {n1_out}", "n2_out", ["n1_out"])]
    ask = HumanAction("approve", "approve", ["query"], "approval", question="Go on with {query}?")
    outputs = {"approval": "json", "n2_out": "text"}
    holder["mas"] = mas = people(agents, [("n1", "n2")], [ask], llm=MockLLM(late), outputs=outputs)
    events: list[Event] = []
    done = run(mas, {"query": "q"}, "e1", events)
    assert done.status == "ok"
    assert {"pause", "resume"}.isdisjoint(e.kind for e in events)
    with pytest.raises(ValueError, match="answered already"):
        mas.answer("e1", {"e1/approve/1": {"approved": False, **TOM}})
    with pytest.raises(ValueError, match="no question of run"):
        mas.answer("e1", {"e1/approve/9": {"approved": False, **TOM}})

    # An answer that lands after the wait looked and before the run returns: resumed in place.
    from langgraph.types import interrupt as real

    gated_mas = gated()

    def landing(value: Any) -> Any:
        for question in value["questions"]:
            if cast("Any", gated_mas.runs().question(question)).answer is None:
                gated_mas.answer("e2", {question: {"approved": True, **TOM}})
        return real(value)

    monkeypatch.setattr("xmas.runtime.interrupt", landing)
    assert run(gated_mas, {"query": "q"}, "e2").status == "ok"
    monkeypatch.undo()

    # Eight items paused at once, answered from eight threads.
    mas = chained(side=False)
    keys = [f"k{i}" for i in range(8)]
    paused = run(mas, plan(*keys), "e3")
    assert len(open_ids(paused)) == 8
    threads = [
        threading.Thread(target=mas.answer, args=("e3", {qid: {"approved": True, **TOM}})) for qid in open_ids(paused)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done = resume(mas, "e3")
    assert [fixed["approved"] for fixed in done.outputs["fixed"]] == [True] * 8
    assert sorted(cast("str", r.item_key) for r in done["fix"]) == sorted(keys)


@pytest.fixture
def serde_events() -> Iterator[list[SerdeEvent]]:
    seen: list[SerdeEvent] = []
    unregister = register_serde_event_listener(seen.append)
    yield seen
    unregister()


def test_store_keeps_deadlines_totals_and_threads(tmp_path: Path) -> None:
    mas = chained(side=False, store=Store(tmp_path / "runs.db"))
    run(mas, plan("a", "b"), "s1")
    store = mas.runs()
    (row,) = store.runs("paused")
    assert sorted(row.paused) == ["s1/fix/a/1", "s1/fix/b/1"]  # in the order the items paused, which threads decide
    assert row.totals is not None
    assert row.totals["calls"] == 2
    question = cast("Any", store.question("s1/fix/b/1"))
    assert question.payload["item_key"] == "b"
    assert question.deadline is None
    threads = {t for (t,) in store.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    assert threads == {"s1", "s1/drafter/1/0", "s1/drafter/1/1"}
    assert store.prune_runs(0) == ()  # a paused run is kept
    resume(mas, "s1", {"s1/fix/a/1": {"approved": True, **TOM}, "s1/fix/b/1": {"approved": True, **TOM}})
    assert store.prune_runs(1) == ()
    assert store.prune_runs(0) == ("s1",)
    assert store.run("s1") is None
    assert store.execute("SELECT COUNT(*) FROM checkpoints") == [(0,)]


def test_checkpoints_hold_xmas_types(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, serde_events: list[SerdeEvent]
) -> None:
    @tool
    def look(word: str) -> str:
        """Look a word up."""
        return word.upper()

    llm = MockLLM(lambda p: "not json", turns=[("", [req("look", '{"word": "cat"}')])], default=('{"n": 1}', []))
    finder = Agent("finder", "Find {query}", "found", output_type="json", tools=["look"])
    inner = people(
        [finder, Agent("writer", "Write {found}", "draft", ["found"])],
        [("finder", "writer"), ("writer", "approve")],
        [gate()],
        llm=llm,
        tools=[look],
    )
    box = MASModule("review", inner, reads={"query": "query"}, writes={"approval": "approval"})
    path = tmp_path / "runs.db"
    with caplog.at_level(logging.DEBUG):
        mas = people([], modules=[box], store=Store(path), outputs={"approval": "json"})
        paused = run(mas, {"query": "q"}, "x1")
        assert open_ids(paused) == ["x1/review/1/approve/1"]
        again = people([], modules=[box], store=Store(path), outputs={"approval": "json"})
        done = resume(again, "x1", {"x1/review/1/approve/1": {"approved": True, **TOM}})
    inner_trace = cast("Any", done["review"][0].inner)
    (found,) = inner_trace["finder"]
    assert found.tool_calls[0].result == "CAT"
    assert isinstance(found.tool_calls, tuple)
    assert serde_events == []
    assert not [r for r in caplog.records if "unregistered" in r.getMessage()]


def test_events_and_compiled_graphs() -> None:
    events: list[Event] = []
    mas = gated()
    run(mas, {"query": "q"}, "v1", events)
    resume(mas, "v1", {"v1/approve/1": {"approved": True, **TOM}}, events)
    human = [(e.kind, e.question, e.visit) for e in events if e.node == "approve"]
    assert human == [("start", None, 1), ("pause", "v1/approve/1", 1), ("resume", "v1/approve/1", 1), ("end", None, 1)]
    (resumed,) = [e for e in events if e.kind == "resume"]
    assert resumed.wait_s is not None
    assert resumed.wait_s >= 0
    assert {e.run_id for e in events} == {"v1"}

    # A human is two nodes, its edges leave from the wait; a system without one compiles as it did.
    nodes = set(mas.compiled.get_graph().nodes)
    assert {"approve", "approve.wait"} <= nodes
    plain = MAS([Agent("writer", "Write {query}", "draft")], llm=MockLLM())
    assert set(plain.compiled.get_graph().nodes) == {"__start__", "writer", "__end__"}
    assert "asks" not in plain.compiled.channels
    assert "asks" in mas.compiled.channels


def test_a_model_without_parts_fails_clearly(png: Path) -> None:
    llm = SeeingLLM()
    mas = MAS([Agent("reviewer", "Review {page}", "review", ["page"])], inputs={"page": "image"}, llm=llm)
    llm.full_response = False  # after the check: the parts path is off when the run calls it
    trace = mas.run_state({"page": png})
    assert trace.status == "failed"
    assert "SeeingLLM cannot send pictures or sound" in str(trace.error)


def test_a_stopped_visit_keeps_its_record() -> None:
    """The model raises the stop on the visit's second call, after a part of its answer; the record sums both."""
    cut = Stopped("stopped", CallMeta(latency_s=0.1, cost_usd=0.002), '{"half')
    llm = MockLLM(answers=["not json", cut], meta=True)
    events: list[Event] = []
    mas = MAS([Agent("judge", "Judge {query}", "verdict", output_type="json")], llm=llm)
    trace = mas.run_state({"query": "q"}, on_event=events.append)
    assert trace.status == "stopped"
    (record,) = trace.records
    assert (record.error_kind, record.route, record.raw_response) == (ErrorKind.STOPPED, None, '{"half')
    assert record.cost_usd == pytest.approx(0.012)
    assert [(e.kind, e.status) for e in events] == [("start", None), ("end", "stopped")]
    holder: dict[str, MAS] = {}

    def press(prompt: str) -> str:
        holder["mas"].stop("s2")  # the writer's call ends; the judge's never starts
        return "a draft"

    agents = [Agent("writer", "Write {query}", "draft"), Agent("judge", "Judge {draft}", "verdict", ["draft"])]
    holder["mas"] = pressed = MAS(agents, [("writer", "judge")], llm=MockLLM(press))
    stopped = pressed.run_state({"query": "q"}, run_id="s2")
    assert stopped.status == "stopped"
    assert "judge" not in {record.agent for record in stopped.records}


def test_a_paused_and_resumed_run_is_in_the_run_log() -> None:
    mas = gated()
    paused = run(mas, {"query": "cats"}, "logged")
    store = mas.runs()
    assert store.path is not None
    kept = store.trace("logged")
    assert kept == paused
    assert kept.status == "paused"
    assert open_ids(kept) == open_ids(paused)
    done = resume(mas, "logged", {open_ids(paused)[0]: {"approved": True, **TOM}})
    assert store.trace("logged") == done
    assert done.status == "ok"
    assert len(done.records) == 3
    assert [row.answer is not None for row in store.questions("logged")] == [True]
    assert {"pause", "resume"} <= {event.kind for event in store.events("logged")}
