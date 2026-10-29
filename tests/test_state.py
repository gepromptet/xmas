"""State in, several outputs out: ``run_state``, declared outputs, retries, reserved reads, events, caps, pictures.

The models are scripted doubles, so no test spends money.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from xmas import MAS, Agent, Budget, ErrorKind, MASModule, TokenUsage
from xmas.edges import Edge, EdgeKind
from xmas.media import Image, Ref
from xmas.nodes import Code
from xmas.tools import tool
from xmas.trace import Trace

from .conftest import PNG, MockLLM, PaintingLLM, SeeingLLM


def deck_notes(llm: MockLLM) -> MAS:
    """``summarizer`` and ``noter`` run in parallel on the plan; END joins them; both are outputs."""
    return MAS(
        [
            Agent("summarizer", "Summarize {plan} in the style {house}", "summary", ["plan", "house"]),
            Agent("noter", "Write notes for {plan}", "notes", ["plan"]),
        ],
        outputs={"summary": "text", "notes": "text"},
        inputs={"plan": "json", "house": "text"},
        llm=llm,
    )


def test_run_state_checks_inputs_and_returns_outputs() -> None:
    llm = MockLLM(lambda p: "A summary." if p.startswith("Summarize") else "Some notes.", meta=True)
    mas = deck_notes(llm)

    # every problem at once, before any call
    with pytest.raises(ValueError, match="do not fit the system's inputs") as caught:
        mas.run_state({"plan": {1, 2}, "query": "x"})
    lines = str(caught.value).splitlines()[1:]
    assert len(lines) == 3
    assert llm.calls == []
    assert lines[0] == "- input 'house' is missing"
    assert lines[1].startswith("- 'query' is no input")
    assert lines[2].startswith("- input 'plan' is not json")

    values = {"plan": {"slides": 2}, "house": "plain"}
    trace = mas.run_state(values)
    assert trace.error is None
    assert trace.outputs == {"summary": "A summary.", "notes": "Some notes."}
    assert trace.output == "A summary."  # the first output, which a task scores
    assert sorted(trace.agents) == ["noter", "summarizer"]  # END joined both branches
    assert trace.query == json.dumps(values)
    assert re.fullmatch("[0-9a-f]{32}", trace.run_id or "")
    assert trace.graph_id == mas.describe()["id"]
    again = mas.run_state(values)
    assert again.run_id != trace.run_id
    assert again.graph_id == trace.graph_id
    assert again == trace  # the run id is left out of ==

    # a failure is a failed trace, with no outputs
    broken = deck_notes(MockLLM(answers=[RuntimeError("boom"), RuntimeError("boom")]))
    failed = broken.run_state(values)
    assert failed.error is not None
    assert failed.outputs == {}

    # version 1: text values, a preset beside the query; equal to run(query) record for record
    v1 = MAS([Agent("a", "Answer {query} {style}", "answer", ["query", "style"])], llm=MockLLM())
    plain = v1.run_state({"query": "cats"})
    assert plain.records == v1.run("cats").records
    assert plain.query == "cats"
    assert plain.run_id is None
    assert plain.graph_id is None
    assert plain.records[0].visit is None
    assert v1.run_state({"query": "cats", "style": "tersely"}).output == "echo:Answer cats tersely"
    assert v1("cats") == "echo:Answer cats "
    with pytest.raises(ValueError, match="'query' is not text"):
        v1.run_state({"query": 3})

    # a module's inner trace is no run of its own
    inner = MAS([Agent("cut", "Cut {page}", "short", ["page"])], inputs={"page": "text"}, llm=MockLLM())
    boxed = MAS(
        [],
        [("summarize", "fin")],
        modules=[MASModule("summarize", inner, reads={"page": "query"}, writes={"short": "short"})],
        code_nodes=[Code("fin", "shout")],
        tools=[shout],
        llm=inner.llm,
    )
    outer: Any = boxed.run("cats")
    assert outer.run_id is not None
    assert outer.records[0].inner.run_id is None


@tool(writes={"loud": "text"})
def shout(short: str) -> str:
    """Shout a text."""
    return short.upper()


def down_at_first(fails: int) -> tuple[Any, list[str]]:
    """A fetch code node's callable that meets an outage on its first ``fails`` calls, and the queries it got."""
    calls: list[str] = []

    @tool(writes={"page": "text"})
    def fetch(query: str) -> str:
        """Fetch a page; the server is down at first."""
        calls.append(query)
        if len(calls) <= fails:
            raise ConnectionError("the server is down")
        return f"page on {query}"

    return fetch, calls


def test_retry_and_reserved_reads() -> None:
    # a code node whose callable fails once with an outage is run again: one record, one retry
    for fails, ok in ((1, True), (2, False)):
        fetch, calls = down_at_first(fails)
        mas = MAS([], code_nodes=[Code("fetch", "fetch")], retry={"fetch": {"outage": 1}}, tools=[fetch])
        trace: Any = mas.run("cats")
        (record,) = trace["fetch"]
        assert (trace.error is None, record.retries, len(calls)) == (ok, 1, 2)
        assert record.error_kind is (None if ok else ErrorKind.OUTAGE)

    # a retried agent costs what every attempt paid: a refused answer, a lost call, then a good answer
    judge = Agent("judge", "Judge {query}", "verdict", output_type="json", schema={"fits": {"type": "bool"}})
    llm = MockLLM(answers=["not json", ConnectionError("down"), '{"fits": true}'], meta=True)
    retried: Any = MAS([judge], retry={"judge": {"outage": 1}}, llm=llm).run("cats")
    (judged,) = retried["judge"]
    assert judged.retries == 1
    assert judged.outputs == {"verdict": '{"fits": true}'}
    assert judged.cost_usd == pytest.approx(0.02)
    assert judged.tokens == TokenUsage(20, 4)

    # error: the failure routed to a fallback, as {kind, message, node}
    @tool(writes={"page": "text"})
    def slow(query: str) -> str:
        """Time out."""
        raise TimeoutError("the server took too long")

    sorry = Agent("sorry", "Sorry: {error}", "answer", ["error"])
    routed = MAS(
        [sorry],
        [Edge("fetch", "sorry", EdgeKind.ERROR, errors=("timeout",))],
        code_nodes=[Code("fetch", "slow")],
        tools=[slow],
        llm=MockLLM(),
    ).run("cats")
    error = {"kind": "timeout", "message": "TimeoutError: the server took too long", "node": "fetch"}
    assert routed["sorry"][0].rendered_prompt == f"Sorry: {json.dumps(error)}"

    # run: the spend so far, read by an agent, a code node and a predicate
    @tool(writes={"spent": "json"})
    def spend(run: dict[str, Any]) -> dict[str, Any]:
        """Keep what the run has spent."""
        return run

    looked = MAS(
        [
            Agent("a", "Answer {query}", "answer"),
            Agent("b", "So far: {run}", "note", ["run"]),
            Agent("more", "More on {query}", "extra"),
        ],
        [
            ("a", "b"),
            ("b", "spend"),
            Edge("spend", "more", EdgeKind.WHEN, when="run.records >= 3 and run.calls == 2"),
            Edge("spend", "fin", EdgeKind.ELSE),
        ],
        code_nodes=[Code("spend", "spend"), Code("fin", "spend", writes={"spent": "final"})],
        tools=[spend],
        llm=MockLLM(meta=True),
    ).run("cats")
    note = json.loads(looked["b"][0].rendered_prompt.removeprefix("So far: "))
    assert (note["calls"], note["records"], note["cost_usd"]) == (1, 1, pytest.approx(0.01))
    spent = json.loads(looked["spend"][0].outputs["spent"])
    assert (spent["calls"], spent["records"], spent["cost_usd"]) == (2, 2, pytest.approx(0.02))
    assert spent["elapsed_s"] >= 0
    assert looked["spend"][0].route == ("fin",)

    reader = Agent("b", "So far: {run}", "note", ["run"])
    uncosted = MAS([Agent("a", "Answer {query}", "answer"), reader], [("a", "b")], llm=MockLLM(), version=2).run("cats")
    assert json.loads(uncosted["b"][0].rendered_prompt.removeprefix("So far: "))["cost_usd"] is None


def test_events_and_the_parallel_call_cap() -> None:
    from .test_chains import pages, plan_of

    def writer(prompt: str) -> str:
        return "a broken page" if '"b"' in prompt else "Page in four words"

    mas, _ = pages(writer)
    events: list[Any] = []
    trace = mas.run_state(plan_of("a", "b"), on_event=events.append)
    assert {event.run_id for event in events} == {trace.run_id}
    per_item = [[(e.kind, e.node, e.visit, e.status) for e in events if e.item == item] for item in (0, 1)]
    assert per_item[0] == [
        ("item_start", "writer", None, None),
        ("start", "writer", 1, None),
        ("end", "writer", 1, "ok"),
        ("start", "count", 1, None),
        ("end", "count", 1, "ok"),
        ("item_end", "writer", None, "ok"),
    ]
    assert [step[3] for step in per_item[1]] == [None, None, "ok", None, "failed", "failed"]
    assert {e.item_key for e in events if e.item == 1} == {"b"}
    fin = [e for e in events if e.node == "fin"]
    assert [e.kind for e in fin] == ["start", "end"]
    assert fin[1].item is None
    assert list(fin[1].to_dict()) == ["kind", "node", "run_id", "at_s", "visit", "status", "cost_usd"]
    item_end = next(e for e in events if e.kind == "item_end" and e.item == 0)
    assert item_end.cost_usd == pytest.approx(0.01)  # the writer's one call; the count is free

    # a listener that raises is warned about once and dropped; the run goes on
    heard: list[Any] = []

    def broken(event: Any) -> None:
        heard.append(event)
        raise RuntimeError("the viewer broke")

    two = MAS(
        [Agent("a", "Answer {query}", "answer"), Agent("b", "Check {answer}", "check", ["answer"])],
        [("a", "b")],
        llm=MockLLM(),
    )
    with pytest.warns(RuntimeWarning, match="the viewer broke") as warned:
        heard_trace = two.run_state({"query": "cats"}, on_event=broken)
    assert heard_trace.error is None
    assert len(heard) == 1
    assert len(warned) == 1

    # eight agents in parallel never hold more than two calls at once
    lock = threading.Lock()
    now = {"in": 0, "most": 0}

    def slow(prompt: str) -> str:
        with lock:
            now["in"] += 1
            now["most"] = max(now["most"], now["in"])
        time.sleep(0.03)
        with lock:
            now["in"] -= 1
        return "done"

    names = [f"a{i}" for i in range(8)]
    agents = [Agent(name, "Answer {query}", name) for name in names] + [Agent("sum", "Sum", "total", names)]
    for cap in (None, 2):
        now["most"] = 0
        wide = MAS(agents, [(name, "sum") for name in names], llm=MockLLM(slow), budget=Budget(max_parallel_calls=cap))
        assert wide.run("cats").error is None
        assert now["most"] > 2 if cap is None else now["most"] == cap
    for wrong in (0, 1.5, True):  # a float cap would hold no call back
        with pytest.raises(ValueError, match="max_parallel_calls must be an integer >= 1"):
            Budget(max_parallel_calls=cast("int", wrong))


def assert_plain_json(trace: Trace) -> None:
    """A trace with pictures is plain JSON, and no picture's bytes are in it."""
    text = json.dumps(trace.to_dict())
    assert "base64," not in text
    assert json.loads(text)["records"]


def test_run_state_with_pictures(png: Path) -> None:
    judge = Agent("judge", "Look: {png}", "verdict", ["png"])
    seeing = SeeingLLM("fits")
    looked = MAS([judge], inputs={"png": "list of image"}, llm=seeing).run_state({"png": [png]})
    ref = Ref.from_path(png)
    assert looked.error is None
    assert seeing.parts == [["Look: \n[png 0] ", ref]]
    (record,) = looked.records
    assert record.rendered_prompt == "Look: \n[png 0] [image a.png, image/png, 6 B]"
    assert record.inputs == {"png": json.dumps([ref.to_dict()])}  # the model sees bytes, the record a reference
    assert json.loads(looked.query)["png"] == [ref.to_dict()]
    assert_plain_json(looked)

    # a list of lists, such as each slide's pages gathered by a chain, labels a picture by its place
    nested = SeeingLLM()
    pages = MAS([judge], inputs={"png": "list of list of image"}, llm=nested).run_state({"png": [[png, None], [png]]})
    assert pages.error is None
    assert nested.parts == [["Look: \n[png 0.0] ", ref, "\n[png 0.1] (none)\n[png 1.0] ", ref]]

    # one picture is one part; a json judge's format block and retry note follow it as text
    form = {"fits": {"type": "bool"}}
    strict = Agent("judge", "Look: {png}", "verdict", ["png"], output_type="json", schema=form)
    retried = SeeingLLM(answers=["not json", '{"fits": true}'])
    trace = MAS([strict], inputs={"png": "image"}, llm=retried).run_state({"png": png})
    assert trace.error is None
    assert trace.records[0].outputs == {"verdict": '{"fits": true}'}
    first, second = retried.parts
    assert first[:2] == ["Look: ", ref]
    assert first[2].startswith("\n\nAnswer")
    assert len(first) == 3
    assert second[:2] == first[:2]
    assert second[2].startswith(first[2])
    assert len(second) == 3
    assert trace.records[0].rendered_prompt.startswith("Look: [image a.png, image/png, 6 B]\n\nAnswer")

    # a file that changed since it was given is refused when the call reads it
    png.write_bytes(PNG + b"!")
    changed = MAS([judge], inputs={"png": "list of image"}, llm=SeeingLLM()).run_state({"png": [ref]})
    assert "a.png changed since it was referenced" in (changed.error or "")


@pytest.mark.usefixtures("png")  # the store lives beside the picture
def test_a_painting_model_keeps_its_picture_in_the_store() -> None:
    painter = Agent("painter", "Paint {query}", "picture", output_type="image")
    painted = MAS([painter], llm=PaintingLLM(), outputs={"picture": "image"}).run_state({"query": "a cat"})
    picture = painted.outputs["picture"]
    assert painted.error is None
    assert isinstance(picture, Ref)
    assert picture.read_bytes() == PNG + b"\x00"
    assert Path(picture.location).parent.parent == Path(os.environ["XMAS_STORE"])
    assert painted.records[0].outputs == {"picture": json.dumps(picture.to_dict())}
    assert painted.records[0].raw_response == "drawn"
    many = Agent("painter", "Paint {query}", "pictures", output_type="list of image")
    drawn = MAS([many], llm=PaintingLLM(count=2), outputs={"pictures": "list of image"}).run_state({"query": "cats"})
    assert [ref.read_bytes() for ref in drawn.outputs["pictures"]] == [PNG + b"\x00", PNG + b"\x01"]
    for count, wanted in ((2, "expected one image, got 2"), (0, "expected one image, got 0")):
        refused = MAS([painter], llm=PaintingLLM(count=count), version=2).run("a cat")
        assert refused.error_kind is ErrorKind.BAD_FORMAT
        assert wanted in (refused.error or "")
    nothing = MAS([many], llm=PaintingLLM(count=0), version=2).run("cats")
    assert "expected at least one image, got 0" in (nothing.error or "")

    for trace in (painted, drawn):
        assert_plain_json(trace)
    assert painted.to_dict()["outputs"] == {"picture": picture.to_dict()}


def test_a_step_returning_a_path_writes_a_stored_ref(png: Path) -> None:
    @tool(writes={"png": "image"})
    def draw(query: str) -> Image:
        """Draw a page to the same file every time."""
        png.write_bytes(query.encode())
        return cast("Image", png)

    mas = MAS(
        [Agent("judge", "Look: {png}", "verdict", ["png"])],
        [("draw", "judge")],
        code_nodes=[Code("draw", "draw")],
        tools=[draw],
        llm=SeeingLLM(),
        outputs={"png": "image", "verdict": "text"},
    )
    first, second = (mas.run_state({"query": text}).outputs["png"] for text in ("one", "two"))
    assert (first.read_bytes(), second.read_bytes()) == (b"one", b"two")  # a copy each, not the rewritten file
    assert Path(first.location).parent.parent == Path(os.environ["XMAS_STORE"])


def test_the_run_log_keeps_a_run_that_dies_midway() -> None:
    @tool(writes={"loud": "text"})
    def publish(draft: str) -> str:
        """Publish a draft; the process ends here."""
        raise KeyboardInterrupt

    mas = MAS(
        [Agent("writer", "Write {query}", "draft")],
        [("writer", "publish")],
        code_nodes=[Code("publish", "publish")],
        tools=[publish],
        llm=MockLLM(meta=True),
    )
    with pytest.raises(KeyboardInterrupt):
        mas.run_state({"query": "cats"}, run_id="dies")
    store = mas.runs()
    died = store.trace("dies")
    assert [record.agent for record in died.records] == ["writer"]
    assert died.records[0].outputs == {"draft": "echo:Write cats"}
    assert cast("Any", store.run("dies")).status == "running"
    assert "no end in the store" in str(died.error)
    assert [(event.kind, event.node) for event in store.events("dies")] == [
        ("start", "writer"),
        ("end", "writer"),
        ("start", "publish"),
    ]
    assert [call["cost_usd"] for call in store.calls("dies")] == [0.01]
