"""``for each`` chains in the engine: one inner run per item behind ``Send``, gathered in item order.

The models are scripted doubles, so no test spends money.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from langgraph.errors import GraphBubbleUp

from xmas import MAS, Agent, Budget, ErrorKind, MASModule, RunStatus
from xmas.budget import RunMeter
from xmas.edges import Edge, EdgeKind
from xmas.media import Image, Ref
from xmas.nodes import Code
from xmas.optimization.graph_updater import GraphUpdater
from xmas.optimization.synthesis import MechanicalSynthesizer
from xmas.optimization.topology import TopologyBudget
from xmas.runtime import initial_state, run_config
from xmas.tools import tool

from .conftest import (
    DECK,
    LONG_PAGE,
    PLAN,
    MockLLM,
    SeeingLLM,
    assemble_deck,
    deck,
    deck_llms,
    render_page,
)


class OutOfMoneyError(Exception):
    """The key is out of money."""

    kind = ErrorKind.MONEY


def most_in_flight(events: list[Any], head: str) -> int:
    """Return the most items of a chain that ran at once, from its item events."""
    now = most = 0
    for event in events:
        if event.node == head and event.kind in ("item_start", "item_end"):
            now += 1 if event.kind == "item_start" else -1
            most = max(most, now)
    return most


def test_the_deck_runs_per_item_and_gathers_in_order() -> None:
    mas = deck()
    events: list[Any] = []
    trace = mas.run_state({"plan": PLAN, "house": "plain"}, on_event=events.append)
    assert trace.error is None

    # every write comes back as a list aligned to the items
    built = trace.outputs["deck"]
    assert built == {
        "pages": ["Short page.", "Short page.", "Short page.", LONG_PAGE, "Short page."],
        "captions": ["A caption.", "A caption."],
        "fits": [True, True, True, False, True],
    }
    assert trace.outputs["notes"] == "Notes."
    assert trace.output == json.dumps(built, ensure_ascii=False)

    # one redraw per page, the visits counted per item; "limits" still does not fit, so its loop is capped
    pages = [slide["id"] for slide in PLAN["slides"]]
    drawn = [r for r in trace.records if r.agent in ("drafter", "render", "checker")]
    assert [(r.item_key, r.agent, r.visit) for r in drawn] == [
        (page, node, visit) for page in pages for visit in (1, 2) for node in ("drafter", "render", "checker")
    ]
    assert [r.item for r in drawn] == [index for index in range(5) for _ in range(6)]
    assert [r.capped for r in trace["checker"] if r.visit == 2] == [False, False, False, True, False]
    assert [r.route for r in trace["checker"]][:2] == [("drafter",), ("__end__",)]
    figures = [(r.item, r.item_key, r.agent) for r in trace.records if r.agent in ("captioner", "figure_checker")]
    assert figures == [
        (0, "f1", "captioner"),
        (0, "f1", "figure_checker"),
        (1, "f2", "captioner"),
        (1, "f2", "figure_checker"),
    ]

    # "intro" finished last, yet its records come first: records keep the items' order
    finished = [event.item_key for event in events if event.kind == "item_end" and event.node == "drafter"]
    assert finished[-1] == "intro"
    assert sorted(finished) == sorted(pages)

    # assemble runs once, after both chains; noter runs beside them; neither is an item
    names = [r.agent for r in trace.records]
    assert names.count("assemble") == 1
    assert names.count("noter") == 1
    assert names.index("assemble") > max(i for i, name in enumerate(names) if name in ("checker", "figure_checker"))
    assert (trace["assemble"][0].item, trace["assemble"][0].visit) == (None, 1)
    assert trace["noter"][0].item is None

    # at most four pages at once
    assert 1 < most_in_flight(events, "drafter") <= 4
    assert trace.cost_usd == pytest.approx(0.01 * (5 * 4 + 2 * 2 + 1))
    health = trace.health
    assert health is not None
    assert health.items["drafter"]["items"] == 5
    assert health.items["drafter"]["failed"] == 0


def pages(
    responder: Any = None, *, budget: Budget | None = None, loop: bool = False, key: bool = True
) -> tuple[MAS, MockLLM]:
    """A writer and a word count per page, then ``fin`` adds the counts up.

    A page whose draft says "broken" cannot be counted. With ``loop``, ``trim``
    drops the last page and goes back to the writer while pages are left, at
    most two passes, then on to ``fin``.
    """

    @tool(writes={"words": "json"})
    def count(draft: str) -> int:
        """Count a draft's words."""
        if "broken" in draft:
            raise ValueError("cannot count a broken page")
        return len(draft.split())

    @tool(writes={"total": "json"})
    def add_up(words: list[Any]) -> dict[str, Any]:
        """Add the counts up, and say how many are missing."""
        return {"words": sum(w for w in words if w is not None), "missing": words.count(None)}

    @tool(writes={"plan": "json"})
    def trim(plan: dict[str, Any]) -> dict[str, Any]:
        """Drop the plan's last page."""
        return {**plan, "slides": plan["slides"][:-1]}

    llm = MockLLM(responder or (lambda p: f"Page {json.loads(p.split(' ', 2)[2])['id']} in words"), meta=True)
    edges: list[tuple[str, str] | Edge] = [("writer", "count")]
    code_nodes = [Code("count", "count"), Code("fin", "add_up")]
    if loop:
        code_nodes.append(Code("trim", "trim"))
        edges += [
            ("count", "trim"),
            Edge("trim", "writer", EdgeKind.WHILE, when="len(plan.slides) > 0", max_visits=2),
            Edge("trim", "fin", EdgeKind.ELSE),
        ]
    else:
        edges.append(("count", "fin"))
    head = {"item": "page", "in": "plan.slides", "key": "id"} if key else "page in plan.slides"
    mas = MAS(
        [Agent("writer", "Write page {page}", "draft", ["page"])],
        edges,
        each={"writer": head, "count": "page in plan.slides"},
        outputs={"words": "list of json", "total": "json"},
        code_nodes=code_nodes,
        inputs={"plan": "json"},
        state={"plan": {"merge": "last"}} if loop else None,
        llm=llm,
        tools=[count, add_up, trim],
        budget=budget,
    )
    return mas, llm


def plan_of(*ids: str) -> dict[str, Any]:
    return {"plan": {"slides": [{"id": name} for name in ids]}}


def test_a_failed_item_keeps_the_others() -> None:
    def writer(prompt: str) -> str:
        page = json.loads(prompt.split(" ", 2)[2])["id"]
        if page == "d" and "money" in prompt:
            raise OutOfMoneyError("Key limit exceeded")
        return "a broken page" if page == "c" else f"Page {page} in words"

    mas, _ = pages(writer)
    trace = mas.run_state(plan_of("a", "b", "c", "d"))
    assert trace.error is None
    assert trace.status is RunStatus.OK
    assert trace.outputs == {"words": [4, 4, None, 4], "total": {"words": 12, "missing": 1}}
    assert trace["fin"][0].inputs == {"words": "[4, 4, null, 4]"}
    (failed,) = [r for r in trace["count"] if r.error is not None]
    assert (failed.item, failed.item_key, failed.error) == (2, "c", "ValueError: cannot count a broken page")
    assert failed.error_kind is ErrorKind.UNKNOWN
    assert failed.route is None
    assert [(r.agent, r.item) for r in trace.records][:8] == [(n, i) for i in range(4) for n in ("writer", "count")]
    health = trace.health
    assert health is not None
    assert health.items["writer"]["failed"] == 1
    assert "item_failed" in {flag.signal for flag in health.flags}

    # money in one item ends the run: the siblings finished and were paid, and keep their records
    money = {"plan": {"slides": [{"id": n, "say": "money"} for n in "abd"]}}
    broke = pages(writer)[0].run_state(money)
    assert broke.error == "OutOfMoneyError: Key limit exceeded"
    assert broke.error_kind is ErrorKind.MONEY
    kept = sorted((r.item, r.agent, r.error is None) for r in broke.records)
    assert kept == [
        (0, "count", True),
        (0, "writer", True),
        (1, "count", True),
        (1, "writer", True),
        (2, "writer", False),
    ]
    assert "fin" not in broke.agents

    # so does a budget refusal: three calls for four pages
    over = pages(budget=Budget(max_calls=3))[0].run_state(plan_of("a", "b", "c", "d"))
    assert over.status is RunStatus.OVER_BUDGET
    assert len({r.item for r in over["writer"]}) == len(over["writer"]) == 3  # one record per item
    assert all(r.item is not None and r.error is None for r in over.records)


@pytest.mark.parametrize("case", ["zero items", "outer loop", "bad keys", "a dict at the path"])
def test_zero_items_outer_loop_and_key_problems(case: str) -> None:
    if case == "zero items":
        empty: list[dict[str, Any]] = [plan_of(), {"plan": {"slides": None}}, {"plan": {}}]
        for values in empty:
            mas, llm = pages()
            trace = mas.run_state(values)
            assert trace.error is None
            assert llm.calls == []
            assert [r.agent for r in trace.records] == ["fin"]
            assert trace["fin"][0].inputs == {"words": "[]"}
            assert trace.outputs == {"words": [], "total": {"words": 0, "missing": 0}}
        keyless = pages(key=False)[0].run_state(plan_of("a", "b"))  # without a key, an item is its index
        assert [(r.item, r.item_key) for r in keyless["writer"]] == [(0, None), (1, None)]
    elif case == "outer loop":
        mas, llm = pages(loop=True)
        trace = mas.run_state(plan_of("a", "b", "c"))
        assert trace.error is None
        # pass 1 writes three pages, pass 2 the two left; the loop into the head is capped on its passes
        assert [(r.item_key, r.visit) for r in trace["writer"]] == [("a", 1), ("b", 1), ("c", 1), ("a", 1), ("b", 1)]
        assert [(r.route, r.capped) for r in trace["trim"]] == [(("writer",), False), (("fin",), True)]
        assert [r.visit for r in trace["trim"]] == [1, 2]
        assert trace.outputs["words"] == [4, 4]  # no stale third page
        assert len(llm.calls) == 5
    else:
        cases: dict[str, list[tuple[dict[str, Any], str]]] = {
            "bad keys": [
                (plan_of("a", "a"), "plan.slides repeats the key 'id': ['a']"),
                ({"plan": {"slides": [{"id": "a"}, {"name": "b"}]}}, "item 1 of plan.slides has no key 'id'"),
                ({"plan": {"slides": [{"id": "a"}, {"id": True}]}}, "item 1 of plan.slides has no key 'id'"),
            ],
            "a dict at the path": [
                ({"plan": {"slides": {"id": "a"}}}, "plan.slides holds dict, not a list"),
                ({"plan": [{"slides": []}]}, "plan holds list, not a mapping with 'slides'"),
            ],
        }
        for values, problem in cases[case]:
            mas, llm = pages()
            trace = mas.run_state(values)
            assert trace.error is not None
            assert trace.error.startswith("ValueError: for each page in plan.slides: ")
            assert problem in trace.error
            assert llm.calls == []
            assert trace.records == ()


def test_a_box_and_shared_prompts_in_a_chain() -> None:
    llm = MockLLM(lambda p: p.split(": ", 1)[1].upper() if p.startswith("Cut") else f"draft {json.loads(p[6:])['id']}")
    cutter = Agent("cutter", "Cut: {page}", output="short", inputs=["page"])
    inner = MAS([cutter], inputs={"page": "text"}, llm=llm)
    mas = MAS(
        [Agent("writer", "Write {page}", "draft", ["page"])],
        [("writer", "shorten")],
        each={"writer": {"item": "page", "in": "plan.slides", "key": "id"}, "shorten": "page in plan.slides"},
        outputs={"short": "list of text"},
        modules=[MASModule("shorten", inner, reads={"page": "draft"}, writes={"short": "short"})],
        inputs={"plan": "json"},
        llm=llm,
    )
    trace = mas.run_state(plan_of("one", "two"))
    assert trace.error is None
    assert trace.outputs == {"short": ["DRAFT ONE", "DRAFT TWO"]}
    # the module's record and the records nested in it carry the item
    modules = trace["shorten"]
    assert [(r.item, r.item_key, r.kind) for r in modules] == [(0, "one", "module"), (1, "two", "module")]
    assert [[(r.agent, r.item, r.item_key) for r in cast("Any", b.inner).records] for b in modules] == [
        [("cutter", 0, "one")],
        [("cutter", 1, "two")],
    ]
    assert [(path, r.agent, r.item) for path, r in trace.walk()] == [
        ((), "writer", 0),
        ((), "shorten", 0),
        (("shorten",), "cutter", 0),
        ((), "writer", 1),
        ((), "shorten", 1),
        (("shorten",), "cutter", 1),
    ]

    # a chain inside a module keeps its items' records in the module's inner trace
    boxed = MAS(
        [], modules=[MASModule("all", mas, reads={"plan": "plan"}, writes={"short": "short"})], inputs={"plan": "json"}
    )
    (whole,) = boxed.run_state(plan_of("one", "two"))["all"]
    assert whole.outputs == {"short": '["DRAFT ONE", "DRAFT TWO"]'}
    assert whole.inner is not None
    assert [(r.agent, r.item_key) for r in whole.inner.records] == [
        ("writer", "one"),
        ("shorten", "one"),
        ("writer", "two"),
        ("shorten", "two"),
    ]

    # shared prompts, and the chain system resolving through the outer one
    shared = deck().copy(prompts={"checker": "Judge {planned} as {shown}"})
    judged = shared.run_state({"plan": PLAN, "house": "plain"})
    assert all(r.rendered_prompt.startswith("Judge") for r in [*judged["checker"], *judged["figure_checker"]])

    asked: list[str] = []

    class Metered(MAS):
        def resolve_llm(self, agent: Agent) -> Any:
            asked.append(agent.name)
            return super().resolve_llm(agent)

    metered = Metered.from_dict(yaml.safe_load(DECK), llms=dict(deck_llms()), tools=[render_page, assemble_deck])
    assert metered.run_state({"plan": PLAN, "house": "plain"}).error is None
    assert {"drafter", "checker", "captioner", "figure_checker", "noter"} <= set(asked)


def test_topology_search_never_breaks_a_chain() -> None:
    @tool(writes={"words": "json"})
    def count_words(draft: str) -> int:
        """Count the words of a draft."""
        return len(draft.split())

    mas = MAS(
        [
            Agent("writer", "Write page {slide}", "draft", inputs={"slide": "page"}),
            Agent("summarizer", "Sum up {words}", "summary", ["words"]),
            Agent("polisher", "Polish {summary}", "final", ["summary"]),
        ],
        [("writer", "count"), ("count", "summarizer"), ("summarizer", "polisher")],
        code_nodes=[Code("count", "count_words")],
        inputs={"plan": "json"},
        each={"writer": {"item": "page", "in": "plan.slides", "key": "id"}, "count": "page in plan.slides"},
        llm=MockLLM(lambda prompt: "one two"),
        tools=[count_words],
    )
    updater = GraphUpdater(MechanicalSynthesizer(), TopologyBudget(min_agents=2, max_agents=5, max_edges=8), seed=0)
    fixed = updater.fixed_of(mas)
    assert fixed is not None
    assert {"writer", "count"} <= fixed.nodes  # chain members are held
    candidates = updater.propose(mas, {}, 4)
    assert candidates
    for candidate in candidates:
        assert candidate.each == mas.each
        assert candidate.agents["writer"].reads == {"slide": "page"}
        assert {(e.source, e.target) for e in candidate.edges} >= {("writer", "count"), ("count", "summarizer")}
        assert candidate.run_state(plan_of("a", "b")).error is None


def test_boxes_in_items_keep_apart() -> None:
    def cut(prompt: str) -> str:
        if "bad" in prompt:
            time.sleep(0.1)  # the siblings' inner records land on the same list meanwhile
            raise RuntimeError("boom")
        return prompt.upper()

    def counted(prompt: str) -> str:
        tag = prompt.split(maxsplit=1)[0]
        with lock:
            now[tag] += 1
            most[tag] = max(most[tag], now[tag])
        time.sleep(0.03)
        with lock:
            now[tag] -= 1
        return "ok"

    inner = MAS(
        [Agent("first", "First: {page}", "mid", ["page"]), Agent("cutter", "Cut: {mid}", "short", ["mid"])],
        [("first", "cutter")],
        inputs={"page": "text"},
        llm=MockLLM(cut),
    )
    box = MASModule("shorten", inner, reads={"page": "page"}, writes={"short": "short"})
    pages = {"pages": "list of text"}
    mas = MAS([], modules=[box], each={"shorten": "page in pages"}, outputs={"short": "list of text"}, inputs=pages)
    trace = mas.run_state({"pages": ["a", "bad", "c", "d"]})
    (failed,) = [r for r in trace.records if r.error is not None]
    assert failed.item == 1
    assert failed.inner is not None
    assert [(r.agent, r.item) for r in failed.inner.records] == [("first", 1)]  # no sibling's records

    # two chains with the same head name keep their own at once
    lock, now, most = threading.Lock(), {"Narrow": 0, "Wide": 0}, {"Narrow": 0, "Wide": 0}

    def system(tag: str, at_once: int) -> MAS:
        each = {"w": {"item": "page", "in": "pages", "at once": at_once}}
        writer = Agent("w", tag + " {page}", "d", ["page"])
        return MAS([writer], each=each, outputs={"d": "list of text"}, llm=MockLLM(counted), inputs=pages)

    both = MAS(
        [],
        modules=[
            MASModule("narrow", system("Narrow", 1), reads={"pages": "pages"}, writes={"d": "narrow"}),
            MASModule("wide", system("Wide", 4), reads={"pages": "pages"}, writes={"d": "wide"}),
        ],
        outputs={"narrow": "list of text", "wide": "list of text"},
        inputs=pages,
    )
    assert both.run_state({"pages": list("abcdefgh")}).error is None
    assert most == {"Narrow": 1, "Wide": 4}


def test_langgraph_signals_pass_the_wrapper() -> None:
    @tool(writes={"done": "text"})
    def pause(draft: str) -> str:
        """Pause the run, as an interrupt does."""
        raise GraphBubbleUp

    mas = MAS(
        [Agent("writer", "Write {page}", "draft", ["page"])],
        [("writer", "halt")],
        code_nodes=[Code("halt", "pause")],
        each={"writer": "page in pages", "halt": "page in pages"},
        inputs={"pages": "list of text"},
        outputs={"done": "list of text"},
        llm=MockLLM(lambda prompt: "a draft"),
        tools=[pause],
    )
    meter = RunMeter(owner=mas)
    with pytest.raises(GraphBubbleUp):
        mas.compiled.invoke(initial_state({"pages": ["a", "b"]}, mas.prompts()), config=run_config(mas, meter))
    assert [(r.agent, r.error) for r in meter.records] == [("writer", None), ("writer", None)]  # no failed record


@tool(writes={"png": "image"})
def render_slide(page: dict[str, Any]) -> Image:
    """Draw one slide as a picture in the store; a broken one cannot be drawn."""
    if page.get("broken"):
        raise ValueError("cannot draw a broken slide")
    return Ref.from_bytes(f"slide {page['id']}".encode(), "image/png", f"{page['id']}.png")


@tool(writes={"deck": "json"})
def assemble_pictures(png: list[Image]) -> dict[str, Any]:
    """Put the drawn slides together."""
    return {"pages": [picture.size for picture in png]}


def slides(*, assemble: bool) -> tuple[MAS, SeeingLLM, SeeingLLM]:
    """Each slide is rendered and judged on its own picture; ``reviewer``, then ``assemble``, take all of them."""
    eye = SeeingLLM(lambda parts: f"fits {parts[1].read_bytes().decode()}")
    desk = SeeingLLM("the deck holds")
    code_nodes, edges = [Code("render", "render_slide")], [("render", "judge"), ("judge", "reviewer")]
    outputs = {"review": "text", "verdict": "list of text"}
    if assemble:
        code_nodes.append(Code("assemble", "assemble_pictures"))
        edges.append(("reviewer", "assemble"))
        outputs["deck"] = "json"
    mas = MAS(
        [
            Agent("judge", "Judge this slide: {png}", "verdict", ["png"], model="eye"),
            Agent("reviewer", "Review the deck: {png}", "review", ["png"], model="desk"),
        ],
        edges,
        llms={"eye": eye, "desk": desk},
        code_nodes=code_nodes,
        tools=[render_slide, assemble_pictures],
        inputs={"plan": "json"},
        each={"render": {"item": "page", "in": "plan.slides", "key": "id"}, "judge": "page in plan.slides"},
        outputs=outputs,
    )
    return mas, eye, desk


def test_the_judge_sees_each_slide(png: Path) -> None:  # the fixture keeps the store apart
    mas, eye, desk = slides(assemble=True)
    assert str(mas.checked().types["png"]) == "list of image"
    trace = mas.run_state(plan_of("intro", "method", "result"))
    assert trace.error is None
    assert trace.outputs["verdict"] == ["fits slide intro", "fits slide method", "fits slide result"]
    assert sorted(parts[1].read_bytes() for parts in eye.parts) == [b"slide intro", b"slide method", b"slide result"]
    assert all(parts[0] == "Judge this slide: " and len(parts) == 2 for parts in eye.parts)
    pictures = [record.outputs["png"] for record in trace["render"]]
    assert [json.loads(line)["media_type"] for line in pictures] == ["image/png"] * 3
    assert trace.outputs["deck"] == {"pages": [11, 12, 12]}
    assert json.loads(trace["assemble"][0].inputs["png"]) == [json.loads(line) for line in pictures]
    (parts,) = desk.parts
    assert parts[0::2] == ["Review the deck: \n[png 0] ", "\n[png 1] ", "\n[png 2] "]
    assert [part.read_bytes() for part in parts[1::2]] == [b"slide intro", b"slide method", b"slide result"]

    # a slide that cannot be drawn is no picture: the reviewer sees (none) in its place
    broken, eye, desk = slides(assemble=False)
    plan = {"plan": {"slides": [{"id": "intro"}, {"id": "method", "broken": True}, {"id": "result"}]}}
    failed = broken.run_state(plan)
    assert failed.error is None
    assert failed.outputs["verdict"] == ["fits slide intro", None, "fits slide result"]
    assert len(eye.parts) == 2
    (parts,) = desk.parts
    assert [type(part).__name__ for part in parts] == ["str", "Ref", "str", "Ref"]
    assert parts[2] == "\n[png 1] (none)\n[png 2] "
    assert "[png 1] (none)\n[png 2] [image " in failed["reviewer"][0].rendered_prompt
    assert json.loads(failed["reviewer"][0].inputs["png"])[1] is None
