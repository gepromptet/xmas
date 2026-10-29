"""The guide's pages under docs/guide, executed as written.

Each test mirrors one code block of a guide page line for line, with a scripted
``MockLLM`` in place of ``APILLM``, and checks the prints the page shows. Each
test names its page in its docstring. If a test needs editing to keep passing,
the page changed with it. The last tests pin claims that have no block of their
own, and that nothing the roadmap calls planned exists yet.
"""

from __future__ import annotations

import inspect
import itertools
import json
import re
import shlex
import shutil
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from promptolution.llms.base_llm import BaseLLM

import xmas
import xmas.config
import xmas.mas
import xmas.tasks
import xmas.verify
from xmas import (
    MAS,
    Agent,
    Budget,
    Code,
    Edge,
    EdgeKind,
    ErrorKind,
    Event,
    ForEach,
    GSM8KTask,
    HealthSettings,
    Image,
    MASModule,
    Ref,
    TokenUsage,
)
from xmas.bundle import parse_prompt, read
from xmas.cli import main
from xmas.config import PARAMS, Catalogue, Config, ModelEntry, build_llms, closest
from xmas.convert import document_text, table_rows
from xmas.examples import CHECK_KINDS, parse_examples
from xmas.llms import APILLM
from xmas.predicates import Context, parse_predicate
from xmas.spec import validate_spec_v2
from xmas.tools import Tool, ToolSource, tool
from xmas.verify import BUDGET_USD, PER_EXAMPLE_USD, VerifyFailed, VerifyWarning

from .conftest import PNG, MockLLM, PaintingLLM, SeeingLLM, req
from .test_stop import PollingLLM

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guide"


def page(name: str) -> str:
    """Return the text of the guide page ``name``."""
    return (GUIDE / name).read_text(encoding="utf-8")


STEPS = page("steps-and-values.md")
ANSWER_FORMS = page("answer-forms.md")
CONDITIONS = page("conditions-and-loops.md")
READING = page("reading-a-run.md")
MEDIA = page("media.md")
PEOPLE = page("people.md")
BUNDLES = page("bundles.md")
ROADMAP = page("roadmap.md")
QUERY = "Why is the sky blue?"
DRAFT = "Sunlight scatters off air."
FITS = '{"fits": true, "action": "publish"}'
SHORTEN = '{"fits": false, "action": "shorten"}'


def scripted(judge: Callable[[], str] | None = None, review: str = "A fine answer.") -> MockLLM:
    """A model that drafts DRAFT for the writer and answers the judge from ``judge``."""

    def respond(prompt: str) -> str:
        if prompt.startswith("Write"):
            return DRAFT
        if prompt.startswith("Cut"):
            return "Air scatters blue light."
        if "Is it a good answer?" in prompt:
            return review
        return judge() if judge is not None else FITS

    return MockLLM(respond)


def answers(*replies: str) -> Callable[[], str]:
    """Answer the judge with ``replies`` in order, repeating the last one."""
    stream: Iterator[str] = itertools.chain(replies, itertools.repeat(replies[-1]))
    return lambda: next(stream)


# ------------------------------------------------------------- the guide's definitions


@tool
def count_words(draft: str) -> int:
    """Count the words of a draft."""
    return len(draft.split())


@tool
def publish(draft: str) -> str:
    """Publish a draft."""
    return f"Published: {draft}"


@tool
def apologize(query: str) -> str:
    """Say sorry when no draft fits."""
    return f"Sorry, no short answer to: {query}"


verdict = {
    "fits": {"type": "bool", "means": "the draft answers the question in under 50 words"},
    "action": {"type": "text", "one of": ["publish", "shorten"]},
}


def json_judge() -> Agent:
    """The judge of 'Answers with a shape'."""
    return Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
        inputs=["query", "draft", "count_words"],
        output="verdict",
        output_type="json",
        schema=verdict,
    )


def looping_system(llm: MockLLM, counter: Tool = count_words) -> MAS:
    """The system of 'Conditions and loops'."""
    judge = json_judge()
    writer = Agent(
        "writer",
        prompt="Write a short answer to: {query}\nThe judge's last verdict, empty at first: {verdict}",
        inputs=["query", "verdict"],
        output="draft",
    )
    return MAS(
        [writer, judge],
        code_nodes=[
            Code("count", uses="count_words"),
            Code("publish", uses="publish"),
            Code("sorry", uses="apologize"),
        ],
        edges=[
            ("writer", "count"),
            ("count", "judge"),
            Edge("judge", "writer", kind=EdgeKind.WHILE, when="not verdict.fits", max_visits=3),
            Edge("judge", "publish", kind=EdgeKind.SWITCH, switch="verdict.action"),
            Edge("judge", "sorry", kind=EdgeKind.ELSE),
            Edge("count", "sorry", kind=EdgeKind.ERROR, errors=("timeout", "outage")),
        ],
        tools=[counter, publish, apologize],
        llm=llm,
    )


def quickstart_agents() -> tuple[Agent, Agent]:
    """The quickstart's intern and boss."""
    intern = Agent("intern", prompt="Answer this question, showing your working: {query}", output="draft")
    boss = Agent(
        "boss",
        prompt="Question: {query}\nYour intern's draft: {draft}\nGive the final answer in one sentence, as your own.",
        inputs=["query", "draft"],
        output="prediction",
    )
    return intern, boss


def quickstart_llm() -> MockLLM:
    """The quickstart test's model."""
    return MockLLM(lambda prompt: "Rayleigh scattering." if "as your own" in prompt else "Short waves scatter more.")


# -------------------------- docs/guide/steps-and-values.md: Code between agents: steps


def test_steps_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/steps-and-values.md."""
    llm = scripted(review="A fine answer.")

    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    judge = Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nIs it a good answer? Say why in one sentence.",
        inputs=["query", "draft", "count_words"],
        output="review",
    )
    mas = MAS(
        [writer, judge],
        code_nodes=[Code("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )

    print(mas("Why is the sky blue?"))  # the judge's review
    assert mas.last_trace is not None
    print([record.agent for record in mas.last_trace.records])  # ['writer', 'count', 'judge']

    assert capsys.readouterr().out.splitlines() == ["A fine answer.", "['writer', 'count', 'judge']"]
    count = mas.last_trace["count"][0]
    assert (count.kind, count.outputs, count.cost_usd) == ("code", {"count_words": "4"}, 0.0)
    assert mas.last_trace.cost_usd is None, "the writer's cost is unknown, so the sum is too"
    assert f"Draft, 4 words: {DRAFT}" in llm.calls[-1]
    assert len(llm.calls) == 2, "a code node makes no model call"


def test_a_step_writes_text_for_str_json_otherwise_and_renames_both_sides() -> None:
    """Mirrors docs/guide/steps-and-values.md."""

    @tool
    def shout(answer: str) -> str:
        """Shout an answer."""
        return answer.upper()

    llm = scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="answer")
    judge = Agent("judge", prompt="{words} words: {shout}", inputs=["words", "shout"], output="review")
    mas = MAS(
        [writer, judge],
        code_nodes=[
            Code("count", uses="count_words", reads={"draft": "answer"}, writes={"count_words": "words"}),
            Code("loud", uses="shout"),
        ],
        edges=[("writer", "count"), ("writer", "loud"), ("count", "judge"), ("loud", "judge")],
        tools=[count_words, shout],
        llm=llm,
    )

    trace = mas.run(QUERY)
    assert trace.error is None
    assert trace["count"][0].outputs == {"words": "4"}
    described = {node["name"]: node for node in mas.describe()["nodes"]}
    assert described["count"]["writes"] == {"words": "json"}
    assert described["loud"]["writes"] == {"shout": "text"}


def test_a_tool_needs_a_docstring_and_a_hint_on_every_parameter() -> None:
    """Mirrors docs/guide/steps-and-values.md."""

    def no_doc(draft: str) -> int:
        return len(draft)

    def no_hint(draft) -> int:  # type: ignore[no-untyped-def]
        """Count."""
        return len(draft)

    with pytest.raises(ValueError, match="docstring"):
        tool(no_doc)
    with pytest.raises(ValueError, match="has no type hint"):
        tool(no_hint)


def test_the_same_function_is_a_step_and_an_agents_tool() -> None:
    """Mirrors docs/guide/steps-and-values.md."""
    model = MockLLM(turns=[("", [req("count_words", '{"draft": "one two three"}')])], default=(DRAFT, []))
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft", tools=["count_words"])
    judge = Agent("judge", prompt="{count_words}: {draft}", inputs=["draft", "count_words"], output="review")
    mas = MAS(
        [writer, judge],
        code_nodes=[Code("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=model,
    )

    trace = mas.run(QUERY)
    assert trace.error is None
    assert [(call.tool, call.result) for call in trace["writer"][0].tool_calls] == [("count_words", "3")]
    assert trace["count"][0].outputs == {"count_words": "4"}


class RemotePages(ToolSource):
    """A tool source standing in for an MCP server."""

    def tools(self) -> list[Tool]:
        """Offer one remote tool."""
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        return [Tool("web_search", "Search the web.", schema)]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Answer every call with the same page."""
        return f"A page about {arguments['query']}"


def test_an_mcp_style_tool_runs_as_a_step() -> None:
    """Mirrors docs/guide/steps-and-values.md."""
    llm = scripted(review="ok")
    judge = Agent("judge", prompt="Is it a good answer? {web_search}", inputs=["web_search"], output="review")
    mas = MAS(
        [judge],
        code_nodes=[Code("search", uses="web_search")],
        edges=[("search", "judge")],
        tools=[RemotePages()],
        llm=llm,
    )

    trace = mas.run(QUERY)
    assert trace["search"][0].outputs == {"web_search": f"A page about {QUERY}"}


# ------------------------------------ docs/guide/answer-forms.md: Answers with a shape


def test_answer_form_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/answer-forms.md."""
    llm = scripted(answers("Looks good to me.", FITS))
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")

    judge = Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
        inputs=["query", "draft", "count_words"],
        output="verdict",
        output_type="json",
        schema=verdict,
    )
    mas = MAS(
        [writer, judge],
        code_nodes=[Code("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )

    trace = mas.run("Why is the sky blue?")
    print(trace["judge"][0].rendered_prompt)

    shown = (
        "Question: Why is the sky blue?\n"
        "Draft, 4 words: Sunlight scatters off air.\n"
        "Judge the draft.\n"
        "\n"
        "Answer with one JSON object and nothing else. Its fields:\n"
        "- fits (true or false): the draft answers the question in under 50 words\n"
        '- action (one of "publish", "shorten")\n'
    )
    assert capsys.readouterr().out == shown
    assert f"```text\n{shown}```" in ANSWER_FORMS

    print(trace.output)  # {"fits": true, "action": "publish"}
    print(trace["judge"][0].rejected)  # the answer refused before the retry, if there was one
    out = capsys.readouterr().out.splitlines()
    assert out[0] == '{"fits": true, "action": "publish"}'
    assert "Looks good to me." in out[1]
    assert [rejected.raw_response for rejected in trace["judge"][0].rejected] == ["Looks good to me."]
    judge_calls = [call for call in llm.calls if call.startswith("Question:")]
    assert len(judge_calls) == 2, "exactly one retry"
    assert "Your previous answer did not match the required format" in judge_calls[1]


def test_two_misses_end_the_run_with_bad_format() -> None:
    """Mirrors docs/guide/answer-forms.md."""
    llm = scripted(answers("Looks good.", "Still looks good."))
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    mas = MAS(
        [writer, json_judge()],
        code_nodes=[Code("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )

    trace = mas.run(QUERY)
    assert (trace.status, trace.error_kind) == ("failed", "bad_format")
    assert len([call for call in llm.calls if call.startswith("Question:")]) == 2
    with pytest.raises(RuntimeError, match="BadFormat"):
        mas(QUERY)


def test_the_form_dialect_the_guide_names() -> None:
    """Mirrors docs/guide/answer-forms.md."""
    schema = {
        "fits": {"type": "bool", "means": "whether it fits"},
        "words": {"type": "int", "optional": True},
        "score": {"type": "number"},
        "extra": {"type": "json"},
        "where": {"type": "object", "fields": {"line": {"type": "int"}}},
        "problems": {"type": "list of object", "fields": {"text": {"type": "text"}}},
        "tags": {"type": "list of text"},
        "action": {"type": "text", "one of": ["publish", "shorten"]},
    }
    Agent("judge", prompt="{query}", output="verdict", output_type="json", schema=schema)
    with pytest.raises(ValueError, match="unknown type 'boolean'"):
        Agent("judge", prompt="{query}", output="verdict", output_type="json", schema={"fits": {"type": "boolean"}})


# ---------------------- docs/guide/steps-and-values.md: Mistakes found before any call


def test_mistakes_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/steps-and-values.md."""
    llm = scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")

    judge_with_typos = Agent(
        "judge",
        prompt="Draft: {draft}\nIt has {count_words} words. The style guide: {style}",
        inputs=["draft", "notes"],
        output="verdict",
    )
    try:
        MAS(
            [writer, judge_with_typos],
            code_nodes=[Code("count", uses="count_word")],
            edges=[("writer", "count"), ("count", "judge")],
            tools=[count_words],
            llm=llm,
        )
    except ValueError as problems:
        print(problems)

    shown = (
        "The system does not check (3 problems):\n"
        "- agent 'judge': prompt placeholders ['count_words', 'style'] are not among its reads ['draft', 'notes']\n"
        "- Code 'count' uses unknown callable 'count_word'; available: ['count_words'].\n"
        "- 'judge' reads 'notes', which is neither an input nor written by any node\n"
    )
    assert capsys.readouterr().out == shown
    assert f"```text\n{shown}```" in STEPS
    assert llm.calls == [], "no model call"


def test_the_check_refuses_two_writers_a_type_clash_and_a_bad_condition() -> None:
    """Mirrors docs/guide/steps-and-values.md."""

    @tool
    def shout(draft: str) -> str:
        """Shout a draft."""
        return draft.upper()

    llm = scripted()
    two_writers = [Agent("a", "{query}", output="draft"), Agent("b", "{query}", output="draft")]
    with pytest.raises(ValueError, match="draft"):
        MAS(two_writers, code_nodes=[Code("s", uses="shout")], tools=[shout], llm=llm)

    json_writer = Agent("judge", "{query}", output="draft", output_type="json")
    with pytest.raises(ValueError, match="json"):
        MAS([json_writer], code_nodes=[Code("s", uses="shout")], edges=[("judge", "s")], tools=[shout], llm=llm)

    looping = [Agent("writer", "{query}", output="draft"), json_judge()]
    with pytest.raises(ValueError, match="position"):
        MAS(
            looping,
            code_nodes=[Code("count", uses="count_words"), Code("publish", uses="publish")],
            edges=[
                ("writer", "count"),
                ("count", "judge"),
                Edge("judge", "publish", kind=EdgeKind.WHEN, when="verdict.fits = true"),
            ],
            tools=[count_words, publish],
            llm=llm,
        )


# ---------------------------- docs/guide/conditions-and-loops.md: Conditions and loops


def test_conditions_block_publishes_a_draft_that_fits(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""
    mas = looping_system(scripted(answers(SHORTEN, FITS)))

    print(mas("Why is the sky blue?"))  # "Published: ..." or "Sorry, no short answer to: ..."

    assert capsys.readouterr().out == f"Published: {DRAFT}\n"
    assert mas.last_trace is not None
    assert [record.agent for record in mas.last_trace.records] == ["writer", "count", "judge"] * 2 + ["publish"]
    assert mas.last_trace["judge"][0].route == ("writer",)


def test_at_the_cap_the_else_edge_runs_even_when_the_switch_would_match() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""
    trace = looping_system(scripted(answers('{"fits": false, "action": "publish"}'))).run(QUERY)

    assert [record.agent for record in trace.records] == ["writer", "count", "judge"] * 3 + ["sorry"]
    assert [record.capped for record in trace["judge"]] == [False, False, True]
    assert trace.output == f"Sorry, no short answer to: {QUERY}"


def test_an_action_outside_the_cases_falls_through_to_else() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""
    trace = looping_system(scripted(answers('{"fits": true, "action": "shorten"}'))).run(QUERY)

    assert [record.agent for record in trace.records] == ["writer", "count", "judge", "sorry"]


def test_a_timeout_in_the_count_is_routed_and_keeps_the_failed_record() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""

    @tool(name="count_words")
    def slow_count(draft: str) -> int:
        """Count the words of a draft, too slowly."""
        raise TimeoutError("the counter took too long")

    trace = looping_system(scripted(), counter=slow_count).run(QUERY)

    assert trace.error is None
    assert [record.agent for record in trace.records] == ["writer", "count", "sorry"]
    assert trace["count"][0].error is not None
    assert trace.output == f"Sorry, no short answer to: {QUERY}"


def test_an_unrouted_error_fails_the_run_and_budget_and_stop_are_never_routable() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""

    @tool(name="count_words")
    def broken_count(draft: str) -> int:
        """Count the words of a draft, badly."""
        raise RuntimeError("no idea")

    trace = looping_system(scripted(), counter=broken_count).run(QUERY)
    assert trace.status == "failed"
    assert [record.agent for record in trace.records] == ["writer", "count"]

    for kind in ("budget", "stopped"):
        with pytest.raises(ValueError, match=kind):
            MAS(
                [Agent("writer", "{query}", output="draft")],
                code_nodes=[Code("sorry", uses="apologize")],
                edges=[Edge("writer", "sorry", kind=EdgeKind.ERROR, errors=(kind,))],
                tools=[apologize],
                llm=scripted(),
            )


def test_the_operator_table_matches_the_edge_kinds_and_the_spec_forms() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""
    assert {kind.value for kind in EdgeKind} == {"plain", "when", "else", "while", "switch", "error", "first of"}
    assert Edge("a", "b", kind=EdgeKind.SWITCH, switch="verdict.action").case == "b"
    forms = {
        "a -> b: {when: verdict.fits}": Edge("a", "b", kind=EdgeKind.WHEN, when="verdict.fits"),
        "a -> b: else": Edge("a", "b", kind=EdgeKind.ELSE),
        "a -> b: {when: not verdict.fits, max_visits: 3}": Edge(
            "a", "b", kind=EdgeKind.WHILE, when="not verdict.fits", max_visits=3
        ),
        "a -> b: {switch: verdict.action}": Edge("a", "b", kind=EdgeKind.SWITCH, switch="verdict.action"),
        "a -> b: {on error: [timeout]}": Edge("a", "b", kind=EdgeKind.ERROR, errors=("timeout",)),
        "a -> b: first of": Edge("a", "b", kind=EdgeKind.FIRST),
    }
    from xmas.edges import parse_edges

    for text, edge in forms.items():
        assert parse_edges(yaml.safe_load(f"- {text}")) == [edge], text


def test_every_condition_the_guide_names_parses_and_behaves() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""
    values = {"verdict": {"fits": False, "action": "publish"}, "draft": "four words right here"}
    ctx = Context(values=values, visits={"writer": 2, "judge": 1}, node="judge", cost_usd=0.2, elapsed_s=12.0)
    holds = {
        "not verdict.fits": True,
        "verdict.action in ['publish', 'shorten']": True,
        "exists(verdict.action)": True,
        "exists(verdict.notes)": False,
        "len(draft) > 200": False,
        "visits == 1 and visits(writer) >= 2": True,
        "cost < 0.5 and elapsed < 60": True,
        "verdict.fits == false or null == true": True,
        '"publish" == verdict.action': True,
    }
    for text, expected in holds.items():
        assert parse_predicate(text).evaluate(ctx) is expected, text
    with pytest.raises(ValueError, match="position 13"):
        parse_predicate("verdict.fits = true")


def test_a_join_runs_once_after_both_edges() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""
    llm = scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    critic = Agent("critic", prompt="Critic of {draft}", inputs=["draft"], output="notes")
    judge = Agent("judge", prompt="{draft} {notes} {count_words}", inputs=["draft", "notes", "count_words"], output="v")
    mas = MAS(
        [writer, critic, judge],
        code_nodes=[Code("count", uses="count_words")],
        edges=[("writer", "count"), ("writer", "critic"), ("count", "judge"), ("critic", "judge")],
        tools=[count_words],
        llm=llm,
    )

    trace = mas.run(QUERY)
    order = [record.agent for record in trace.records]
    assert order.count("judge") == 1
    assert order.index("judge") > max(order.index("count"), order.index("critic"))
    assert mas.describe()["joins"]["judge"] == "join"
    assert looping_system(scripted()).describe()["joins"]["sorry"] == "merge"


# ------------------------------------------ docs/guide/reading-a-run.md: Reading a run


def never_accepting() -> MAS:
    """The looping system with a judge that never accepts a draft."""
    return looping_system(scripted(answers(SHORTEN)))


def test_reading_a_run_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/reading-a-run.md."""
    mas = never_accepting()

    trace = mas.run("Why is the sky blue?")
    print(trace.status)  # ok
    print([record.agent for record in trace.records])
    # ['writer', 'count', 'judge', 'writer', 'count', 'judge', 'writer', 'count', 'judge', 'sorry']
    print(trace["judge"][-1].capped)  # True
    assert trace.health is not None
    print(trace.health.loops)  # {'judge -> writer': {'taken': 2, 'cap_hit': True}}

    assert capsys.readouterr().out.splitlines() == [
        "ok",
        "['writer', 'count', 'judge', 'writer', 'count', 'judge', 'writer', 'count', 'judge', 'sorry']",
        "True",
        "{'judge -> writer': {'taken': 2, 'cap_hit': True}}",
    ]
    record = trace["judge"][0]
    for name in ("inputs", "outputs", "rendered_prompt", "latency_s", "cost_usd", "tokens", "finish_reason"):
        assert hasattr(record, name), name
    for name in ("error_kind", "served_model", "route", "capped", "rejected", "kind"):
        assert hasattr(record, name), name
    assert record.cost_usd is None, "a model that reports no cost leaves it unknown, never 0"
    assert trace["count"][0].kind == "code"
    assert {status.value for status in xmas.RunStatus} >= {"ok", "failed", "over_budget"}


def test_health_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/reading-a-run.md."""
    trace = never_accepting().run(QUERY)
    assert trace.health is not None

    print(sorted({(flag.signal, flag.node) for flag in trace.health.flags}))
    # [('stuck', 'count'), ('stuck', 'judge'), ('stuck', 'writer')]

    assert capsys.readouterr().out == "[('stuck', 'count'), ('stuck', 'judge'), ('stuck', 'writer')]\n"


def test_health_signal_names_and_a_healthy_run() -> None:
    """Mirrors docs/guide/reading-a-run.md."""
    named = ["retried", "bad_format", "error_routed", "empty", "stuck", "unfilled", "leak", "long", "short"]
    named += ["item_failed", "item_over_budget"]
    sentence = READING.split("The signals are ", 1)[1].split("`trace.health.loops`", 1)[0]
    assert re.findall(r"`(\w+)`", sentence) == named

    healthy = looping_system(scripted(), counter=count_words)
    trace = healthy.run(QUERY)
    assert "health" not in trace.to_dict()

    empty = MAS(
        [Agent("writer", "{query}", output="draft")],
        llm=MockLLM(lambda prompt: ""),
        health=HealthSettings(length_factor=4),
    )
    flagged = empty.run(QUERY)
    assert flagged.health is not None
    assert [flag.signal for flag in flagged.health.flags] == ["empty"]


def test_budget_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/reading-a-run.md."""
    llm = quickstart_llm()
    intern, boss = quickstart_agents()

    thrifty = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_calls=1))
    spent = thrifty.run("Why is the sky blue?")
    print(spent.status)  # over_budget
    print(spent.error)  # BudgetExceeded: calls 1 >= max 1 before call 2
    print(spent["intern"][0].outputs)  # the intern's draft is kept

    assert capsys.readouterr().out.splitlines() == [
        "over_budget",
        "BudgetExceeded: calls 1 >= max 1 before call 2",
        "{'draft': 'Short waves scatter more.'}",
    ]
    assert spent.error_kind == "budget"


def test_a_zero_budget_spends_nothing_and_an_unknown_cost_stops_a_cost_limit() -> None:
    """Mirrors docs/guide/reading-a-run.md."""
    intern, boss = quickstart_agents()
    llm = quickstart_llm()
    nothing = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_cost_usd=0))
    assert nothing.run(QUERY).status == "over_budget"
    assert llm.calls == []

    unknown = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_cost_usd=0.5))
    trace = unknown.run(QUERY)
    assert (trace.status, len(llm.calls)) == ("over_budget", 1)
    assert trace["intern"][0].cost_usd is None
    Budget(max_cost_usd=0.5, max_wall_s=600, max_calls=40)


# ------------------------------------ docs/guide/drawing-a-system.md: Drawing a system


def test_describe_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/drawing-a-system.md."""
    mas = looping_system(scripted())

    picture = mas.describe()
    print([(node["name"], node["kind"]) for node in picture["nodes"]])
    # [('writer', 'agent'), ('count', 'code'), ('judge', 'agent'), ('publish', 'code'), ('sorry', 'code')]
    print([edge["kind"] for edge in picture["edges"] if not edge.get("implicit")])
    # ['plain', 'plain', 'while', 'switch', 'else', 'error']
    mas.agents["judge"].prompt += "\nBe strict."
    print(mas.describe()["id"] == picture["id"])  # True

    assert capsys.readouterr().out.splitlines() == [
        "[('writer', 'agent'), ('count', 'code'), ('judge', 'agent'), ('publish', 'code'), ('sorry', 'code')]",
        "['plain', 'plain', 'while', 'switch', 'else', 'error']",
        "True",
    ]
    rewired = looping_system(scripted()).copy(edges=[edge for edge in mas.edges if edge.kind is not EdgeKind.ERROR])
    assert rewired.describe()["id"] != picture["id"]
    node = picture["nodes"][2]
    assert {"model", "reads", "writes", "detached"} <= set(node)


# --------------------------------------- docs/guide/boxes.md: A system inside a system


def test_box_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/boxes.md."""
    llm = MockLLM(
        lambda prompt: (
            "Air scatters blue light." if prompt.startswith("Cut") else "Sunlight hits air, which scatters blue."
        )
    )

    cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
    shorten = MASModule(
        "shorten",
        MAS([cutter], inputs={"page": "text"}, llm=llm),
        reads={"page": "draft"},
        writes={"short": "summary"},
    )
    first_draft = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    outer = MAS([first_draft], modules=[shorten], edges=[("writer", "shorten")], llm=llm)

    trace = outer.run("Why is the sky blue?")
    print(trace.output)  # the cutter's shorter draft
    print([(path, record.agent) for path, record in trace.walk()])
    # [((), 'writer'), ((), 'shorten'), (('shorten',), 'cutter')]

    assert capsys.readouterr().out.splitlines() == [
        "Air scatters blue light.",
        "[((), 'writer'), ((), 'shorten'), (('shorten',), 'cutter')]",
    ]
    box = trace["shorten"][0]
    assert box.kind == "module"
    assert box.inner is not None
    assert box.cost_usd is None, "a model that reports no cost leaves every sum unknown"
    assert trace.cost_usd is None, "a model that reports no cost leaves every sum unknown"
    assert (
        box.inner["cutter"][0].rendered_prompt == "Cut this to half its length: Sunlight hits air, which scatters blue."
    )


def test_a_box_costs_its_inner_records_sum() -> None:
    """Mirrors docs/guide/boxes.md."""
    llm = MockLLM(lambda prompt: "Air scatters blue light.", meta=True)
    cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
    shorten = MASModule(
        "shorten", MAS([cutter], inputs={"page": "text"}, llm=llm), {"page": "draft"}, {"short": "summary"}
    )
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    trace = MAS([writer], modules=[shorten], edges=[("writer", "shorten")], llm=llm).run(QUERY)

    (box,) = trace["shorten"]
    assert box.cost_usd == pytest.approx(0.01)
    assert box.tokens == TokenUsage(10, 2)
    assert trace.cost_usd == pytest.approx(0.02)


def test_a_box_spends_from_the_outer_budget() -> None:
    """Mirrors docs/guide/boxes.md."""
    llm = scripted()
    cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
    shorten = MASModule(
        "shorten", MAS([cutter], inputs={"page": "text"}, llm=llm), {"page": "draft"}, {"short": "summary"}
    )
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    outer = MAS([writer], modules=[shorten], edges=[("writer", "shorten")], llm=llm, budget=Budget(max_calls=1))

    assert outer.run(QUERY).status == "over_budget"


# ------------------------------------------------- docs/guide/chains.md: Once per item


PLANNED_PAGES = [
    {"id": "light", "topic": "sunlight"},
    {"id": "air", "topic": "the air"},
    {"id": "eye", "topic": "the eye"},
]


def pages_llm() -> MockLLM:
    """Plan three pages and write them; the call for the page ``air`` times out."""

    def respond(prompt: str) -> str:
        if prompt.startswith("Plan"):
            return json.dumps({"pages": PLANNED_PAGES})
        if '"air"' in prompt:
            raise TimeoutError("the model took too long")
        return "Light scatters off air." if '"light"' in prompt else "Eyes see blue best."

    return MockLLM(respond)


@tool
def total_words(count_words: list[int | None]) -> int:
    """Add up the word counts of the pages that were written."""
    return sum(count for count in count_words if count is not None)


def test_for_each_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/chains.md."""
    llm = pages_llm()

    planner = Agent(
        "planner",
        prompt="Plan a short answer in three pages: {query}",
        output="plan",
        output_type="json",
        schema={"pages": {"type": "list of object", "fields": {"id": {"type": "text"}, "topic": {"type": "text"}}}},
    )
    page_writer = Agent("page_writer", prompt="Write one page on: {page}", inputs=["page"], output="draft")
    deck = MAS(
        [planner, page_writer],
        code_nodes=[Code("count", uses="count_words"), Code("total", uses="total_words")],
        edges=[("planner", "page_writer"), ("page_writer", "count"), ("count", "total")],
        each={"page_writer": ForEach("page", "plan.pages", at_once=2, key="id"), "count": "page in plan.pages"},
        tools=[count_words, total_words],
        llm=llm,
    )

    trace = deck.run("Why is the sky blue?")
    print([(record.agent, record.item_key) for record in trace.records])
    # [('planner', None), ('page_writer', 'light'), ('count', 'light'), ('page_writer', 'air'), ...]
    print(trace["total"][0].inputs)  # {'count_words': '[4, null, 4]'}: the page 'air' failed

    assert capsys.readouterr().out.splitlines() == [
        (
            "[('planner', None), ('page_writer', 'light'), ('count', 'light'), ('page_writer', 'air'), "
            "('page_writer', 'eye'), ('count', 'eye'), ('total', None)]"
        ),
        "{'count_words': '[4, null, 4]'}",
    ]
    assert trace.status == "ok"
    assert trace.error is None
    assert trace.output == "8"
    assert trace["page_writer"][1].error_kind is ErrorKind.TIMEOUT
    assert trace["page_writer"][1].item == 1
    assert deck.describe()["groups"] == [
        {
            "name": "page_writer",
            "for each": "page in plan.pages",
            "nodes": ["page_writer", "count"],
            "at once": 2,
            "key": "id",
        }
    ]
    assert trace.health is not None

    print(trace.health.items["page_writer"]["failed"])  # 1, of 3 items
    print([flag.signal for flag in trace.health.flags])  # ['item_failed']

    assert capsys.readouterr().out.splitlines() == ["1", "['item_failed']"]
    assert trace.health.items["page_writer"]["items"] == 3


def test_for_each_claims_without_a_block() -> None:
    """Mirrors docs/guide/chains.md."""
    page_writer = Agent("page_writer", prompt="Write one page on: {page}", inputs=["page"], output="draft")
    llm = pages_llm()

    def chain(form: ForEach, **kwargs: Any) -> MAS:
        return MAS(
            [page_writer],
            code_nodes=[Code("count", uses="count_words")],
            edges=[("page_writer", "count")],
            each={"page_writer": form, "count": "page in plan.pages"},
            inputs={"plan": "json"},
            outputs={"count_words": "list of json"},
            tools=[count_words],
            llm=llm,
            **kwargs,
        )

    # an item over its budget is flagged, never stopped
    costly = MAS(
        [page_writer],
        each={"page_writer": ForEach("page", "plan.pages", item_budget={"cost_usd": 0.005})},
        inputs={"plan": "json"},
        outputs={"draft": "list of text"},
        llm=MockLLM(lambda prompt: "A page.", meta=True),
    )
    over = costly.run_state({"plan": {"pages": [{"id": "light"}]}})
    assert over.error is None
    assert over.health is not None
    assert [flag.signal for flag in over.health.flags] == ["item_over_budget"]

    # the spec form: each node carries the short form, the first may write the long one
    spec = chain(ForEach("page", "plan.pages", key="id")).to_dict()
    assert spec["agents"][0]["for each"] == {"item": "page", "in": "plan.pages", "key": "id"}
    assert spec["code_nodes"][0]["for each"] == "page in plan.pages"
    assert ForEach.parse("page in plan.pages") == ForEach("page", "plan.pages")

    # a budget stop ends the whole run
    stopped = chain(ForEach("page", "plan.pages"), budget=Budget(max_calls=1))
    assert stopped.run_state({"plan": {"pages": PLANNED_PAGES}}).status == "over_budget"

    # chains do not nest
    inner = chain(ForEach("page", "plan.pages"))
    with pytest.raises(ValueError, match="chains do not nest"):
        MAS(
            [],
            modules=[MASModule("pages", inner, reads={"plan": "plan"}, writes={"draft": "drafts"})],
            each={"pages": "deck in decks"},
            inputs={"decks": "json"},
        )


# ----------------------------- docs/guide/run-state.md: Several values in, several out


def test_run_state_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/run-state.md."""
    llm = pages_llm()
    page_writer = Agent("page_writer", prompt="Write one page on: {page}", inputs=["page"], output="draft")

    pages = MAS(
        [page_writer],
        code_nodes=[Code("count", uses="count_words"), Code("total", uses="total_words")],
        edges=[("page_writer", "count"), ("count", "total")],
        each={"page_writer": ForEach("page", "plan.pages", key="id"), "count": "page in plan.pages"},
        inputs={"plan": "json"},
        outputs={"total_words": "json", "count_words": "list of json"},
        tools=[count_words, total_words],
        llm=llm,
    )

    heard: list[Event] = []
    trace = pages.run_state({"plan": {"pages": [{"id": "light"}, {"id": "eye"}]}}, on_event=heard.append)
    print(trace.outputs)  # {'total_words': 8, 'count_words': [4, 4]}
    print(trace.output)  # 8: the first output, which a task scores
    print(sorted({event.kind for event in heard}))  # ['end', 'item_end', 'item_start', 'start']

    assert capsys.readouterr().out.splitlines() == [
        "{'total_words': 8, 'count_words': [4, 4]}",
        "8",
        "['end', 'item_end', 'item_start', 'start']",
    ]
    assert {(event.node, event.item_key, event.status) for event in heard if event.kind == "item_end"} == {
        ("page_writer", "light", "ok"),
        ("page_writer", "eye", "ok"),
    }
    assert trace.run_id is not None
    assert trace.graph_id == pages.describe()["id"]
    with pytest.raises(ValueError, match="do not fit the system's inputs") as caught:
        pages.run_state({"plan": {1, 2}, "style": "short"})
    assert len(str(caught.value).splitlines()) == 3  # two problems, no call
    assert len(llm.calls) == 2
    with pytest.raises(ValueError, match="run_state"):
        pages.run(QUERY)


def test_run_state_claims_without_a_block() -> None:
    """Mirrors docs/guide/run-state.md."""
    # several branches end the run once all of them have
    both = MAS(
        [Agent("short", "Short: {query}", "brief"), Agent("long", "Long: {query}", "full")],
        outputs={"brief": "text", "full": "text"},
        llm=MockLLM(lambda prompt: prompt.split(":")[0]),
    )
    assert both.run_state({"query": QUERY}).outputs == {"brief": "Short", "full": "Long"}

    # a retry: one record counts the retries
    calls: list[str] = []

    @tool
    def flaky(draft: str) -> int:
        """Count the words, after one outage."""
        calls.append(draft)
        if len(calls) == 1:
            raise ConnectionError("down")
        return len(draft.split())

    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    retried = MAS(
        [writer],
        code_nodes=[Code("count", uses="flaky")],
        edges=[("writer", "count")],
        tools=[flaky],
        llm=scripted(),
        retry={"count": {"outage": 1}},
    ).run(QUERY)
    assert [record.retries for record in retried["count"]] == [1]

    # the reserved reads, and a read map
    looking = Agent("looking", prompt="Spent so far: {run}", inputs=["run"], output="note")
    spent = MAS([writer, looking], edges=[("writer", "looking")], llm=scripted(), version=2).run(QUERY)
    assert set(json.loads(spent["looking"][0].rendered_prompt.removeprefix("Spent so far: "))) == {
        "cost_usd",
        "elapsed_s",
        "calls",
        "records",
    }
    slide = Agent("slide", prompt="Write a slide on {slide}", inputs={"slide": "page"}, output="draft")
    assert slide.reads == {"slide": "page"}
    assert slide.inputs == ["page"]


# ---------------------------- docs/guide/media.md: Pictures, documents and other files


@tool(writes={"cover": "image"})
def keep_cover(pages: list[Image]) -> Path:
    """Keep the first page as the deck's cover."""
    return Path(pages[0].location)


@pytest.fixture
def slides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two slides of 182 KB and 96 KB in the working folder, and a store beside them."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XMAS_STORE", str(tmp_path / "store"))
    Path("slide-1.png").write_bytes(PNG.ljust(182 * 1024, b"1"))
    Path("slide-2.png").write_bytes(PNG.ljust(96 * 1024, b"2"))
    return tmp_path


def test_a_judge_looks_at_the_pictures(slides: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/media.md."""
    reviewer = Agent("reviewer", prompt="Review these slides.\n{pages}", inputs=["pages"], output="review")
    vision = SeeingLLM("Clear slides. The second needs a title.")
    deck = MAS(
        [reviewer],
        code_nodes=[Code("cover", uses="keep_cover")],
        inputs={"pages": "list of image"},
        outputs={"review": "text", "cover": "image"},
        tools=[keep_cover],
        llm=vision,
    )

    trace = deck.run_state({"pages": [Path("slide-1.png"), Path("slide-2.png")]})
    print(trace["reviewer"][0].rendered_prompt)
    print(trace.outputs["cover"].location)  # a copy in the store, named by its checksum

    cover = trace.outputs["cover"]
    assert capsys.readouterr().out.splitlines() == [
        "Review these slides.",
        "",
        "[pages 0] [image slide-1.png, image/png, 182 KB]",
        "[pages 1] [image slide-2.png, image/png, 96 KB]",
        cover.location,
    ]
    assert [type(part).__name__ for part in vision.parts[0]] == ["str", "Ref", "str", "Ref"]
    assert Path(cover.location).parent.parent == slides / "store"
    assert cover.read_bytes() == Path("slide-1.png").read_bytes()
    assert trace.outputs["review"] == "Clear slides. The second needs a title."
    dumped = json.dumps(trace.to_dict())
    assert "base64," not in dumped
    assert json.loads(dumped)["outputs"]["cover"] == cover.to_dict()

    with pytest.raises(ValueError, match="does not check") as caught:
        MAS([reviewer], inputs={"pages": "list of image"}, llm=quickstart_llm())
    assert str(caught.value).splitlines() == [
        "The system does not check (1 problem):",
        (
            "- agent 'reviewer' reads image 'pages', but the default model accepts only text; "
            'give it accepts = ("text", "image") or convert it first with a code node'
        ),
    ]


class Blind(SeeingLLM):
    """A model that sends content parts but says of itself that it accepts only text."""

    accepts = ("text",)


def test_media_claims_refs_and_bytes(slides: Path) -> None:
    """Mirrors docs/guide/media.md."""
    # a referenced file that changes is refused; a code node's Path is copied; bytes are refused
    page = Ref.from_path("slide-2.png")
    Path("slide-2.png").write_bytes(PNG)
    with pytest.raises(ValueError, match="changed"):
        page.read_bytes()
    assert Ref.from_bytes(PNG, "image/png") == Ref.from_bytes(PNG, "image/png")
    assert len(list((slides / "store").rglob("*.png"))) == 1

    @tool(writes={"cover": "image"})
    def raw_cover(pages: list[Image]) -> bytes:
        """Return the first page's bytes."""
        return pages[0].read_bytes()

    raw = MAS(
        [Agent("reviewer", prompt="Review {pages}", inputs=["pages"], output="review")],
        code_nodes=[Code("cover", uses="raw_cover")],
        inputs={"pages": "list of image"},
        outputs={"review": "text", "cover": "image"},
        tools=[raw_cover],
        llm=SeeingLLM(),
    ).run_state({"pages": [Path("slide-1.png")]})
    assert raw.status == "failed"
    assert "Ref.from_bytes stores bytes" in str(raw.error)


def test_media_claims_models_section(slides: Path) -> None:
    """Mirrors docs/guide/media.md."""
    # the models section: the page's entries, models over the model's own word, nothing else
    models = yaml.safe_load("models:" + MEDIA.split("```yaml\nmodels:", 1)[1].split("```", 1)[0])
    assert models == {"models": {"vision-model": {"accepts": ["image"]}, "painter-model": {"produces": ["image"]}}}
    spec = {
        "version": 2,
        "inputs": {"pages": "list of image", "title": "text"},
        "outputs": {"review": "text", "picture": "image"},
        "agents": [
            {"name": "reviewer", "model": "vision-model", "reads": ["pages"], "writes": {"review": "text"}},
            {"name": "painter", "model": "painter-model", "reads": ["title"], "writes": {"picture": "image"}},
        ],
        **models,
        "prompts": {"reviewer": "Review {pages}", "painter": "Draw a cover for {title}"},
    }
    llms: dict[str, BaseLLM] = {"vision-model": Blind(), "painter-model": PaintingLLM()}
    built = MAS.from_dict(spec, llms=llms)
    assert built.to_dict()["models"] == models["models"]
    painted = built.run_state({"pages": [Path("slide-1.png")], "title": "Light"})
    assert painted.outputs["picture"].location.startswith(str(slides / "store"))
    with pytest.raises(ValueError, match="accepts only text"):
        MAS.from_dict({**spec, "models": {"painter-model": {"produces": ["image"]}}}, llms=llms)
    tuned = MAS.from_dict(
        {**spec, "models": {**models["models"], "vision-model": {"accepts": ["image"], "temperature": 0}}}, llms=llms
    )
    assert tuned.to_dict()["models"]["vision-model"] == {"accepts": ["image"], "temperature": 0}
    with pytest.raises(ValueError, match="has key 'colour'; it takes model, accepts, produces and the parameters"):
        MAS.from_dict({**spec, "models": {"vision-model": {"accepts": ["image"], "colour": "red"}}}, llms=llms)
    with pytest.raises(ValueError, match="models names 'other', which no agent binds"):
        MAS.from_dict({**spec, "models": {**models["models"], "other": {"accepts": ["image"]}}}, llms=llms)
    reviewer = Agent("reviewer", prompt="Review {pages}", inputs=["pages"], output="review", model="vision-model")
    vision_only = {"vision-model": models["models"]["vision-model"]}
    seen = MAS([reviewer], inputs={"pages": "list of image"}, llms={"vision-model": Blind()}, models=vision_only)
    assert seen.capabilities(reviewer).takes("image")
    with pytest.raises(ValueError, match="accepts only text"):
        MAS([reviewer], inputs={"pages": "list of image"}, llms={"vision-model": Blind()})
    with pytest.raises(ValueError, match="cannot send content parts"):
        MAS([reviewer], inputs={"pages": "list of image"}, llms={"vision-model": MockLLM()}, models=vision_only)


@pytest.mark.usefixtures("slides")
def test_media_claims_conversion_steps() -> None:
    """Mirrors docs/guide/media.md."""
    # documents and tables go through a conversion code node
    Path("house.md").write_text("# House rules\nOne idea per slide.", encoding="utf-8")
    writer = Agent("writer", prompt="Follow these rules: {house_rules}", inputs=["house_rules"], output="draft")
    with pytest.raises(
        ValueError, match=r"through a conversion code node, such as one using xmas\.convert\.document_text"
    ):
        MAS([Agent("writer", "Follow {house}", "draft", inputs=["house"])], inputs={"house": "document"}, llm=MockLLM())
    ruled = MAS(
        [writer],
        code_nodes=[Code("rules", uses="document_text", reads={"source": "house"}, writes={"content": "house_rules"})],
        edges=[("rules", "writer")],
        inputs={"house": "document"},
        tools=[document_text],
        llm=MockLLM(lambda prompt: prompt),
    ).run_state({"house": Path("house.md")})
    assert ruled.output == "Follow these rules: # House rules\nOne idea per slide."
    Path("deck.csv").write_text("page,words\nlight,4\n", encoding="utf-8")
    assert table_rows(Ref.from_path("deck.csv")) == [{"page": "light", "words": "4"}]
    with pytest.raises(ValueError, match="with a step of your own"):
        document_text(Ref.from_bytes(b"%PDF", "application/pdf"))


@pytest.mark.usefixtures("slides")
def test_media_claims_image_answers() -> None:
    """Mirrors docs/guide/media.md."""
    # an image answer needs a model that produces images and no tools
    painter = Agent("painter", prompt="Draw {query}", output="picture", output_type="image")
    with pytest.raises(ValueError, match="produces only text"):
        MAS([painter], llm=scripted())
    assert isinstance(MAS([painter], llm=PaintingLLM()).run(QUERY)["painter"][0].outputs["picture"], str)
    with pytest.raises(ValueError, match="has tools; an image answer takes none"):
        MAS(
            [Agent("painter", "Draw {query}", "picture", output_type="image", tools=["count_words"])],
            tools=[count_words],
            llm=PaintingLLM(),
        )


@pytest.mark.usefixtures("slides")
def test_media_claims_remote_and_mcp_tools() -> None:
    """Mirrors docs/guide/media.md."""
    # a remote tool reads no pictures or files; an MCP tool may write a picture
    with pytest.raises(ValueError, match="remote tools read no pictures or files"):
        MAS(
            [Agent("judge", prompt="Is it a good answer? {web_search}", inputs=["web_search"], output="review")],
            code_nodes=[Code("search", uses="web_search", reads={"query": "pages"})],
            edges=[("search", "judge")],
            inputs={"pages": "list of image"},
            tools=[RemotePages()],
            llm=scripted(),
        )
    mcp = pytest.importorskip("xmas.tools.mcp")
    assert "writes" in inspect.signature(mcp.MCPTools).parameters


# ------------------------------------------- docs/guide/people.md: People in the graph


def test_people_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/people.md."""
    from xmas import END, HumanAction

    llm = scripted()

    editor = HumanAction(
        "editor",
        "approve",
        reads=["draft"],
        writes="approval",
        question="Publish this answer? {draft}",
        labels={"yes": "Publish", "no": "Hold it"},
    )
    gated = MAS(
        [Agent("writer", prompt="Write a short answer to: {query}", output="draft")],
        code_nodes=[Code("publish", uses="publish")],
        human_actions=[editor],
        edges=[
            ("writer", "editor"),
            Edge("editor", "publish", kind=EdgeKind.WHEN, when="approval.approved"),
            Edge("editor", END, kind=EdgeKind.ELSE),
        ],
        tools=[publish],
        llm=llm,
    )

    trace = gated.run_state({"query": "Why is the sky blue?"}, run_id="sky-1")
    print(trace.status, [question.id for question in trace.pause.questions])  # type: ignore[union-attr]  # paused ['sky-1/editor/1']
    print(trace.pause.questions[0].options)  # type: ignore[union-attr]  # ('Publish', 'Hold it')
    done = gated.resume("sky-1", {"sky-1/editor/1": {"approved": True, "who": "tom"}})
    print(done.status, done.output)  # ok Published: Sunlight scatters off air.

    assert capsys.readouterr().out.splitlines() == [
        "paused ['sky-1/editor/1']",
        "('Publish', 'Hold it')",
        f"ok Published: {DRAFT}",
    ]
    (record,) = done["editor"]
    assert (record.kind, record.cost_usd, record.inputs) == ("human_action", 0.0, {"draft": DRAFT})
    assert json.loads(record.raw_response)["who"] == "tom"
    assert record.wait_s is not None


def test_people_claims_without_a_block() -> None:
    """Mirrors docs/guide/people.md."""
    from xmas import HumanAction, ReplayAsker
    from xmas.human import ANSWER_FIELDS, COMMON_FIELDS, FORMS
    from xmas.llms.api_llm import APILLM

    assert FORMS == ("ask", "judge", "approve", "edit")
    assert ANSWER_FIELDS == {
        "ask": ("option",),
        "judge": ("pick",),
        "approve": ("approved", "option"),
        "edit": ("changed",),
    }
    assert COMMON_FIELDS == ("who", "asked_at", "answered_at", "comment", "timed_out")
    for form in FORMS:
        assert f"| `{form}` |" in PEOPLE
    for name in ("XMAS_RUNS", "trace.pause", "ReplayAsker(traces)", "withdraw(question, reason)", "stream=True"):
        assert name in PEOPLE, name
    assert inspect.signature(APILLM).parameters["stream"].default is False

    # a timeout takes the default; an unattended run takes it too, or fails by name
    def gate(**settings: Any) -> MAS:
        approve = HumanAction("approve", "approve", ["draft"], "approval", question="Ship {draft}?", **settings)
        return MAS(
            [Agent("writer", "Write {query}", "draft")],
            [("writer", "approve")],
            human_actions=[approve],
            llm=scripted(),
        )

    timed = gate(timeout_s=0, default={"approved": False}).run_state({"query": QUERY})
    assert timed.status == "ok"
    assert json.loads(timed["approve"][0].raw_response)["timed_out"] is True
    assert gate(default={"approved": True}).run(QUERY).status == "ok"
    failed = gate().run(QUERY)
    assert failed.status == "failed"
    assert "human action 'approve' got no answer" in str(failed.error)

    # an asker that replays recorded answers lets a system with people run unattended
    asked = gate()
    paused = asked.run_state({"query": QUERY}, run_id="r1")
    done = asked.resume("r1", {paused.pause.questions[0].id: {"approved": True, "who": "tom"}})  # type: ignore[union-attr]
    replayed = gate()
    replayed.asker = ReplayAsker([done])
    again = json.loads(replayed.run(QUERY).output)
    assert (again["approved"], again["who"]) == (True, "tom")


# ---------------------------------------------------------- docs/guide/people.md: Stop


def test_stop_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/people.md."""
    import threading

    llm = PollingLLM()

    essay = MAS(
        [Agent("writer", prompt="Write a long essay on: {query}", output="draft")],
        outputs={"draft": "text"},
        llm=llm,
    )
    threading.Timer(1.0, essay.stop, args=("essay-1",)).start()
    trace = essay.run_state({"query": "Why is the sky blue?"}, run_id="essay-1")
    print(trace.status, trace.error)  # stopped Stopped: stopped

    assert capsys.readouterr().out == "stopped Stopped: stopped\n"
    (record,) = trace.records
    assert (record.error_kind, record.raw_response) == (ErrorKind.STOPPED, "half")
    assert hasattr(xmas, "cancel_flag")


# ------------------------------------------------------ docs/guide/people.md: first of


def test_first_of_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/people.md."""
    llm = PollingLLM(lambda prompt: 3.0 if prompt.startswith("A careful") else 0.0)

    outline = Agent("outline", prompt="Outline an answer to: {query}", output="outline")
    quick = Agent("quick", prompt="One line from: {outline}", inputs=["outline"], output="draft")
    careful = Agent("careful", prompt="A careful answer from: {outline}", inputs=["outline"], output="draft")
    racing = MAS(
        [outline, quick, careful],
        edges=[Edge("outline", "quick", kind=EdgeKind.FIRST), Edge("outline", "careful", kind=EdgeKind.FIRST)],
        llm=llm,
    )

    trace = racing.run("Why is the sky blue?")
    print(sorted(record.agent for record in trace.records if record.lost))  # ['careful']
    print(trace.output)  # the quick draft

    assert capsys.readouterr().out.splitlines() == ["['careful']", trace["quick"][0].raw_response]
    assert trace["careful"][0].cost_usd is not None  # the loser paid for what it used


# ----------------------------------------------- docs/guide/bundles.md: Save and share


def test_version_1_save_block_runs_as_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors docs/guide/bundles.md."""
    monkeypatch.chdir(tmp_path)
    llm = quickstart_llm()
    intern, boss = quickstart_agents()

    team = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)  # the quickstart's system
    team.to_yaml("team.yaml")
    team = MAS.from_yaml("team.yaml", llm=llm)

    assert team("Why is the sky blue?") == "Rayleigh scattering."
    text = (tmp_path / "team.yaml").read_text(encoding="utf-8")
    assert "http" not in text
    assert "key" not in text.lower()


def test_version_2_save_block_runs_as_written(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Mirrors docs/guide/bundles.md."""
    llm = scripted()
    mas = looping_system(llm)

    spec = mas.to_dict()  # version, agents, code nodes, edges, callables, prompts
    rebuilt = MAS.from_dict(spec, llm=llm, tools=[count_words, publish, apologize])
    print(rebuilt.describe()["id"] == mas.describe()["id"])  # True

    assert capsys.readouterr().out == "True\n"
    assert list(spec) == ["version", "agents", "code_nodes", "edges", "callables", "prompts"]
    assert spec["version"] == 2
    shown = (
        "- writer -> count -> judge\n"
        "- judge -> writer: {when: not verdict.fits, max_visits: 3}\n"
        "- judge -> publish: {switch: verdict.action}\n"
        "- judge -> sorry: else\n"
        "- count -> sorry: {on error: [timeout, outage]}\n"
    )
    assert f"```yaml\n{shown}```" in BUNDLES
    assert yaml.safe_load(shown) == spec["edges"]
    from_yaml_edges = MAS.from_dict(
        {**spec, "edges": yaml.safe_load(shown)}, llm=llm, tools=[count_words, publish, apologize]
    )
    assert from_yaml_edges.describe()["id"] == mas.describe()["id"]
    assert "http" not in yaml.safe_dump(spec)
    assert "api_key" not in yaml.safe_dump(spec)
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    from_file = MAS.from_yaml(tmp_path / "spec.yaml", llm=llm, tools=[count_words, publish, apologize])
    assert (from_file.version, from_file.describe()["id"]) == (2, mas.describe()["id"])
    with pytest.raises(ValueError, match=r"\.mas bundle .*to_dict\(\) gives its spec"):
        mas.to_yaml(tmp_path / "team.yaml")


def block(fence: str, start: str) -> str:
    """Return the bundles page code block that opens with ``fence`` and whose text starts with ``start``."""
    return start + BUNDLES.split(f"```{fence}\n{start}", 1)[1].split("```", 1)[0]


def configured_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MockLLM:
    """The guide's model behind a configured endpoint: built by load, named in the endpoint's catalogue."""
    monkeypatch.chdir(tmp_path)
    for name in ("XMAS_BASE_URL", "XMAS_API_KEY", "XMAS_MODELS", "XMAS_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    llm = scripted()
    llm.full_response = True  # each call reports its cost
    llm.model_id = "llama-3.1-8b"  # type: ignore[attr-defined]  # the default model's name, as APILLM's

    def listed(config: Config, **_: Any) -> Catalogue:
        return Catalogue(("llama-3.1-8b", "text-model"), "endpoint")

    monkeypatch.setattr(xmas.config, "read_catalogue", listed)
    monkeypatch.setattr(xmas.verify, "read_catalogue", listed)
    monkeypatch.setattr(xmas.mas, "build_llms", lambda names, models, config: dict.fromkeys(names, llm))
    return llm


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MockLLM:
    """The guide's model behind a configured endpoint, in a fresh folder."""
    return configured_endpoint(monkeypatch, tmp_path)


def test_the_bundle_block_runs_as_written(endpoint: MockLLM, capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/bundles.md."""
    mas = looping_system(endpoint)

    mas.save("team.mas")  # save("team.mas", zipped=True) writes one zip file
    team = MAS.load("team.mas")
    print(team.describe()["id"] == mas.describe()["id"])  # True

    assert capsys.readouterr().out == "True\n"
    files = read("team.mas")
    assert sorted(files) == ["callables.yaml", "prompts/judge.md", "prompts/writer.md", "xmas.lock", "xmas.yaml"]
    listed = [line.split()[0] for line in block("text", "team.mas/").splitlines()[1:]]
    assert listed == ["xmas.yaml", "callables.yaml", "prompts/", "modules/", "examples/", "optimize.yaml", "xmas.lock"]
    assert yaml.safe_load(block("yaml", "version: 2\nagents:")) == yaml.safe_load(files["xmas.yaml"])
    shown = block("markdown", "---\nschema:")
    assert parse_prompt(shown, "judge.md") == parse_prompt(files["prompts/judge.md"].decode(), "judge.md")
    located = yaml.safe_load(block("yaml", "count_words:\n  kind: python"))
    saved = yaml.safe_load(files["callables.yaml"])
    assert located["count_words"] == {**saved["count_words"], "import": "mytools.text:count_words"}
    validate_spec_v2({**mas.to_dict(), "callables": {**saved, "web_search": located["web_search"]}})
    assert not any(b"://" in data or b"key" in data for data in files.values())

    entry = ModelEntry.parse(yaml.safe_load(block("yaml", "models:\n  editor:"))["models"]["editor"], "editor")
    editor = build_llms(["editor"], {"editor": entry}, Config(base_url="http://localhost:8000/v1"))["editor"]
    assert isinstance(editor, APILLM)
    assert (editor.model_id, editor.call_timeout_s) == ("llama-3.1-70b", 120)
    assert editor.call_params() == {"max_tokens": 2000, "temperature": 0.2}
    assert "`" + "`, `".join(PARAMS).replace(", `reasoning_effort", " and\n`reasoning_effort") + "`" in BUNDLES
    zipped = mas.save("team-zip.mas", zipped=True)
    assert zipped.is_file()
    assert MAS.load(zipped).describe()["id"] == mas.describe()["id"]
    (Path("team.mas") / "callables.yaml").write_text("# hand notes\n" + files["callables.yaml"].decode(), "utf-8")
    team.agents["writer"].prompt = "Write a very short answer to: {query}\nThe judge's last verdict: {verdict}"
    team.save("team.mas")  # only the changed prompt file and the lock are rewritten; the comment stays
    assert read("team.mas")["callables.yaml"].startswith(b"# hand notes\n")
    with pytest.raises(ValueError, match=r"save\(path\) writes it"):
        mas.to_yaml("team.yaml")


def test_the_verify_block_runs_as_written(endpoint: MockLLM, capsys: pytest.CaptureFixture[str]) -> None:
    """Mirrors docs/guide/bundles.md."""
    looping_system(endpoint).save("team.mas")
    team = MAS.load("team.mas")

    trace = team.run("Why is the sky blue?")
    team.save_examples(trace, nodes=["count", "judge"])
    print(team.verify(2))

    def steady(text: str) -> str:
        return re.sub(r"(<0\.1|\d+\.\d) s", "_ s", re.sub(r"bundle [0-9a-f]{16}", "bundle _", text))

    assert steady(capsys.readouterr().out) == steady(block("text", "level 0:"))
    hand = Path("hand.yaml")
    hand.write_text(block("yaml", "version: 1\nbudget:"), encoding="utf-8")
    problems: list[str] = []
    parse_examples({"examples/hand.yaml": hand.read_bytes()}, None, problems=problems)
    assert problems == []
    report = team.verify(2, examples=hand)
    assert report.ok
    assert [(r.name, r.status) for r in report.examples] == [("judge", "pass"), ("short-answer", "pass")]
    kinds = BUNDLES.split("A check is one of ", 1)[1].split(". Its `path`", 1)[0]
    assert {word.split(":")[0] for word in re.findall(r"`([^`]+)`", kinds)} - {"within"} == set(CHECK_KINDS)
    assert (BUDGET_USD, PER_EXAMPLE_USD) == (0.05, 0.02)
    assert [r.name for r in team.verify(2, only=["judge"]).examples] == ["judge"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MAS.load("team.mas", verify=1)
    (Path("team.mas") / "prompts" / "writer.md").write_text("Write: {query} {verdict}\n", encoding="utf-8")
    with pytest.warns(VerifyWarning, match="level 0: changed: prompts/writer.md"):
        team.verify(0)
    with pytest.raises(VerifyFailed, match="not ok: level 0 differs"):
        team.verify(0, strict=True)


def test_the_command_line_block_runs_as_written(
    endpoint: MockLLM, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mirrors docs/guide/bundles.md."""
    commands = [shlex.split(line.split("  #", 1)[0]) for line in block("bash", "xmas init").splitlines()]
    assert [command[:2] for command in commands] == [
        ["xmas", "init"],
        ["xmas", "check"],
        ["xmas", "show"],
        ["xmas", "run"],
        ["xmas", "diff"],
        ["xmas", "verify"],
    ]
    endpoint.responder = lambda prompt: "Rayleigh scattering."
    monkeypatch.setenv("XMAS_BASE_URL", "http://localhost:8000/v1")
    for command in commands:
        assert main(command[1:]) == 0, command
        if command[1] == "init":
            shutil.copytree("team.mas", "copy.mas")
    out = capsys.readouterr().out
    assert "ok: 1 node, 0 edges, 0 chains" in out
    assert "Rayleigh scattering." in out
    assert "same files" in out
    assert out.rstrip().endswith("ok: 1 example passed, 0.0100 USD")


def test_the_configuration_block_loads(tmp_path: Path) -> None:
    """Mirrors docs/guide/bundles.md."""
    path = tmp_path / "config.toml"
    path.write_text(block("toml", "[endpoint]"), encoding="utf-8")
    path.chmod(0o600)
    keys = {"GATEWAY_KEY": "sk-readme", "WEB_TOOLS_KEY": "w"}
    config = Config.load(path, environ=keys)
    assert (config.base_url, config.api_key, config.catalogue) == (
        "http://localhost:8000/v1",
        "sk-readme",
        ("llama-3.1-8b",),
    )
    assert (config.stream_of("llama-3.1-8b"), config.stream_of("other"), config.tools["web"].key) == (True, False, "w")
    assert (config.media_store, config.run_store) == (
        Path("~/.cache/xmas/store").expanduser(),
        Path("~/.cache/xmas/runs.sqlite").expanduser(),
    )
    assert Config.load(path, environ={**keys, "XMAS_BASE_URL": "http://env/v1"}).base_url == "http://env/v1"
    assert Config.load(path, environ=keys, base_url="http://given/v1").base_url == "http://given/v1"
    table = BUNDLES.split("| Variable | Meaning |", 1)[1].split("\n\n", 1)[0]
    named = set(re.findall(r"^\| `(XMAS_[A-Z_]+)`", table, flags=re.MULTILINE))
    assert named == {
        f"XMAS_{name}" for name in ("BASE_URL", "API_KEY", "MODELS", "CONFIG", "TOOLS_KEY", "STORE", "RUNS")
    }
    assert closest("llama-3-1-8b", ["llama-3.1-70b", "llama-3.1-8b"])[0] == "llama-3.1-8b"
    assert "sk-readme" not in repr(config)


# ----------------------------------- docs/guide/optimization.md: Optimization built in


def test_cantante_block_runs_as_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors docs/guide/optimization.md."""
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame({"question": ["What is 2 + 2?", "What is 1 + 3?"], "answer": ["#### 4", "#### 4"]})
    monkeypatch.setattr(GSM8KTask, "from_huggingface", classmethod(lambda cls, split, n: cls(frame.head(n))))
    llm = MockLLM(lambda prompt: "<prompt>Solve: {query}</prompt>" if "<prompt>" in prompt else "The answer is 4.")
    intern, boss = quickstart_agents()
    team = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)

    from promptolution.optimizers import OPRO

    from xmas.optimization import CantanteOptimizer, ContrastiveAttributor

    task = GSM8KTask.from_huggingface(split="train", n=50)  # needs: uv run --with datasets
    optimizer = CantanteOptimizer(team, task, ContrastiveAttributor(llm), OPRO, optimizer_kwargs={"meta_llm": llm})
    best = optimizer.optimize(n_steps=5)
    best.to_yaml("team-optimized.yaml")

    assert MAS.from_yaml("team-optimized.yaml", llm=llm).agents.keys() == {"intern", "boss"}


def test_a_detached_node_is_never_an_optimizer_target_and_makes_the_system_version_2() -> None:
    """Mirrors docs/guide/optimization.md."""
    from xmas.optimization.updaters import movable_agents

    intern, boss = quickstart_agents()
    boss.detached = True
    detached = MAS([intern, boss], edges=[("intern", "boss")], llm=quickstart_llm())

    assert movable_agents(detached) == ["intern"]
    assert detached.version == 2
    assert detached.detached == frozenset({"boss"})


# ---------------------------------- docs/guide/optimization.md: xmas and promptolution


def test_a_one_agent_system_encodes_to_its_bare_prompt_with_or_without_steps() -> None:
    """Mirrors docs/guide/optimization.md."""
    solver = Agent(
        "solver", prompt="Solve, using {count_words} words: {query}", inputs=["query", "count_words"], output="answer"
    )
    with_step = MAS(
        [solver],
        code_nodes=[Code("count", uses="count_words", reads={"draft": "query"})],
        edges=[("count", "solver")],
        tools=[count_words],
        llm=scripted(),
    )
    alone = MAS([Agent("solver", prompt="Solve: {query}", output="answer")], llm=scripted())

    assert with_step.encode() == solver.prompt
    assert alone.encode() == "Solve: {query}"


# ------------------------------------------------- claims without a block of their own


def test_error_kinds_the_guide_names_are_the_real_ones() -> None:
    """Mirrors docs/guide/conditions-and-loops.md."""
    assert {kind.value for kind in ErrorKind} == {
        "money",
        "rate_limited",
        "outage",
        "timeout",
        "length_cut",
        "bad_format",
        "refusal",
        "stopped",
        "budget",
        "unknown",
    }
    for kind in ("timeout", "outage", "rate_limited", "money", "length_cut", "bad_format", "refusal"):
        assert f"`{kind}`" in CONDITIONS


def test_nothing_the_guide_calls_planned_exists_yet() -> None:
    """Mirrors docs/guide/roadmap.md."""
    assert not any("preference" in name.lower() for name in [*dir(xmas), *dir(xmas.tasks)])
    assert "langgraph" not in " ".join(inspect.signature(MAS).parameters).lower()


def test_the_guide_keeps_the_rulings() -> None:
    """Mirrors every page under docs/guide."""
    pages = {path.name: path.read_text(encoding="utf-8") for path in sorted(GUIDE.glob("*.md"))}
    assert len(pages) == 13
    for name, text in pages.items():
        assert ".xmas" not in text, name
        assert "langgraph" not in text.lower(), name
        for block in text.split("```yaml")[1:]:
            assert "http" not in block.split("```", 1)[0], name
            assert "key" not in block.split("```", 1)[0], name
    assert "name.mas/" in BUNDLES
    assert "> **Planned, not in the code yet.**" in ROADMAP
