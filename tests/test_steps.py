"""Version 2 in the engine: typed values, steps, one callable registry, answer forms with one retry.

No network: the models are scripted doubles.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from xmas import MAS, Agent, Budget, ErrorKind, TokenUsage, ToolCall
from xmas.edges import Edge, EdgeKind
from xmas.errors import EmptyResponse, XmasError
from xmas.nodes import Step
from xmas.schemas import RETRY_NOTE
from xmas.tools import Tool, ToolSource, index_tools, tool
from xmas.trace import Rejected
from xmas.values import JSON, TEXT

from .conftest import META, MockLLM, req


@tool(writes={"counts": "json"})
def count_points(plan: dict[str, Any]) -> dict[str, int]:
    """Count a plan's points and their words."""
    return {"points": len(plan["points"]), "words": sum(len(point.split()) for point in plan["points"])}


@tool
def shout(text: str) -> str:
    """Shout a text."""
    return text.upper() + "!"


FORM = {"topic": {"type": "text"}, "points": {"type": "list of text"}}
FORMAT = "Answer with one JSON object and nothing else. Its fields:\n- topic (text)\n- points (list of text)"
PLAN = {"topic": "cats", "points": ["sleep a lot", "purr"]}
PLAN_TEXT = '{"topic": "cats", "points": ["sleep a lot", "purr"]}'
COUNTS_TEXT = '{"points": 2, "words": 4}'


def planned_note(**kwargs: Any) -> tuple[MAS, MockLLM]:
    """A planner answering a json plan, the step ``count`` counting its points, and a writer reading both."""

    def answer(prompt: str) -> str:
        return "```json\n" + json.dumps(PLAN) + "\n```" if prompt.startswith("Plan") else "A note."

    llm = MockLLM(answer)
    planner = Agent("planner", "Plan a note on {query}.", output="plan", output_type="json", schema=FORM)
    writer = Agent("writer", "Write about {plan} using {counts}.", output="note", inputs=["plan", "counts"])
    mas = MAS(
        [planner, writer],
        # writer is fed by planner and by count, at unequal depth: a barrier, so it runs once
        [("planner", "count"), ("count", "writer"), ("planner", "writer")],
        steps=[Step("count", "count_points")],
        llm=llm,
        tools=[count_points],
        **kwargs,
    )
    return mas, llm


def test_a_step_runs_between_agents_with_typed_values() -> None:
    mas, llm = planned_note()
    assert count_points.reads == {"plan": JSON}
    assert count_points.writes == {"counts": JSON}
    trace = mas.run("cats")

    assert trace.error is None
    assert [(r.agent, r.kind) for r in trace.records] == [("planner", "agent"), ("count", "step"), ("writer", "agent")]
    assert llm.calls == [f"Plan a note on cats.\n\n{FORMAT}", f"Write about {PLAN_TEXT} using {COUNTS_TEXT}."]
    planner, step, writer = trace.records
    assert planner.outputs == {"plan": PLAN_TEXT}
    assert planner.raw_response.startswith("```json")
    assert step.prompt == step.rendered_prompt == ""
    assert step.inputs == {"plan": PLAN_TEXT}
    assert step.outputs == {"counts": COUNTS_TEXT}
    assert step.raw_response == COUNTS_TEXT
    assert step.model is None
    assert step.cost_usd == 0.0
    assert step.latency_s is not None
    assert writer.inputs == {"plan": PLAN_TEXT, "counts": COUNTS_TEXT}
    assert trace.output == "A note."
    assert trace.to_dict()["records"][1]["kind"] == "step"


def test_a_step_spends_no_model_call() -> None:
    """With one call allowed, the step still runs, and the writer is refused."""
    limited, _ = planned_note(budget=Budget(max_calls=1))
    failed = limited.run("cats")
    assert failed.error_kind == ErrorKind.BUDGET
    assert [r.agent for r in failed.records] == ["planner", "count"]


def test_a_system_whose_inputs_are_not_a_query_does_not_run_on_one() -> None:
    boxed, _ = planned_note(inputs={"query": TEXT, "style": TEXT})
    with pytest.raises(ValueError, match=r"only runs on one text query; use it as a box\."):
        boxed.run("cats")


def test_one_callable_is_a_tool_and_a_step() -> None:
    llm = MockLLM(turns=[("", [req("shout", '{"text": "hi"}')]), ("said hi", [])])
    writer = Agent("writer", "Greet {query}.", output="draft", tools=["shout"])
    loud = Step("loud", "shout", reads={"text": "draft"}, writes={"shout": "loud"})
    mas = MAS([writer], [("writer", "loud")], steps=[loud], llm=llm, tools=[shout])
    assert list(index_tools(mas.tools)) == ["shout"]
    assert shout.reads == {"text": TEXT}
    assert shout.writes == {"shout": TEXT}

    trace = mas.run("Tom")
    assert trace.error is None
    called, stepped = trace.records
    assert called.tool_calls == (ToolCall("shout", {"text": "hi"}, "HI!"),)
    assert (stepped.kind, stepped.inputs, stepped.outputs) == ("step", {"draft": "said hi"}, {"loud": "SAID HI!"})
    assert trace.output == "SAID HI!"


VERDICT = {"fits": {"type": "bool", "means": "the draft fits"}}
RENDERED = (
    "Judge this.\n\nAnswer with one JSON object and nothing else. Its fields:\n- fits (true or false): the draft fits"
)


def judge_mas(llm: MockLLM, tools: list[Any] | None = None) -> MAS:
    """A judge answering in the form ``VERDICT``, with ``shout`` as a tool when ``tools`` offers it."""
    judge = Agent(
        "judge",
        "Judge {query}.",
        output="verdict",
        output_type="json",
        schema=VERDICT,
        tools=["shout"] if tools else None,
    )
    return MAS([judge], [], llm=llm, tools=tools or [], budget=Budget(max_calls=10))


def calls_spent(trace: Any) -> int:
    """The model calls the run's meter counted."""
    assert trace.health is not None
    spent: int = trace.health.budget["calls"]["spent"]
    return spent


def test_an_answer_outside_the_form_is_asked_once_more() -> None:
    llm = MockLLM(answers=['Sure: {"fits": true}', '{"fits": true}'], meta=True)
    trace = judge_mas(llm).run("this")
    assert trace.error is None
    assert trace.output == '{"fits": true}'
    (record,) = trace.records
    (rejected,) = record.rejected
    assert rejected.raw_response == 'Sure: {"fits": true}'
    assert re.fullmatch(r"<root>: .+", rejected.error)
    assert llm.calls == [RENDERED, f"{RENDERED}\n\n{RETRY_NOTE.format(problem=rejected.error)}"]
    assert record.raw_response == '{"fits": true}'
    assert record.rendered_prompt == RENDERED
    assert record.outputs == {"verdict": '{"fits": true}'}
    assert record.cost_usd == pytest.approx(0.02)
    assert record.tokens == TokenUsage(20, 4)
    assert calls_spent(trace) == 2
    assert trace.to_dict()["records"][0]["rejected"] == [
        {"raw_response": 'Sure: {"fits": true}', "error": rejected.error}
    ]


def test_a_second_answer_outside_the_form_is_a_bad_format() -> None:
    trace = judge_mas(MockLLM(answers=["Sure.", '{"fits": "perhaps"}'], meta=True)).run("this")
    assert trace.error_kind == ErrorKind.BAD_FORMAT
    assert trace.error is not None
    assert re.fullmatch(r"BadFormat: Bad format from 'judge': <root>: .+; after the retry: fits: .+", trace.error)
    assert calls_spent(trace) == 2
    (failed,) = trace.records  # both paid attempts stay in the judge's failed record
    assert (failed.agent, failed.error_kind) == ("judge", ErrorKind.BAD_FORMAT)
    assert failed.cost_usd == pytest.approx(0.02)


def test_the_retry_passes_the_meter() -> None:
    """With one call allowed, the retry is refused."""
    limited = judge_mas(MockLLM(answers=["Sure.", '{"fits": true}'], meta=True))
    limited.budget = Budget(max_calls=1)
    assert limited.run("this").error_kind == ErrorKind.BUDGET


def test_a_tool_agent_answers_outside_the_form_and_gets_one_more_turn() -> None:
    llm = MockLLM(meta=True, turns=[("", [req("shout", '{"text": "hi"}')]), ("Sure.", []), ('{"fits": true}', [])])
    trace = judge_mas(llm, tools=[shout]).run("this")
    assert trace.error is None
    assert [names for _, names in llm.seen] == [["shout"], ["shout"], []]
    (record,) = trace.records
    (rejected,) = record.rejected
    assert rejected == Rejected("Sure.", rejected.error)
    assert re.fullmatch(r"<root>: .+", rejected.error)
    last = llm.seen[2][0]
    assert last[1] == {"role": "user", "content": RENDERED}
    assert last[-2:] == [
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": RETRY_NOTE.format(problem=rejected.error)},
    ]
    assert record.tool_calls == (ToolCall("shout", {"text": "hi"}, "HI!"),)
    assert record.raw_response == '{"fits": true}'
    assert calls_spent(trace) == 3
    assert record.cost_usd == pytest.approx(0.03)

    # a second miss fails the run with both problems
    again = judge_mas(MockLLM(meta=True, turns=[("Sure.", []), ("Sure!", [])]), tools=[shout]).run("this")
    assert again.error_kind == ErrorKind.BAD_FORMAT
    assert again.error is not None
    assert re.fullmatch(r"BadFormat: Bad format from 'judge': <root>: .+; after the retry: <root>: .+", again.error)


def test_a_step_costs_nothing() -> None:
    loud = Agent("writer", "Write {query}", output="draft")
    mas = MAS(
        [loud],
        [("writer", "shout")],
        steps=[Step("shout", "shout", {"text": "draft"})],
        tools=[shout],
        llm=MockLLM(answers=["a draft"], meta=True),
    )
    trace = mas.run("cats")
    assert [(r.agent, r.cost_usd) for r in trace.records] == [("writer", 0.01), ("shout", 0.0)]
    assert trace.cost_usd == pytest.approx(0.01)
    assert trace.output == "A DRAFT!"


class Refusing(ToolSource):
    """Offers shout, and refuses every call of it."""

    def tools(self) -> list[Tool]:
        """Offer shout."""
        return [Tool("shout", "Shout.", {"type": "object", "properties": {"text": {"type": "string"}}})]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Refuse the call."""
        raise XmasError("forbidden (403)", status=403)


CUT = EmptyResponse("Empty content from model", kind=ErrorKind.LENGTH_CUT, meta=META)


@pytest.mark.parametrize(
    ("judge", "llm", "tools", "kind", "paid"),
    [
        pytest.param(
            Agent("judge", "Judge {query}", output="verdict", output_type="json", schema={"fits": {"type": "bool"}}),
            MockLLM(
                answers=[
                    "Sure.",
                    EmptyResponse("Empty content from model", kind=ErrorKind.LENGTH_CUT, meta=META),
                    "Sorry.",
                ],
                meta=True,
            ),
            [],
            ErrorKind.LENGTH_CUT,
            0.02,
            id="retry cut",
        ),
        pytest.param(
            Agent("judge", "Judge {query}", output="verdict", tools=["shout"]),
            MockLLM(answers=["Sorry."], meta=True, turns=[("", [req("shout", '{"text": "hi"}')])]),
            [Refusing()],
            ErrorKind.REFUSAL,
            0.01,
            id="tool refused",
        ),
    ],
)
def test_a_failed_visit_keeps_every_paid_call(
    judge: Agent, llm: MockLLM, tools: list[Any], kind: ErrorKind, paid: float
) -> None:
    """The failed record holds the retries and tool turns of the visit, and the trace costs what the meter paid."""
    sorry = Agent("sorry", "Sorry about {query}", output="said")
    error = Edge("judge", "sorry", EdgeKind.ERROR, errors=("any",))
    trace = MAS([judge, sorry], [error], llm=llm, tools=tools, budget=Budget(max_cost_usd=1.0)).run("cats")
    failed, apology = trace.records
    assert (failed.agent, failed.error_kind, failed.route) == ("judge", kind, ("sorry",))
    calls = round(paid / 0.01)
    assert failed.cost_usd == pytest.approx(paid)
    assert failed.tokens == TokenUsage(10 * calls, 2 * calls)
    assert apology.cost_usd == pytest.approx(0.01)
    assert trace.cost_usd == pytest.approx(paid + 0.01)
    assert trace.health is not None
    assert trace.health.budget["cost_usd"]["spent"] == pytest.approx(paid + 0.01)


def test_an_optional_text_parameter_reads_text() -> None:
    @tool
    def tidy(draft: str | None = None) -> str:
        """Tidy a draft."""
        return (draft or "").strip().upper()

    assert tidy.reads == {"draft": TEXT}  # so a text value may feed it
    writer = Agent("writer", "Write {query}", output="draft")
    steps = [Step("tidy", "tidy")]
    mas = MAS([writer], [("writer", "tidy")], steps=steps, tools=[tidy], llm=MockLLM(answers=[" cats "], meta=True))
    assert mas.run("q").output == "CATS"


def test_a_json_null_is_a_value_and_renders_as_null() -> None:
    """A value not there yet renders empty instead: test_operators."""
    judge = Agent("judge", "Judge {query}", output="verdict", output_type="json")
    reader = Agent("reader", "Read {verdict}", output="note", inputs=["verdict"])
    llm = MockLLM(answers=["null", "noted"], meta=True)
    trace = MAS([judge, reader], [("judge", "reader")], llm=llm, inputs={"query": "text"}).run("q")
    assert trace["judge"][0].outputs == {"verdict": "null"}
    assert trace["reader"][0].inputs == {"verdict": "null"}
    assert llm.calls[1] == "Read null"
    alone = MAS([judge], llm=MockLLM(answers=["null"], meta=True), inputs={"query": "text"}).run("q")
    assert alone.output == "null"
    assert alone.health is not None
    assert alone.health.flags == ()
