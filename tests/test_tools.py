"""xmas.tools: describing tools, resolving them in a MAS, and the per-agent tool loop.

No network: the model is a scripted double and the tool source a recording one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Annotated, Any

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion
from pydantic import Field

from xmas import END, MAS, START, Agent, ToolCall
from xmas.llms.api_llm import APILLM, RateLimitedError
from xmas.llms.telemetry import CallLog, timestamp
from xmas.optimization.graph_updater import GraphUpdater
from xmas.optimization.synthesis import MechanicalSynthesizer
from xmas.optimization.topology import TopologyBudget
from xmas.tools import (
    LocalTools,
    SupportsToolCalls,
    Tool,
    ToolAccessError,
    ToolCallRequest,
    ToolError,
    ToolSource,
    tool,
)

from .conftest import MockLLM

# --------------------------------------------------------------------- doubles

Turn = tuple[str, list[ToolCallRequest]]


class ToolMockLLM(MockLLM):
    """A MockLLM that can also take tool turns, scripted or produced by a function."""

    def __init__(self, turns: list[Turn] | None = None, default: Turn = ("done", [])) -> None:
        """Script the tool turns; ``default`` answers once the script runs out."""
        super().__init__()
        self.turns = list(turns or [])
        self.default = default
        self.seen: list[tuple[list[dict[str, Any]], list[str]]] = []

    def get_tool_response(self, messages: list[dict[str, Any]], tools: Sequence[Tool]) -> Turn:
        """Record a snapshot of the history and the tools offered, and answer from the script."""
        self.seen.append((json.loads(json.dumps(messages)), [t.name for t in tools]))
        return self.turns.pop(0) if self.turns else self.default


class FakeToolSource:
    """A ToolSource with scripted answers; records every call."""

    def __init__(self, names: Sequence[str] = ("run_python",), answers: dict[str, Any] | None = None) -> None:
        """Offer ``names``; ``answers`` maps a name to its result, or to an exception to raise."""
        self.specs = [Tool(name, f"{name} tool", {"type": "object", "properties": {}}) for name in names]
        self.answers = answers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.listed = 0

    def tools(self) -> list[Tool]:
        """Return the tools, counting how often they were listed."""
        self.listed += 1
        return list(self.specs)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Record the call and answer it."""
        self.calls.append((name, arguments))
        answer = self.answers.get(name, f"{name} ok")
        if isinstance(answer, Exception):
            raise answer
        return str(answer)


def req(name: str, arguments: str = "{}", call_id: str | None = None) -> ToolCallRequest:
    return ToolCallRequest(call_id or f"call_{name}", name, arguments)


def coder(tools: list[str] | None = None) -> Agent:
    return Agent("coder", "Compute {query}", output="answer", tools=["run_python"] if tools is None else tools)


# -------------------------------------------------------------- schema: @tool


@tool
def add(a: int, b: int = 2) -> int:
    """Add two integers.

    Longer prose that is not part of the description.

    Args:
        a: The first summand.
        b: The second summand,
            continued on a second line.

    Returns:
        The sum.
    """
    return a + b


def test_tool_derives_name_description_and_schema() -> None:
    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two integers."
    assert add.parameters == {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "The first summand."},
            "b": {"type": "integer", "default": 2, "description": "The second summand, continued on a second line."},
        },
        "required": ["a"],
        "additionalProperties": False,
    }


def test_tool_keeps_annotated_constraints_and_takes_a_name() -> None:
    @tool(name="clip")
    def clip_value(json: Annotated[int, Field(ge=0)], schema: str = "x") -> str:
        """Clip a value."""
        return f"{json}{schema}"

    assert clip_value.name == "clip"
    assert clip_value.parameters["properties"]["json"] == {"type": "integer", "minimum": 0}
    assert clip_value.parameters["required"] == ["json"]
    assert clip_value.validate({"json": "3"}) == {"json": 3, "schema": "x"}  # names that shadow BaseModel work
    with pytest.raises(ToolError, match="json"):
        clip_value.validate({"json": -1})


def test_decorated_function_stays_callable() -> None:
    assert add(1) == 3
    assert add(a=1, b=5) == 6
    with pytest.raises(TypeError, match="remote"):
        Tool("remote", "r", {"type": "object"})()


def _no_doc(a: int) -> int:
    return a


def _no_hint(a):  # type: ignore[no-untyped-def]
    """Doc."""
    return a


def _star_args(*a: int) -> int:
    """Doc."""
    return 0


def _star_kwargs(**a: int) -> int:
    """Doc."""
    return 0


def _positional(a: int, /) -> int:
    """Doc."""
    return a


def _unschemable(a: object) -> int:
    """Doc."""
    return 0


@pytest.mark.parametrize(
    ("fn", "message"),
    [
        (_no_doc, "docstring"),
        (_no_hint, "no type hint"),
        (_star_args, r"\*args"),
        (_star_kwargs, r"\*\*kwargs"),
        (_positional, "positional-only"),
    ],
)
def test_tool_rejects_what_it_cannot_describe(fn: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tool(fn)


def test_tool_rejects_a_name_openai_would_refuse() -> None:
    with pytest.raises(ValueError, match="must match"):
        tool(name="has space")(add.fn)  # type: ignore[arg-type]


def test_to_openai_shape() -> None:
    assert add.to_openai() == {
        "type": "function",
        "function": {"name": "add", "description": "Add two integers.", "parameters": add.parameters},
    }


def test_validate_coerces_and_reports() -> None:
    assert add.validate({"a": "4"}) == {"a": 4, "b": 2}
    with pytest.raises(ToolError, match="invalid arguments for add"):
        add.validate({"a": "four"})
    with pytest.raises(ToolError, match="c: Extra inputs"):
        add.validate({"a": 1, "c": 3})
    remote = Tool(
        "r", "r", {"type": "object", "properties": {"x": {}}, "required": ["x"], "additionalProperties": False}
    )
    with pytest.raises(ToolError, match="missing"):
        remote.validate({})
    with pytest.raises(ToolError, match="unexpected"):
        remote.validate({"x": 1, "y": 2})
    assert remote.validate({"x": 1}) == {"x": 1}


# -------------------------------------------------------------- LocalTools


def test_local_tools_stringify_results_and_wrap_exceptions() -> None:
    @tool
    def listing(n: int) -> list[int]:
        """List numbers."""
        return list(range(n))

    @tool
    def boom() -> str:
        """Fail."""
        raise KeyError("nope")

    source = LocalTools([add, listing, boom])
    assert isinstance(source, ToolSource)
    assert source.call("add", {"a": 1}) == "3"
    assert source.call("listing", {"n": 3}) == "[0, 1, 2]"
    with pytest.raises(ToolError, match="KeyError"):
        source.call("boom", {})
    with pytest.raises(ToolError, match="invalid arguments"):
        source.call("add", {})
    with pytest.raises(ToolError, match="unknown tool"):
        source.call("nope", {})


def test_local_tools_reject_remote_tools_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="no local implementation"):
        LocalTools([Tool("r", "r", {})])

    @tool(name="add")
    def other(a: int) -> int:
        """Other add."""
        return a

    with pytest.raises(ValueError, match="Two local tools"):
        LocalTools([add, other])


# ------------------------------------------------------- resolution in a MAS


def test_unknown_tool_name_fails_at_construction() -> None:
    with pytest.raises(ValueError, match=r"unknown tools \['nope'\]; available: \['run_python'\]"):
        MAS(agents=[coder(["nope"])], llm=ToolMockLLM(), tools=[FakeToolSource()])


def test_tool_names_need_a_source_at_all() -> None:
    with pytest.raises(ValueError, match="unknown tools"):
        MAS(agents=[coder()], llm=ToolMockLLM())


def test_duplicate_tool_names_across_sources_fail() -> None:
    with pytest.raises(ValueError, match="offered by both"):
        MAS(agents=[coder()], llm=ToolMockLLM(), tools=[FakeToolSource(), FakeToolSource()])

    @tool(name="run_python")
    def local_python(code: str) -> str:
        """Run locally."""
        return code

    with pytest.raises(ValueError, match="offered by both"):
        MAS(agents=[coder()], llm=ToolMockLLM(), tools=[local_python, FakeToolSource()])


def test_the_same_object_listed_twice_counts_once() -> None:
    source = FakeToolSource()
    mas = MAS(agents=[coder(["run_python", "add"])], llm=ToolMockLLM(), tools=[source, add, source, add])
    assert set(mas.resolve_tools(mas.agents["coder"])) == {"run_python", "add"}


def test_local_and_remote_tools_mix() -> None:
    source = FakeToolSource()
    llm = ToolMockLLM([("", [req("add", '{"a": 40}'), req("run_python")])])
    mas = MAS(agents=[coder(["add", "run_python"])], llm=llm, tools=[add, source])
    trace = mas.run("q")
    assert trace.error is None
    assert [c.result for c in trace["coder"][0].tool_calls] == ["42", "run_python ok"]


def test_a_model_without_tool_calls_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="get_tool_response"):
        MAS(agents=[coder()], llm=MockLLM(), tools=[FakeToolSource()])


def test_a_mas_without_tool_agents_never_touches_its_sources() -> None:
    source = FakeToolSource()
    mas = MAS(agents=[Agent("a", "{query}", output="out")], llm=MockLLM(), tools=[source])
    assert mas.run("q").output == "echo:q"
    assert source.listed == 0


def test_bad_tools_items_and_rounds_are_rejected() -> None:
    with pytest.raises(TypeError, match="Tool objects and tool sources"):
        MAS(agents=[coder()], llm=ToolMockLLM(), tools=["run_python"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="max_tool_rounds"):
        MAS(agents=[coder()], llm=ToolMockLLM(), tools=[FakeToolSource()], max_tool_rounds=-1)


def test_copy_from_dict_and_rebuild_keep_tools() -> None:
    source = FakeToolSource()
    mas = MAS(agents=[coder()], llm=ToolMockLLM(), tools=[source], max_tool_rounds=2)

    copied = mas.copy(prompts={"coder": "Again {query}"})
    assert copied.tools == (source,)
    assert copied.max_tool_rounds == 2

    restored = MAS.from_dict(dict(mas.to_dict()), llm=ToolMockLLM(), tools=[source])
    assert restored.agents["coder"].tools == ["run_python"]
    with pytest.raises(ValueError, match="unknown tools"):
        MAS.from_dict(dict(mas.to_dict()), llm=ToolMockLLM())

    updater = GraphUpdater(MechanicalSynthesizer(), TopologyBudget(min_agents=1, max_agents=3, max_edges=6))
    rebuilt = updater.rebuild(mas, ["coder", "helper"], [("helper", "coder")])
    assert rebuilt.agents["coder"].tools == ["run_python"]
    assert rebuilt.agents["helper"].tools == []
    assert rebuilt.tools == (source,)
    assert rebuilt.max_tool_rounds == 2


def test_from_yaml_takes_tools(tmp_path: Any) -> None:
    source = FakeToolSource()
    mas = MAS(agents=[coder()], llm=ToolMockLLM(), tools=[source])
    path = tmp_path / "mas.yaml"
    mas.to_yaml(path)
    assert MAS.from_yaml(path, llm=ToolMockLLM(), tools=[source]).agents["coder"].tools == ["run_python"]


def test_editing_agent_tools_recompiles() -> None:
    source = FakeToolSource()
    llm = ToolMockLLM(default=("final", []))
    mas = MAS(agents=[Agent("coder", "{query}", output="answer")], llm=llm, tools=[source])
    assert mas.run("q").output == "echo:q"
    before = mas.structure
    mas.agents["coder"].tools = ["run_python"]
    assert mas.structure != before
    assert mas.run("q").output == "final"


# ---------------------------------------------------------------- tool loop


def test_one_round_feeds_the_result_back_as_a_tool_message() -> None:
    source = FakeToolSource(answers={"run_python": "1267650600228229401496703205376"})
    llm = ToolMockLLM([("thinking", [req("run_python", '{"code": "print(2**100)"}', "c1")]), ("2**100 is big", [])])
    trace = MAS(agents=[coder()], llm=llm, tools=[source]).run("2**100")

    assert trace.error is None
    assert trace.output == "2**100 is big"
    record = trace["coder"][0]
    assert record.tool_calls == (ToolCall("run_python", {"code": "print(2**100)"}, "1267650600228229401496703205376"),)
    assert record.raw_response == "2**100 is big"
    assert record.rendered_prompt == "Compute 2**100"
    assert record.usage.input_tokens > 0

    first, second = llm.seen
    assert first[1] == ["run_python"]
    assert [m["role"] for m in first[0]] == ["system", "user"]
    assert first[0][1]["content"] == "Compute 2**100"
    history = second[0]
    assert history[2] == {
        "role": "assistant",
        "content": "thinking",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "run_python", "arguments": '{"code": "print(2**100)"}'},
            }
        ],
    }
    assert history[3] == {"role": "tool", "tool_call_id": "c1", "content": "1267650600228229401496703205376"}
    assert source.calls == [("run_python", {"code": "print(2**100)"})]


def test_calls_in_one_round_run_in_order() -> None:
    source = FakeToolSource(names=("a", "b"))
    llm = ToolMockLLM([("", [req("b", call_id="1"), req("a", call_id="2")])])
    trace = MAS(agents=[coder(["a", "b"])], llm=llm, tools=[source]).run("q")
    assert [name for name, _ in source.calls] == ["b", "a"]
    assert [c.tool for c in trace["coder"][0].tool_calls] == ["b", "a"]
    assert llm.seen[1][0][2]["content"] is None  # an empty assistant text goes out as null
    assert [m["tool_call_id"] for m in llm.seen[1][0][3:]] == ["1", "2"]


@pytest.mark.parametrize("rounds", [5, 2, 0])
def test_endless_requests_stop_after_max_rounds_plus_a_final_turn_without_tools(rounds: int) -> None:
    llm = ToolMockLLM(default=("again", [req("run_python")]))
    source = FakeToolSource()
    trace = MAS(agents=[coder()], llm=llm, tools=[source], max_tool_rounds=rounds).run("q")

    assert trace.error is None
    assert trace.output == "again"
    assert [tools for _, tools in llm.seen] == [["run_python"]] * rounds + [[]]
    assert len(trace["coder"][0].tool_calls) == rounds
    assert len(source.calls) == rounds


def test_default_is_five_rounds() -> None:
    llm = ToolMockLLM(default=("again", [req("run_python")]))
    MAS(agents=[coder()], llm=llm, tools=[FakeToolSource()]).run("q")
    assert len(llm.seen) == 6


def test_failed_calls_are_fed_back_and_the_run_goes_on() -> None:
    source = FakeToolSource(answers={"run_python": ToolError("exit_code=1 killed=none\nTraceback")})
    llm = ToolMockLLM(
        [
            ("", [req("run_python", "{not json", "1"), req("run_python", "[1]", "2")]),
            ("", [req("nope", "{}", "3"), req("run_python", '{"code": "x"}', "4")]),
            ("gave up", []),
        ]
    )
    trace = MAS(agents=[coder()], llm=llm, tools=[source]).run("q")

    assert trace.error is None
    assert trace.output == "gave up"
    calls = trace["coder"][0].tool_calls
    assert all(call.failed for call in calls)
    assert calls[0].arguments == {"_raw": "{not json"}
    assert calls[0].result.startswith("invalid JSON arguments")
    assert calls[1].result == "invalid JSON arguments: not an object"
    assert calls[2].result == "unknown tool; available: run_python"
    assert calls[3] == ToolCall("run_python", {"code": "x"}, "exit_code=1 killed=none\nTraceback", failed=True)
    fed_back = [m["content"] for m in llm.seen[2][0] if m["role"] == "tool"]
    assert fed_back == [call.result for call in calls]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ToolAccessError("tool_not_in_scope: nope"), "ToolAccessError: tool_not_in_scope: nope"),
        (RateLimitedError("too_many_concurrent: 4 in flight", retry_after=1.0), "RateLimitedError: too_many"),
    ],
)
def test_refusals_end_the_run(error: Exception, expected: str) -> None:
    source = FakeToolSource(answers={"run_python": error})
    llm = ToolMockLLM([("", [req("run_python")])])
    mas = MAS(agents=[coder()], llm=llm, tools=[source])

    trace = mas.run("q")
    assert trace.error is not None
    assert trace.error.startswith(expected)
    assert trace.records == ()

    llm.turns = [("", [req("run_python")])]
    with pytest.raises(RuntimeError, match="MAS run failed"):
        mas("q")


def test_tools_inside_a_cycle_start_fresh_each_visit() -> None:
    llm = ToolMockLLM()
    llm.turns = [("", [req("run_python")]), ("<continue>", [])] * 5
    agents = [
        Agent("solver", "Solve {query} {feedback}", output="draft", inputs=["query", "feedback"]),
        Agent("checker", "Check {draft}", output="feedback", inputs=["draft"], tools=["run_python"]),
    ]
    edges = [(START, "solver"), ("solver", "checker"), ("checker", "solver"), ("checker", END)]
    mas = MAS(agents=agents, edges=edges, llm=llm, tools=[FakeToolSource()], max_steps=50)
    trace = mas.run("q")

    assert trace.error is None
    visits = trace["checker"]
    assert len(visits) == 3  # MAX_NODE_VISITS still bounds the cycle
    assert all(len(record.tool_calls) == 1 for record in visits)
    first_turns = [messages for messages, _ in llm.seen if len(messages) == 2]
    assert len(first_turns) == 3  # every visit opens with system + user only


def test_a_plain_agent_in_a_mixed_mas_sends_exactly_what_it_did_before() -> None:
    def build(llm: MockLLM, with_tools: bool) -> MAS:
        agents = [
            Agent("planner", "Plan {query}", output="plan"),
            Agent("coder", "Do {plan}", output="answer", inputs=["plan"], tools=["run_python"] if with_tools else []),
        ]
        return MAS(agents=agents, edges=[("planner", "coder")], llm=llm, tools=[FakeToolSource()])

    plain, mixed = MockLLM(), ToolMockLLM(default=("tool answer", []))
    plain_trace = build(plain, with_tools=False).run("q")
    mixed_trace = build(mixed, with_tools=True).run("q")
    assert mixed.calls == plain.calls[:1] == ["Plan q"]
    assert mixed_trace["planner"][0] == plain_trace["planner"][0]
    assert mixed.seen[0][0][1]["content"] == f"Do {plain_trace['planner'][0].raw_response}"


def test_run_batch_with_tools() -> None:
    source = FakeToolSource()
    llm = ToolMockLLM(default=("done", []))
    traces = MAS(agents=[coder()], llm=llm, tools=[source]).run_batch(["a", "b", "c"])
    assert [t.output for t in traces] == ["done"] * 3
    assert all(t.error is None for t in traces)


# ------------------------------------------------------ APILLM.get_tool_response


def completion(content: str | None, calls: list[dict[str, Any]] | None = None, usage: bool = True) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    body: dict[str, Any] = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    return ChatCompletion.model_validate(body)


def api_llm(monkeypatch: pytest.MonkeyPatch, answers: list[Any]) -> tuple[APILLM, list[dict[str, Any]], CallLog]:
    log = CallLog()
    llm = APILLM(model="m", base_url="https://gateway.example/v1", api_key="k", telemetry=log, retry_base_delay_s=0)
    sent: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> ChatCompletion:
        sent.append(kwargs)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        assert isinstance(answer, ChatCompletion)
        return answer

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    return llm, sent, log


def test_api_llm_offers_tools_and_parses_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}
    llm, sent, log = api_llm(monkeypatch, [completion(None, [call])])
    assert isinstance(llm, SupportsToolCalls)

    messages = [{"role": "user", "content": "hi"}]
    text, requests = llm.get_tool_response(messages, [add])
    assert text == ""
    assert requests == [ToolCallRequest("c1", "add", '{"a": 1}')]
    assert sent[0]["tools"] == [add.to_openai()]
    assert sent[0]["messages"] == messages
    assert sent[0]["model"] == "m"
    assert "tool_choice" not in sent[0]
    record = log.records[0]
    assert (record.ok, record.input_tokens, record.output_tokens) == (True, 11, 7)


def test_api_llm_omits_tools_on_the_final_turn_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, sent, _ = api_llm(monkeypatch, [RuntimeError("flaky"), completion("42", usage=False)])
    text, requests = llm.get_tool_response([{"role": "user", "content": "hi there"}], [])
    assert (text, requests) == ("42", [])
    assert len(sent) == 2
    assert "tools" not in sent[1]


def test_api_llm_records_a_failed_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, _, log = api_llm(monkeypatch, [RuntimeError("down")] * 6)
    with pytest.raises(RuntimeError, match="down"):
        llm.get_tool_response([{"role": "user", "content": "hi"}], [add])
    assert log.records[-1].ok is False


def test_api_llm_tool_turn_records_carry_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    llm, _, log = api_llm(monkeypatch, [completion("ok"), *([RuntimeError("down")] * 6)])
    before = timestamp()
    llm.get_tool_response([{"role": "user", "content": "hi"}], [add])
    with pytest.raises(RuntimeError, match="down"):
        llm.get_tool_response([{"role": "user", "content": "hi"}], [add])
    good, failed = log.records
    assert good.response_id == "x"
    assert failed.response_id is None
    for record in (good, failed):
        assert record.params == {"max_tokens": llm.max_tokens}
        assert record.ts >= before
    assert good.ts <= failed.ts


def test_api_llm_drives_a_whole_tool_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 40}'}}
    llm, sent, _ = api_llm(monkeypatch, [completion("", [call]), completion("42")])
    trace = MAS(agents=[coder(["add"])], llm=llm, tools=[add]).run("q")
    assert trace.error is None
    assert trace.output == "42"
    assert trace["coder"][0].tool_calls == (ToolCall("add", {"a": 40}, "42"),)
    assert sent[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "42"}


URL = "https://gateway.example/v1/chat/completions"


def status_error(cls: type[openai.APIStatusError], status: int, message: str = "") -> openai.APIStatusError:
    response = httpx2.Response(status, request=httpx2.Request("POST", URL))
    return cls(message or f"Error code: {status}", response=response, body=None)


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the backoff delays instead of waiting them out."""
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return delays


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (openai.BadRequestError, 400),
        (openai.AuthenticationError, 401),
        (openai.PermissionDeniedError, 403),
        (openai.NotFoundError, 404),
        (openai.UnprocessableEntityError, 422),
    ],
)
def test_api_llm_does_not_retry_a_permanent_client_error(
    monkeypatch: pytest.MonkeyPatch, slept: list[float], cls: type[openai.APIStatusError], status: int
) -> None:
    llm, sent, log = api_llm(monkeypatch, [status_error(cls, status), completion("unreachable")])
    llm.retry_base_delay_s = 1.0
    with pytest.raises(cls):
        llm.get_tool_response([{"role": "user", "content": "hi"}], [add])
    assert len(sent) == 1
    assert slept == []
    assert log.records[-1].ok is False


def test_a_tool_less_route_fails_the_run_after_one_call(monkeypatch: pytest.MonkeyPatch, slept: list[float]) -> None:
    refused = status_error(openai.NotFoundError, 404, "No endpoints found that support tool use (Tool Compatibility)")
    llm, sent, _ = api_llm(monkeypatch, [refused, completion("unreachable")])
    llm.retry_base_delay_s = 1.0
    trace = MAS(agents=[coder(["add"])], llm=llm, tools=[add]).run("q")
    assert trace.error is not None
    assert trace.error.startswith("NotFoundError: No endpoints found")
    assert len(sent) == 1
    assert slept == []


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.RateLimitError, 429),
        status_error(openai.InternalServerError, 500),
        status_error(openai.InternalServerError, 503),
        status_error(openai.APIStatusError, 408),
        status_error(openai.ConflictError, 409),
        openai.APIConnectionError(request=httpx2.Request("POST", URL)),
        openai.APITimeoutError(request=httpx2.Request("POST", URL)),
    ],
    ids=["429", "500", "503", "408", "409", "connection", "timeout"],
)
def test_api_llm_retries_transient_errors_up_to_the_limit(
    monkeypatch: pytest.MonkeyPatch, slept: list[float], error: Exception
) -> None:
    llm, sent, log = api_llm(monkeypatch, [error] * 6)
    llm.retry_base_delay_s = 1.0
    assert llm.max_retries == 5
    with pytest.raises(type(error)):
        llm.get_tool_response([{"role": "user", "content": "hi"}], [add])
    assert len(sent) == 6
    assert slept == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert log.records[-1].ok is False


def test_api_llm_recovers_after_a_connection_error(monkeypatch: pytest.MonkeyPatch, slept: list[float]) -> None:
    dropped = openai.APIConnectionError(request=httpx2.Request("POST", URL))
    llm, sent, _ = api_llm(monkeypatch, [dropped, completion("42")])
    assert llm.get_tool_response([{"role": "user", "content": "hi"}], []) == ("42", [])
    assert len(sent) == 2
    assert len(slept) == 1
