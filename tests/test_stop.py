"""Stop: a run ends within a moment, its calls in flight cut; streaming opt-in; the flags of a ``first of`` race."""

from __future__ import annotations

import base64
import dataclasses
import gc
import json
import os
import pickle
import re
import select
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import httpx2
import pytest
from openai.types.chat import ChatCompletion

from xmas import MAS, Agent, Budget, TokenUsage
from xmas.budget import RUNNING, SLICE_S, RunMeter, cancel_flag, contestant
from xmas.edges import Edge, EdgeKind
from xmas.errors import BudgetExceeded, ErrorKind, RaceLost, RunStatus, Stopped, classify
from xmas.graph import END
from xmas.human import HumanAction, Question
from xmas.llms.api_llm import APILLM, assemble
from xmas.llms.meta import CallMeta, TokenPrice
from xmas.nodes import Code, MASModule
from xmas.runtime import BATCH_QUERIES
from xmas.tools import ToolCallRequest, tool
from xmas.trace import Event

from .conftest import META, MockLLM, add, answer

CUT_USD, META_USD = 0.001, 0.01
CUT = CallMeta(latency_s=0.1, tokens=TokenUsage(3, 1), cost_usd=CUT_USD)
"""What the polling model's cut call used; a finished call uses ``META``, of ``META_USD``."""


class PollingLLM(MockLLM):
    """A model thinking ``seconds(prompt)`` per prompt, polling its call's cancel flag as a streaming wrapper does."""

    def __init__(self, seconds: Callable[[str], float] = lambda prompt: 5.0) -> None:
        """Answer "done" after the prompt's time, unless the flag is set first."""
        super().__init__(meta=True)
        self.seconds = seconds

    def generate_with_meta(self, prompt: str, system_prompt: str = "") -> tuple[str, CallMeta]:
        """Wait out the prompt's time; a set flag cuts the call with :data:`CUT` and the text so far."""
        self.calls.append(prompt)
        flag = cancel_flag()
        assert flag is not None, "a metered call sees its flag"
        ends = time.monotonic() + self.seconds(prompt)
        while time.monotonic() < ends:
            if flag.wait(0.01):
                flag.check(CUT, "half")
        return "done", META

    def started(self, count: int) -> None:
        """Wait until ``count`` calls started; a slow host may take its time, so the cap is generous."""
        ends = time.monotonic() + 30
        while len(self.calls) < count and time.monotonic() < ends:
            time.sleep(0.005)
        assert len(self.calls) == count


def stop_system(model: PollingLLM) -> MAS:
    """A planner, then 8 items four at once beside a side agent with an error edge and a module.

    The planner is quick.
    """
    inner = MAS([Agent("inner", "Inner {pages}", "short", ["pages"])], inputs={"pages": "list of text"}, llm=model)
    return MAS(
        [
            Agent("planner", "Plan {pages}", "plan", ["pages"]),
            Agent("drafter", "Draft {page}", "draft", ["page"]),
            Agent("noter", "Note {pages}", "note", ["pages"]),
            Agent("fallback", "Fallback {pages}", "apology", ["pages"]),
        ],
        [("planner", "noter"), ("planner", "boxed"), Edge("noter", "fallback", EdgeKind.ERROR, errors=("any",))],
        modules=[MASModule("boxed", inner, reads={"pages": "pages"}, writes={"short": "short"})],
        each={"drafter": {"item": "page", "in": "pages", "at once": 4}},
        inputs={"pages": "list of text"},
        outputs={"draft": "list of text", "note": "text", "apology": "text", "short": "text"},
        llm=model,
        budget=Budget(max_cost_usd=1.0),
    )


@pytest.fixture
def one_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the host has one core, where LangGraph's default pool would have five threads for the run's seven."""
    monkeypatch.setattr(os, "cpu_count", lambda: 1)


PAGES = {"pages": list("abcdefgh")}
LISTED = json.dumps(PAGES["pages"])


@pytest.mark.usefixtures("one_core")
def test_stop_ends_a_run_within_a_second() -> None:
    model = PollingLLM(lambda prompt: 0.0 if prompt.startswith("Plan") else 5.0)
    mas = stop_system(model)
    pressed: list[float] = []

    def press() -> None:
        model.started(7)  # the planner, four items, the side agent and the module's agent
        (meter,) = [meter for meter in list(RUNNING.values()) if meter.owner is mas]
        pressed.append(time.monotonic())
        meter.stop("Tom pressed Stop")

    presser = threading.Thread(target=press)
    presser.start()
    trace = mas.run_state(PAGES)
    returned = time.monotonic()
    presser.join()

    assert returned - pressed[0] < 1.0  # mostly under 0.01 s; one in ten took 0.2 s on two busy cores
    assert (trace.status, trace.error_kind) == (RunStatus.STOPPED, ErrorKind.STOPPED)
    assert trace.error == "Stopped: Tom pressed Stop"
    # the queued items never started, the module's inner run stopped too, and no error edge routed the stop
    assert sorted(model.calls) == [
        "Draft a",
        "Draft b",
        "Draft c",
        "Draft d",
        "Inner " + LISTED,
        "Note " + LISTED,
        "Plan " + LISTED,
    ]
    # the six cut calls count what they used on the run's meter
    assert trace.health is not None
    assert trace.health.budget["cost_usd"]["spent"] == pytest.approx(META_USD + 6 * CUT_USD)


@pytest.mark.usefixtures("one_core")
def test_mas_stop_keeps_cut_records_and_one_event() -> None:
    model = PollingLLM(lambda prompt: 0.0 if prompt.startswith("Plan") else 5.0)
    mas = stop_system(model)
    events: list[Any] = []
    done: dict[str, Any] = {}
    runner = threading.Thread(
        target=lambda: done.update(trace=mas.run_state(PAGES, run_id="turn-7", on_event=events.append))
    )
    runner.start()
    model.started(7)
    with pytest.raises(ValueError, match="in use"):
        mas.run_state(PAGES, run_id="turn-7")
    pressed = time.monotonic()
    assert mas.stop("turn-7", "Tom pressed Stop") is True
    runner.join(5)
    assert time.monotonic() - pressed < 1.0
    trace = done["trace"]
    assert (trace.status, trace.run_id) == (RunStatus.STOPPED, "turn-7")
    assert [(event.node, event.reason) for event in events if event.kind == "stop"] == [("", "Tom pressed Stop")]
    cut = {(record.agent, record.item): record for record in trace.records if record.error_kind is ErrorKind.STOPPED}
    assert sorted(cut) == [("boxed", None), *[("drafter", item) for item in range(4)], ("noter", None)]
    noter, boxed = cut["noter", None], cut["boxed", None]
    assert (noter.raw_response, noter.cost_usd, noter.tokens) == ("half", CUT.cost_usd, CUT.tokens)
    assert noter.error == "Stopped: Tom pressed Stop"
    assert boxed.inner is not None
    assert [(record.agent, record.raw_response) for record in boxed.inner.records] == [("inner", "half")]
    ended = sorted(event.node for event in events if event.kind == "end" and event.status == "stopped")
    assert ended == ["boxed", "drafter", "drafter", "drafter", "drafter", "inner", "noter"]
    assert mas.stop("turn-7") is False  # ended: no longer running
    assert mas.stop("no-such-run") is False


# ------------------------------------------------------------ a local endpoint


def sse(chunk: dict[str, Any]) -> bytes:
    """One server-sent event carrying ``chunk``."""
    return f"data: {json.dumps(chunk)}\n\n".encode()


def word_chunk(index: int) -> dict[str, Any]:
    """The streamed chunk carrying word ``index``."""
    delta = {"index": 0, "delta": {"content": f"w{index} "}, "finish_reason": None}
    return {"id": "s", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [delta]}


class SlowServer:
    """A local endpoint that answers slowly: one word every 0.1 s when streamed, else nothing for 10 s.

    It notes each request body and the moment the client went away.
    """

    def __init__(self) -> None:
        """Listen on a free local port."""
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}/v1"
        self.bodies: list[dict[str, Any]] = []
        self.gone: list[float] = []
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self) -> None:
        """Accept connections until the socket closes, each answered on its own thread."""
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self.answer, args=(conn,), daemon=True).start()

    def answer(self, conn: socket.socket) -> None:
        """Note one request, answer it slowly and note when the client goes away."""
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, _, body = data.partition(b"\r\n\r\n")
            found = re.search(rb"(?i)content-length:\s*(\d+)", head)
            assert found is not None
            while len(body) < int(found.group(1)):
                body += conn.recv(65536)
            request = json.loads(body)
            self.bodies.append(request)
            if request.get("stream"):
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            for index in range(100):
                if request.get("stream"):
                    try:
                        conn.sendall(sse(word_chunk(index)))
                    except OSError:
                        break
                readable, _, _ = select.select([conn], [], [], 0.1)
                if readable and not conn.recv(1):
                    break
            else:
                return
            self.gone.append(time.monotonic())

    def gone_by(self, count: int) -> list[float]:
        """The moments the client went away, once ``count`` are noted."""
        ends = time.monotonic() + 2
        while len(self.gone) < count and time.monotonic() < ends:
            time.sleep(0.005)
        return self.gone


@pytest.fixture
def server() -> Iterator[SlowServer]:
    endpoint = SlowServer()
    yield endpoint
    endpoint.sock.close()


def cut_after(meter: RunMeter, seconds: float, action: Callable[[], object]) -> list[float]:
    """Run ``action`` after ``seconds`` on a timer; return the list its moment lands in."""
    moments: list[float] = []

    def act() -> None:
        moments.append(time.monotonic())
        action()

    threading.Timer(seconds, act).start()
    return moments


PRICE = TokenPrice(input_per_m=1.0, output_per_m=2.0)


def test_apillm_cancels_in_flight(server: SlowServer) -> None:
    for stream, price in ((True, PRICE), (True, None), (False, PRICE)):
        llm = APILLM(model="m", base_url=server.url, api_key="k", stream=stream, price=price, max_retries=0)
        meter = RunMeter()
        pressed = cut_after(meter, 0.5, meter.stop)
        with pytest.raises(Stopped) as raised:
            meter.metered(lambda llm=llm: llm.generate_with_meta("hi"))  # type: ignore[misc]
        returned = time.monotonic()
        gone = server.gone_by(len(server.bodies))
        assert returned - pressed[0] < 0.3
        assert gone[-1] - pressed[0] < 0.3, "the connection closed"
        cut = raised.value
        assert (str(cut), cut.reason) == ("stopped", "stopped")
        assert cut.meta is not None
        assert cut.meta.finish_reason is None
        assert meter.calls == 1
        if not stream:  # only a stream keeps what arrived; the cut call's use is unknown
            assert (cut.text, cut.meta.tokens, cut.meta.cost_usd, cut.meta.served_model) == ("", None, None, None)
            assert meter.uncosted_calls == 1
            continue
        words = cut.text.split()
        assert 2 <= len(words) <= 8
        assert words == [f"w{index}" for index in range(len(words))]
        assert cut.meta.tokens == TokenUsage(5 + 1, len(words))  # "You are a helpful assistant." and "hi"
        assert cut.meta.served_model == "m"
        if price is None:
            assert (cut.meta.cost_usd, meter.uncosted_calls) == (None, 1)
        else:
            assert cut.meta.cost_usd == pytest.approx((6 * 1.0 + len(words) * 2.0) / 1e6)
            assert meter.cost_usd == cut.meta.cost_usd
    assert len(server.bodies) == 3, "a cut call is not sent again"

    # a tool turn is cut alike, and outside a metered call nothing polls
    llm = APILLM(model="m", base_url=server.url, api_key="k", stream=True, max_retries=0)
    meter = RunMeter()
    cut_after(meter, 0.3, meter.stop)
    with pytest.raises(Stopped) as raised:
        meter.metered(lambda: llm.get_tool_response_meta([{"role": "user", "content": "hi"}], [add])[::2])
    assert raised.value.text.startswith("w0 ")
    assert server.bodies[-1]["tools"] == [add.to_openai()]
    assert cancel_flag() is None

    # a stop in the wait before a retry ends the call with the paid attempt; a watched call still times out
    usage = {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1, "cost": 0.002}
    llm, sent = wired([answer(None, usage=usage).model_dump(mode="json")] * 2, retry_base_delay_s=5.0)
    meter = RunMeter()
    pressed = cut_after(meter, 0.2, meter.stop)
    with pytest.raises(Stopped) as raised:
        meter.metered(lambda: llm.generate_with_meta("hi"))
    assert time.monotonic() - pressed[0] < 0.3
    assert raised.value.meta is not None
    assert (len(sent), raised.value.meta.cost_usd, meter.cost_usd) == (1, 0.002, 0.002)
    slow = APILLM(model="m", base_url=server.url, api_key="k", max_retries=0, gather_timeout_s=-4.8)
    with pytest.raises(TimeoutError, match=r"^model call timed out after 0\.2"):
        RunMeter().metered(lambda: slow.generate_with_meta("hi"))


# ------------------------------------------------------------ streaming


def wired(replies: list[bytes | dict[str, Any]], **kwargs: Any) -> tuple[APILLM, list[bytes]]:
    """An APILLM whose transport keeps each request body and answers the next reply: SSE bytes or a JSON body."""
    sent: list[bytes] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        body = replies.pop(0)
        if isinstance(body, bytes):
            return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=body)

    client = {"http_client": httpx2.AsyncClient(transport=httpx2.MockTransport(reply))}
    return APILLM(model="m", base_url="https://gateway.example/v1", api_key="k", client_kwargs=client, **kwargs), sent


def streamed(*deltas: dict[str, Any], finish: str = "stop", usage: dict[str, Any] | None = None, **top: Any) -> bytes:
    """A stream of ``deltas`` as a provider sends it: the role first, the finish reason, then usage and ``top``."""
    base = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": top.pop("model", "m")}
    chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
    chunks += [{**base, "choices": [{"index": 0, "delta": delta}]} for delta in deltas]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    chunks.append({**base, "choices": [], **({"usage": usage} if usage else {}), **top})
    return b"".join(sse(chunk) for chunk in chunks) + b"data: [DONE]\n\n"


DEFAULT_BYTES = (
    b'{"model":"m","messages":[{"role":"system","content":"You are a helpful assistant."},'
    b'{"role":"user","content":"hi"}],"max_tokens":4096}'
)
"""The default request for "hi", byte for byte."""

STREAMED_BYTES = (
    b'{"model":"m","stream":true,"messages":[{"role":"system","content":"You are a helpful assistant."},'
    b'{"role":"user","content":"hi"}],"max_tokens":4096,"stream_options":{"include_usage":true}}'
)
"""The same request streamed: ``stream`` placed by the client, ``stream_options`` last, nothing else."""

USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 40,
    "total_tokens": 140,
    "prompt_tokens_details": {"cached_tokens": 60},
    "completion_tokens_details": {"reasoning_tokens": 25},
    "cost": 0.0123,
}

CALLS = [
    {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 40}'}},
    {"id": "c2", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'}},
]


def test_streaming_is_opt_in_and_equivalent() -> None:
    # off by default: the plain bytes; on: the same body plus stream and stream_options
    plain, sent = wired([answer("ok").model_dump(mode="json")])
    assert plain.stream is False
    assert plain.generate_with_meta("hi")[0] == "ok"
    explicit, sent_too = wired([answer("ok").model_dump(mode="json")], stream=False)
    assert explicit.generate_with_meta("hi")[0] == "ok"
    assert sent == sent_too == [DEFAULT_BYTES]
    on, sent = wired([streamed({"content": "ok"})], stream=True)
    assert on.generate_with_meta("hi")[0] == "ok"
    assert sent == [STREAMED_BYTES]
    assert json.loads(STREAMED_BYTES) == {**json.loads(DEFAULT_BYTES), **on.call_params()} | {
        "messages": json.loads(DEFAULT_BYTES)["messages"]
    }
    assert on.call_params() == {"max_tokens": 4096, "stream": True, "stream_options": {"include_usage": True}}
    assert "stream" not in plain.call_params()

    # a streamed answer gives the plain answer's text and meta: usage, cost, finish reason, served model
    whole = answer("half an ans", finish="length", usage=USAGE, model="vendor/m-0613").model_dump(mode="json")
    pieces = streamed(
        {"content": "half "},
        {"content": "an "},
        {"content": "ans"},
        finish="length",
        usage=USAGE,
        model="vendor/m-0613",
    )
    plain, _ = wired([whole])
    on, _ = wired([pieces], stream=True)
    (text, meta), (streamed_text, streamed_meta) = plain.generate_with_meta("hi"), on.generate_with_meta("hi")
    assert streamed_text == text == "half an ans"
    assert dataclasses.replace(streamed_meta, latency_s=0) == dataclasses.replace(meta, latency_s=0)
    assert (meta.cost_usd, meta.error_kind, meta.served_model) == (0.0123, ErrorKind.LENGTH_CUT, "vendor/m-0613")

    # tool turns: the call deltas joined by index, the request the tool turn's plus the stream keys
    messages = [{"role": "user", "content": "hi"}]
    plain, plain_sent = wired([answer("", calls=CALLS, usage=USAGE).model_dump(mode="json")])
    deltas = (
        {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "add", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"a": '}}]},
        {"tool_calls": [{"index": 1, "id": "c2", "type": "function", "function": {"name": "add"}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": "40}"}}]},
        {"tool_calls": [{"index": 1, "function": {"arguments": '{"a": 1, "b": 1}'}}]},
    )
    on, on_sent = wired([streamed(*deltas, finish="tool_calls", usage=USAGE)], stream=True)
    turn, streamed_turn = plain.get_tool_response_meta(messages, [add]), on.get_tool_response_meta(messages, [add])
    requests = [ToolCallRequest("c1", "add", '{"a": 40}'), ToolCallRequest("c2", "add", '{"a": 1, "b": 1}')]
    assert turn[:2] == streamed_turn[:2] == ("", requests)
    assert dataclasses.replace(streamed_turn[2], latency_s=0) == dataclasses.replace(turn[2], latency_s=0)
    sent_body, streamed_body = json.loads(plain_sent[0]), json.loads(on_sent[0])
    assert streamed_body == {**sent_body, "stream": True, "stream_options": {"include_usage": True}}

    # a stream of usage alone has no choice and is asked again; promptolution's batch path streams too
    usage_only = sse({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": []})
    on, sent = wired([usage_only + b"data: [DONE]\n\n", streamed({"content": "ok"})], stream=True, retry_base_delay_s=0)
    assert on.generate_with_meta("hi")[0] == "ok"
    assert len(sent) == 2
    on, sent = wired([streamed({"content": "ok"})], stream=True)
    assert on.get_response(["hi"]) == ["ok"]
    assert sent == [STREAMED_BYTES]

    # images as OpenRouter streams them, and a top-level cost
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()
    on, _ = wired([streamed({"images": [{"type": "image_url", "image_url": {"url": url}}]}, cost=0.5)], stream=True)
    text, images, meta = on.generate_image_with_meta("draw")
    assert (text, images, meta.cost_usd) == ("", [(b"\x89PNG", "image/png")], 0.5)
    assert assemble([]).choices == []
    assert isinstance(assemble([]), ChatCompletion)

    # call_kwargs that name the stream keys give way to them, as call_params records it
    on, sent = wired([streamed({"content": "ok"})], stream=True, call_kwargs={"stream_options": {"x": 1}})
    assert on.generate_with_meta("hi")[0] == "ok"
    assert sent == [STREAMED_BYTES]


# ------------------------------------------------------------ runs that are not running


def test_stop_paused_and_unknown_runs() -> None:
    assert "no-such-run" not in RUNNING
    meter = RunMeter(run_id="turn-1")
    assert RUNNING["turn-1"] is meter
    with pytest.raises(ValueError, match="'turn-1' is in use"):
        RunMeter(run_id="turn-1")
    meter.close()
    meter.close()
    assert "turn-1" not in RUNNING
    again = RunMeter(run_id="turn-1")  # an ended run's id may be taken again
    del again
    gc.collect()
    assert "turn-1" not in RUNNING  # a dropped meter leaves the registry too
    assert len({RunMeter().run_id for _ in range(3)}) == 3


class Withdrawing:
    """A host asker that never answers at once and notes each question it withdraws."""

    def __init__(self) -> None:
        """Note nothing yet."""
        self.withdrawn: list[tuple[str, str]] = []

    def ask(self, question: Question) -> None:
        """Leave the question open."""

    def withdraw(self, question: Question, reason: str) -> None:
        """Note the withdrawn question."""
        self.withdrawn.append((question.id, reason))


def test_stop_a_paused_run() -> None:
    asker = Withdrawing()
    gate = HumanAction("approve", "approve", ["draft"], "approval", question="Ship {draft}?")
    mas = MAS(
        [Agent("writer", "Write {query}", "draft")],
        [("writer", "approve")],
        human_actions=[gate],
        llm=MockLLM(lambda prompt: "a draft"),
        asker=asker,
    )
    assert mas.run_state({"query": "q"}, run_id="turn-2").status == "paused"
    with pytest.raises(ValueError, match="'turn-2' is taken"):
        mas.run_state({"query": "q"}, run_id="turn-2")
    assert mas.stop("turn-2", "Tom pressed Stop") is True
    assert asker.withdrawn == [("turn-2/approve/1", "Tom pressed Stop")]
    row = mas.runs().run("turn-2")
    assert row is not None
    assert row.status == "stopped"
    approved = {"turn-2/approve/1": {"approved": True, "who": "tom"}}
    with pytest.raises(ValueError, match="is stopped, not paused"):
        mas.resume("turn-2", approved)
    with pytest.raises(ValueError, match="is stopped: it takes no answers"):
        mas.answer("turn-2", approved)
    assert mas.stop("turn-2") is False
    with pytest.raises(ValueError, match="holds a '/'"):  # it would share the threads of turn-2's items and modules
        mas.run_state({"query": "q"}, run_id="turn-2/approve")


class Asking(Withdrawing):
    """A withdrawing asker that also notes each question it was asked."""

    def __init__(self) -> None:
        """Note nothing yet."""
        super().__init__()
        self.asked: list[str] = []

    def ask(self, question: Question) -> None:
        """Note the question and leave it open."""
        self.asked.append(question.id)


def test_a_stop_while_running_ends_the_run_even_at_a_pause() -> None:
    def gated(model: MockLLM, asker: Asking, *, code_nodes: tuple[Code, ...] = (), after: str = "") -> MAS:
        side = (
            [] if code_nodes else [Agent("pre", "Pre {query}", "pre"), Agent("noter", "Note {pre}", "notes", ["pre"])]
        )
        side += [Agent("pre", "Pre {query}", "pre")] if after else []
        return MAS(
            [Agent("writer", "Write {query}", "draft"), *side],
            [("writer", "approve"), *([] if code_nodes else [("pre", "noter")]), *([("pre", after)] if after else [])],
            code_nodes=list(code_nodes),
            human_actions=[HumanAction("approve", "approve", ["draft"], "approval")],
            tools=[slow],
            outputs={"approval": "json", ("thought" if code_nodes else "notes"): "text"},
            llm=model,
            asker=asker,
        )

    # a stop during a code node beside the human: no question is put after it, and the run ends stopped
    asker = Asking()
    mas = gated(
        MockLLM(lambda prompt: "a draft"),
        asker,
        code_nodes=(Code("think", "slow", reads={"outline": "query"}, writes={"slow": "thought"}),),
    )
    events: list[Event] = []
    threading.Timer(0.1, mas.stop, args=("turn-3", "Tom pressed Stop")).start()
    trace = mas.run_state({"query": "q"}, run_id="turn-3", on_event=events.append)
    assert (trace.status, trace.error) == ("stopped", "Stopped: Tom pressed Stop")
    assert asker.asked == []
    assert "pause" not in [event.kind for event in events]
    row = mas.runs().run("turn-3")
    assert row is not None
    assert (row.status, row.paused) == ("stopped", ())
    with pytest.raises(ValueError, match="is stopped, not paused"):
        mas.resume("turn-3", {"turn-3/approve/1": {"approved": True, "who": "tom"}})

    # a question put before the stop, while a call on another branch runs, is withdrawn when the run ends
    asker = Asking()
    model = PollingLLM(lambda prompt: 5.0 if prompt.startswith("Note") else 0.0)
    mas = gated(model, asker)
    threading.Timer(0.3, mas.stop, args=("turn-4", "Tom pressed Stop")).start()
    trace = mas.run_state({"query": "q"}, run_id="turn-4")
    assert trace.status == "stopped"
    assert asker.asked == ["turn-4/approve/1"]
    assert asker.withdrawn == [("turn-4/approve/1", "Tom pressed Stop")]

    # a stop during a code node that runs after the question was put: the pause it reaches ends stopped
    asker = Asking()
    think = Code("think", "slow", reads={"outline": "pre"}, writes={"slow": "thought"})
    mas = gated(MockLLM(lambda prompt: "a draft"), asker, code_nodes=(think,), after="think")
    threading.Timer(0.1, mas.stop, args=("turn-5", "Tom pressed Stop")).start()
    trace = mas.run_state({"query": "q"}, run_id="turn-5")
    assert (trace.status, trace.pause, asker.asked) == ("stopped", None, ["turn-5/approve/1"])
    assert asker.withdrawn == [("turn-5/approve/1", "Tom pressed Stop")]
    assert [record.agent for record in trace.records] == ["pre", "writer", "think"]
    row = mas.runs().run("turn-5")
    assert row is not None
    assert (row.status, row.paused) == ("stopped", ())


# ------------------------------------------------------------ first of


def test_first_of_races(server: SlowServer) -> None:
    meter = RunMeter()
    race = meter.race("outline:0:1")
    assert meter.race("outline:0:1") is race
    assert not race.is_set()
    outcome: dict[str, tuple[str, float]] = {}
    losses: dict[str, tuple[str, CallMeta | None, ErrorKind]] = {}
    started = time.monotonic()

    def contest(name: str, seconds: float) -> None:
        model = PollingLLM(lambda prompt: seconds)
        with contestant(race):
            try:
                meter.metered(lambda: model.generate_with_meta(name))
            except RaceLost as error:
                losses[name] = (error.text, error.meta, error.kind)
                outcome[name] = ("lost", time.monotonic() - started)
                return
        outcome[name] = ("won" if meter.win("outline:0:1", name) else "dropped", time.monotonic() - started)

    racers = [threading.Thread(target=contest, args=args) for args in (("fast", 0.3), ("mid", 1.0), ("slow", 3.0))]
    for racer in racers:
        racer.start()
    for racer in racers:
        racer.join()
    assert {name: result for name, (result, _) in outcome.items()} == {"fast": "won", "mid": "lost", "slow": "lost"}
    assert losses == {"mid": ("half", CUT, ErrorKind.STOPPED), "slow": ("half", CUT, ErrorKind.STOPPED)}
    assert max(at for _, at in outcome.values()) < 0.3 + 0.2, "the wall is the fastest contestant's"
    assert meter.calls == 3
    assert meter.cost_usd == pytest.approx(META_USD + 2 * CUT_USD)  # the losers paid for what they used
    assert meter.win("outline:0:1", "slow") is False  # a contestant that finishes later is dropped
    with contestant(race), pytest.raises(RaceLost):
        meter.before_call()  # a lost contestant starts no more calls
    assert meter.flag().reason is None  # outside the race the run goes on
    meter.before_call()

    # a real call of a losing contestant is cut, and keeps what it streamed
    llm = APILLM(model="m", base_url=server.url, api_key="k", stream=True, max_retries=0)
    second = meter.race("outline:0:2")
    cut_after(meter, 0.3, lambda: meter.win("outline:0:2", "quick"))
    with contestant(second), pytest.raises(RaceLost) as raised:
        meter.metered(lambda: llm.generate_with_meta("hi"))
    assert raised.value.text.startswith("w0 ")
    assert server.gone_by(1)

    # stopping the run sets every race; a contestant then stops rather than loses
    third = meter.race("outline:0:3")
    meter.stop()
    assert third.is_set()
    assert meter.race("outline:0:4").is_set()
    with contestant(third), pytest.raises(Stopped) as stopped:
        meter.before_call()
    assert type(stopped.value) is Stopped
    assert meter.win("outline:0:3", "fast") is True  # the engine reads the stop at its next boundary


SECONDS = {"Outline": 0.0, "Side": 0.0, "Draft": 0.0, "Quick": 0.05, "Careful": 3.0, "Polish": 0.0}
"""How long the racing system's model thinks per prompt, by its first word."""


def racing(model: MockLLM, *, careful: Code | None = None) -> MAS:
    """``quick`` and ``careful`` race from ``outline`` to write the notes; ``polish`` joins them with a side branch."""
    agents = [
        Agent("outline", "Outline {query}", "outline"),
        Agent("quick", "Quick {outline}", "notes", ["outline"]),
        Agent("side", "Side {query}", "aside"),
        Agent("polish", "Polish {notes} {aside}", "final", ["notes", "aside"]),
    ]
    if careful is None:
        agents.append(Agent("careful", "Careful {outline}", "notes", ["outline"]))
    edges: list[tuple[str, str] | Edge] = [
        Edge("outline", "quick", EdgeKind.FIRST),
        Edge("outline", "careful", EdgeKind.FIRST),
        ("quick", "polish"),
        ("careful", "polish"),
        ("side", "polish"),
    ]
    code_nodes = [] if careful is None else [careful]
    return MAS(agents, edges, code_nodes=code_nodes, tools=[slow], outputs={"final": "text"}, llm=model)


@tool
def slow(outline: str) -> str:
    """Write notes slowly, without a model."""
    time.sleep(0.3)
    return f"slow notes on {outline}"


def test_the_first_of_a_race_writes() -> None:
    model = PollingLLM(lambda prompt: SECONDS[prompt.split()[0]])
    events: list[Event] = []
    mas = racing(model)
    assert mas.checked().joins["polish"].kind == "join"  # the race is one group of the join
    started = time.monotonic()
    trace = mas.run_state({"query": "q"}, on_event=events.append)
    assert time.monotonic() - started < 1.0, "the wall is the fastest contestant's"
    assert trace.status == "ok"
    assert trace.outputs == {"final": "done"}
    by = {record.agent: record for record in trace.records}
    assert [record.agent for record in trace.records].count("polish") == 1
    assert (by["quick"].lost, by["quick"].outputs) == (False, {"notes": "done"})
    careful = by["careful"]
    assert (careful.lost, careful.outputs, careful.error, careful.raw_response) == (True, {}, None, "half")
    assert (careful.cost_usd, careful.route) == (CUT_USD, ())
    assert ("careful", "lost") in [(event.node, event.status) for event in events if event.kind == "end"]

    # without the side branch the node after the race is a merge, and runs once too
    merged = MAS(
        [
            *(mas.agents[name] for name in ("outline", "quick", "careful")),
            Agent("polish", "Polish {notes}", "final", ["notes"]),
        ],
        [edge for edge in mas.edges if edge.source != "side"],
        outputs={"final": "text"},
        llm=PollingLLM(lambda prompt: SECONDS[prompt.split()[0]]),
    )
    assert merged.checked().joins["polish"].kind == "merge"
    trace = merged.run_state({"query": "q"})
    assert trace.outputs == {"final": "done"}
    assert [r.agent for r in trace.records].count("polish") == 1

    # a code node contestant cannot be cut: it is waited for, and its writes are dropped
    stepped = racing(
        PollingLLM(lambda prompt: SECONDS[prompt.split()[0]]), careful=Code("careful", "slow", writes={"slow": "notes"})
    )
    trace = stepped.run_state({"query": "q"})
    by = {record.agent: record for record in trace.records}
    assert trace.outputs == {"final": "done"}
    assert by["careful"].lost
    assert by["careful"].cost_usd == 0.0
    assert [record.agent for record in trace.records].count("polish") == 1


def test_a_race_fails_when_every_contestant_fails() -> None:
    def answer(prompt: str) -> str:
        if prompt.startswith(("Quick", "Careful")):
            raise ValueError(f"no notes from {prompt.split(maxsplit=1)[0]}")
        return "done"

    trace = racing(MockLLM(answer)).run_state({"query": "q"})
    assert trace.status == "failed"
    assert "no notes from" in str(trace.error)
    assert sorted(record.agent for record in trace.records if record.error) == ["careful", "quick"]
    assert "polish" not in {record.agent for record in trace.records}

    def one(prompt: str) -> str:
        if prompt.startswith("Quick"):
            raise ValueError("no quick notes")
        return "done"

    trace = racing(MockLLM(one)).run_state({"query": "q"})  # the other contestant wins
    assert trace.status == "ok"
    assert trace.outputs == {"final": "done"}


def test_a_race_inside_a_chain_item() -> None:
    model = PollingLLM(lambda prompt: SECONDS[prompt.split()[0]])
    mas = MAS(
        [
            Agent("drafter", "Draft {page}", "draft", ["page"]),
            Agent("quick", "Quick {draft}", "notes", ["draft"]),
            Agent("careful", "Careful {draft}", "notes", ["draft"]),
        ],
        [Edge("drafter", "quick", EdgeKind.FIRST), Edge("drafter", "careful", EdgeKind.FIRST)],
        each={"drafter": {"item": "page", "in": "pages"}, "quick": "page in pages", "careful": "page in pages"},
        inputs={"pages": "list of text"},
        outputs={"notes": "list of text"},
        llm=model,
    )
    started = time.monotonic()
    trace = mas.run_state({"pages": ["a", "b"]})
    assert time.monotonic() - started < 1.0
    assert trace.status == "ok"
    assert trace.outputs == {"notes": ["done", "done"]}
    lost = sorted((record.agent, record.item) for record in trace.records if record.lost)
    assert lost == [("careful", 0), ("careful", 1)]


def test_each_inner_run_races_on_its_own() -> None:
    def inner() -> MAS:
        return MAS(
            [
                Agent("outline", "Outline {topic}", "outline", ["topic"]),
                Agent("quick", "Quick {outline}", "notes", ["outline"]),
                Agent("careful", "Careful {outline}", "notes", ["outline"]),
            ],
            [Edge("outline", "quick", EdgeKind.FIRST), Edge("outline", "careful", EdgeKind.FIRST)],
            inputs={"topic": "text"},
            outputs={"notes": "text"},
            llm=PollingLLM(lambda prompt: SECONDS[prompt.split()[0]]),
        )

    # two modules whose systems race under the same names, on one run's meter
    modules = MAS(
        [Agent("join", "Join {a} {b}", "final", ["a", "b"])],
        [("one", "join"), ("two", "join")],
        modules=[
            MASModule("one", inner(), reads={"topic": "query"}, writes={"notes": "a"}),
            MASModule("two", inner(), reads={"topic": "query"}, writes={"notes": "b"}),
        ],
        outputs={"a": "text", "b": "text"},
        llm=MockLLM(lambda prompt: "joined"),
    )
    trace = modules.run_state({"query": "q"})
    assert trace.status == "ok"
    assert trace.outputs == {"a": "done", "b": "done"}
    for name in ("one", "two"):
        (record,) = trace[name]
        inner_records = record.inner.records  # type: ignore[union-attr]
        assert sorted((r.agent, r.lost) for r in inner_records) == [
            ("careful", True),
            ("outline", False),
            ("quick", False),
        ]

    # a chain item's race on the chain's second pass
    looped = MAS(
        [
            Agent("drafter", "Draft {page}", "draft", ["page"]),
            Agent("quick", "Quick {draft}", "notes", ["draft"]),
            Agent("careful", "Careful {draft}", "notes", ["draft"]),
            Agent("polish", "Polish {notes}", "final", ["notes"]),
        ],
        [
            Edge("drafter", "quick", EdgeKind.FIRST),
            Edge("drafter", "careful", EdgeKind.FIRST),
            ("drafter", "polish"),
            Edge("polish", "drafter", EdgeKind.WHILE, when="run.records < 99", max_visits=2),
            Edge("polish", END, EdgeKind.ELSE),
        ],
        each={"drafter": {"item": "page", "in": "pages"}, "quick": "page in pages", "careful": "page in pages"},
        inputs={"pages": "list of text"},
        outputs={"notes": "list of text"},
        llm=PollingLLM(lambda prompt: SECONDS[prompt.split()[0]]),
    )
    trace = looped.run_state({"pages": ["a"]})
    assert trace.status == "ok"
    assert trace.outputs == {"notes": ["done"]}
    assert [r.agent for r in trace.records if r.lost] == ["careful", "careful"]


# ------------------------------------------------------------ budgets, gates and the flag


@contextmanager
def holding(meter: RunMeter, place: Callable[[], Any]) -> Iterator[threading.Event]:
    """Hold ``place`` on another thread until the block ends; yield once it is held."""
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with place():
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    try:
        yield held
    finally:
        release.set()
        holder.join()


def test_stop_with_budgets_and_gates() -> None:
    # a stop beats a budget refusal; code nodes and calls alike are refused, with the reason
    meter = RunMeter(Budget(max_calls=0))
    with pytest.raises(BudgetExceeded):
        meter.before_call()
    meter.stop("halt")
    meter.stop("again")  # a second stop changes nothing
    assert (meter.stopped, meter.halted.is_set()) == ("halt", True)
    for check in (meter.before_call, meter.check_boundary):
        with pytest.raises(Stopped, match=r"^halt$"):
            check()

    # waiters for a call slot or a chain's place leave within a slice of the stop
    for place in ("slot", "gate"):
        meter = RunMeter(Budget(max_parallel_calls=1))
        enter = meter.slot if place == "slot" else lambda meter=meter: meter.gate("each:1", 1)
        with holding(meter, enter):
            pressed = cut_after(meter, 0.2, meter.stop)
            with pytest.raises(Stopped), enter():
                pass  # pragma: no cover (never entered)
            assert time.monotonic() - pressed[0] < SLICE_S + 0.05
    meter = RunMeter(Budget(max_parallel_calls=1))
    with meter.slot(), meter.gate("each:2", 2), meter.gate("each:2", 9):
        pass  # free places are taken at once
    assert meter.gate("each:3", None).__enter__() is None

    # the flag: none outside a call; inside, it reports and waits; Stopped and RaceLost survive a pickle
    meter = RunMeter()
    assert cancel_flag() is None
    seen: list[tuple[str | None, bool]] = []

    def call() -> tuple[str, CallMeta]:
        flag = cancel_flag()
        assert flag is not None
        seen.append((flag.reason, flag.wait(0.01)))
        flag.check()  # not set: nothing raised
        return "ok", META

    assert meter.metered(call) == ("ok", META)
    assert seen == [(None, False)]
    for error in (Stopped("halt", CUT, "half"), RaceLost("lost", CUT, "half")):
        restored = pickle.loads(pickle.dumps(error))
        assert (type(restored), restored.reason, restored.meta, restored.text) == (
            type(error),
            "halt" if type(error) is Stopped else "lost",
            CUT,
            "half",
        )
        assert classify(error) is ErrorKind.STOPPED
    assert str(Stopped()) == "stopped"

    # totals and seed: the counts carry over, the wall counts active time only
    first = RunMeter(Budget(max_wall_s=10.0))
    first.start()
    first.before_call()
    first.after_call(META)
    first.after_call(None)
    totals = first.totals()
    assert {key: totals[key] for key in ("calls", "cost_usd", "uncosted_calls")} == {
        "calls": 1,
        "cost_usd": META.cost_usd,
        "uncosted_calls": 1,
    }
    resumed = RunMeter(Budget(max_wall_s=10.0))
    resumed.seed({**totals, "active_s": 9.99})
    resumed.start()  # the seeded clock is kept
    assert 9.99 <= resumed.elapsed_s < 10.5
    assert resumed.totals()["calls"] == 1
    time.sleep(0.02)
    with pytest.raises(BudgetExceeded, match=r"^wall_s"):
        resumed.check_boundary()


def test_a_batch_runs_at_most_batch_queries_at_once() -> None:
    lock, running, most = threading.Lock(), [0], [0]

    def counting(prompt: str) -> str:
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return "ok"

    mas = MAS([Agent("solver", "Solve {query}", output="answer")], llm=MockLLM(counting))
    traces = mas.run_batch([str(item) for item in range(3 * BATCH_QUERIES)])
    assert [trace.error for trace in traces] == [None] * (3 * BATCH_QUERIES)
    assert most[0] == BATCH_QUERIES
