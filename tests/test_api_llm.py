"""APILLM: construction ergonomics, the teacher-forced scoring path, and call metadata."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest
from openai.types.chat import ChatCompletion

from xmas import MAS, Agent, AgentRecord, Budget, TokenUsage
from xmas.budget import RunMeter
from xmas.errors import EmptyResponse, ErrorKind, XmasError
from xmas.llms.api_llm import APILLM
from xmas.llms.call_log import CallLog
from xmas.llms.meta import TokenPrice, supports_images, supports_meta, supports_parts
from xmas.llms.parts import Part, content_text, has_media, marker, render_parts, to_content
from xmas.media import MAX_PART_BYTES, Ref
from xmas.prompts import safe_format
from xmas.tools import ToolCallRequest, index_tools, run_tool_loop
from xmas.utils import approximate_tokens

from .conftest import BASE_URL, add, answer, recording_llm, requires_gateway


def build_llm() -> APILLM:
    """An APILLM pointed at a URL nothing will actually call."""
    return APILLM(model="llama-3.1-8b", base_url=BASE_URL, api_key="sk-test")


def fake_response(status: int, body: dict[str, object]) -> httpx.Response:
    """An httpx response with no network behind it."""
    return httpx.Response(status_code=status, json=body, request=httpx.Request("POST", BASE_URL))


def test_accepts_the_xmas_spelling_of_model_and_base_url() -> None:
    llm = build_llm()
    assert llm.model == "llama-3.1-8b"
    assert llm.model_id == "llama-3.1-8b", "the parent's attribute must still be set"
    assert llm.api_url == BASE_URL


def test_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XMAS_BASE_URL", BASE_URL)
    monkeypatch.setenv("XMAS_API_KEY", "sk-from-env")
    llm = APILLM(model="llama-3.1-8b")
    assert llm.api_url == BASE_URL
    assert llm.api_key == "sk-from-env"


def test_requires_a_model() -> None:
    with pytest.raises(ValueError, match="needs a model"):
        APILLM(base_url=BASE_URL)


def test_requires_a_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XMAS_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="needs a base_url"):
        APILLM(model="llama-3.1-8b")


def test_score_parses_a_successful_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return fake_response(
            200,
            {"tokens": [" forty", "-two"], "token_logprobs": [-0.25, -0.75], "sum_logprob": -1.0},
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    result = build_llm().score("The answer is", " forty-two")

    assert captured["url"] == f"{BASE_URL}/score"
    assert captured["json"] == {
        "model": "llama-3.1-8b",
        "prompt": "The answer is",
        "continuation": " forty-two",
    }
    assert result.tokens == (" forty", "-two")
    assert result.token_logprobs == (-0.25, -0.75)
    assert result.sum_logprob == -1.0
    assert result.token_count == 2


def test_score_raises_a_named_error_when_the_backend_cannot_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway answers 501 when no configured model exposes prompt logprobs."""
    body: dict[str, object] = {
        "error": {
            "message": "`llama-3.1-8b` cannot score a continuation: it does not expose prompt "
            "log-probabilities (echo). Supported by: none of the configured backends.",
            "code": "capability_not_supported",
        }
    }
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: fake_response(501, body))

    with pytest.raises(XmasError, match="does not expose prompt log-probabilities") as raised:
        build_llm().score("context", "continuation")
    assert raised.value.status == 501


def test_score_surfaces_a_tokenisation_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    body: dict[str, object] = {"error": {"message": "The continuation does not align with token boundaries."}}
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: fake_response(422, body))

    with pytest.raises(XmasError, match="422"):
        build_llm().score("context", "continuation")


def test_score_rejects_an_empty_continuation() -> None:
    with pytest.raises(ValueError, match="empty continuation"):
        build_llm().score("context", "")


@requires_gateway
def test_score_against_the_live_gateway() -> None:
    """Env-gated. Skips while the gateway cannot score.

    Every model in the gateway's registry declares ``echo_score: false``, so
    ``/v1/score`` answers 501. This test is written so that it starts genuinely
    verifying the moment a backend gains the capability, rather than being
    deleted and forgotten.
    """
    llm = APILLM(model="llama-3.1-8b")
    try:
        result = llm.score("The capital of France is", " Paris")
    except XmasError as error:
        if error.status != 501:
            raise
        pytest.skip(f"gateway cannot score yet: {error}")

    assert result.token_count > 0
    assert result.sum_logprob < 0.0


# ------------------------------------------------- full_response: call metadata


def wire(requests: list[dict[str, Any]]) -> list[str]:
    """Requests as text, so key order counts as well as keys and values."""
    return [json.dumps(request) for request in requests]


ADD_CALL = {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 40}'}}


def first_request(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request promptolution sends for the planner's prompt, literally."""
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Plan q"},
        ],
        "max_tokens": 4096,
    }
    request.update(call_kwargs)
    return request


CALL_KWARGS: tuple[dict[str, Any], ...] = ({}, {"temperature": 0.2})


def test_full_response_sends_promptolutions_request_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    # (i) one direct call: promptolution's get_response against generate_with_meta
    for call_kwargs in CALL_KWARGS:
        old, old_sent = recording_llm(monkeypatch, [answer("a plan")], full_response=False, call_kwargs=call_kwargs)
        new, new_sent = recording_llm(monkeypatch, [answer("a plan")], full_response=True, call_kwargs=call_kwargs)
        assert old.get_response(["Plan q"]) == ["a plan"]
        assert new.generate_with_meta("Plan q")[0] == "a plan"
        assert old_sent == new_sent == [first_request(call_kwargs)]
        assert wire(old_sent) == wire(new_sent) == wire([first_request(call_kwargs)])

    # (ii) through a MAS: two plain agents, then a plain planner and a coder with a tool
    for call_kwargs in CALL_KWARGS:
        for coder_tools in ([], ["add"]):
            agents = [
                Agent("planner", "Plan {query}", output="plan"),
                Agent("coder", "Do {plan}", output="answer", inputs=["plan"], tools=coder_tools),
            ]
            coder_turns = [answer("", finish="tool_calls", calls=[ADD_CALL]), answer("42")] if coder_tools else []
            runs: dict[bool, tuple[list[dict[str, Any]], Any]] = {}
            for full_response in (False, True):
                script = [answer("a plan"), *(coder_turns or [answer("42")])]
                llm, sent = recording_llm(monkeypatch, script, full_response=full_response, call_kwargs=call_kwargs)
                trace = MAS(agents=agents, edges=[("planner", "coder")], llm=llm, tools=[add]).run("q")
                assert trace.error is None
                assert trace.output == "42"
                runs[full_response] = (sent, trace)
            (old_sent, old_trace), (new_sent, new_trace) = runs[False], runs[True]
            assert len(new_sent) == (3 if coder_tools else 2)
            assert old_sent == new_sent
            assert wire(old_sent) == wire(new_sent)
            assert wire(new_sent[:1]) == wire([first_request(call_kwargs)])
            # the opted-in run really took the meta path: the engine (plain) and the tool loop read the meta
            assert [record.served_model for record in old_trace.records] == [None, None]
            assert [record.served_model for record in new_trace.records] == ["m", "m"]


USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 40,
    "total_tokens": 140,
    "prompt_tokens_details": {"cached_tokens": 60},
    "completion_tokens_details": {"reasoning_tokens": 25},
}


RICH = TokenUsage(input_tokens=100, output_tokens=40, cached_tokens=60, reasoning_tokens=25)
"""The counts in ``USAGE``, as xmas reads them."""


class OneShotLLM(APILLM):
    """A subclass with helpers of its own under plain names, as a gateway's metered model has."""

    def ask(self, prompt: str, system_prompt: str = "", timeout_s: float | None = None) -> str:
        """Its own ask, not APILLM's."""
        return "own ask"

    def request(self) -> str:
        """Its own request, not APILLM's."""
        return "own request"

    def generate(self) -> str:
        """Its own generate, not APILLM's."""
        return "own generate"

    def wait(self) -> str:
        """Its own wait, not APILLM's."""
        return "own wait"

    def log(self) -> str:
        """Its own log, not APILLM's."""
        return "own log"


def test_a_subclass_may_name_its_own_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    # APILLM's own steps go by names a subclass is unlikely to pick, so a plain ``ask`` does not replace one
    llm, sent = recording_llm(monkeypatch, [answer("hi"), answer("again")], cls=OneShotLLM)
    assert llm.generate_with_meta("hello")[0] == "hi"
    assert llm.get_tool_response_meta([{"role": "user", "content": "hello"}], [])[0] == "again"
    assert len(sent) == 2


def test_full_response_keeps_usage_cost_and_finish(monkeypatch: pytest.MonkeyPatch) -> None:

    # the provider's cost, cache reads, reasoning and served model reach the meta and the record keywords
    llm, _ = recording_llm(monkeypatch, [answer("hi", usage={**USAGE, "cost": 0.0123}, model="vendor/m-0613")])
    assert llm.full_response is True
    assert supports_meta(llm)
    text, meta = llm.generate_with_meta("hello")
    assert text == "hi"
    assert (meta.cost_usd, meta.tokens, meta.served_model, meta.finish_reason) == (
        0.0123,
        RICH,
        "vendor/m-0613",
        "stop",
    )
    assert meta.error_kind is None
    fields = meta.record_fields()
    assert fields == {
        "cost_usd": 0.0123,
        "tokens": RICH,
        "finish_reason": "stop",
        "error_kind": None,
        "served_model": "vendor/m-0613",
    }
    record = AgentRecord("a", 0, "p", "p", {}, {}, **fields)
    assert (record.cost_usd, record.tokens, record.served_model) == (0.0123, RICH, "vendor/m-0613")
    assert llm.get_token_count() == {
        "input_tokens": 1 + 5,
        "output_tokens": 1,
        "total_tokens": 7,
    }  # as get_response counts

    # a top-level cost is the second source; a bool is never a cost
    llm, _ = recording_llm(monkeypatch, [answer("hi", usage=USAGE, cost=0.5)])
    assert llm.generate_with_meta("hello")[1].cost_usd == 0.5
    llm, _ = recording_llm(monkeypatch, [answer("hi", usage={**USAGE, "cost": True})])
    assert llm.generate_with_meta("hello")[1].cost_usd is None

    # TokenPrice is the fallback; reasoning tokens are inside completion_tokens and not priced twice
    price = TokenPrice(input_per_m=1.0, output_per_m=4.0, cached_input_per_m=0.5)
    llm, _ = recording_llm(monkeypatch, [answer("hi", usage=USAGE)], price=price)
    assert llm.generate_with_meta("hello")[1].cost_usd == pytest.approx((40 * 1.0 + 60 * 0.5 + 40 * 4.0) / 1e6)
    # more cache reads than input means the provider counted them apart: priced as such, never below 0
    assert price.cost_usd(TokenUsage(10, 0, cached_tokens=1000)) == pytest.approx((10 * 1.0 + 1000 * 0.5) / 1e6)

    # neither a reported cost nor a price: unknown, never 0; no usage at all: no tokens either
    llm, _ = recording_llm(monkeypatch, [answer("hi", usage=USAGE), answer("hi")])
    assert llm.generate_with_meta("hello")[1].cost_usd is None
    assert llm.generate_with_meta("hello")[1].tokens is None


def test_full_response_asks_again_after_an_empty_or_choiceless_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    price = TokenPrice(input_per_m=1.0, output_per_m=4.0, cached_input_per_m=0.5)
    # the client does not validate responses: a usage block without counts gives no tokens, not a TypeError
    gap = ChatCompletion.construct(**{**answer("hi").model_dump(), "usage": {}})
    llm, _ = recording_llm(monkeypatch, [gap, gap], price=price)
    text, meta = llm.generate_with_meta("hello")
    assert (text, meta.tokens, meta.cost_usd) == ("hi", None, None)
    assert llm.get_tool_response_meta([{"role": "user", "content": "hello"}], [])[2].tokens is None
    # an empty answer is asked again, and both paid attempts are summed
    llm, sent = recording_llm(monkeypatch, [answer(None, usage=USAGE), answer("ok", usage=USAGE)])
    text, meta = llm.generate_with_meta("hello")
    assert (text, len(sent), meta.tokens) == ("ok", 2, RICH + RICH)

    # so is a response without choices, as promptolution asks again, in a plain call and in a tool turn
    no_choice = ChatCompletion.model_validate({**answer("x").model_dump(), "choices": []})
    llm, sent = recording_llm(monkeypatch, [no_choice, answer("ok")])
    assert (llm.generate_with_meta("hello")[0], len(sent)) == ("ok", 2)
    llm, sent = recording_llm(monkeypatch, [no_choice, answer("", finish="tool_calls", calls=[ADD_CALL])])
    turn = llm.get_tool_response([{"role": "user", "content": "hello"}], [add])
    assert (turn, len(sent)) == (("", [ToolCallRequest("c1", "add", '{"a": 40}')]), 2)


def test_full_response_marks_an_answer_cut_by_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    # an empty answer cut by max_tokens is not sent again
    llm, sent = recording_llm(monkeypatch, [answer(None, finish="length")] * 6)
    with pytest.raises(EmptyResponse, match=r"^Empty content from model") as raised:
        llm.generate_with_meta("hello")
    assert raised.value.kind is ErrorKind.LENGTH_CUT
    assert len(sent) == 1
    restored = pickle.loads(pickle.dumps(raised.value))  # the kind and the paid attempts survive a pickle
    assert raised.value.meta is not None
    assert (restored.kind, restored.meta) == (ErrorKind.LENGTH_CUT, raised.value.meta)

    # a text answer cut by max_tokens is kept, and marked
    llm, _ = recording_llm(monkeypatch, [answer("half an ans", finish="length")])
    text, meta = llm.generate_with_meta("hello")
    assert (text, meta.error_kind) == ("half an ans", ErrorKind.LENGTH_CUT)
    assert meta.record_fields()["error_kind"] is ErrorKind.LENGTH_CUT


def test_full_response_spares_a_subclass_that_hooks_promptolution(monkeypatch: pytest.MonkeyPatch) -> None:
    # a subclass that hooks promptolution's path keeps it, unless it opts in explicitly
    hooked_prompts: list[str] = []

    class Hooked(APILLM):
        async def _ainvoke_once(self, prompt: str, system_prompt: str) -> ChatCompletion:
            hooked_prompts.append(prompt)
            return await super()._ainvoke_once(prompt, system_prompt)

    hooked, sent = recording_llm(monkeypatch, [answer("done")], cls=Hooked)
    assert hooked.full_response is False
    assert not supports_meta(hooked)
    trace = MAS(agents=[Agent("solver", "Solve {query}", output="answer")], llm=hooked).run("q")
    assert trace.error is None
    assert trace.output == "done"
    assert hooked_prompts == ["Solve q"]
    assert len(sent) == 1
    assert recording_llm(monkeypatch, [], cls=Hooked, full_response=True)[0].full_response is True
    assert recording_llm(monkeypatch, [], full_response=False)[0].full_response is False


def test_the_paid_attempts_behind_an_empty_answer_reach_the_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    paid = {**USAGE, "cost": 0.01}
    no_choice = ChatCompletion.model_validate({**answer("x", usage=paid).model_dump(), "choices": []})

    # the error carries the six paid attempts
    llm, sent = recording_llm(monkeypatch, [answer(None, usage=paid)] * 6)
    with pytest.raises(EmptyResponse, match="after 6 attempts") as raised:
        llm.generate_with_meta("hello")
    assert raised.value.meta is not None
    assert len(sent) == 6
    assert raised.value.meta.cost_usd == pytest.approx(6 * 0.01)

    # a plain agent and a tool agent: the run fails, and its meter counted one call that cost what the six
    # attempts cost, over a cost limit of 0.05 only because of them
    plain = Agent("solver", "Solve {query}", output="answer")
    with_tools = Agent("coder", "Do {query}", output="answer", tools=["add"])
    # the plain call fails as promptolution's batch call does; a tool turn raises its own error
    for agent, script, form in (
        (plain, [answer(None, usage=paid)] * 6, "RuntimeError: LLM batch failed: "),
        (with_tools, [no_choice] * 6, "EmptyResponse: "),
    ):
        llm, sent = recording_llm(monkeypatch, list(script))
        trace = MAS([agent], llm=llm, tools=[add], budget=Budget(max_calls=10, max_cost_usd=0.05)).run("q")
        assert trace.error is not None
        assert trace.error.startswith(f"{form}Empty content from model after 6 attempts")
        assert (len(sent), trace.records) == (6, ())
        assert trace.health is not None
        assert trace.health.budget == {
            "calls": {"spent": 1, "limit": 10},
            "cost_usd": {"spent": pytest.approx(6 * 0.01), "limit": 0.05},
        }


# ------------------------------------------------------- pictures and sound


def picture(tmp_path: Path, name: str, data: bytes) -> Ref:
    """A file on disk, referenced where it is."""
    path = tmp_path / name
    path.write_bytes(data)
    return Ref.from_path(path)


def test_parts_request_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # render_parts substitutes as safe_format does, labels list items 0-based and marks a failed one
    a, b, c = (picture(tmp_path, f"{name}.png", name.encode()) for name in ("a", "b", "c"))
    template = 'Plan: {plan} {{lit}} {"k": 1}\nPages: {pages}\nCover: {cover}{gone}{empty} {unknown}'
    media: dict[str, Any] = {"pages": [a, None, b], "cover": c, "gone": None, "empty": []}
    parts, shown = render_parts(template, {"plan": "p"}, media)
    assert parts == [
        'Plan: p {{lit}} {"k": 1}\nPages: \n[pages 0] ',
        a,
        "\n[pages 1] (none)\n[pages 2] ",
        b,
        "\nCover: ",
        c,
        " {unknown}",
    ]
    assert shown == (
        'Plan: p {{lit}} {"k": 1}\nPages: \n[pages 0] [image a.png, image/png, 1 B]\n[pages 1] (none)'
        "\n[pages 2] [image b.png, image/png, 1 B]\nCover: [image c.png, image/png, 1 B] {unknown}"
    )
    text_only = "Plan: {plan} {{lit}} {other}"
    rendered = safe_format(text_only, {"plan": "p"})
    assert render_parts(text_only, {"plan": "p"}, {}) == ([rendered], rendered)
    one = (["\n[pages 0] ", a], "\n[pages 0] [image a.png, image/png, 1 B]")
    assert render_parts("{pages}", {}, {"pages": [a]}) == one
    assert (has_media(parts), has_media(["x"])) == (True, False)
    assert marker(Ref("/s/x.png", "image/png", 182 * 1024)) == "[image x.png, image/png, 182 KB]"
    assert marker(Ref("https://h/d/x.wav", "audio/wav")) == "[audio x.wav, audio/wav]"
    assert marker(Ref("/s/x.png", "image/png", 3 * 1024 * 1024 + 400_000)) == "[image x.png, image/png, 3.4 MB]"

    # to_content: data URIs, URLs as they are, wav and mp3, nothing a model reads through a conversion step
    wav, mp3 = picture(tmp_path, "v.wav", b"RIFF"), picture(tmp_path, "v.mp3", b"ID3")
    remote = Ref("https://cdn.example/p.png", "image/png")
    assert to_content(["Look", a, remote, wav, mp3]) == [
        {"type": "text", "text": "Look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"a").decode()}},
        {"type": "image_url", "image_url": {"url": "https://cdn.example/p.png"}},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(b"RIFF").decode(), "format": "wav"}},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(b"ID3").decode(), "format": "mp3"}},
    ]
    refused = [
        (Ref("/s/big.png", "image/png", MAX_PART_BYTES + 1), r"\[image big.png, image/png, 20.0 MB\] is over"),
        (Ref("https://h/a.wav", "audio/wav"), "audio is sent as bytes, and xmas reads no network"),
        (picture(tmp_path, "v.ogg", b"Ogg"), "a model takes wav or mp3 audio, not audio/ogg"),
        (picture(tmp_path, "d.pdf", b"%PDF"), "a model reads a document through a conversion step"),
        (picture(tmp_path, "t.csv", b"a,b"), "a model reads a table through a conversion step"),
    ]
    for ref, match in refused:
        with pytest.raises(ValueError, match=match):
            to_content([ref])
    monkeypatch.setattr("xmas.llms.parts.MAX_PART_BYTES", 4)
    with pytest.raises(ValueError, match=r"over the 4 B a content part may carry"):
        to_content([Ref(str(picture(tmp_path, "five.png", b"12345").location), "image/png")])  # size unknown
    monkeypatch.undo()

    # one request through the same path: the same keys in the same order, then call_kwargs; the record shows markers
    log = CallLog(capture_text=True)
    llm, sent = recording_llm(monkeypatch, [answer("seen")], call_kwargs={"temperature": 0.2}, call_log=log)
    text, meta = llm.generate_parts_with_meta(parts, system_prompt="Judge.")
    assert (text, meta.served_model) == ("seen", "m")
    assert list(sent[0]) == ["model", "messages", "max_tokens", "temperature"]
    assert sent[0]["messages"] == [
        {"role": "system", "content": "Judge."},
        {"role": "user", "content": to_content(parts)},
    ]
    assert [part["type"] for part in sent[0]["messages"][1]["content"]] == ["text", "image_url"] * 3 + ["text"]
    (record,) = log.records
    assert (record.prompt, record.system_prompt, record.response) == (shown, "Judge.", "seen")
    assert "base64," not in json.dumps(record.to_dict())
    assert content_text(sent[0]["messages"][1]["content"]) == "".join(part for part in parts if isinstance(part, str))


B1_BYTES = (
    b'{"model":"m","messages":[{"role":"system","content":"You are a helpful assistant."},'
    b'{"role":"user","content":"Look: {png}"}],"max_tokens":4096}'
)
"""A text-only request as promptolution sends it, byte for byte."""


def wired_llm(**kwargs: Any) -> tuple[APILLM, list[bytes]]:
    """An APILLM whose HTTP transport keeps each request body's bytes and answers "ok"."""
    sent: list[bytes] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        return httpx2.Response(200, json=answer("ok").model_dump(mode="json"))

    client = {"http_client": httpx2.AsyncClient(transport=httpx2.MockTransport(reply))}
    return APILLM(model="m", base_url=BASE_URL, api_key="k", client_kwargs=client, **kwargs), sent


def test_text_only_requests_keep_their_bytes() -> None:
    assert hashlib.sha256(B1_BYTES).hexdigest().startswith("2d432b7b068dd094")
    llm, sent = wired_llm()
    assert llm.generate_with_meta("Look: {png}")[0] == "ok"
    assert llm.generate_parts_with_meta(["Look: ", "{png}"])[0] == "ok"
    assert sent == [B1_BYTES, B1_BYTES]

    # a model that sees, serving an agent that reads text and one with a tool: text requests, as before
    seeing, sent = wired_llm(accepts=("text", "image"))
    assert (seeing.accepts, seeing.produces) == (("text", "image"), ("text",))
    plain = Agent("looker", "Look: {query}", output="answer")
    assert MAS([plain], llm=seeing).run("{png}").output == "ok"
    with_tool = Agent("looker", "Look: {query}", output="answer", tools=["add"])
    assert MAS([with_tool], llm=seeing, tools=[add]).run("{png}").output == "ok"
    tools = json.dumps([add.to_openai()], separators=(",", ":")).encode()
    assert sent == [B1_BYTES, B1_BYTES[:-1] + b',"tools":' + tools + b"}"]


def painted(content: str | None, *urls: str, **kwargs: Any) -> ChatCompletion:
    """A chat completion whose message carries OpenRouter's ``images`` list."""
    body = answer(content, **kwargs).model_dump()
    body["choices"][0]["message"]["images"] = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
    return ChatCompletion.model_validate(body)


PNG_URI = "data:image/png;base64," + base64.b64encode(b"\x89PNG-o").decode()


def test_image_answers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    usage = {**USAGE, "cost": 0.0002}
    log = CallLog(capture_text=True)
    llm, sent = recording_llm(monkeypatch, [painted("ok", PNG_URI, usage=usage, model="m-served")], call_log=log)
    text, images, meta = llm.generate_image_with_meta("Paint a house")
    assert (text, images) == ("ok", [(b"\x89PNG-o", "image/png")])
    assert (meta.cost_usd, meta.served_model, meta.finish_reason) == (0.0002, "m-served", "stop")
    assert list(sent[0]) == ["model", "messages", "max_tokens", "modalities"]
    assert sent[0]["modalities"] == ["image", "text"]
    assert sent[0]["messages"][1] == {"role": "user", "content": "Paint a house"}
    assert log.records[0].params == {"max_tokens": 4096, "modalities": ["image", "text"]}

    # modalities only on the image path
    llm, sent = recording_llm(monkeypatch, [answer("words")])
    llm.generate_with_meta("Paint a house")
    assert "modalities" not in sent[0]

    # images without text are an answer, asked once; a picture in the prompt goes as a part
    ref = picture(tmp_path, "sketch.png", b"sketch")
    llm, sent = recording_llm(monkeypatch, [painted(None, PNG_URI, PNG_URI)])
    text, images, _ = llm.generate_image_with_meta(["Finish ", ref])
    assert (text, len(images), len(sent)) == ("", 2, 1)
    assert sent[0]["messages"][1]["content"] == to_content(["Finish ", ref])

    # neither text nor images is empty, as for text; an image xmas would have to fetch is refused
    llm, sent = recording_llm(monkeypatch, [painted(None), answer("ok")])
    assert llm.generate_image_with_meta("Paint")[:2] == ("ok", [])
    assert len(sent) == 2
    llm, _ = recording_llm(monkeypatch, [painted("ok", "https://cdn.example/p.png")])
    with pytest.raises(EmptyResponse, match="not a base64 data URI") as refused:
        llm.generate_image_with_meta("Paint")
    assert refused.value.kind is ErrorKind.BAD_FORMAT
    assert refused.value.meta is not None
    llm, _ = recording_llm(monkeypatch, [painted("ok", "data:image/png;base64,!!")])
    with pytest.raises(XmasError, match="base64 is invalid"):
        llm.generate_image_with_meta("Paint")

    # that refused picture was paid for: a run's meter counts its cost
    fetched = painted("ok", "https://cdn.example/p.png", usage={**USAGE, "cost": 0.04})
    llm, _ = recording_llm(monkeypatch, [fetched], produces=("text", "image"))
    painter = Agent("painter", "Paint {query}", "picture", output_type="image")
    trace = MAS([painter], llm=llm, budget=Budget(max_cost_usd=1.0), version=2).run("a house")
    assert trace.error_kind is ErrorKind.BAD_FORMAT
    assert trace.health is not None
    assert trace.health.budget["cost_usd"]["spent"] == pytest.approx(0.04)

    # what a model takes and gives are words a model may state
    for wrong, words in (
        ("image", r"APILLM accepts is a list of words such as \[image\], not 'image'"),
        (("text", "video"), "APILLM accepts 'video'; a model may accept image, audio"),
    ):
        with pytest.raises(ValueError, match=words):
            APILLM(model="m", base_url=BASE_URL, api_key="k", accepts=wrong)


def test_a_hooked_subclass_cannot_send_parts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Hooked(APILLM):
        async def _ainvoke_once(self, prompt: str, system_prompt: str) -> ChatCompletion:
            return await super()._ainvoke_once(prompt, system_prompt)

    hooked, _ = recording_llm(monkeypatch, [], cls=Hooked, accepts=("text", "image"))
    assert (supports_parts(hooked), supports_images(hooked)) == (False, False)
    plain, _ = recording_llm(monkeypatch, [])
    assert (supports_parts(plain), supports_images(plain)) == (True, True)
    assert not supports_parts(object())

    # the tool loop sends the parts in the first user message, and counts their text only
    ref = picture(tmp_path, "slide.png", b"x" * 3000)
    parts: list[Part] = ["Check ", ref]
    llm, sent = recording_llm(monkeypatch, [answer("", calls=[ADD_CALL]), answer("42")])
    text, calls, usage, meta = run_tool_loop(
        llm, parts, index_tools([add]), 2, "System.", approximate_tokens, RunMeter()
    )
    assert (text, [call.result for call in calls]) == ("42", ["42"])
    assert meta is not None
    assert meta.served_model == "m"
    assert sent[0]["messages"][1] == {"role": "user", "content": to_content(parts)}
    assert sent[1]["messages"][1] == sent[0]["messages"][1]
    assert [message["role"] for message in sent[1]["messages"]] == ["system", "user", "assistant", "tool"]
    assert usage.input_tokens < 20
    llm, sent = recording_llm(monkeypatch, [answer("done")])
    run_tool_loop(llm, ["Check ", "this"], index_tools([add]), 2, "System.", approximate_tokens, RunMeter())
    assert sent[0]["messages"][1] == {"role": "user", "content": "Check this"}
