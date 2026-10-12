"""Call-level statistics, and the limiter wall that must not be guessed at.

Both exist for an experiment whose headline is a wall-time and token comparison, so
what is pinned here is that the numbers mean what the report will say they mean.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from xmas.errors import XmasError
from xmas.llms.api_llm import APILLM
from xmas.llms.call_log import CallLog, CallRecord
from xmas.utils import approximate_tokens

from .conftest import MockLLM

BASE_URL = "https://gateway.example/v1"


def build_llm(call_log: CallLog | None = None) -> APILLM:
    """A gateway model with no network behind it."""
    return APILLM(model="llama-3.1-8b", base_url=BASE_URL, api_key="sk-test", call_log=call_log)


def score_response(status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    """A `/score` answer carrying two scored tokens."""
    body: dict[str, object] = (
        {"tokens": [" forty", "-two"], "token_logprobs": [-0.25, -0.75]}
        if status == 200
        else {"error": {"message": "slow down"}}
    )
    return httpx.Response(
        status_code=status,
        json=body,
        headers=headers or {},
        request=httpx.Request("POST", BASE_URL),
    )


# ------------------------------------------------------------------ the collector


def test_an_empty_log_still_has_the_shape_a_summary_reader_expects() -> None:
    """A run that never probed must not produce a summary missing the probe block."""
    summary = CallLog().summary()

    assert summary["generate"]["calls"] == 0
    assert summary["score"]["calls"] == 0
    assert summary["total"]["latency_s"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}


def test_the_summary_separates_the_two_operations() -> None:
    log = CallLog()
    log.record(CallRecord("generate", "m", "gateway", latency_s=1.0, input_tokens=10, output_tokens=5))
    log.record(CallRecord("score", "m", "gateway", latency_s=0.1, input_tokens=30, scored_tokens=4))

    summary = log.summary()

    assert summary["generate"]["calls"] == 1
    assert summary["generate"]["decoded_output_tokens"] == 5
    assert summary["score"]["calls"] == 1
    assert summary["score"]["scored_tokens"] == 4
    assert summary["score"]["decoded_output_tokens"] == 0, "a probe decodes nothing"
    assert summary["total"]["calls"] == 2


def test_latency_is_reported_as_a_distribution_not_a_mean() -> None:
    log = CallLog()
    for latency in (0.1, 0.2, 0.3, 10.0):
        log.record(CallRecord("score", "m", "gateway", latency_s=latency))

    block = log.summary()["score"]["latency_s"]

    assert block["p50"] == pytest.approx(0.25)
    assert block["max"] == 10.0
    assert block["mean"] > block["p50"], "one slow tail call must not hide in the mean"


def test_spend_is_estimated_from_the_configured_price() -> None:
    log = CallLog(input_price_per_1k=0.5, output_price_per_1k=1.5)
    log.record(CallRecord("generate", "m", "gateway", latency_s=1.0, input_tokens=2000, output_tokens=1000))

    assert log.summary()["generate"]["estimated_cost"] == pytest.approx(2.5)


def test_a_log_without_prices_reports_zero_rather_than_inventing_a_number() -> None:
    log = CallLog()
    log.record(CallRecord("generate", "m", "gateway", latency_s=1.0, input_tokens=2000, output_tokens=1000))

    assert log.summary()["generate"]["estimated_cost"] == 0.0


def test_the_log_dumps_one_json_object_per_call(tmp_path: Path) -> None:
    log = CallLog()
    log.record(CallRecord("generate", "m", "gateway", latency_s=1.0, n_prompts=4))
    log.record(CallRecord("score", "m", "cluster", latency_s=0.1, scored_tokens=7))

    path = log.to_jsonl(tmp_path / "nested" / "calls.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["operation"] for row in rows] == ["generate", "score"]
    assert rows[0]["n_prompts"] == 4
    assert rows[1]["backend"] == "cluster", "two serving paths must stay separable"


def test_the_token_count_is_promptolutions_approximation() -> None:
    """Named rather than hidden: whitespace words undercount subwords by about a third."""
    assert approximate_tokens("the answer is forty-two") == 4


# ------------------------------------------------------------------ the backend


def test_a_probe_records_what_it_billed_and_what_it_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response())
    log = CallLog()

    build_llm(log).score("The answer is", " forty-two")

    (record,) = log.records
    assert record.operation == "score"
    assert record.scored_tokens == 2, "exact: the server returns the tokens it scored"
    assert record.input_tokens == approximate_tokens("The answer is") + approximate_tokens(" forty-two")
    assert record.output_tokens == 0
    assert record.latency_s >= 0.0
    assert record.ok


def test_a_failed_probe_is_still_a_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that lost half its probes to errors must not look like a fast run."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(status=400))
    log = CallLog()

    with pytest.raises(XmasError):
        build_llm(log).score("context", "continuation")

    (record,) = log.records
    assert not record.ok
    assert record.error is not None
    assert log.summary()["score"]["failed"] == 1


def test_generation_records_latency_and_the_prompt_count(monkeypatch: pytest.MonkeyPatch) -> None:
    log = CallLog()
    llm = build_llm(log)
    monkeypatch.setattr(type(llm), "_get_response", lambda self, prompts, system_prompts: ["ok"] * len(prompts))

    llm.get_response(["a b c", "d e"])

    (record,) = log.records
    assert record.operation == "generate"
    assert record.n_prompts == 2
    assert record.input_tokens > 0
    assert record.output_tokens > 0


def test_recording_is_off_unless_a_collector_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary use pays no attention to instrumentation."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response())

    result = build_llm().score("The answer is", " forty-two")

    assert result.token_count == 2


# ------------------------------------------------------------------ provenance


def chat_llm(log: CallLog, sent: list[dict[str, Any]], **kwargs: Any) -> APILLM:
    """A gateway model whose chat endpoint is a mock transport under the real openai client.

    Each answer is derived from the user prompt, and so is its completion id, so a
    test can check that text and id stay with the prompt they belong to even when a
    batch completes out of order.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        user = body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "id": f"cmpl-{user.replace(' ', '-')}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": f"answer to {user}"},
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return APILLM(
        model="llama-3.1-8b",
        base_url=BASE_URL,
        api_key="sk-test",
        call_log=log,
        max_retries=0,
        client_kwargs={"http_client": client},
        **kwargs,
    )


def test_capture_is_off_by_default_and_leaves_the_text_out() -> None:
    """Existing users get provenance, not a log that suddenly holds every prompt."""
    log = CallLog()
    sent: list[dict[str, Any]] = []

    chat_llm(log, sent).get_response("what is six times seven", "be terse")

    (record,) = log.records
    assert log.capture_text is False
    assert record.prompt is None
    assert record.system_prompt is None
    assert record.response is None
    assert record.prompts is None
    assert record.responses is None
    assert record.exchanges() == []
    assert record.response_id == "cmpl-what-is-six-times-seven", "the id is provenance, not text"
    assert record.params == {"max_tokens": 4096}


def test_capture_records_the_exact_strings_sent_and_returned() -> None:
    log = CallLog(capture_text=True)
    sent: list[dict[str, Any]] = []
    llm = chat_llm(
        log,
        sent,
        max_tokens=64,
        call_kwargs={"temperature": 0.7, "extra_headers": {"X-Secret": "do-not-log"}},
    )

    (answer,) = llm.get_response("what is six times seven", "be terse")

    (request,) = sent
    (record,) = log.records
    assert record.system_prompt == request["messages"][0]["content"] == "be terse"
    assert record.prompt == request["messages"][1]["content"] == "what is six times seven"
    assert record.response == answer == "answer to what is six times seven"
    assert type(answer) is str, "the caller gets a plain string back"
    assert record.response_id == "cmpl-what-is-six-times-seven"
    assert record.params == {"max_tokens": 64, "temperature": 0.7}
    assert request["max_tokens"] == 64
    assert request["temperature"] == 0.7, "params record what was actually sent"
    assert record.n_prompts == 1
    assert record.prompts is None, "a one-prompt call uses the singular fields"


def test_the_default_system_prompt_is_recorded_as_sent() -> None:
    """A caller that passed no system prompt still sent one, and the record says which."""
    log = CallLog(capture_text=True)
    sent: list[dict[str, Any]] = []

    chat_llm(log, sent).get_response("hello")

    (record,) = log.records
    assert record.system_prompt == sent[0]["messages"][0]["content"]
    assert record.system_prompt


def test_a_captured_batch_is_one_record_with_aligned_lists() -> None:
    """A batch stays one call, so the rate statistics do not move when capture is on."""
    log = CallLog(capture_text=True)
    sent: list[dict[str, Any]] = []
    prompts = ["first question", "second question", "third question"]

    answers = chat_llm(log, sent).get_response(prompts, "be terse")

    (record,) = log.records
    assert record.n_prompts == 3
    assert record.prompt is None, "a batch uses the list fields"
    assert record.prompts == prompts
    assert record.system_prompts == ["be terse"] * 3
    assert record.responses == answers == [f"answer to {prompt}" for prompt in prompts]
    assert record.response_ids == [f"cmpl-{prompt.replace(' ', '-')}" for prompt in prompts]
    assert [exchange["prompt"] for exchange in record.exchanges()] == prompts
    assert record.exchanges()[1]["response_id"] == "cmpl-second-question"


def test_a_failed_generation_keeps_what_was_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    log = CallLog(capture_text=True)
    llm = build_llm(log)

    def refuse(self: APILLM, prompts: list[str], system_prompts: list[str]) -> list[str]:
        raise TimeoutError("batch timed out")

    monkeypatch.setattr(type(llm), "_get_response", refuse)

    with pytest.raises(TimeoutError):
        llm.get_response("a question", "be terse")

    (record,) = log.records
    assert not record.ok
    assert record.prompt == "a question"
    assert record.system_prompt == "be terse"
    assert record.response is None


def test_every_record_gets_its_own_call_id_and_a_parseable_start_time(monkeypatch: pytest.MonkeyPatch) -> None:
    log = CallLog()
    llm = build_llm(log)
    monkeypatch.setattr(type(llm), "_get_response", lambda self, prompts, system_prompts: ["ok"] * len(prompts))
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response())

    llm.get_response("a")
    llm.get_response(["b", "c"])
    llm.score("The answer is", " forty-two")
    log.record(CallRecord("generate", "m", "gateway", latency_s=1.0))

    assert len({record.call_id for record in log.records}) == len(log.records) == 4
    for record in log.records:
        started = datetime.fromisoformat(record.ts)
        assert started.utcoffset() is not None, "a timestamp without an offset is ambiguous"


def test_to_jsonl_appends_when_asked_and_replaces_otherwise(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    first = CallLog()
    first.record(CallRecord("generate", "m", "gateway", latency_s=1.0, params={"max_tokens": 8}))
    first.record(CallRecord("score", "m", "gateway", latency_s=0.1))
    second = CallLog()
    second.record(CallRecord("generate", "m", "gateway", latency_s=2.0))

    first.to_jsonl(path)
    second.to_jsonl(path, append=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["latency_s"] for row in rows] == [1.0, 0.1, 2.0]
    assert rows[0]["params"] == {"max_tokens": 8}
    assert [row["call_id"] for row in rows] == [record.call_id for record in first.records + second.records]

    second.to_jsonl(path)

    assert len(path.read_text().splitlines()) == 1, "the default still overwrites"


# ------------------------------------------------------------------ the limiter


def test_a_429_raises_a_typed_error_carrying_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wall is the system working; the only correct response is to wait as asked."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(429, {"Retry-After": "12"}))

    with pytest.raises(XmasError) as caught:
        build_llm().score("context", "continuation")

    assert caught.value.retry_after == 12.0
    assert caught.value.status == 429
    assert "slow down" in str(caught.value)


def test_a_429_without_a_header_says_so_rather_than_guessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(429))

    with pytest.raises(XmasError) as caught:
        build_llm().score("context", "continuation")

    assert caught.value.retry_after is None


def test_an_unparseable_retry_after_is_absent_rather_than_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HTTP-date form is legal; reading it as seconds would be a very short wait."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(429, {"Retry-After": "Wed, 21 Oct 2026"}))

    with pytest.raises(XmasError) as caught:
        build_llm().score("context", "continuation")

    assert caught.value.retry_after is None


def test_a_rate_limit_is_not_mistaken_for_a_capability_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """501 and 429 both refuse the call, and only one of them is worth waiting out."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(501))

    with pytest.raises(Exception) as caught:
        build_llm().score("context", "continuation")

    assert isinstance(caught.value, XmasError)
    assert caught.value.status == 501


def test_the_mock_model_is_untouched_by_any_of_this(mock_llm: MockLLM) -> None:
    """Instrumentation is opt-in on one backend, not a change to the LLM contract."""
    assert mock_llm.get_response(["hello"]) == ["echo:hello"]
