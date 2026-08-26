"""Call-level statistics, and the limiter wall that must not be guessed at.

Both exist for an experiment whose headline is a wall-time and token comparison, so
what is pinned here is that the numbers mean what the report will say they mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from xmas.llms.api_llm import APILLM, RateLimitedError
from xmas.llms.telemetry import GENERATE, SCORE, CallLog, CallRecord, Price, approximate_tokens

from .conftest import MockLLM

BASE_URL = "https://gateway.example/v1"


def build_llm(telemetry: CallLog | None = None) -> APILLM:
    """A gateway model with no network behind it."""
    return APILLM(model="llama-3.1-8b", base_url=BASE_URL, api_key="sk-test", telemetry=telemetry)


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

    assert summary[GENERATE]["calls"] == 0
    assert summary[SCORE]["calls"] == 0
    assert summary["total"]["latency_s"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}


def test_the_summary_separates_the_two_operations() -> None:
    log = CallLog()
    log.record(CallRecord(GENERATE, "m", "gateway", latency_s=1.0, input_tokens=10, output_tokens=5))
    log.record(CallRecord(SCORE, "m", "gateway", latency_s=0.1, input_tokens=30, scored_tokens=4))

    summary = log.summary()

    assert summary[GENERATE]["calls"] == 1
    assert summary[GENERATE]["decoded_output_tokens"] == 5
    assert summary[SCORE]["calls"] == 1
    assert summary[SCORE]["scored_tokens"] == 4
    assert summary[SCORE]["decoded_output_tokens"] == 0, "a probe decodes nothing"
    assert summary["total"]["calls"] == 2


def test_the_summary_names_the_echo_billing_artifact() -> None:
    """The dollar column prices the protocol, not the method, and has to say so."""
    caveats = CallLog().summary()["caveats"]

    assert "prompt plus continuation" in caveats["score_tokens"]
    assert "budget rail" in caveats["estimated_cost"]


def test_latency_is_reported_as_a_distribution_not_a_mean() -> None:
    log = CallLog()
    for latency in (0.1, 0.2, 0.3, 10.0):
        log.record(CallRecord(SCORE, "m", "gateway", latency_s=latency))

    block = log.summary()[SCORE]["latency_s"]

    assert block["p50"] == pytest.approx(0.25)
    assert block["max"] == 10.0
    assert block["mean"] > block["p50"], "one slow tail call must not hide in the mean"


def test_spend_is_estimated_from_the_configured_price() -> None:
    log = CallLog(price=Price(input_per_1k=0.5, output_per_1k=1.5))
    log.record(CallRecord(GENERATE, "m", "gateway", latency_s=1.0, input_tokens=2000, output_tokens=1000))

    assert log.summary()[GENERATE]["estimated_cost"] == pytest.approx(2.5)


def test_a_log_without_prices_reports_zero_rather_than_inventing_a_number() -> None:
    log = CallLog()
    log.record(CallRecord(GENERATE, "m", "gateway", latency_s=1.0, input_tokens=2000, output_tokens=1000))

    assert log.summary()[GENERATE]["estimated_cost"] == 0.0


def test_the_log_dumps_one_json_object_per_call(tmp_path: Path) -> None:
    log = CallLog()
    log.record(CallRecord(GENERATE, "m", "gateway", latency_s=1.0, n_prompts=4))
    log.record(CallRecord(SCORE, "m", "cluster", latency_s=0.1, scored_tokens=7))

    path = log.to_jsonl(tmp_path / "nested" / "calls.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["operation"] for row in rows] == [GENERATE, SCORE]
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
    assert record.operation == SCORE
    assert record.scored_tokens == 2, "exact: the server returns the tokens it scored"
    assert record.input_tokens == approximate_tokens("The answer is") + approximate_tokens(" forty-two")
    assert record.output_tokens == 0
    assert record.latency_s >= 0.0
    assert record.ok


def test_a_failed_probe_is_still_a_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that lost half its probes to errors must not look like a fast run."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(status=400))
    log = CallLog()

    with pytest.raises(ValueError):
        build_llm(log).score("context", "continuation")

    (record,) = log.records
    assert not record.ok
    assert record.error is not None
    assert log.summary()[SCORE]["failed"] == 1


def test_generation_records_latency_and_the_prompt_count(monkeypatch: pytest.MonkeyPatch) -> None:
    log = CallLog()
    llm = build_llm(log)
    monkeypatch.setattr(type(llm), "_get_response", lambda self, prompts, system_prompts: ["ok"] * len(prompts))

    llm.get_response(["a b c", "d e"])

    (record,) = log.records
    assert record.operation == GENERATE
    assert record.n_prompts == 2
    assert record.input_tokens > 0
    assert record.output_tokens > 0


def test_recording_is_off_unless_a_collector_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary use pays no attention to instrumentation."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response())

    result = build_llm().score("The answer is", " forty-two")

    assert result.token_count == 2


# ------------------------------------------------------------------ the limiter


def test_a_429_raises_a_typed_error_carrying_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wall is the system working; the only correct response is to wait as asked."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(429, {"Retry-After": "12"}))

    with pytest.raises(RateLimitedError) as caught:
        build_llm().score("context", "continuation")

    assert caught.value.retry_after == 12.0
    assert "slow down" in str(caught.value)


def test_a_429_without_a_header_says_so_rather_than_guessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(429))

    with pytest.raises(RateLimitedError) as caught:
        build_llm().score("context", "continuation")

    assert caught.value.retry_after is None


def test_an_unparseable_retry_after_is_absent_rather_than_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HTTP-date form is legal; reading it as seconds would be a very short wait."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(429, {"Retry-After": "Wed, 21 Oct 2026"}))

    with pytest.raises(RateLimitedError) as caught:
        build_llm().score("context", "continuation")

    assert caught.value.retry_after is None


def test_a_rate_limit_is_not_mistaken_for_a_capability_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """501 and 429 both refuse the call, and only one of them is worth waiting out."""
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: score_response(501))

    with pytest.raises(Exception) as caught:
        build_llm().score("context", "continuation")

    assert not isinstance(caught.value, RateLimitedError)


def test_the_mock_model_is_untouched_by_any_of_this(mock_llm: MockLLM) -> None:
    """Instrumentation is opt-in on one backend, not a change to the LLM contract."""
    assert mock_llm.get_response(["hello"]) == ["echo:hello"]
