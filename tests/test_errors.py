"""Error kinds and the trace format: classification, status, and v1-compatible dicts."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx2
import openai
import pydantic
import pytest
import yaml

from xmas.errors import BudgetExceeded, ErrorKind, RunStatus, XmasError, classify, kind_of_finish
from xmas.trace import AgentRecord, TokenUsage, Trace

from .conftest import URL, status_error


def validation_error() -> pydantic.ValidationError:
    try:
        pydantic.TypeAdapter(int).validate_python("not a number")
    except pydantic.ValidationError as error:
        return error
    raise AssertionError("validate_python accepted a string")


def raised_from(cause: BaseException) -> RuntimeError:
    """Build promptolution's batch failure the way it raises it: ``... from first_exc``."""
    error = RuntimeError(f"LLM batch failed: {cause}")
    error.__cause__ = cause
    return error


def cyclic(error: BaseException) -> BaseException:
    error.__context__ = error
    return error


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (status_error(openai.APIStatusError, 402), ErrorKind.MONEY),
        (status_error(openai.PermissionDeniedError, 403, "Key limit exceeded (weekly limit)"), ErrorKind.MONEY),
        (
            status_error(openai.RateLimitError, 429, body={"error": {"code": "insufficient_quota"}}),
            ErrorKind.MONEY,
        ),
        (status_error(openai.RateLimitError, 429), ErrorKind.RATE_LIMITED),
        (status_error(openai.InternalServerError, 503), ErrorKind.OUTAGE),
        (openai.APIConnectionError(request=httpx2.Request("POST", URL)), ErrorKind.OUTAGE),
        (openai.APITimeoutError(request=httpx2.Request("POST", URL)), ErrorKind.TIMEOUT),
        (
            status_error(openai.BadRequestError, 400, body={"error": {"message": "Input was flagged by moderation"}}),
            ErrorKind.REFUSAL,
        ),
        (validation_error(), ErrorKind.BAD_FORMAT),
        (BudgetExceeded("calls", 1, 1, 1), ErrorKind.BUDGET),
        (raised_from(status_error(openai.APIStatusError, 402)), ErrorKind.MONEY),
        (cyclic(RuntimeError("boom")), ErrorKind.UNKNOWN),
        (httpx2.ConnectError("refused", request=httpx2.Request("POST", URL)), ErrorKind.OUTAGE),
        (httpx2.ReadTimeout("slow", request=httpx2.Request("POST", URL)), ErrorKind.TIMEOUT),
        (raised_from(httpx2.RemoteProtocolError("closed", request=httpx2.Request("POST", URL))), ErrorKind.OUTAGE),
        (XmasError("mcp: unauthorized (401): bad key", status=401), ErrorKind.REFUSAL),
        (XmasError("mcp: tool_not_in_scope (403): nope", status=403), ErrorKind.REFUSAL),
        (XmasError("mcp: too_many_concurrent (429): 4 in flight", status=429, retry_after=1.0), ErrorKind.RATE_LIMITED),
    ],
    ids=[
        "402",
        "403-key-limit",
        "429-insufficient-quota-in-body",
        "429",
        "503",
        "connection",
        "timeout",
        "400-moderation",
        "validation",
        "budget",
        "batch-failed-from-402",
        "plain-runtime-error-in-a-cycle",
        "httpx2-connect",
        "httpx2-timeout",
        "mcp-failure-from-httpx2",
        "tool-source-401",
        "tool-source-403",
        "tool-source-429",
    ],
)
def test_classify_reads_types_statuses_bodies_and_chains(error: BaseException, kind: ErrorKind) -> None:
    assert classify(error) is kind


def test_kind_of_finish_reads_cuts_and_refusals() -> None:
    assert kind_of_finish("length") is ErrorKind.LENGTH_CUT
    assert kind_of_finish("content_filter") is ErrorKind.REFUSAL
    assert kind_of_finish("stop", refused=True) is ErrorKind.REFUSAL
    assert kind_of_finish("stop") is None
    assert kind_of_finish(None) is None


V1_RECORD_KEYS = [
    "agent",
    "step",
    "prompt",
    "rendered_prompt",
    "inputs",
    "outputs",
    "tool_calls",
    "usage",
    "model",
    "raw_response",
]
V1_TRACE_KEYS = ["query", "output", "records", "error"]
NEWER_RECORD_FIELDS: dict[str, Any] = {
    "latency_s": 0.25,
    "cost_usd": 0.0012,
    "tokens": TokenUsage(11, 7, cached_tokens=4),
    "finish_reason": "length",
    "error_kind": ErrorKind.LENGTH_CUT,
    "served_model": "vendor/model-1",
    "logprobs": (("4", -0.1),),
    "response_id": "cmpl-1",
    "system_prompt": "Be brief.",
}


def test_trace_dicts_keep_v1_keys_and_add_only_set_fields() -> None:
    record = AgentRecord("solver", 0, "Solve {query}", "Solve q", {"query": "q"}, {"draft": "4"})
    assert list(record.to_dict()) == V1_RECORD_KEYS
    assert list(record.to_dict()["usage"]) == ["input_tokens", "output_tokens", "total_tokens"]

    for name, value in NEWER_RECORD_FIELDS.items():
        assert list(replace(record, **{name: value}).to_dict()) == [*V1_RECORD_KEYS, name]
    full = replace(record, **NEWER_RECORD_FIELDS).to_dict()
    assert list(full) == V1_RECORD_KEYS + list(NEWER_RECORD_FIELDS)
    assert type(full["error_kind"]) is str
    assert full["error_kind"] == "length_cut"
    assert full["logprobs"] == [["4", -0.1]]
    assert full["tokens"] == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18, "cached_tokens": 4}
    assert (TokenUsage(1, 2, reasoning_tokens=1) + TokenUsage(3, 4)).to_dict()["reasoning_tokens"] == 1
    assert list((TokenUsage(1, 2) + TokenUsage(3, 4)).to_dict()) == ["input_tokens", "output_tokens", "total_tokens"]

    ok = Trace("q", "4")
    assert list(ok.to_dict()) == V1_TRACE_KEYS
    assert ok.status is RunStatus.OK
    assert ok.cost_usd is None  # no records: the cost is unknown, not 0

    failed = Trace("q", "", records=(record,), error="RuntimeError: boom")
    assert failed.status is RunStatus.FAILED
    assert list(failed.to_dict()) == [*V1_TRACE_KEYS, "status"]
    assert type(failed.to_dict()["status"]) is str
    assert failed.to_dict()["status"] == "failed"
    assert failed.cost_usd is None

    costed = replace(record, **NEWER_RECORD_FIELDS)
    over = Trace(
        "q",
        "",
        records=(costed, costed),
        error="BudgetExceeded: calls 2 >= max 2 before call 3",
        error_kind=ErrorKind.BUDGET,
    )
    assert over.status is RunStatus.OVER_BUDGET
    assert list(over.to_dict()) == [*V1_TRACE_KEYS, "status", "error_kind"]
    assert (over.to_dict()["status"], over.to_dict()["error_kind"]) == ("over_budget", "budget")
    assert over.cost_usd == pytest.approx(0.0024)

    for data in (record.to_dict(), full, ok.to_dict(), failed.to_dict(), over.to_dict()):
        assert yaml.safe_load(yaml.safe_dump(data)) == data
        assert json.loads(json.dumps(data)) == data
