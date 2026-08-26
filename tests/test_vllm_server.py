"""VLLMServerLLM: the echo-scoring path, without a server behind it.

The live equivalent lives in `test_live_gateway.py`'s neighbourhood and needs a
cluster; what is pinned here is the arithmetic and the refusals, which are what
would silently corrupt a probe tensor rather than fail loudly.
"""

from __future__ import annotations

import httpx
import pytest

from xmas.llms.api_llm import RateLimitedError
from xmas.llms.scoring import SupportsScore
from xmas.llms.telemetry import SCORE, CallLog
from xmas.llms.vllm_server import BACKEND_TAG, VLLMServerLLM

BASE_URL = "https://releagpu01.example/v1"

# A Qwen3-style rendered context: chat template applied, think block closed empty,
# assistant turn open. The scorer never builds this; the caller hands it over.
CONTEXT = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CONTINUATION = "The answer is 4."

PROMPT_TOKENS = [
    "<|im_start|>",
    "user",
    "\n",
    "What",
    " is",
    " 2",
    "+",
    "2",
    "?",
    "<|im_end|>",
    "\n",
    "<|im_start|>",
    "assistant",
    "\n",
    "<think>",
    "\n\n",
    "</think>",
    "\n\n",
]
SPAN_TOKENS = ["The", " answer", " is", " 4", "."]
SPAN_LOGPROBS = [-0.1, -0.2, -0.3, -0.4, -0.5]


def build_llm(**kwargs: object) -> VLLMServerLLM:
    """A backend pointed at a URL nothing will actually call."""
    return VLLMServerLLM(model="qwen3-8b", base_url=BASE_URL, **kwargs)  # type: ignore[arg-type]


def echo_choice(index: int, tokens: list[str], logprobs: list[float | None]) -> dict[str, object]:
    """One `choices[i]` of a `{"echo": true, "logprobs": 0, "max_tokens": 0}` answer."""
    return {"index": index, "text": "".join(tokens), "logprobs": {"tokens": tokens, "token_logprobs": logprobs}}


def echo_response(pairs: list[tuple[list[str], list[float | None]]], status: int = 200) -> httpx.Response:
    """A server answer carrying one echoed prompt per requested prompt."""
    choices = [echo_choice(index, tokens, logprobs) for index, (tokens, logprobs) in enumerate(pairs)]
    return httpx.Response(
        status_code=status,
        json={"choices": choices},
        request=httpx.Request("POST", BASE_URL),
    )


def full_echo() -> tuple[list[str], list[float | None]]:
    """The whole prompt echoed: context tokens, then the continuation's."""
    tokens = [*PROMPT_TOKENS, *SPAN_TOKENS]
    logprobs: list[float | None] = [None, *[-1.0] * (len(PROMPT_TOKENS) - 1), *SPAN_LOGPROBS]
    return tokens, logprobs


def capture(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> dict[str, object]:
    """Replace the transport and record what was sent."""
    seen: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        seen["url"] = url
        seen["json"] = kwargs.get("json")
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    return seen


# ------------------------------------------------------------------ the request


def test_it_satisfies_the_scoring_protocol() -> None:
    """The probing layer depends on the operation, not on this class.

    Checked by assignment rather than ``isinstance``: ``SupportsScore`` is not
    runtime-checkable, and a static check is the stronger one anyway, since it
    covers the signature and not merely the presence of an attribute. mypy runs
    over ``tests/``, so this line fails the type job if the contract drifts.
    """
    scorer: SupportsScore = build_llm()
    assert callable(scorer.score)


def test_the_probe_asks_for_an_echo_and_decodes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = capture(monkeypatch, echo_response([full_echo()]))

    build_llm().score(CONTEXT, CONTINUATION)

    assert seen["url"] == f"{BASE_URL}/completions"
    assert seen["json"] == {
        "model": "qwen3-8b",
        "prompt": [CONTEXT + CONTINUATION],
        "echo": True,
        "logprobs": 0,
        "max_tokens": 0,
        "temperature": 0.0,
    }


def test_the_span_is_cut_out_of_the_echo_rather_than_summed_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """The context's own log-probabilities must not enter the score."""
    capture(monkeypatch, echo_response([full_echo()]))

    result = build_llm().score(CONTEXT, CONTINUATION)

    assert result.tokens == tuple(SPAN_TOKENS)
    assert result.token_logprobs == tuple(SPAN_LOGPROBS)
    assert result.sum_logprob == pytest.approx(sum(SPAN_LOGPROBS))
    assert result.token_count == len(SPAN_TOKENS)


def test_many_continuations_ride_one_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probes are prefill-only, so batching them is the workload a server schedules well."""
    seen = capture(monkeypatch, echo_response([full_echo(), full_echo()]))

    results = build_llm().score_many([(CONTEXT, CONTINUATION), (CONTEXT, CONTINUATION)])

    assert len(results) == 2
    assert isinstance(seen["json"], dict)
    assert len(seen["json"]["prompt"]) == 2, "both pairs must travel in one call"


def test_a_batch_is_chunked_to_the_configured_size(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        calls.append(len(payload["prompt"]))
        return echo_response([full_echo()] * len(payload["prompt"]))

    monkeypatch.setattr(httpx, "post", fake_post)

    build_llm(score_batch_size=2).score_many([(CONTEXT, CONTINUATION)] * 5)

    assert calls == [2, 2, 1]


def test_a_batch_size_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="score_batch_size"):
        build_llm(score_batch_size=0)


# ------------------------------------------------------------------ the refusals


def test_an_out_of_order_answer_is_reordered_by_its_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe attributed to the wrong continuation is invisible in the tensor."""
    first = ([*PROMPT_TOKENS, "one"], [None, *[-1.0] * (len(PROMPT_TOKENS) - 1), -0.25])
    second = ([*PROMPT_TOKENS, "two"], [None, *[-1.0] * (len(PROMPT_TOKENS) - 1), -0.75])
    response = httpx.Response(
        status_code=200,
        json={"choices": [echo_choice(1, *second), echo_choice(0, *first)]},
        request=httpx.Request("POST", BASE_URL),
    )
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: response)

    results = build_llm().score_many([(CONTEXT, "one"), (CONTEXT, "two")])

    assert [r.sum_logprob for r in results] == [-0.25, -0.75]


def test_a_choice_count_that_does_not_match_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    capture(monkeypatch, echo_response([full_echo()]))

    with pytest.raises(ValueError, match="one-to-one"):
        build_llm().score_many([(CONTEXT, CONTINUATION), (CONTEXT, CONTINUATION)])


def test_an_answer_without_an_echo_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silently scoring nothing would read downstream as a very confident model."""
    response = httpx.Response(
        status_code=200,
        json={"choices": [{"index": 0, "text": "", "logprobs": None}]},
        request=httpx.Request("POST", BASE_URL),
    )
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: response)

    with pytest.raises(ValueError, match="missing echoed tokens"):
        build_llm().score(CONTEXT, CONTINUATION)


def test_a_continuation_off_the_token_boundaries_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """No exact sequence log-probability exists, and a plausible one is worse than none."""
    capture(monkeypatch, echo_response([full_echo()]))

    with pytest.raises(ValueError, match="does not align"):
        build_llm().score(CONTEXT, "he answer is 4.")


def test_an_empty_continuation_is_refused() -> None:
    with pytest.raises(ValueError, match="empty continuation"):
        build_llm().score(CONTEXT, "")


def test_an_error_status_names_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(status_code=400, text="bad request", request=httpx.Request("POST", BASE_URL))
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: response)

    with pytest.raises(ValueError, match="Scoring failed \\(400\\)"):
        build_llm().score(CONTEXT, CONTINUATION)


# ------------------------------------------------------------------ cost and limiter


def test_a_batched_request_is_recorded_as_one_call_carrying_every_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    capture(monkeypatch, echo_response([full_echo(), full_echo()]))
    log = CallLog()

    build_llm(telemetry=log).score_many([(CONTEXT, CONTINUATION), (CONTEXT, CONTINUATION)])

    assert len(log) == 1
    record = log.records[0]
    assert record.operation == SCORE
    assert record.backend == BACKEND_TAG, "probes from the vLLM path must not pool with the gateway's"
    assert record.n_prompts == 2
    assert record.output_tokens == 0
    assert record.scored_tokens == 2 * len(SPAN_TOKENS)
    expected_input = 2 * (len(CONTEXT.split()) + len(CONTINUATION.split()))
    assert record.input_tokens == expected_input


def test_a_failed_request_is_recorded_before_it_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(status_code=400, text="bad request", request=httpx.Request("POST", BASE_URL))
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: response)
    log = CallLog()

    with pytest.raises(ValueError, match="Scoring failed"):
        build_llm(telemetry=log).score(CONTEXT, CONTINUATION)

    assert len(log) == 1
    assert not log.records[0].ok


def test_a_429_raises_the_typed_limiter_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(
        status_code=429,
        json={"error": {"message": "slow down"}},
        headers={"Retry-After": "7"},
        request=httpx.Request("POST", BASE_URL),
    )
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: response)

    with pytest.raises(RateLimitedError) as caught:
        build_llm().score(CONTEXT, CONTINUATION)

    assert caught.value.retry_after == 7.0


def test_score_batch_rides_the_batched_echo_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = capture(monkeypatch, echo_response([full_echo(), full_echo()]))

    results = build_llm().score_batch([CONTEXT, CONTEXT], [CONTINUATION, CONTINUATION])

    assert len(results) == 2
    payload = seen["json"]
    assert isinstance(payload, dict)
    assert len(payload["prompt"]) == 2
