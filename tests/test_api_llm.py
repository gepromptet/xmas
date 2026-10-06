"""APILLM: construction ergonomics and the teacher-forced scoring path."""

from __future__ import annotations

import httpx
import pytest

from xmas.errors import XmasError
from xmas.llms.api_llm import APILLM

from .conftest import requires_gateway

BASE_URL = "https://gateway.example/v1"


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
    """Env-gated. Skips when the gateway cannot score, which is true today.

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
