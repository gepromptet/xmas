"""Shared test fixtures.

The mock model is the reason CI can run the whole framework end to end without a
key: it satisfies promptolution's LLM contract exactly, so nothing under test has
to know it is not talking to a real endpoint.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
from promptolution.llms.base_llm import BaseLLM


class MockLLM(BaseLLM):
    """A deterministic model that records every prompt it was given."""

    def __init__(self, responder: Callable[[str], str] | None = None) -> None:
        """Initialize the mock.

        Args:
            responder: Maps a rendered prompt to a response. Defaults to echoing
                the prompt back, which makes substitution bugs visible in output.
        """
        super().__init__()
        self.responder = responder or (lambda prompt: f"echo:{prompt}")
        self.calls: list[str] = []

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        """Return one scripted response per prompt."""
        self.calls.extend(prompts)
        return [self.responder(prompt) for prompt in prompts]


@pytest.fixture
def mock_llm() -> MockLLM:
    """A model that echoes its prompt."""
    return MockLLM()


def gateway_configured() -> bool:
    """Whether the environment has what a live gateway call needs."""
    return bool(os.environ.get("XMAS_BASE_URL")) and bool(os.environ.get("XMAS_API_KEY"))


requires_gateway = pytest.mark.skipif(
    not gateway_configured(),
    reason="set XMAS_BASE_URL and XMAS_API_KEY to run live gateway tests",
)
"""Skip marker for tests that need Tom's gateway. CI has no key, so these skip."""
