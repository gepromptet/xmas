"""The assistant-token masking utility.

Teacher-forced scoring is only meaningful if the right tokens are scored, so the
alignment rules get their own tests: an approximate span would produce a number
that looks correct and is not.
"""

from __future__ import annotations

import pytest

from xmas.llms.masking import assistant_span, mask_logprobs


def test_span_finds_a_continuation_on_token_boundaries() -> None:
    tokens = ["The", " answer", " is", " forty", "-two"]
    assert assistant_span(tokens, " forty-two") == (3, 5)


def test_span_covers_the_whole_sequence_when_it_all_matches() -> None:
    assert assistant_span(["ab", "cd"], "abcd") == (0, 2)


def test_span_rejects_a_continuation_that_straddles_a_token() -> None:
    """`orty-two` starts mid-token, so no exact sequence log-probability exists."""
    with pytest.raises(ValueError, match="does not align"):
        assistant_span(["The", " answer", " is", " forty", "-two"], "orty-two")


def test_mask_returns_only_the_continuation_logprobs() -> None:
    tokens = ["Q", ":", " 2+2", " is", " four"]
    logprobs = [None, -0.1, -0.2, -0.3, -0.4]
    assert mask_logprobs(tokens, logprobs, " is four") == [-0.3, -0.4]


def test_mask_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="3 tokens but 2"):
        mask_logprobs(["a", "b", "c"], [-0.1, -0.2], "c")


def test_mask_rejects_an_unconditioned_first_token() -> None:
    """The first prompt token has no log-probability, so it cannot be scored."""
    with pytest.raises(ValueError, match="no log-probability"):
        mask_logprobs(["ab", "cd"], [None, -0.2], "abcd")
