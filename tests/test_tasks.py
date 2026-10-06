"""GSM8K scoring: numeric where it should be, tolerant where it must be."""

from __future__ import annotations

import pandas as pd
import pytest

from xmas.tasks.gsm8k import GSM8KTask, extract_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 42", 42.0),
        ("18", 18.0),
        ("Janet sells 16 - 3 - 4 = 9 eggs, so she makes 9 * 2 = 18", 18.0),
        ("$1,234.50", 1234.5),
        ("-7 degrees", -7.0),
        ("Natalia sold 48/2 = 24 clips.\n#### 72", 72.0),
        ("no digits here", None),
    ],
)
def test_extract_number(text: str, expected: float | None) -> None:
    assert extract_number(text) == expected


def gsm8k_frame() -> pd.DataFrame:
    """A GSM8K-shaped slice, in the hub's column layout."""
    return pd.DataFrame(
        {
            "question": ["What is 17 * 23?", "What is 100 - 42?"],
            "answer": ["17 * 23 = 391\n#### 391", "100 - 42 = 58\n#### 58"],
        }
    )


def test_scores_a_correct_answer_regardless_of_surrounding_text() -> None:
    task = GSM8KTask(gsm8k_frame())
    scores = task.reward(
        xs=["", ""],
        ys=["17 * 23 = 391\n#### 391", "100 - 42 = 58\n#### 58"],
        preds=["The answer is 391.", "58"],
    )
    assert scores.tolist() == [1.0, 1.0]


def test_numeric_tolerance_accepts_a_differently_formatted_number() -> None:
    """The seed repos' exact-match scorer marked this wrong; formatting is not arithmetic."""
    task = GSM8KTask(gsm8k_frame())
    assert task.reward(xs=[""], ys=["#### 18"], preds=["18.0"]).tolist() == [1.0]
    assert task.reward(xs=[""], ys=["#### 1234"], preds=["1,234"]).tolist() == [1.0]


def test_a_wrong_answer_scores_zero() -> None:
    task = GSM8KTask(gsm8k_frame())
    assert task.reward(xs=[""], ys=["#### 391"], preds=["390"]).tolist() == [0.0]


def test_unparseable_output_falls_back_to_string_equality() -> None:
    task = GSM8KTask(gsm8k_frame())
    assert task.reward(xs=[""], ys=["yes"], preds=["yes"]).tolist() == [1.0]
    assert task.reward(xs=[""], ys=["yes"], preds=["no"]).tolist() == [0.0]


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="has no 'question' column"):
        GSM8KTask(pd.DataFrame({"q": ["x"], "a": ["y"]}))


def test_task_description_defaults_to_something_an_optimizer_can_use() -> None:
    assert "word problem" in (GSM8KTask(gsm8k_frame()).task_description or "")
