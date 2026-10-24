"""GSM8K: grade-school maths word problems."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from xmas.tasks.base import MASTask

__all__ = [
    "GSM8KTask",
    "extract_number",
    "is_correct",
]


class GSM8KTask(MASTask):
    """GSM8K with numeric-tolerant scoring.

    Answers are numbers, so scoring compares them as numbers. Exact string
    equality would mark ``18.0`` wrong against ``18``, which measures formatting
    rather than arithmetic and makes an optimizer chase output shape instead of
    correctness. This is the fix the mas-optimization students landed in
    ``a1b8dcb``, carried here as the default rather than as a patch.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tolerance: float = 1e-3,
        task_description: str = "Solve a grade-school maths word problem and give the final numeric answer.",
        **kwargs: Any,
    ) -> None:
        """Initialize the task.

        Args:
            df: Dataset with ``question`` and ``answer`` columns.
            tolerance: Relative and absolute tolerance for the numeric comparison.
            task_description: Natural-language description, used by optimizers that write prompts.
            **kwargs: Forwarded to :class:`~xmas.tasks.base.MASTask`.

        Raises:
            ValueError: If the expected columns are missing.
        """
        for column in ["question", "answer"]:
            if column not in df.columns:
                raise ValueError(f"GSM8K data has no {column!r} column.")
        self.tolerance = tolerance
        super().__init__(df=df, x_column="question", y_column="answer", task_description=task_description, **kwargs)

    @classmethod
    def from_huggingface(cls, split: str = "test", n: int | None = None, **kwargs: Any) -> GSM8KTask:
        """Load GSM8K from the Hugging Face hub.

        Args:
            split: Dataset split to load.
            n: Number of examples to keep, or ``None`` for all.
            **kwargs: Forwarded to the constructor.

        Returns:
            The loaded task.
        """
        from datasets import load_dataset  # noqa: PLC0415  optional dependency, imported only when this loader runs

        data = load_dataset("openai/gsm8k", "main", split=split)
        df = pd.DataFrame(data)
        if n is not None:
            df = df.head(n)
        return cls(df, **kwargs)

    # xs unused: GSM8K is scored against the reference alone
    def reward(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:  # noqa: ARG002
        """Score predictions numerically, falling back to string equality.

        Args:
            xs: The questions.
            ys: The reference answers.
            preds: The MAS outputs.

        Returns:
            ``1.0`` for a correct answer and ``0.0`` otherwise, per datapoint.
        """
        return np.array([is_correct(y, pred, self.tolerance) for y, pred in zip(ys, preds, strict=True)], dtype=float)


def extract_number(text: str) -> float | None:
    """Return the last number in a piece of text.

    The last one rather than the first: a model that shows its working states
    intermediate values before the answer, and the answer is what it ends on.

    Args:
        text: Free-form model output or a reference answer.

    Returns:
        The parsed number, or ``None`` if the text contains none.
    """
    # GSM8K references end in `#### 18`; the worked solution before it is not the answer
    if "####" in text:
        text = text.rsplit("####", maxsplit=1)[-1]
    # numbers as models write them: 1,234.5 / -7 / $18 / 42
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return float(matches[-1].replace(",", "").rstrip("."))


def is_correct(reference: str, prediction: str, tolerance: float) -> bool:
    """Whether one prediction matches one reference, numerically when both hold a number."""
    target = extract_number(reference)
    predicted = extract_number(prediction)
    if target is None or predicted is None:
        return reference.strip() == prediction.strip()
    return math.isclose(target, predicted, rel_tol=tolerance, abs_tol=tolerance)
