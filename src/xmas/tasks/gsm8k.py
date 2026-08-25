"""GSM8K: grade-school maths word problems."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from xmas.tasks.base import MASTask

QUESTION_COLUMN = "question"
"""Column holding the word problem."""

ANSWER_COLUMN = "answer"
"""Column holding the reference answer."""

TASK_DESCRIPTION = "Solve a grade-school maths word problem and give the final numeric answer."

# GSM8K reference answers arrive as `... #### 18`. Everything before the marker
# is the worked solution, which is not what a prediction is scored against.
_FINAL_ANSWER = re.compile(r"####\s*(.+)\s*$")

# Numbers as models write them: 1,234.5 / -7 / $18 / 42.
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str) -> float | None:
    """Return the last number in a piece of text.

    The last one rather than the first: a model that shows its working states
    intermediate values before the answer, and the answer is what it ends on.

    Args:
        text: Free-form model output or a reference answer.

    Returns:
        The parsed number, or ``None`` if the text contains none.
    """
    marked = _FINAL_ANSWER.search(text)
    if marked:
        text = marked.group(1)

    matches = _NUMBER.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


class GSM8KTask(MASTask):
    """GSM8K with numeric-tolerant scoring.

    Answers are numbers, so scoring compares them as numbers. Exact string
    equality would mark ``18.0`` wrong against ``18``, which measures formatting
    rather than arithmetic and makes an optimizer chase output shape instead of
    correctness. This is the fix the mas-optimization students landed in
    ``a1b8dcb``, carried here as the default rather than as a patch.
    """

    def __init__(self, df: pd.DataFrame, tolerance: float = 1e-3, **kwargs: Any) -> None:
        """Initialize the task.

        Args:
            df: Dataset with ``question`` and ``answer`` columns.
            tolerance: Relative and absolute tolerance for the numeric comparison.
            **kwargs: Forwarded to :class:`~xmas.tasks.base.MASTask`.

        Raises:
            ValueError: If the expected columns are missing.
        """
        missing = {QUESTION_COLUMN, ANSWER_COLUMN} - set(df.columns)
        if missing:
            raise ValueError(f"GSM8K data is missing columns: {sorted(missing)}.")

        self.tolerance = tolerance
        kwargs.setdefault("task_description", TASK_DESCRIPTION)
        super().__init__(df=df, x_column=QUESTION_COLUMN, y_column=ANSWER_COLUMN, **kwargs)

    @classmethod
    def from_huggingface(cls, split: str = "test", n: int | None = None, **kwargs: Any) -> GSM8KTask:
        """Load GSM8K from the Hugging Face hub.

        ``datasets`` is imported here rather than declared as a dependency: it
        pulls a large tree, and a framework that only sometimes loads a public
        benchmark should not make every install pay for it.

        Args:
            split: Dataset split to load.
            n: Number of examples to keep, or ``None`` for all.
            **kwargs: Forwarded to the constructor.

        Returns:
            The loaded task.

        Raises:
            ImportError: If ``datasets`` is not installed.
        """
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "GSM8KTask.from_huggingface needs the `datasets` package. "
                "Install it, or build the task from a DataFrame with question/answer columns."
            ) from error

        data = load_dataset("openai/gsm8k", "main", split=split)
        df = pd.DataFrame(data)
        if n is not None:
            df = df.head(n)
        return cls(df, **kwargs)

    # `xs` is unused: GSM8K is scored against the reference answer alone. The
    # signature is promptolution's, and a task that did need the inputs (a judge,
    # say) would use it, so the parameter stays.
    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:  # noqa: ARG002
        """Score predictions numerically, falling back to string equality.

        Args:
            xs: The questions.
            ys: The reference answers.
            preds: The MAS outputs.

        Returns:
            ``1.0`` for a correct answer and ``0.0`` otherwise, per datapoint.
        """
        return np.array([self._score_one(y, pred) for y, pred in zip(ys, preds, strict=True)], dtype=float)

    def _score_one(self, reference: str, prediction: str) -> float:
        """Score one prediction against one reference."""
        target = extract_number(reference)
        predicted = extract_number(prediction)
        if target is None or predicted is None:
            return float(reference.strip() == prediction.strip())
        return float(math.isclose(target, predicted, rel_tol=self.tolerance, abs_tol=self.tolerance))
