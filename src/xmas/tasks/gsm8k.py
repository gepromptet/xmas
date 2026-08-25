"""GSM8K: grade-school maths word problems."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xmas.tasks.base import MASTask


class GSM8KTask(MASTask):
    """GSM8K with numeric-tolerant scoring.

    Answers are numbers, so scoring compares them as numbers. Exact string
    equality would mark ``18.0`` wrong against ``18``, which measures formatting
    rather than arithmetic and makes an optimizer chase output shape instead of
    correctness. This is the fix the mas-optimization students landed in
    ``a1b8dcb``, carried here as the default rather than as a patch.
    """

    def __init__(self, df: pd.DataFrame, tolerance: float = 1e-3, **kwargs: object) -> None:
        """Initialize the task.

        Args:
            df: Dataset with ``question`` and ``answer`` columns.
            tolerance: Relative and absolute tolerance for the numeric comparison.
            **kwargs: Forwarded to :class:`~xmas.tasks.base.MASTask`.
        """
        raise NotImplementedError

    @classmethod
    def from_huggingface(cls, split: str = "test", n: int | None = None, **kwargs: object) -> GSM8KTask:
        """Load GSM8K from the Hugging Face hub.

        Args:
            split: Dataset split to load.
            n: Number of examples to keep, or ``None`` for all.
            **kwargs: Forwarded to the constructor.

        Returns:
            The loaded task.
        """
        raise NotImplementedError

    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """Score predictions numerically, falling back to string equality.

        Args:
            xs: The questions.
            ys: The reference answers.
            preds: The MAS outputs.

        Returns:
            ``1.0`` for a correct answer and ``0.0`` otherwise, per datapoint.
        """
        raise NotImplementedError
