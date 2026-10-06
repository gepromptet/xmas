"""The task a MAS is optimized against."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from promptolution.tasks.base_task import BaseTask


class MASTask(BaseTask):
    """A dataset plus a reward over MAS outputs.

    Subclasses promptolution's ``BaseTask`` and inherits its narrow contract
    unchanged: subsampling, caching and ``evaluate(prompts, predictor)`` all come
    from there. That is the point. An optimizer talks to a task through
    ``evaluate`` and nothing else, so any optimizer runs against any task, and a
    later probing task can wrap this one without a single optimizer noticing.

    What a subclass owns is :meth:`reward`: given inputs, targets and the
    MAS's predictions, return a score per datapoint.
    """

    def __init__(self, df: pd.DataFrame, x_column: str, y_column: str, **kwargs: Any) -> None:
        """Initialize the task.

        Args:
            df: The dataset.
            x_column: Column holding the inputs.
            y_column: Column holding the targets.
            **kwargs: Forwarded to promptolution's ``BaseTask``, e.g. ``task_description``.
        """
        super().__init__(df=df, x_column=x_column, y_column=y_column, **kwargs)
        self.task_type = "reward"

    def reward(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """Score outputs; subclasses define it. Probing calls it on references decoded earlier, so it is public.

        Args:
            xs: The inputs.
            ys: The targets.
            preds: The outputs to score.

        Returns:
            One score per datapoint.
        """
        raise NotImplementedError("A MASTask subclass must define its reward.")

    # promptolution's abstract hook; the underscore name is promptolution's and cannot change.
    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        return self.reward(xs, ys, preds)
