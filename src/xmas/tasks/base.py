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

    What a subclass owns is :meth:`_evaluate`: given inputs, targets and the
    MAS's predictions, return a score per datapoint.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        task_description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the task.

        Args:
            df: The dataset.
            x_column: Column holding the inputs.
            y_column: Column holding the targets.
            task_description: Natural-language description, used by optimizers
                that write prompts.
            **kwargs: Forwarded to promptolution's ``BaseTask``.
        """
        super().__init__(
            df=df,
            x_column=x_column,
            y_column=y_column,
            task_description=task_description,
            **kwargs,
        )
        self.task_type = "reward"

    def reward(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """Score outputs the MAS did not necessarily just produce.

        The public name for :meth:`_evaluate`. promptolution keeps the reward
        protected because generation-based evaluation only ever scores what the
        predictor just returned, so nothing outside the task needed it. Probing
        does: it scores a pool of references decoded earlier, by other
        candidates, and the reward that grades them has to be this task's own or
        the probe is estimating a different quantity than the task defines.

        Args:
            xs: The inputs.
            ys: The targets.
            preds: The outputs to score.

        Returns:
            One score per datapoint.
        """
        return self._evaluate(xs, ys, preds)

    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """Score predictions against targets.

        Args:
            xs: The inputs.
            ys: The targets.
            preds: The MAS outputs.

        Returns:
            One score per datapoint.

        Raises:
            NotImplementedError: Always; subclasses define the reward.
        """
        raise NotImplementedError("A MASTask subclass must define its reward.")
