"""The task a MAS is optimized against."""

from __future__ import annotations

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
        **kwargs: object,
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
        raise NotImplementedError

    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """Score predictions against targets.

        Args:
            xs: The inputs.
            ys: The targets.
            preds: The MAS outputs.

        Returns:
            One score per datapoint.
        """
        raise NotImplementedError
