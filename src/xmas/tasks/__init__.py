"""Tasks a MAS can be optimized against."""

from xmas.tasks.base import MASTask
from xmas.tasks.gsm8k import GSM8KTask
from xmas.tasks.probing import (
    Estimator,
    ImportanceSampling,
    ProbeTensor,
    ProbingTask,
    Reference,
    ReferencePool,
    importance_weights,
    rank_agreement,
)

__all__ = [
    "Estimator",
    "GSM8KTask",
    "ImportanceSampling",
    "MASTask",
    "ProbeTensor",
    "ProbingTask",
    "Reference",
    "ReferencePool",
    "importance_weights",
    "rank_agreement",
]
