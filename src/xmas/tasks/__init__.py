"""Tasks a MAS can be optimized against."""

from xmas.tasks.base import MASTask
from xmas.tasks.gsm8k import GSM8KTask
from xmas.tasks.probing import (
    AgreementReadout,
    Estimator,
    ImportanceSampling,
    ProbeTensor,
    ProbingTask,
    Reference,
    ReferencePool,
    importance_weights,
    materialize,
    measure_agreement,
    negative_control,
    rank_agreement,
)

__all__ = [
    "AgreementReadout",
    "Estimator",
    "GSM8KTask",
    "ImportanceSampling",
    "MASTask",
    "ProbeTensor",
    "ProbingTask",
    "Reference",
    "ReferencePool",
    "importance_weights",
    "materialize",
    "measure_agreement",
    "negative_control",
    "rank_agreement",
]
