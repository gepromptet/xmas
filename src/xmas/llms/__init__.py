"""Model backends.

Each backend subclasses its promptolution counterpart and adds exactly one thing:
``score``, the teacher-forced sibling of generation.
"""

from xmas.llms.api_llm import APILLM, RateLimitedError, ScoringUnsupportedError
from xmas.llms.masking import assistant_span, mask_logprobs
from xmas.llms.scoring import ScoredSequence, SupportsScore
from xmas.llms.telemetry import CallLog, CallRecord, Price

__all__ = [
    "APILLM",
    "CallLog",
    "CallRecord",
    "Price",
    "RateLimitedError",
    "ScoredSequence",
    "ScoringUnsupportedError",
    "SupportsScore",
    "assistant_span",
    "mask_logprobs",
]
