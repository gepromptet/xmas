"""Model backends: promptolution's APILLM plus ``score`` (teacher-forced log-probabilities) and a per-call log.

:class:`APILLM` can also return call metadata (:class:`CallMeta`) next to its answers, read
prompts with pictures and sound in them (:mod:`xmas.llms.parts`), and answer with images.
"""

from xmas.llms.api_llm import APILLM
from xmas.llms.call_log import CallLog, CallRecord
from xmas.llms.meta import CallMeta, SupportsImages, SupportsMeta, SupportsParts, TokenPrice
from xmas.llms.parts import Part
from xmas.llms.scoring import ScoredSequence

__all__ = [
    "APILLM",
    "CallLog",
    "CallMeta",
    "CallRecord",
    "Part",
    "ScoredSequence",
    "SupportsImages",
    "SupportsMeta",
    "SupportsParts",
    "TokenPrice",
]
