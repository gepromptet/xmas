"""Model backends.

Each backend subclasses its promptolution counterpart and adds exactly one thing:
``score``, the teacher-forced sibling of generation.
"""

from xmas.llms.api_llm import APILLM, ScoringUnsupportedError
from xmas.llms.masking import assistant_span, mask_logprobs
from xmas.llms.scoring import ScoredSequence
from xmas.llms.vllm import VLLM

__all__ = [
    "APILLM",
    "VLLM",
    "ScoredSequence",
    "ScoringUnsupportedError",
    "assistant_span",
    "mask_logprobs",
]
