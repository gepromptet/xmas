"""vLLM-backed model with teacher-forced scoring."""

from __future__ import annotations

from typing import Any

from promptolution.llms import vllm as _promptolution

from xmas.llms.scoring import ScoredSequence


class VLLM(_promptolution.VLLM):
    """A locally served model that can also score a fixed continuation.

    Generation is inherited from promptolution. :meth:`score` is added on top and
    uses vLLM's ``prompt_logprobs``, which reports log-probabilities for the
    tokens of the prompt itself: supply ``context + continuation`` as the prompt
    and the continuation's tokens are scored without being generated.
    """

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        """Initialize the model.

        Args:
            model: Model identifier. Alias of the parent's ``model_id``.
            **kwargs: Forwarded to promptolution's ``VLLM``.
        """
        raise NotImplementedError

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        Args:
            context: The text the continuation is conditioned on.
            continuation: The exact text to score.

        Returns:
            The continuation's per-token and aggregate log-probabilities.

        Raises:
            ValueError: If the continuation does not align with token boundaries.
        """
        raise NotImplementedError

    def score_batch(self, contexts: list[str], continuations: list[str]) -> list[ScoredSequence]:
        """Score several continuations in one engine batch.

        Args:
            contexts: The conditioning texts.
            continuations: The texts to score, aligned to ``contexts``.

        Returns:
            One result per pair, in the order given.
        """
        raise NotImplementedError
