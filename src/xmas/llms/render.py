"""One rendering for both primitives, so the probe measures the prompt that ran.

A teacher-forced probe estimates p(continuation | context). That is only the
quantity in the estimator if `context` is byte-for-byte the string the model
conditioned on when references were generated. The gateway cannot guarantee that
today: /v1/chat/completions applies the model's chat template server-side, while
/v1/score accepts raw text and applies none. Byte-identical inputs from the client
would still reach the model differently.

So the run renders the chat template exactly once, here, on the client, with
`enable_thinking=False`, and sends the rendered string down BOTH paths raw:

    generation  ->  /v1/completions   with the rendered context
    probing     ->  /v1/score         with the same rendered context, raw

No-think falls out of the template rather than being bolted on: with
`enable_thinking=False` Qwen3's own template closes the assistant turn with the
empty think block, so generation starts after it and a probe's continuation starts
at the same position. Template identity is then a property of this module having
exactly one rendering function, and `identical` states it as a checkable fact.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def _tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(model)


def context(prompt: str, model: str) -> str:
    """Render one user prompt into the exact string the model conditions on.

    The single rendering both primitives share: `add_generation_prompt` opens the
    assistant turn, `enable_thinking=False` closes the think block inside it, and
    what generation appends from there is precisely what a probe scores from there.
    """
    rendered: str = _tokenizer(model).apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return rendered


NO_THINK_TAIL = "<think>\n\n</think>\n\n"
"""How the assistant turn ends under no-think: the template's own empty block."""


def no_think_active(prompt: str, model: str) -> bool:
    """Whether the rendered context ends where no-think generation begins.

    Template identity between the primitives is structural (both call `context`,
    and there is no second renderer), so what remains checkable is that the
    rendering actually carries the no-think ruling. This is the gate the dry run
    asserts, so a tokenizer or template update that changes the tail fails loudly
    instead of silently moving where every probe starts scoring.
    """
    return context(prompt, model).endswith(NO_THINK_TAIL)
