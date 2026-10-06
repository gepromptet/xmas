"""Prompt synthesis: writing an agent's prompt after its inputs change.

Topology search moves information around. When an agent gains a predecessor it
starts receiving a variable it has no words for, and a prompt that never mentions
the variable renders without it, so the new edge changes the schedule and nothing
else. Something has to write the sentence that uses it.

The seed solved this by regenerating every prompt with a meta-model after each
topology change, and by reading each agent's inputs back out of its prompt text
by regex. xmas declares inputs on the agent instead, so the regeneration is
narrower: only the agents whose inputs actually changed are rewritten, and the
rewrite is told exactly which variables to use rather than defining them by what
it happens to mention.
"""

from __future__ import annotations

from dataclasses import dataclass

from promptolution.llms.base_llm import BaseLLM

from xmas.optimization.prompts import REWRITE_PROMPT


@dataclass(frozen=True)
class Rewrite:
    """One agent's prompt, and the inputs it is now supposed to use.

    Attributes:
        agent: The agent's name.
        prompt: Its current prompt template.
        output: The variable it writes.
        inputs: The variables it now receives, after the topology change.
    """

    agent: str
    prompt: str
    output: str
    inputs: list[str]


class LLMPromptSynthesizer:
    """Has a meta-model rewrite each prompt around its new inputs.

    This is the seed's approach, narrowed. The model is told the agent's job, what
    it writes and exactly which variables it now receives, and is asked to keep
    the job the same. A response that does not come back in ``<prompt>`` tags, or
    that fails to reference every input, is rejected in favour of the mechanical
    fallback, because a prompt that silently drops a variable would make the
    topology change inert again and look like it had been handled.

    Attributes:
        llm: The meta-model.
        template: The rewrite prompt.
        fallbacks: How many rewrites fell back, since construction. Above zero
            means the meta-model is not reliably answering as asked.
    """

    def __init__(self, llm: BaseLLM, *, template: str = REWRITE_PROMPT) -> None:
        """Initialize the synthesizer.

        Args:
            llm: The meta-model that writes the prompts.
            template: The rewrite prompt. Defaults to the ported template.
        """
        self.llm = llm
        self.template = template
        self.fallbacks = 0

    def rewrite(self, requests: list[Rewrite]) -> list[str]:
        """Rewrite each agent's prompt around its new inputs.

        Args:
            requests: The agents to rewrite.

        Returns:
            One prompt per request, in the order given.
        """
        if not requests:
            return []
        responses = self.llm.get_response([rewrite_instruction(self.template, request) for request in requests])
        written = []
        for request, response in zip(requests, responses, strict=True):
            candidate = prompt_from_response(response, request)
            if candidate is None:
                self.fallbacks += 1
                candidate = mechanical_prompt(request)
            written.append(candidate)
        return written


class MechanicalSynthesizer:
    """Appends a labelled context block instead of calling a model.

    The same routine :class:`LLMPromptSynthesizer` falls back to, exposed as a
    strategy in its own right. It is deterministic, costs nothing, and always
    references every input, which is what a test or an exactly reproducible run
    needs. What it cannot do is explain to the agent what the new variable is
    *for*, which is why it is not the default.
    """

    def rewrite(self, requests: list[Rewrite]) -> list[str]:
        """Return a mechanically written prompt per request.

        Args:
            requests: The agents to rewrite.

        Returns:
            One prompt per request, in the order given.
        """
        return [mechanical_prompt(request) for request in requests]


def mechanical_prompt(request: Rewrite) -> str:
    """Return a prompt that references every input, written without a model.

    The fallback, and the reason a synthesizer failure degrades rather than
    breaks: the agent's own instruction is preserved and a labelled block naming
    each input is appended, so the new information reaches the model even though
    nothing wrote prose about it.

    Args:
        request: The agent to write for.

    Returns:
        A prompt referencing every input in ``request``.
    """
    body = request.prompt.strip()
    if not missing_inputs(body, request):
        return body
    lines = "\n".join(f"{name}: {{{name}}}" for name in request.inputs)
    return f"{body}\n\nContext:\n{lines}"


def rewrite_instruction(template: str, request: Rewrite) -> str:
    """Render the rewrite instruction for one agent."""
    return (
        template.replace("{agent}", request.agent)
        .replace("{output}", request.output)
        .replace("{inputs}", ", ".join(request.inputs))
        .replace("{prompt}", request.prompt)
    )


def prompt_from_response(response: str, request: Rewrite) -> str | None:
    """Extract the prompt between ``<prompt>`` tags, or None if the model did not deliver one."""
    inside = response.partition("<prompt>")[2]
    if "</prompt>" not in inside:
        return None
    candidate = inside.partition("</prompt>")[0].strip()
    if not candidate:
        return None
    if missing_inputs(candidate, request):
        return None  # a dropped variable would make the topology change inert
    return candidate


def missing_inputs(prompt: str, request: Rewrite) -> list[str]:
    """Return the inputs of ``request`` that ``prompt`` does not reference as placeholders."""
    return [name for name in request.inputs if f"{{{name}}}" not in prompt]
