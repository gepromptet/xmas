"""Prompt rendering and prompt-set encoding.

``safe_format`` and ``extract_placeholders`` are adapted from CANTANTE by Tom
Zehle (https://github.com/finitearth/cantante); they exist because a prompt that
contains JSON or Markdown braces must survive ``str.format``.

The encoding functions solve a different problem. promptolution evaluates a
*prompt string* against a predictor, so a MAS with several agents has to carry
several prompts through one string slot. The encoding below is that carrier, and
it is deliberately an identity for a single-agent MAS, which is what makes the
N=1 degenerate case exact rather than merely analogous.
"""

from __future__ import annotations

AGENT_HEADER = "<<<xmas:agent {name}>>>"
"""Header line that introduces one agent's prompt in an encoded prompt set."""


def extract_placeholders(template: str) -> set[str]:
    """Return the ``{placeholder}`` names appearing in a template.

    Args:
        template: A prompt template.

    Returns:
        The set of simple identifier placeholders found.
    """
    raise NotImplementedError


def safe_format(template: str, values: dict[str, str]) -> str:
    """Substitute ``values`` into ``template`` without choking on stray braces.

    Placeholders that are not present in ``values``, and brace runs that are not
    placeholders at all (JSON bodies, Markdown tables), are passed through
    untouched instead of raising.

    Args:
        template: The prompt template.
        values: Substitutions, keyed by placeholder name.

    Returns:
        The rendered string.
    """
    raise NotImplementedError


def encode_prompts(prompts: dict[str, str]) -> str:
    """Encode a per-agent prompt mapping as one string.

    A mapping with exactly one entry encodes to that entry's prompt verbatim, so
    a single-agent MAS is indistinguishable from a promptolution predictor and
    promptolution's optimizers mutate it directly.

    Args:
        prompts: Prompt text keyed by agent name.

    Returns:
        The encoded prompt set.

    Raises:
        ValueError: If ``prompts`` is empty, or an agent name would collide with
            the header syntax.
    """
    raise NotImplementedError


def decode_prompts(encoded: str, agents: list[str]) -> dict[str, str]:
    """Decode a prompt set produced by :func:`encode_prompts`.

    Args:
        encoded: The encoded prompt set.
        agents: Agent names of the MAS this prompt set belongs to. A bare string
            with no headers is assigned to the single agent named here.

    Returns:
        Prompt text keyed by agent name.

    Raises:
        ValueError: If the encoding is headerless and ``agents`` has more than one
            entry, or if a header names an agent not in ``agents``.
    """
    raise NotImplementedError
