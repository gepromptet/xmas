"""Prompt rendering and prompt-set encoding.

``safe_format`` and ``extract_placeholders`` are adapted from CANTANTE by Tom
Zehle (https://github.com/finitearth/cantante); they exist because a prompt that
contains JSON or Markdown braces must survive substitution.

The encoding functions solve a different problem. promptolution evaluates a
*prompt string* against a predictor, so a MAS with several agents has to carry
several prompts through one string slot. The encoding below is that carrier, and
it is deliberately an identity for a single-agent MAS, which is what makes the
N=1 degenerate case exact rather than merely analogous.
"""

from __future__ import annotations

import re

AGENT_HEADER = "<<<xmas:agent {name}>>>"
"""Header line that introduces one agent's prompt in an encoded prompt set."""

_HEADER_MARKER = "<<<xmas:agent "

# A placeholder is a bare identifier in single braces. The lookarounds leave
# doubled braces alone, so `{{literal}}` survives untouched, and the identifier
# class means a JSON body like {"k": 1} is never mistaken for a placeholder.
_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")

_HEADER_LINE = re.compile(r"^<<<xmas:agent (.+?)>>>$", re.MULTILINE)


def extract_placeholders(template: str) -> set[str]:
    """Return the ``{placeholder}`` names appearing in a template.

    Args:
        template: A prompt template.

    Returns:
        The set of simple identifier placeholders found.
    """
    return {match.group(1) for match in _PLACEHOLDER.finditer(template)}


def safe_format(template: str, values: dict[str, str]) -> str:
    """Substitute ``values`` into ``template`` without choking on stray braces.

    Placeholders absent from ``values``, and brace runs that are not placeholders
    at all, are passed through untouched instead of raising. A prompt is
    user-supplied text and will contain JSON, code and Markdown; refusing to
    render it because of an unrelated brace would make the framework hostile to
    exactly the prompts people write.

    Args:
        template: The prompt template.
        values: Substitutions, keyed by placeholder name.

    Returns:
        The rendered string.
    """

    def substitute(match: re.Match[str]) -> str:
        return values.get(match.group(1), match.group(0))

    return _PLACEHOLDER.sub(substitute, template)


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
        ValueError: If ``prompts`` is empty, or a name or prompt would collide
            with the header syntax.
    """
    if not prompts:
        raise ValueError("Cannot encode an empty prompt set.")

    for name, text in prompts.items():
        if _HEADER_MARKER in text:
            raise ValueError(
                f"The prompt for {name!r} contains the reserved marker {_HEADER_MARKER!r}, "
                "which would be read back as an agent header."
            )
        if ">>>" in name or "\n" in name:
            raise ValueError(f"Agent name {name!r} cannot appear in a prompt-set header.")

    if len(prompts) == 1:
        return next(iter(prompts.values()))

    sections = [f"{AGENT_HEADER.format(name=name)}\n{text}" for name, text in prompts.items()]
    return "\n".join(sections)


def decode_prompts(encoded: str, agents: list[str]) -> dict[str, str]:
    """Decode a prompt set produced by :func:`encode_prompts`.

    Args:
        encoded: The encoded prompt set.
        agents: Agent names of the MAS this prompt set belongs to. A bare string
            with no headers is assigned to the single agent named here.

    Returns:
        Prompt text keyed by agent name.

    Raises:
        ValueError: If the encoding is headerless and ``agents`` has more than
            one entry, or if a header names an agent not in ``agents``.
    """
    if not agents:
        raise ValueError("Cannot decode a prompt set without knowing the agents.")

    matches = list(_HEADER_LINE.finditer(encoded))
    if not matches:
        if len(agents) != 1:
            raise ValueError(
                f"Prompt set has no agent headers, so it can only belong to a single-agent "
                f"MAS, but this one has {len(agents)} agents: {sorted(agents)}."
            )
        return {agents[0]: encoded}

    known = set(agents)
    decoded: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        if name not in known:
            raise ValueError(f"Prompt set names unknown agent {name!r}; expected one of {sorted(known)}.")
        start = match.end() + 1  # skip the newline that follows the header
        end = matches[index + 1].start() - 1 if index + 1 < len(matches) else len(encoded)
        decoded[name] = encoded[start : max(start, end)]

    return decoded
