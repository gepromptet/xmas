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

__all__ = [
    "decode_prompts",
    "encode_prompts",
    "extract_placeholders",
    "safe_format",
]


# A placeholder is a bare identifier in single braces. The lookarounds leave
# doubled braces alone, so `{{literal}}` survives untouched, and the identifier
# class means a JSON body like {"k": 1} is never mistaken for a placeholder.
PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
HEADER = "<<<xmas:agent "


def extract_placeholders(template: str) -> set[str]:
    """Return the ``{placeholder}`` names appearing in a template."""
    return {match.group(1) for match in PLACEHOLDER.finditer(template)}


def safe_format(template: str, values: dict[str, str]) -> str:
    """Substitute ``values`` into ``template`` without choking on stray braces.

    Placeholders absent from ``values``, and brace runs that are not placeholders
    at all, are passed through untouched instead of raising. A prompt is
    user-supplied text and will contain JSON, code and Markdown; refusing to
    render it because of an unrelated brace would make the framework hostile to
    exactly the prompts people write.
    """
    return PLACEHOLDER.sub(lambda match: values.get(match.group(1), match.group(0)), template)


def encode_prompts(prompts: dict[str, str]) -> str:
    """Encode a per-agent prompt mapping as one string.

    A mapping with exactly one entry encodes to that entry's prompt verbatim, so
    a single-agent MAS is indistinguishable from a promptolution predictor and
    promptolution's optimizers mutate it directly.

    Raises:
        ValueError: If ``prompts`` is empty, or a name or prompt would collide
            with the header syntax.
    """
    if not prompts:
        raise ValueError("Cannot encode an empty prompt set.")

    for name, text in prompts.items():
        if HEADER in text:
            raise ValueError(
                f"The prompt for {name!r} contains the reserved marker {HEADER!r}, "
                "which would be read back as an agent header."
            )
        if ">>>" in name or "\n" in name:
            raise ValueError(f"Agent name {name!r} cannot appear in a prompt-set header.")

    if len(prompts) == 1:
        return next(iter(prompts.values()))

    return "\n".join(f"{HEADER}{name}>>>\n{text}" for name, text in prompts.items())


def decode_prompts(encoded: str, agents: list[str]) -> dict[str, str]:
    """Decode a prompt set produced by :func:`encode_prompts` for a MAS of ``agents``.

    A bare string with no headers is assigned to the single agent named in ``agents``.

    Raises:
        ValueError: If the encoding is headerless and ``agents`` has more than
            one entry, or if a header names an agent not in ``agents``.
    """
    if not agents:
        raise ValueError("Cannot decode a prompt set without knowing the agents.")

    decoded: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in encoded.split("\n"):
        if line.startswith(HEADER) and line.endswith(">>>"):
            name = line.removeprefix(HEADER).removesuffix(">>>")
            if name not in agents:
                raise ValueError(f"Prompt set names unknown agent {name!r}; expected one of {sorted(agents)}.")
            current = []
            decoded[name] = current
        elif current is not None:
            current.append(line)

    if not decoded:
        if len(agents) != 1:
            raise ValueError(
                f"Prompt set has no agent headers, so it can only belong to a single-agent "
                f"MAS, but this one has {len(agents)} agents: {sorted(agents)}."
            )
        return {agents[0]: encoded}

    return {name: "\n".join(lines) for name, lines in decoded.items()}
