"""The serialized form of a MAS.

A spec is plain data: dicts, lists, strings and ints, and nothing else. It holds
prompts, topology and model bindings, which per the design principles is the
framework-agnostic artefact an optimization run produces. No LLM object, no
compiled graph and no callable is ever part of it.
"""

from __future__ import annotations

from typing import Any, TypedDict

SPEC_VERSION = 1
"""Schema version written into every spec, so old files stay readable."""


class AgentSpec(TypedDict, total=False):
    """Serialized form of one agent."""

    name: str
    prompt: str
    inputs: list[str]
    output: str
    tools: list[str]
    model: str | None


class MASSpec(TypedDict, total=False):
    """Serialized form of a whole MAS."""

    version: int
    agents: list[AgentSpec]
    edges: list[list[str]]
    default_model: str | None


def validate_spec(data: dict[str, Any]) -> MASSpec:
    """Check that a mapping is a well-formed spec and return it typed.

    Args:
        data: Candidate spec, typically parsed from YAML.

    Returns:
        The same mapping, narrowed to :class:`MASSpec`.

    Raises:
        ValueError: If a required key is missing, a key is unknown, the version
            is unsupported, or a value has the wrong type.
    """
    raise NotImplementedError
