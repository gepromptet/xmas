"""The serialized form of a MAS.

A spec is plain data: dicts, lists, strings and ints, and nothing else. It holds
prompts, topology and model bindings, which per the design principles is the
framework-agnostic artefact an optimization run produces. No LLM object, no
compiled graph and no callable is ever part of it.
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

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
    unknown = set(data) - {"version", "agents", "edges", "default_model"}
    if unknown:
        raise ValueError(f"Unknown spec keys: {sorted(unknown)}.")

    version = data.get("version", SPEC_VERSION)
    if version != SPEC_VERSION:
        raise ValueError(f"Unsupported spec version {version!r}; this build reads version {SPEC_VERSION}.")

    agents = data.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Spec needs a non-empty 'agents' list.")

    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent {index} is not a mapping.")
        unknown_agent = set(agent) - {"name", "prompt", "inputs", "output", "tools", "model"}
        if unknown_agent:
            raise ValueError(f"Agent {index} has unknown keys: {sorted(unknown_agent)}.")
        for key in ("name", "prompt", "output"):
            if not isinstance(agent.get(key), str):
                raise ValueError(f"Agent {index} needs a string {key!r}.")
        for key in ("inputs", "tools"):
            value = agent.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Agent {agent['name']!r} needs {key!r} to be a list of strings.")
        model = agent.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"Agent {agent['name']!r} needs 'model' to be a string or null.")

    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("Spec needs 'edges' to be a list.")
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or not all(isinstance(node, str) for node in edge):
            raise ValueError(f"Edge {edge!r} must be a two-element list of strings.")

    default_model = data.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise ValueError("Spec needs 'default_model' to be a string or null.")

    return cast(MASSpec, data)
