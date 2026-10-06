"""The serialized form of a MAS.

A spec is plain data: dicts, lists, strings and ints, and nothing else. It holds
prompts, topology and model bindings, which per the design principles is the
framework-agnostic artefact an optimization run produces. No LLM object, no
compiled graph and no callable is ever part of it.
"""

from __future__ import annotations

from typing import Any


def validate_spec(data: dict[str, Any]) -> dict[str, Any]:
    """Check the top-level shape of a spec and return it.

    Each agent mapping is checked by :meth:`xmas.Agent.from_dict`.

    Args:
        data: Candidate spec, typically parsed from YAML.

    Returns:
        The same mapping.

    Raises:
        ValueError: If a key is unknown, the version is unsupported, the agents
            are not a non-empty list of mappings, or an edge or the default model
            has the wrong type.
    """
    unknown = set(data) - {"version", "agents", "edges", "default_model"}
    if unknown:
        raise ValueError(f"Unknown spec keys: {sorted(unknown)}.")

    version = data.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported spec version {version!r}; this build reads version 1.")

    agents = data.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Spec needs a non-empty 'agents' list.")
    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent {index} is not a mapping.")

    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("Spec needs 'edges' to be a list.")
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or not all(isinstance(node, str) for node in edge):
            raise ValueError(f"Edge {edge!r} must be a two-element list of strings.")

    default_model = data.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise ValueError("Spec needs 'default_model' to be a string or null.")

    return data
