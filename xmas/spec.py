"""The serialized form of a MAS.

A spec is plain data: dicts, lists, strings and ints, and nothing else. It holds
prompts, topology and model bindings, which per the design principles is the
framework-agnostic artefact an optimization run produces. No LLM object, no
compiled graph and no callable is ever part of it.

A ``version: 2`` spec adds steps, boxes, typed values, answer forms, the edge
forms of :mod:`xmas.edges`, ``frozen`` nodes and ``health`` settings, and keeps
its prompts in a top-level ``prompts`` section. It may declare ``outputs``; a
node may run ``for each`` item of a list and ``retry`` on some error kinds; an
agent may read values under placeholders of other names and share a prompt
with other agents under a ``prompt`` name. Its ``models`` section says, per
bound model name, which kinds beyond text the model ``accepts`` and
``produces``, the catalogue id it stands for (``model``, default the name)
and model parameters such as ``temperature``; never a URL or a key. A
callable names code by ``import`` or a tool server by name (``server``), and
may bind constant ``params``. Its ``humans`` are the nodes that ask a person
(:class:`xmas.human.Human`), and an edge may be one of a ``first of`` race.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "V2_KEYS",
    "VERSION_2",
    "validate_spec",
    "validate_spec_v2",
]


VERSION_2 = 2
"""The ``version`` value of a version 2 spec."""
EDGE_LENGTH = 2  # a version 1 edge is [source, target]
CALLABLE_KINDS = ("python", "mcp")  # code imported by path, or a tool server named in the configuration

V2_KEYS: dict[str, Any] = {
    "version": int,
    "inputs": dict,
    "outputs": dict,
    "agents": list,
    "steps": list,
    "boxes": list,
    "humans": list,
    "edges": list,
    "state": dict,
    "frozen": list,
    "health": dict,
    "models": dict,
    "callables": dict,
    "default_model": str,
    "prompts": dict,
}
"""Every top-level key of a version 2 spec with the type of its value, in the order a dump writes them."""

ENTRIES: dict[str, tuple[str, frozenset[str], dict[str, Any]]] = {
    "agents": (
        "Agent",
        frozenset({"writes"}),
        {
            "name": str,
            "model": str,
            "prompt": str,
            "for each": (str, dict),
            "reads": (list, dict),
            "writes": dict,
            "tools": list,
            "schema": dict,
            "retry": dict,
        },
    ),
    "steps": (
        "Step",
        frozenset({"uses"}),
        {"name": str, "uses": str, "for each": (str, dict), "reads": dict, "writes": dict, "retry": dict},
    ),
    "boxes": (
        "Box",
        frozenset({"reads", "writes", "system"}),
        {"name": str, "for each": (str, dict), "reads": dict, "writes": dict, "system": dict},
    ),
    "humans": (
        "Human",
        frozenset({"form", "writes"}),
        {
            "name": str,
            "form": str,
            "for each": (str, dict),
            "reads": (list, dict),
            "writes": (str, dict),
            "question": str,
            "options": (list, str),
            "labels": dict,
            "pair": list,
            "edits": str,
            "timeout_s": (int, float),
            "default": dict,
        },
    ),
}
CALLABLE = {"kind": str, "import": str, "server": str, "params": dict, "reads": dict, "writes": dict}
HEALTH = {"length_factor": (int, float), "min_samples": int, "leak": list}
MODEL_KEYS = ("accepts", "produces")
WORDS = {
    int: "an integer",
    (int, float): "a number",
    str: "a string",
    list: "a list",
    dict: "a mapping",
    (str, dict): "a string or a mapping",
    (list, dict): "a list or a mapping",
    (list, str): "a list or a string",
}


def validate_spec(data: dict[str, Any]) -> dict[str, Any]:
    """Check the top-level shape of a version 1 spec and return it; each agent is checked by ``Agent.from_dict``.

    Raises:
        ValueError: If a key is unknown, the version is not 1, the agents are not a non-empty list of mappings,
            or an edge or the default model has the wrong type.
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
        if not isinstance(edge, list) or len(edge) != EDGE_LENGTH or not all(isinstance(node, str) for node in edge):
            raise ValueError(f"Edge {edge!r} must be a two-element list of strings.")
    if not isinstance(data.get("default_model", ""), str | None):
        raise ValueError("Spec needs 'default_model' to be a string or null.")
    return data


def check_fields(
    entry: Any, where: str, fields: dict[str, Any], required: frozenset[str] = frozenset(), note: str = ""
) -> None:
    """Refuse a mapping with an unknown or missing key or a value of the wrong type."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a mapping.")
    unknown, missing = sorted(set(entry) - set(fields)), sorted(required - set(entry))
    if unknown or missing:
        problem = f"has unknown keys {unknown}" if unknown else f"is missing the keys {missing}"
        raise ValueError(f"{where} {problem}; it takes {sorted(fields)}{note if unknown else ''}.")
    for key, value in entry.items():
        if isinstance(value, bool) or not isinstance(value, fields[key]):
            raise ValueError(f"{where} needs {key!r} to be {WORDS[fields[key]]}, got {value!r}.")


def validate_spec_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Check the shape of a version 2 spec and return it.

    The shape is the keys and the type of each value. The names inside the maps
    are checked by the constructors, and whether the system makes sense by
    :func:`xmas.check.check_system`, both when the MAS is built.
    """
    check_fields(data, "Spec", V2_KEYS)
    check_version_is_2(data)
    check_boxes_hold_specs(data)
    check_entries(data)
    check_callables(data)
    check_fields(data.get("health", {}), "Spec 'health'", HEALTH)
    check_models(data)
    check_agent_prompts(data)
    check_no_stray_prompts(data)
    return data


def check_version_is_2(data: dict[str, Any]) -> None:
    """Refuse a spec whose ``version`` is not 2."""
    if data.get("version") != VERSION_2:
        raise ValueError(f"A version 2 spec needs 'version: 2', not {data.get('version')!r}.")


def check_boxes_hold_specs(data: dict[str, Any]) -> None:
    """Refuse a box that names its system by path: from_dict needs the system's spec itself."""
    for entry in data.get("boxes", []):
        if isinstance(entry, dict) and isinstance(entry.get("system"), str):
            raise ValueError(
                f"Box {entry.get('name')!r} names its system {entry['system']!r}: MAS.load reads a bundle's boxes; "
                "from_dict needs the system's spec."
            )


def check_entries(data: dict[str, Any]) -> None:
    """Check the keys and value types of each agent, step, box and human."""
    for section, (label, required, fields) in ENTRIES.items():
        for index, entry in enumerate(data.get(section, [])):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError(f"{label} {index} must be a mapping with a string 'name'.")
            check_fields(entry, f"{label} {entry['name']!r}", fields, required)


def check_callables(data: dict[str, Any]) -> None:
    """Check each callable's keys, its kind, that only python takes params, and that a server is a name."""
    for name, entry in data.get("callables", {}).items():
        check_fields(entry, f"Callable {name!r}", CALLABLE, frozenset({"kind"}), " (use a YAML comment for a note)")
        if entry["kind"] not in CALLABLE_KINDS:
            raise ValueError(f"Callable {name!r} needs 'kind' to be 'python' or 'mcp', not {entry['kind']!r}.")
        if "params" in entry and entry["kind"] != "python":
            raise ValueError(f"Callable {name!r} binds params, which only a python callable takes.")
        if "://" in entry.get("server", ""):
            raise ValueError(
                f"Callable {name!r} names the server {entry['server']!r}: a callable names its server, and its "
                "address goes to the configuration ([tools.<name>] url)."
            )


def check_models(data: dict[str, Any]) -> None:
    """Check that each models entry maps accepts and produces to lists and names a string catalogue id.

    Other keys are refused, with why, where the MAS reads them.
    """
    for name, entry in data.get("models", {}).items():
        if not isinstance(entry, dict) or any(not isinstance(entry.get(key, []), list) for key in MODEL_KEYS):
            raise ValueError(f"Spec 'models' entry {name!r} must map accepts and produces to lists, got {entry!r}.")
        if not isinstance(entry.get("model", name), str):
            raise ValueError(
                f"Spec 'models' entry {name!r} needs 'model' to be a catalogue id, not {entry['model']!r}."
            )


def check_agent_prompts(data: dict[str, Any]) -> None:
    """Check that the prompts are text and each agent writes one value and has a prompt in the section."""
    prompts = data.get("prompts", {})
    if not all(isinstance(text, str) for text in prompts.values()):
        raise ValueError("Spec 'prompts' must map agent and prompt names to prompt text.")
    for agent in data.get("agents", []):
        if len(agent["writes"]) != 1:
            raise ValueError(
                f"Agent {agent['name']!r} needs exactly one write such as {{draft: text}}, got {agent['writes']!r}."
            )
        if "prompt" in agent and agent["prompt"] not in prompts:
            raise ValueError(f"Agent {agent['name']!r}: prompt {agent['prompt']!r} names no entry of 'prompts'.")
        if agent["name"] not in prompts and "prompt" not in agent:
            raise ValueError(f"Agent {agent['name']!r} has no prompt in the 'prompts' section.")


def check_no_stray_prompts(data: dict[str, Any]) -> None:
    """Refuse a prompt that no agent uses, under its own name or a shared ``prompt`` name."""
    used = {agent.get("prompt", agent["name"]) for agent in data.get("agents", [])}
    stray = sorted(set(data.get("prompts", {})) - used)
    if stray:
        raise ValueError(f"Spec 'prompts' has entries no agent uses: {stray}.")
