"""The Agent: one node of a multi-agent system."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, ClassVar

from xmas.schemas import compile_form
from xmas.values import JSON, parse_type

__all__ = [
    "Agent",
]


class Agent:
    """A single LLM node: a prompt, the state it reads, and the state it writes.

    An agent is a plain mutable object. It is addressable through the MAS that
    holds it, so ``mas.agents["solver"].prompt = "..."`` is a supported way to
    change one, and optimizers use :meth:`xmas.MAS.copy` to derive candidates
    without mutating the original.

    Attributes:
        name: Unique identifier, also the node name in the MAS graph.
        prompt: Prompt template. ``{placeholders}`` naming its reads are
            substituted at run time; unknown braces are left alone.
        reads: ``{placeholder: value}``: which value fills each placeholder.
            A list of value names given as ``inputs`` fills each placeholder
            from the value of its own name.
        inputs: The values this agent reads, in order, each once. Defaults to
            ``["query"]``; assigning a list reads each value under its own name.
        output: State variable this agent writes.
        tools: Names of tools this agent may call. Empty means no tool access.
        model: Name of the model to bind this agent to, resolved by the MAS
            against its model registry. ``None`` uses the MAS default. Stored as
            a name rather than an object so that a MAS serializes to plain data.
        output_type: The type of the value the agent writes, ``"text"`` or
            ``"json"``.
        schema: The answer form a json answer must match, in the dialect of
            :mod:`xmas.schemas`, or ``None`` for any JSON.
        detached: Whether the node is detached: no optimizer changes its prompt, model, wiring or edges.
            A detached agent makes the system version 2.
    """

    name: str
    prompt: str
    reads: dict[str, str]
    output: str
    tools: list[str]
    model: str | None
    output_type: str
    schema: dict[str, Any] | None
    detached: bool

    def __init__(  # noqa: PLR0913, PLR0917  the public constructor, kept for version 1 callers
        self,
        name: str,
        prompt: str,
        output: str,
        inputs: list[str] | Mapping[str, str] | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        output_type: str = "text",
        schema: dict[str, Any] | None = None,
        *,
        detached: bool = False,
    ) -> None:
        """Initialize an agent; the fields are described on the class."""
        if not name:
            raise ValueError("An agent needs a name.")
        if not output:
            raise ValueError(f"Agent {name!r} needs an output variable.")
        reads = reads_map(name, inputs if inputs is not None else ["query"])
        try:
            answers_json = parse_type(output_type) == JSON
            if schema is not None:
                compile_form(schema)
        except ValueError as exc:
            raise ValueError(f"Agent {name!r}: {exc}") from exc
        if schema is not None and not answers_json:
            raise ValueError(f"Agent {name!r}: a schema needs output_type 'json', not {output_type!r}.")

        self.name = name
        self.prompt = prompt
        self.reads = reads
        self.output = output
        self.tools = list(tools) if tools is not None else []
        self.model = model
        self.output_type = output_type
        self.schema = schema
        self.detached = detached

    @property
    def inputs(self) -> list[str]:
        """The values this agent reads, in order, each once."""
        return list(dict.fromkeys(self.reads.values()))

    @inputs.setter
    def inputs(self, names: list[str]) -> None:
        self.reads = reads_map(self.name, names)

    @property
    def renamed(self) -> bool:
        """Whether a placeholder reads a value of another name."""
        return any(placeholder != value for placeholder, value in self.reads.items())

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form, which round-trips exactly through :meth:`from_dict`.

        ``inputs`` is a list of value names, or the ``{placeholder: value}``
        map when a placeholder reads a value of another name. ``output_type``
        ``schema`` and ``detached`` appear only when set, so a version 1 agent's
        dict keeps its six keys.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "prompt": self.prompt,
            "inputs": dict(self.reads) if self.renamed else list(self.reads),
            "output": self.output,
            "tools": list(self.tools),
            "model": self.model,
        }
        if self.output_type != "text":
            data["output_type"] = self.output_type
        if self.schema is not None:
            data["schema"] = copy.deepcopy(self.schema)
        if self.detached:
            data["detached"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Rebuild an agent from :meth:`to_dict` output, refusing unknown or missing keys and wrong types."""
        unknown = set(data) - {
            "name",
            "prompt",
            "inputs",
            "output",
            "tools",
            "model",
            "output_type",
            "schema",
            "detached",
        }
        if unknown:
            raise ValueError(f"Unknown agent keys: {sorted(unknown)}.")
        missing = {"name", "prompt", "output"} - set(data)
        if missing:
            raise ValueError(f"Agent is missing required keys: {sorted(missing)}.")
        for key in ("name", "prompt", "output"):
            if not isinstance(data[key], str):
                raise ValueError(f"Agent {data['name']!r} needs a string {key!r}.")
        tools = data.get("tools", [])
        if not isinstance(tools, list) or not all(isinstance(item, str) for item in tools):
            raise ValueError(f"Agent {data['name']!r} needs 'tools' to be a list of strings.")
        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"Agent {data['name']!r} needs 'model' to be a string or null.")
        if not isinstance(data.get("output_type", "text"), str):
            raise ValueError(f"Agent {data['name']!r} needs 'output_type' to be a string.")
        schema = data.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError(f"Agent {data['name']!r} needs 'schema' to be a mapping or null.")
        if not isinstance(data.get("detached", False), bool):
            raise ValueError(f"Agent {data['name']!r} needs 'detached' to be true or false.")
        return cls(
            name=data["name"],
            prompt=data["prompt"],
            output=data["output"],
            inputs=data.get("inputs"),
            tools=data.get("tools"),
            model=data.get("model"),
            output_type=data.get("output_type", "text"),
            schema=data.get("schema"),
            detached=data.get("detached", False),
        )

    def copy(self, **overrides: Any) -> Agent:
        """Return a copy with the given constructor arguments replaced; this agent is left unchanged."""
        fields: dict[str, Any] = {
            "name": self.name,
            "prompt": self.prompt,
            "output": self.output,
            "inputs": dict(self.reads),
            "tools": list(self.tools),
            "model": self.model,
            "output_type": self.output_type,
            "schema": copy.deepcopy(self.schema),
            "detached": self.detached,
        }
        unknown = set(overrides) - set(fields)
        if unknown:
            raise ValueError(f"Unknown agent fields: {sorted(unknown)}.")
        fields.update(overrides)
        return Agent(**fields)

    def __eq__(self, other: object) -> bool:
        """Structural equality over every field."""
        if not isinstance(other, Agent):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    # agents are mutable and compare by value, so they cannot be dict keys or set members
    __hash__: ClassVar[None] = None  # type: ignore[assignment]  # typeshed's idiom for an unhashable class

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return f"Agent(name={self.name!r}, inputs={self.inputs!r}, output={self.output!r}, model={self.model!r})"


def reads_map(name: str, inputs: Any) -> dict[str, str]:
    """Return an agent's ``{placeholder: value}`` map from a list of value names or a map."""
    pairs = (
        list(inputs.items())
        if isinstance(inputs, Mapping)
        else [(value, value) for value in inputs]
        if isinstance(inputs, (list, tuple))
        else None
    )
    if pairs is None or not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in pairs):
        raise ValueError(f"Agent {name!r} needs 'inputs' to be a list of strings or a map of names to names.")
    if not pairs:
        raise ValueError(f"Agent {name!r} needs at least one input variable.")
    return dict(pairs)
