"""The Agent: one node of a multi-agent system."""

from __future__ import annotations

from typing import Any


class Agent:
    """A single LLM node: a prompt, the state it reads, and the state it writes.

    An agent is a plain mutable object. It is addressable through the MAS that
    holds it, so ``mas.agents["solver"].prompt = "..."`` is a supported way to
    change one, and optimizers use :meth:`xmas.MAS.copy` to derive candidates
    without mutating the original.

    Attributes:
        name: Unique identifier, also the node name in the MAS graph.
        prompt: Prompt template. ``{placeholders}`` naming input variables are
            substituted at run time; unknown braces are left alone.
        inputs: State variables this agent reads. Defaults to ``["query"]``.
        output: State variable this agent writes.
        tools: Names of tools this agent may call. Empty means no tool access.
        model: Name of the model to bind this agent to, resolved by the MAS
            against its model registry. ``None`` uses the MAS default. Stored as
            a name rather than an object so that a MAS serializes to plain data.
    """

    name: str
    prompt: str
    inputs: list[str]
    output: str
    tools: list[str]
    model: str | None

    def __init__(
        self,
        name: str,
        prompt: str,
        output: str,
        inputs: list[str] | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize an agent.

        Args:
            name: Unique identifier within the MAS.
            prompt: Prompt template for this agent.
            output: Name of the state variable this agent writes.
            inputs: State variables this agent reads. Defaults to ``["query"]``.
            tools: Names of tools this agent may call.
            model: Model name to bind this agent to, or ``None`` for the default.

        Raises:
            ValueError: If ``name`` or ``output`` is empty, or ``inputs`` is empty.
        """
        if not name:
            raise ValueError("An agent needs a name.")
        if not output:
            raise ValueError(f"Agent {name!r} needs an output variable.")
        resolved_inputs = list(inputs) if inputs is not None else ["query"]
        if not resolved_inputs:
            raise ValueError(f"Agent {name!r} needs at least one input variable.")

        self.name = name
        self.prompt = prompt
        self.inputs = resolved_inputs
        self.output = output
        self.tools = list(tools) if tools is not None else []
        self.model = model

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form of this agent.

        Returns:
            A JSON- and YAML-safe mapping. Round-trips exactly through
            :meth:`from_dict`.
        """
        return {
            "name": self.name,
            "prompt": self.prompt,
            "inputs": list(self.inputs),
            "output": self.output,
            "tools": list(self.tools),
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Rebuild an agent from :meth:`to_dict` output.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed agent.

        Raises:
            ValueError: If a required key is missing, an unknown key is present,
                or a value has the wrong type.
        """
        unknown = set(data) - {"name", "prompt", "inputs", "output", "tools", "model"}
        if unknown:
            raise ValueError(f"Unknown agent keys: {sorted(unknown)}.")
        missing = {"name", "prompt", "output"} - set(data)
        if missing:
            raise ValueError(f"Agent is missing required keys: {sorted(missing)}.")
        for key in ("name", "prompt", "output"):
            if not isinstance(data[key], str):
                raise ValueError(f"Agent {data['name']!r} needs a string {key!r}.")
        for key in ("inputs", "tools"):
            value = data.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Agent {data['name']!r} needs {key!r} to be a list of strings.")
        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"Agent {data['name']!r} needs 'model' to be a string or null.")
        return cls(
            name=data["name"],
            prompt=data["prompt"],
            output=data["output"],
            inputs=data.get("inputs"),
            tools=data.get("tools"),
            model=data.get("model"),
        )

    def copy(self, **overrides: Any) -> Agent:
        """Return a copy with the given attributes replaced.

        Args:
            **overrides: Any constructor argument, by name.

        Returns:
            A new agent; this one is left unchanged.
        """
        fields: dict[str, Any] = {
            "name": self.name,
            "prompt": self.prompt,
            "output": self.output,
            "inputs": list(self.inputs),
            "tools": list(self.tools),
            "model": self.model,
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

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return f"Agent(name={self.name!r}, inputs={self.inputs!r}, output={self.output!r}, model={self.model!r})"
