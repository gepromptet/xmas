"""The Agent: one node of a multi-agent system."""

from __future__ import annotations

from typing import Any

DEFAULT_INPUT = "query"
"""Variable an agent reads when ``inputs`` is not given."""


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
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form of this agent.

        Returns:
            A JSON- and YAML-safe mapping. Round-trips exactly through
            :meth:`from_dict`.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Rebuild an agent from :meth:`to_dict` output.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed agent.

        Raises:
            ValueError: If a required key is missing or an unknown key is present.
        """
        raise NotImplementedError

    def copy(self, **overrides: Any) -> Agent:
        """Return a copy with the given attributes replaced.

        Args:
            **overrides: Any constructor argument, by name.

        Returns:
            A new agent; this one is left unchanged.
        """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        """Structural equality over every field."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        raise NotImplementedError
