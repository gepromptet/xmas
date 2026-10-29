"""Nodes that are not agents: a :class:`Code` runs a callable, a :class:`MASModule` a whole system.

Code nodes live in ``MAS.code_nodes`` and modules in ``MAS.modules``, never in ``MAS.agents``,
so prompts, encodings and prompt optimizers see agents only. Their maps rename
between the inside and the graph: every map is ``{name inside: name in the
graph}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS
    from xmas.values import ValueType

__all__ = [
    "Code",
    "MASModule",
]


@dataclass(frozen=True)
class Code:
    """A node that runs a registered callable (a tool or a code node function) without a model.

    Attributes:
        name: Unique node name in the MAS.
        uses: Name of the callable in the MAS's callables registry (its ``tools=``).
        reads: ``{parameter: value in the graph}`` for parameters whose value has
            another name; a parameter not listed reads the value of its own name.
        writes: ``{callable write: value in the graph}`` for writes stored under
            another name; a write not listed keeps its own name.
        detached: Whether the node is detached: no optimizer changes its prompt, model, wiring or edges.
    """

    name: str
    uses: str
    reads: Mapping[str, str] = field(default_factory=dict)
    writes: Mapping[str, str] = field(default_factory=dict)
    detached: bool = False

    def __post_init__(self) -> None:
        """Check the names and keep plain copies of the maps."""
        if not self.name:
            raise ValueError("A code node needs a name.")
        if not self.uses:
            raise ValueError(f"Code {self.name!r} needs the name of the callable it uses.")
        for label in ("reads", "writes"):
            object.__setattr__(self, label, names_map(f"Code {self.name!r}", label, getattr(self, label)))

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form; ``reads``, ``writes`` and ``detached`` only when set."""
        data: dict[str, Any] = {"name": self.name, "uses": self.uses}
        if self.reads:
            data["reads"] = dict(self.reads)
        if self.writes:
            data["writes"] = dict(self.writes)
        if self.detached:
            data["detached"] = True
        return data


def names_map(owner: str, label: str, mapping: Any) -> dict[str, str]:
    """Return a plain copy of a ``{name: name}`` map, refusing anything else."""
    if not isinstance(mapping, Mapping) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v for k, v in mapping.items()
    ):
        raise ValueError(f"{owner}: {label} must map names to names, got {mapping!r}.")
    return dict(mapping)


@dataclass(frozen=True)
class MASModule:
    """A node that runs a whole version 2 system on values of the outer one.

    The inner system starts from the mapped values, runs on the outer run's
    meter, and its writes come back under their outer names. Types carry over:
    an outer value has the type of the inner value it maps to.

    Attributes:
        name: Unique node name in the outer MAS.
        system: The inner version 2 system; it may hold modules itself.
        reads: ``{inner input: outer value}``, one entry for every input the
            inner system declares.
        writes: ``{inner value: outer value}`` for the inner writes the outer
            system keeps; at least one.
        detached: Whether the node is detached: no optimizer changes its prompt, model, wiring or edges.
    """

    name: str
    system: MAS
    reads: Mapping[str, str]
    writes: Mapping[str, str]
    detached: bool = False

    def __post_init__(self) -> None:
        """Check the maps against the inner system and keep plain copies of them."""
        if not self.name:
            raise ValueError("A module needs a name.")
        where = f"MASModule {self.name!r}"
        info = self.system.checked(f"{where}: the system in a module must be version 2.")
        reads = names_map(where, "reads", self.reads)
        writes = names_map(where, "writes", self.writes)
        inputs = sorted(self.system.inputs)
        if sorted(reads) != inputs:
            raise ValueError(f"{where}: reads must map exactly the inner inputs {inputs}, got {sorted(reads)}.")
        written = sorted(info.writers)
        if not writes or not set(writes) <= set(written):
            raise ValueError(f"{where}: writes must map inner writes, some of {written}, got {sorted(writes)}.")
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)

    def types(self) -> tuple[dict[str, ValueType], dict[str, ValueType]]:
        """Return the outer reads and writes, each with the type of its inner value: ``{outer: type}``."""
        inner = self.system.checked().types
        reads = {outer: self.system.inputs[name] for name, outer in self.reads.items()}
        return reads, {outer: inner[name] for name, outer in self.writes.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form, the inner system as its own version 2 spec."""
        data: dict[str, Any] = {
            "name": self.name,
            "reads": dict(self.reads),
            "writes": dict(self.writes),
            "system": dict(self.system.to_dict()),
        }
        if self.detached:
            data["detached"] = True
        return data
