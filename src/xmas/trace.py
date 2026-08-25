"""Execution records produced by a MAS run.

A trace is the input an :class:`~xmas.optimization.Attributor` computes on, so it
is modelled as data rather than as a log: every field is a plain value, the whole
structure round-trips through :meth:`Trace.to_dict`, and nothing in it is
formatted for human reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Input and output token counts for a single model call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two usages component-wise, so usage aggregates over a trace."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """Return the plain-data form."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation made by an agent."""

    tool: str
    arguments: dict[str, Any]
    result: str
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "result": self.result,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class AgentRecord:
    """Everything one agent did during one run of the graph.

    Attributes:
        agent: Name of the agent, matching a node in the MAS graph.
        step: Zero-based order in which this record was produced within the run.
        prompt: The prompt template the agent held for this run, before rendering.
        rendered_prompt: The prompt actually sent, after placeholder substitution.
        inputs: The state values the agent read, keyed by variable name.
        outputs: The state values the agent wrote, keyed by variable name.
        tool_calls: Tool invocations made while producing the outputs, in order.
        usage: Token usage attributable to this agent for this run.
        model: Name of the model that served the call.
        raw_response: The unparsed model response, kept because output extraction
            is lossy and an attributor may need what was discarded.
    """

    agent: str
    step: int
    prompt: str
    rendered_prompt: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        return {
            "agent": self.agent,
            "step": self.step,
            "prompt": self.prompt,
            "rendered_prompt": self.rendered_prompt,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "usage": self.usage.to_dict(),
            "model": self.model,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class Trace:
    """The record of a single MAS call.

    Attributes:
        query: The input the MAS was called with.
        output: The terminal agent's output, i.e. what ``__call__`` returned.
        records: One :class:`AgentRecord` per agent execution, in execution order.
            An agent that runs twice (a cycle) contributes two records.
        error: Populated when the run failed, so a failed rollout is still data an
            attributor can reason about rather than a raised exception.
    """

    query: str
    output: str
    records: tuple[AgentRecord, ...] = ()
    error: str | None = None

    def __getitem__(self, agent: str) -> tuple[AgentRecord, ...]:
        """Return every record for one agent, in execution order.

        Args:
            agent: The agent name.

        Returns:
            Its records. Empty if the agent did not run, which is a legitimate
            outcome for a conditional branch rather than an error.
        """
        return tuple(record for record in self.records if record.agent == agent)

    @property
    def agents(self) -> tuple[str, ...]:
        """Names of the agents that ran, in first-execution order."""
        seen: dict[str, None] = {}
        for record in self.records:
            seen.setdefault(record.agent, None)
        return tuple(seen)

    @property
    def usage(self) -> TokenUsage:
        """Token usage summed over every record."""
        total = TokenUsage()
        for record in self.records:
            total = total + record.usage
        return total

    @property
    def failed(self) -> bool:
        """Whether the run ended in an error."""
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form, suitable for JSON or YAML."""
        return {
            "query": self.query,
            "output": self.output,
            "records": [record.to_dict() for record in self.records],
            "error": self.error,
        }
