# Contrastive attribution adapted from CANTANTE by Tom Zehle:
# https://github.com/finitearth/cantante
"""Credit assignment: turning one system-level reward into per-component credit."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from xmas.optimization.prompts import ATTRIBUTION_PROMPT
from xmas.trace import Trace

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

UNKNOWN = math.nan
"""Credit for a rollout the attributor could not judge.

Not zero. Zero is a verdict, "this agent did not matter", and an attributor that
returns it after a failed parse is inventing a finding. ``nan`` propagates as an
absence and every consumer in this package filters it out.
"""

_TAG = re.compile(r"<attribution>(.*?)</attribution>", re.DOTALL)


@runtime_checkable
class Attributor(Protocol):
    """Splits a system-level score across the components that produced it.

    An attributor sees rollouts and their rewards and says which component
    deserves the credit. It never proposes a change; that is an
    :class:`~xmas.optimization.updaters.Updater`'s job. Separating the two is
    what lets a new attribution method ship without a new optimizer.

    The contract is deliberately batch-level and does not aggregate: given
    ``n`` rollouts, every returned list has ``n`` entries. Averaging is the
    caller's decision, and an attributor that averaged early would destroy the
    per-rollout signal a later method might want.
    """

    def attribute(
        self,
        traces: list[Trace],
        rewards: list[float],
    ) -> dict[str, list[float]]:
        """Assign credit per component per rollout.

        Args:
            traces: One trace per rollout.
            rewards: The system-level score of each rollout, aligned to ``traces``.

        Returns:
            Credit per rollout, keyed by component name. Every value has the same
            length as ``rewards``.
        """
        ...


def agents_in(traces: list[Trace]) -> list[str]:
    """Return every agent that ran in any of the traces, in a stable order.

    Args:
        traces: The rollouts.

    Returns:
        Agent names, sorted. Sorted rather than in execution order because
        different rollouts may order them differently once a topology branches.
    """
    return sorted({record.agent for trace in traces for record in trace.records})


class UniformAttributor:
    """Gives every agent that ran the system's whole score.

    The null hypothesis of credit assignment, and the baseline any real
    attributor has to beat. Ported from the seed's identity attributer. An agent
    that did not run in a rollout gets :data:`UNKNOWN` for it rather than zero,
    because it earned neither credit nor blame.
    """

    def attribute(self, traces: list[Trace], rewards: list[float]) -> dict[str, list[float]]:
        """Assign each rollout's reward to every agent that ran in it.

        Args:
            traces: One trace per rollout.
            rewards: The system-level score of each rollout.

        Returns:
            Credit per rollout, keyed by agent name.
        """
        if len(traces) != len(rewards):
            raise ValueError(f"Got {len(traces)} traces and {len(rewards)} rewards; they must be parallel.")
        names = agents_in(traces)
        return {
            name: [reward if name in trace.agents else UNKNOWN for trace, reward in zip(traces, rewards, strict=True)]
            for name in names
        }


class ContrastiveAttributor:
    """Asks a model to compare executions of one query and split the credit.

    This is CANTANTE's attribution method. The signal it exploits is contrast:
    several prompt sets are run against the *same* query, and the model is shown
    them side by side with their scores, so it can attribute a score difference
    to the agent whose output differed. A single rollout carries no such signal,
    which is why the method groups by query rather than judging rollouts alone.

    Rollouts of one query are shuffled before grouping, with a seeded generator,
    because an ordering that correlates with score teaches the model to read the
    position instead of the content.

    Attributes:
        llm: The meta-model that does the judging.
        group_size: How many executions of one query are compared at once.
        seed: Seed for the shuffle, so a run reproduces.
        failures: Number of judgements that could not be parsed, from the last
            call. Readable rather than logged, so a caller can act on it.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        template: str | None = None,
        group_size: int = 4,
        max_chars_per_field: int = 2000,
        seed: int = 42,
    ) -> None:
        """Initialize the attributor.

        Args:
            llm: The meta-model. This is a judging call, not a system call, so it
                is deliberately separate from the models the agents run on.
            template: Attribution prompt. Defaults to the ported CANTANTE prompt.
            group_size: Executions compared per judging call. Below two there is
                nothing to contrast.
            max_chars_per_field: Per-value truncation, so one runaway agent output
                cannot crowd the rest of the comparison out of the context.
            seed: Seed for the shuffle. Drawn from once per judging call and
                carried forward, as the seed implementation does, so a whole run
                reproduces from a freshly constructed attributor.

        Raises:
            ValueError: If ``group_size`` is below two.
        """
        if group_size < 2:
            raise ValueError(f"group_size must be at least 2 to contrast anything; got {group_size}.")
        self.llm = llm
        self.template = template if template is not None else ATTRIBUTION_PROMPT
        self.group_size = group_size
        self.max_chars_per_field = max_chars_per_field
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.failures = 0

    # ------------------------------------------------------------- the context

    def rollout(self, trace: Trace, reward: float) -> dict[str, Any]:
        """Render one execution as the data the judge sees.

        This is the seam the seed calls its attribution *context*. Overriding it
        is how a method changes what the judge reasons over without touching the
        grouping, the prompting or the parsing: the seed's env-ledger and
        tool-call-protocol contexts are both this method, written differently.
        The default is the text context, which is every agent's output.

        Args:
            trace: The rollout.
            reward: Its system-level score.

        Returns:
            A JSON-safe mapping for one entry of the ``executions`` list.
        """
        outputs: dict[str, dict[str, str]] = {}
        for record in trace.records:
            outputs.setdefault(record.agent, {})
            for name, value in record.outputs.items():
                outputs[record.agent][name] = self._truncate(value)
        entry: dict[str, Any] = {"system_score": reward, "agent_outputs": outputs}
        if trace.failed:
            entry["error"] = trace.error
        return entry

    def _truncate(self, value: str) -> str:
        """Shorten one field, marking the cut so the judge does not read it as the end."""
        if len(value) <= self.max_chars_per_field:
            return value
        return value[: self.max_chars_per_field] + " ...[truncated]"

    # ------------------------------------------------------------- the method

    def attribute(self, traces: list[Trace], rewards: list[float]) -> dict[str, list[float]]:
        """Compare executions of each query and assign per-agent credit.

        Args:
            traces: One trace per rollout.
            rewards: The system-level score of each rollout.

        Returns:
            Credit per rollout, keyed by agent name, with :data:`UNKNOWN` wherever
            the judge's answer could not be read.

        Raises:
            ValueError: If the inputs are not parallel.
            RuntimeError: If no judgement at all could be parsed, which means the
                meta-model or the prompt is broken rather than the run being noisy.
        """
        if len(traces) != len(rewards):
            raise ValueError(f"Got {len(traces)} traces and {len(rewards)} rewards; they must be parallel.")
        names = agents_in(traces)
        credit: dict[str, list[float]] = {name: [UNKNOWN] * len(traces) for name in names}
        if not names:
            return credit

        groups = self._groups(traces)
        if not groups:
            return credit

        prompts = [self._prompt(group, traces, rewards, names) for group in groups]
        responses = self.llm.get_response(prompts)

        self.failures = 0
        for group, response in zip(groups, responses, strict=True):
            parsed = self._parse(response, names, len(group))
            if parsed is None:
                self.failures += 1
                continue
            for position, index in enumerate(group):
                for name in names:
                    credit[name][index] = parsed[position][name]

        if self.failures == len(groups):
            raise RuntimeError(
                f"None of {len(groups)} attribution judgements could be parsed. "
                "The meta-model returned no readable <attribution> block."
            )
        return credit

    def _groups(self, traces: list[Trace]) -> list[list[int]]:
        """Split rollout indices into comparable groups: same query, chunked, shuffled.

        Chunk before shuffling, and shuffle only within the chunk, which is the
        seed's order and is what the parity fixture pins. A chunk of one is
        dropped rather than sent: one execution offers nothing to contrast, and a
        judgement made without contrast is not the method.
        """
        by_query: dict[str, list[int]] = {}
        for index, trace in enumerate(traces):
            by_query.setdefault(trace.query, []).append(index)

        groups: list[list[int]] = []
        for indices in by_query.values():
            for start in range(0, len(indices), self.group_size):
                chunk = indices[start : start + self.group_size]
                if len(chunk) < 2:
                    continue
                self.rng.shuffle(chunk)
                groups.append(chunk)
        return groups

    def _prompt(self, group: list[int], traces: list[Trace], rewards: list[float], names: list[str]) -> str:
        """Render the judging prompt for one group of executions."""
        payload = {
            "query": traces[group[0]].query,
            "executions": [self.rollout(traces[index], rewards[index]) for index in group],
        }
        # The judge is told which key holds the answer. The seed hardcoded
        # "prediction"; here it is whatever the last agent to write wrote.
        records = traces[group[0]].records
        output_variable = next(iter(records[-1].outputs), "prediction") if records else "prediction"
        return (
            self.template.replace("{executions}", json.dumps(payload, ensure_ascii=False, indent=2))
            .replace("{n_executions}", str(len(group)))
            .replace("{agent_names}", json.dumps(names))
            .replace("{output_variable}", output_variable)
        )

    def _parse(self, response: str, names: list[str], size: int) -> list[dict[str, float]] | None:
        """Read one judgement, or return None if it cannot be read as specified."""
        match = _TAG.search(response or "")
        if match is None:
            return None
        try:
            payload = json.loads(match.group(1).strip())
            executions = payload["executions"]
            if len(executions) != size:
                return None
            return [
                {name: max(-1.0, min(1.0, float(entry["agent_credits"][name]))) for name in names}
                for entry in executions
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
