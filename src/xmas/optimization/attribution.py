# Contrastive attribution adapted from CANTANTE by Tom Zehle:
# https://github.com/finitearth/cantante
"""Credit assignment: turning one system-level reward into per-component credit."""

from __future__ import annotations

import json
import math
from typing import Any, Protocol

import numpy as np
from promptolution.llms.base_llm import BaseLLM

from xmas.trace import Trace


class Attributor(Protocol):
    """Splits a system-level score across the components that produced it.

    An attributor sees rollouts and their rewards and says which component
    deserves the credit. It never proposes a change; that is an updater's job.
    Separating the two is what lets a new attribution method ship without a new
    optimizer.

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


class UniformAttributor:
    """Gives every agent that ran the system's whole score.

    The null hypothesis of credit assignment, and the baseline any real
    attributor has to beat. Ported from the seed's identity attributer. An agent
    that did not run in a rollout gets ``nan`` for it rather than zero,
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
            name: [reward if name in trace.agents else math.nan for trace, reward in zip(traces, rewards, strict=True)]
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

    ``ATTRIBUTION_PROMPT`` is the CANTANTE contrastive prompt with its
    benchmark-specific paragraph removed (it instructed the model to grade
    against MBPP-style ``assert`` tests embedded in the query) and with the
    terminal output variable named rather than hardcoded to ``prediction``.
    A subclass changes the prompt by overriding the attribute.

    Attributes:
        llm: The meta-model that does the judging.
        group_size: How many executions of one query are compared at once.
        failures: Number of judgements that could not be parsed, from the last
            call. Readable rather than logged, so a caller can act on it.
    """

    ATTRIBUTION_PROMPT = """# Task
You are an attribution agent for a multi-agent system.

You will receive multiple executions of the SAME query.
Each execution uses different agent parametrizations but follows the same workflow.

Your goal is to estimate how each agent's behavior contributed to the outcome of its execution by comparing \
differences across executions.

Base your attribution on:
- the system score differences across executions,
- the system prediction
- how agent outputs differ across executions,
- whether these differences improve or degrade downstream reasoning,
- whether an agent introduces useful reasoning or merely follows,
- whether an agent propagates, corrects, or ignores errors.

## Attribution Scale
Assign a float score in [-1.0, 1.0] to each agent for each execution:
- positive = helpful contribution
- negative = harmful contribution
- 0 = neutral or unclear impact

## Critical Reasoning Guidelines
- Identical predictions or system scores do not imply identical contributions; base attribution on differences \
in reasoning and role adherence.

## Input
You will receive JSON with:
- "query": same for all executions
- "executions": list of execution results, each containing:
  - "system_score": this is the reward signal for attribution, computed by the evaluator based on the \
execution's prediction. The goal is to maximize this quantity.
  - "agent_outputs": mapping from agent name to that agent's output. The output written by the terminal agent, \
under the key "{output_variable}", is the final prediction, on which the system is scored.

## Output
First, briefly compare the executions and identify the key behavioral differences that lead to differences in \
the system score.

Then return JSON inside <attribution>...</attribution>.

## Critical Formatting Rules
- The content inside <attribution>...</attribution> must be valid JSON.
- Do not use markdown code fences inside the <attribution> block.
- The top-level JSON object inside <attribution> must have exactly one key: "executions".
- "executions" must be a list with exactly {n_executions} items.
- Preserve the same execution order as given in the input.
- Each item in "executions" must be an object with exactly one key: "agent_credits".
- "agent_credits" must be an object containing exactly these agent names and no others:
  {agent_names}
- Every credit must be a float between -1.0 and 1.0.
- Do not omit any execution.
- Do not put explanations, summaries, or any other text inside the JSON.
- Any reasoning or explanation must be placed outside the <attribution> block.

# Input

{executions}
"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        group_size: int = 4,
        max_chars_per_field: int = 2000,
        seed: int = 42,
    ) -> None:
        """Initialize the attributor.

        Args:
            llm: The meta-model. This is a judging call, not a system call, so it
                is deliberately separate from the models the agents run on.
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
        self.group_size = group_size
        self.max_chars_per_field = max_chars_per_field
        self.rng = np.random.RandomState(seed)
        self.failures = 0

    def attribute(self, traces: list[Trace], rewards: list[float]) -> dict[str, list[float]]:
        """Compare executions of each query and assign per-agent credit.

        Args:
            traces: One trace per rollout.
            rewards: The system-level score of each rollout.

        Returns:
            Credit per rollout, keyed by agent name, with ``nan`` wherever
            the judge's answer could not be read.

        Raises:
            ValueError: If the inputs are not parallel.
            RuntimeError: If no judgement at all could be parsed, which means the
                meta-model or the prompt is broken rather than the run being noisy.
        """
        if len(traces) != len(rewards):
            raise ValueError(f"Got {len(traces)} traces and {len(rewards)} rewards; they must be parallel.")
        names = agents_in(traces)
        # nan, not zero: zero is a verdict, nan is an absence every consumer filters out
        credit: dict[str, list[float]] = {name: [math.nan] * len(traces) for name in names}
        if not names:
            return credit

        chunks = groups(traces, self.group_size, self.rng)
        if not chunks:
            return credit

        payloads = [
            {"query": traces[group[0]].query, "executions": [self.rollout(traces[i], rewards[i]) for i in group]}
            for group in chunks
        ]
        prompts = [
            prompt(self.ATTRIBUTION_PROMPT, payload, traces[group[0]], names)
            for group, payload in zip(chunks, payloads, strict=True)
        ]
        responses = self.llm.get_response(prompts)

        self.failures = 0
        for group, response in zip(chunks, responses, strict=True):
            parsed = parse_judgement(response, names, len(group))
            if parsed is None:
                self.failures += 1
                continue
            for position, index in enumerate(group):
                for name in names:
                    credit[name][index] = parsed[position][name]

        if self.failures == len(chunks):
            raise RuntimeError(
                f"None of {len(chunks)} attribution judgements could be parsed. "
                "The meta-model returned no readable <attribution> block."
            )
        return credit

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
                outputs[record.agent][name] = truncate(value, self.max_chars_per_field)
        entry: dict[str, Any] = {"system_score": reward, "agent_outputs": outputs}
        if trace.failed:
            entry["error"] = trace.error
        return entry


def agents_in(traces: list[Trace]) -> list[str]:
    """Return every agent that ran in any of the traces, in a stable order.

    Args:
        traces: The rollouts.

    Returns:
        Agent names, sorted. Sorted rather than in execution order because
        different rollouts may order them differently once a topology branches.
    """
    return sorted({record.agent for trace in traces for record in trace.records})


def truncate(value: str, limit: int) -> str:
    """Shorten one field, marking the cut so the judge does not read it as the end."""
    if len(value) <= limit:
        return value
    return value[:limit] + " ...[truncated]"


def groups(traces: list[Trace], size: int, rng: np.random.RandomState) -> list[list[int]]:
    """Split rollout indices into comparable groups: same query, chunked, shuffled.

    Chunk before shuffling, and shuffle only within the chunk, which is the
    seed's order and is what the parity fixture pins. A chunk of one is
    dropped rather than sent: one execution offers nothing to contrast, and a
    judgement made without contrast is not the method.
    """
    by_query: dict[str, list[int]] = {}
    for index, trace in enumerate(traces):
        by_query.setdefault(trace.query, []).append(index)

    chunks: list[list[int]] = []
    for indices in by_query.values():
        for start in range(0, len(indices), size):
            chunk = indices[start : start + size]
            if len(chunk) < 2:
                continue
            rng.shuffle(chunk)
            chunks.append(chunk)
    return chunks


def prompt(template: str, payload: dict[str, Any], first: Trace, names: list[str]) -> str:
    """Render the judging prompt for one group of executions."""
    # The judge is told which key holds the answer. The seed hardcoded
    # "prediction"; here it is whatever the last agent to write wrote.
    output_variable = next(iter(first.records[-1].outputs), "prediction") if first.records else "prediction"
    return (
        template.replace("{executions}", json.dumps(payload, ensure_ascii=False, indent=2))
        .replace("{n_executions}", str(len(payload["executions"])))
        .replace("{agent_names}", json.dumps(names))
        .replace("{output_variable}", output_variable)
    )


def parse_judgement(response: str, names: list[str], size: int) -> list[dict[str, float]] | None:
    """Read one judgement, or return None if it cannot be read as specified."""
    rest = (response or "").partition("<attribution>")[2]
    if "</attribution>" not in rest:
        return None
    try:
        executions = json.loads(rest.partition("</attribution>")[0].strip())["executions"]
        if len(executions) != size:
            return None
        return [
            {name: max(-1.0, min(1.0, float(entry["agent_credits"][name]))) for name in names} for entry in executions
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
