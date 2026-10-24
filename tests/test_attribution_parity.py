"""Parity between the xmas contrastive attributor and the seed code it reproduces.

The port changed the input type (traces rather than task state dicts), the prompt
(one benchmark-specific paragraph removed, the terminal variable named rather than
hardcoded) and the failure behaviour (unknown rather than zero). What it did not
change is how credit is computed: which rollouts get compared with which, in what
order, and where each returned number lands.

That is what this pins, against the vendored seed in `tests/seed_parity/` rather
than against a golden file written here. See its README for provenance.
"""

from __future__ import annotations

import json
import math
import re

import pytest
from promptolution.llms.base_llm import BaseLLM

from tests.seed_parity import seed_attribute
from xmas.optimization.attribution import ContrastiveAttributor
from xmas.trace import AgentRecord, Trace

AGENTS = ["drafter", "reviewer"]
QUERIES = ["What is 2+2?", "What is 7*6?"]
VAR_MAPPING = {
    "drafter": {"input_vars": ["query"], "output_vars": ["draft"], "task_description": ""},
    "reviewer": {"input_vars": ["draft"], "output_vars": ["review"], "task_description": ""},
}


class CannedJudge(BaseLLM):
    """Answers every judging call with a well-formed block keyed to its position.

    Each execution in group ``g`` at position ``p`` gets a credit unique to
    ``(g, p, agent)``, so a scatter that puts a number in the wrong cell shows up
    as a mismatch rather than washing out.
    """

    max_tokens = 100_000

    def __init__(self) -> None:
        """Start with an empty record of what it was asked."""
        super().__init__()
        self.prompts: list[str] = []

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        responses = []
        for prompt in prompts:
            group = len(self.prompts)
            self.prompts.append(prompt)
            executions = [
                {
                    "agent_credits": {
                        agent: round(0.1 * group + 0.01 * position + 0.001 * index, 4)
                        for index, agent in enumerate(AGENTS)
                    }
                }
                for position in range(declared_size(prompt))
            ]
            responses.append(f"reasoning\n<attribution>{json.dumps({'executions': executions})}</attribution>")
        return responses


def declared_size(prompt: str) -> int:
    """How many executions the prompt says it wants judged.

    Read from the formatting rules rather than by parsing the payload, because
    both templates also carry a few-shot example, and the seed's contains a JSON
    typo that no parser gets past.
    """
    match = re.search(r"exactly (\d+) items", prompt)
    assert match is not None, "neither template declared its execution count"
    return int(match.group(1))


def payload(prompt: str) -> dict[str, object]:
    """Extract the executions envelope both templates embed.

    Scans for the last well-formed JSON object carrying an ``executions`` key.
    The few-shot example earlier in each template looks the same but is not it,
    and the seed's is not even valid JSON.
    """
    decoder = json.JSONDecoder()
    found: dict[str, object] | None = None
    for position in (index for index, char in enumerate(prompt) if char == "{"):
        try:
            value, _ = decoder.raw_decode(prompt[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "executions" in value and "query" in value:
            found = value
    assert found is not None, "no executions payload in the prompt"
    return found


def outputs_for(variant: str, query: str) -> dict[str, str]:
    """One rollout's agent outputs, distinct per variant so a judge could tell them apart."""
    return {"draft": f"{variant} draft of {query}", "review": f"{variant} review of {query}"}


def reward_for(variant: str) -> float:
    """The system score of a variant. Only ``stepwise`` gets the answer right."""
    return 1.0 if variant == "stepwise" else 0.0


def as_traces(variants: list[str]) -> tuple[list[Trace], list[float]]:
    """The fixture as xmas sees it: one trace per (variant, query), prompt-major."""
    traces, rewards = [], []
    for variant in variants:
        for query in QUERIES:
            values = outputs_for(variant, query)
            records = tuple(
                AgentRecord(
                    agent=agent,
                    step=step,
                    prompt=f"{agent}-{variant}",
                    rendered_prompt=f"{agent}-{variant} on {query}",
                    inputs={"query": query},
                    outputs={key: values[key]},
                    raw_response=values[key],
                )
                for step, (agent, key) in enumerate(zip(AGENTS, ["draft", "review"], strict=True))
            )
            traces.append(Trace(query=query, output=values["review"], records=records))
            rewards.append(reward_for(variant))
    return traces, rewards


def as_seed_inputs(variants: list[str]) -> tuple[list[list[float]], list[list[dict[str, str]]]]:
    """The same fixture as the seed sees it: [run][query] scores and state dicts."""
    scores = [[reward_for(variant) for _ in QUERIES] for variant in variants]
    states = [[{"query": query, **outputs_for(variant, query)} for query in QUERIES] for variant in variants]
    return scores, states


def run_seed(variants: list[str], group_size: int, judge: CannedJudge) -> dict[str, list[list[float]]]:
    """Run the vendored seed attributer on the fixture."""
    scores, states = as_seed_inputs(variants)
    return seed_attribute(judge, AGENTS, VAR_MAPPING, scores, states, group_size=group_size)


def flatten(seed: dict[str, list[list[float]]], agent: str, variants: list[str]) -> list[float]:
    """Reshape the seed's [run][query] credit to the port's prompt-major flat list."""
    return [seed[agent][run][query] for run in range(len(variants)) for query in range(len(QUERIES))]


# ---------------------------------------------------------------------- parity


@pytest.mark.parametrize(
    ("variants", "group_size"),
    [
        (["terse", "verbose", "stepwise"], 3),
        (["terse", "verbose", "stepwise", "tabular"], 2),
        (["terse", "verbose", "stepwise", "tabular"], 4),
    ],
)
def test_credit_lands_in_the_same_cells_as_the_seed(variants: list[str], group_size: int) -> None:
    traces, rewards = as_traces(variants)
    mine = ContrastiveAttributor(CannedJudge(), group_size=group_size).attribute(traces, rewards)
    seed = run_seed(variants, group_size, CannedJudge())

    for agent in AGENTS:
        assert mine[agent] == pytest.approx(flatten(seed, agent, variants)), agent


def test_both_implementations_compare_the_same_rollouts() -> None:
    # Credit could coincide by luck even if the grouping differed, so the
    # judging payloads themselves are compared, group for group.
    variants = ["terse", "verbose", "stepwise", "tabular"]
    mine_judge, seed_judge = CannedJudge(), CannedJudge()

    traces, rewards = as_traces(variants)
    ContrastiveAttributor(mine_judge, group_size=2).attribute(traces, rewards)
    run_seed(variants, 2, seed_judge)

    assert len(mine_judge.prompts) == len(seed_judge.prompts) == 4
    for mine_prompt, seed_prompt in zip(mine_judge.prompts, seed_judge.prompts, strict=True):
        assert payload(mine_prompt) == payload(seed_prompt)


# ------------------------------------------------------------ the divergences


def test_a_group_of_one_is_skipped_rather_than_judged_without_contrast() -> None:
    # Three rollouts of a query at group_size two leaves a group of one. The seed
    # sends it anyway, and since one execution offers nothing to contrast, it
    # gets back whatever the judge invents. xmas declines to ask.
    variants = ["terse", "verbose", "stepwise"]
    mine_judge, seed_judge = CannedJudge(), CannedJudge()

    traces, rewards = as_traces(variants)
    mine = ContrastiveAttributor(mine_judge, group_size=2).attribute(traces, rewards)
    run_seed(variants, 2, seed_judge)

    assert len(seed_judge.prompts) == 4  # two per query, one of them a singleton
    assert len(mine_judge.prompts) == 2  # the singletons are not asked about
    for agent in AGENTS:
        # The lone third variant is left unknown, in both queries.
        assert [math.isnan(value) for value in mine[agent]] == [False, False, False, False, True, True]


class Mute(CannedJudge):
    """A judge whose answers can never be parsed."""

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        self.prompts.extend(prompts)
        return ["I would rather not." for _ in prompts]


def test_an_unreadable_judgement_is_unknown_rather_than_zero() -> None:
    # The seed logs a warning and falls back to zeros, which reads downstream as
    # "this agent did not matter" rather than "we could not tell". Both are
    # exercised here so the difference is on the record and not just in a
    # docstring.
    variants = ["terse", "verbose"]
    traces, rewards = as_traces(variants)

    seed = run_seed(variants, 2, Mute())
    assert seed["drafter"] == [[0.0, 0.0], [0.0, 0.0]]

    with pytest.raises(RuntimeError, match="could be parsed"):
        ContrastiveAttributor(Mute(), group_size=2).attribute(traces, rewards)
