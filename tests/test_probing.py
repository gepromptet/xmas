"""Probing: the estimator's arithmetic, the pool, and the claim that P2 makes."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from promptolution.utils.prompt import Prompt

from xmas import MAS, Agent
from xmas.llms.api_llm import APILLM, ScoringUnsupportedError
from xmas.llms.scoring import ScoredSequence
from xmas.optimization import PrefixSearchOptimizer
from xmas.tasks.gsm8k import GSM8KTask
from xmas.tasks.probing import (
    ImportanceSampling,
    ProbeTensor,
    ProbingTask,
    Reference,
    ReferencePool,
    importance_weights,
    measure_agreement,
    rank_agreement,
)

from .conftest import MockLLM, requires_gateway

SLICE = pd.DataFrame(
    {
        "question": ["What is 2+2?", "What is 3+3?", "What is 5+5?"],
        "answer": ["#### 4", "#### 6", "#### 10"],
    }
)


class ScoringLLM(MockLLM):
    """A mock that can also teacher-force, with a caller-supplied log-probability."""

    def __init__(
        self,
        logprob: Callable[[str, str], float] | None = None,
        responder: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the mock, defaulting to a length-based log-probability."""
        super().__init__(responder)
        self.logprob = logprob or (lambda context, continuation: -float(len(continuation)))
        self.scored: list[tuple[str, str]] = []

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return a scripted teacher-forced score and record the call."""
        self.scored.append((context, continuation))
        tokens = tuple(continuation.split()) or (continuation,)
        per_token = self.logprob(context, continuation) / len(tokens)
        return ScoredSequence(tokens=tokens, token_logprobs=tuple(per_token for _ in tokens))


def single_agent(llm: MockLLM, prompt: str = "Solve: {query}") -> MAS:
    return MAS([Agent("solver", prompt=prompt, output="prediction")], llm=llm)


def pool_of(*rows: tuple[str, str, float]) -> ReferencePool:
    return ReferencePool(
        [Reference(x=x, inputs={"query": x}, completion=completion, reward=reward) for x, completion, reward in rows]
    )


# --------------------------------------------------------------- the weights


def test_weights_are_a_population_share_that_averages_to_one() -> None:
    """Every column's weights average to 1 over the candidates: that is the normalization."""
    log_p = np.log(np.array([[0.6, 0.4], [0.2, 0.8]]))

    weights = importance_weights(log_p)

    assert weights[:, 0] == pytest.approx([1.5, 0.5])
    assert weights[:, 1] == pytest.approx([2 / 3, 4 / 3])
    assert weights.mean(axis=0) == pytest.approx([1.0, 1.0])


def test_weights_survive_probabilities_that_underflow_float64() -> None:
    """A few hundred negative nats is an ordinary sequence log-probability.

    exp() of it is exactly 0.0, so anything that exponentiates before dividing gets
    0/0. The ratio is well conditioned even when neither side is representable.
    """
    log_p = np.array([[-800.0, -801.0], [-802.0, -800.0]])

    weights = importance_weights(log_p)

    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert weights.mean(axis=0) == pytest.approx([1.0, 1.0])


def test_an_unprobed_cell_drops_out_of_its_column_rather_than_scoring_zero() -> None:
    log_p = np.array([[math.log(0.5), math.log(0.5)], [np.nan, math.log(0.5)]])

    weights = importance_weights(log_p)

    assert np.isnan(weights[1, 0])
    assert weights[0, 0] == pytest.approx(1.0), "a lone candidate is the whole mixture"
    assert weights[:, 1] == pytest.approx([1.0, 1.0])


# ------------------------------------------------------------- the estimator


def test_the_estimator_credits_probability_placed_on_rewarded_references() -> None:
    """Two candidates, one input, one good reference and one bad one."""
    probe = ProbeTensor(
        log_p=np.log(np.array([[0.6, 0.4], [0.2, 0.8]])),
        reward=np.array([1.0, 0.0]),
        input_index=np.array([0, 0]),
        inputs=("q",),
        probed_tokens=np.array([1.0, 1.0]),
    )

    scores = ImportanceSampling()(probe)

    # candidate 0 puts 1.5x the population's mass on the rewarded reference,
    # candidate 1 puts 0.5x; the unrewarded column contributes nothing to either.
    assert scores[:, 0] == pytest.approx([0.75, 0.25])


def test_the_estimator_reports_an_unmeasurable_input_as_nan_not_as_zero() -> None:
    probe = ProbeTensor(
        log_p=np.array([[math.log(0.5)], [np.nan]]),
        reward=np.array([1.0]),
        input_index=np.array([0]),
        inputs=("q",),
        probed_tokens=np.array([1.0]),
    )

    scores = ImportanceSampling()(probe)

    assert scores[0, 0] == pytest.approx(1.0)
    assert np.isnan(scores[1, 0]), "no probe is missing measurement, not a score of zero"


# -------------------------------------------------------------- the pool


def test_a_reference_with_an_empty_completion_is_refused() -> None:
    with pytest.raises(ValueError, match="empty completion"):
        pool_of(("q", "", 1.0))


def test_the_budget_takes_the_same_first_k_references_for_every_input() -> None:
    pool = pool_of(("a", "a1", 1.0), ("a", "a2", 0.0), ("b", "b1", 1.0), ("b", "b2", 0.0))

    columns = pool.columns(["a", "b"], k=1)

    assert [reference.completion for reference in columns] == ["a1", "b1"]
    assert len(pool.columns(["a", "b"])) == 4


def test_a_pool_that_covers_none_of_the_inputs_says_so() -> None:
    pool = pool_of(("a", "a1", 1.0))

    with pytest.raises(ValueError, match="covers none of the"):
        pool.columns(["z"])


def test_generate_records_the_reward_each_reference_actually_earned() -> None:
    llm = ScoringLLM(responder=lambda prompt: "4" if "2+2" in prompt else "wrong")
    mas = single_agent(llm)
    task = GSM8KTask(SLICE)

    pool = ReferencePool.generate(task, [mas], task.xs, task.ys, refs_per_input=1, seed=0)

    rewards = {reference.x: reference.reward for reference in pool.references}
    assert rewards["What is 2+2?"] == 1.0
    assert rewards["What is 3+3?"] == 0.0


def test_generate_draws_sources_uniformly_over_the_whole_population() -> None:
    """The mixture the estimator divides by is the one references came from."""
    llm = ScoringLLM()
    candidates = [single_agent(llm, f"v{index}: {{query}}") for index in range(3)]
    task = GSM8KTask(SLICE)

    pool = ReferencePool.generate(task, candidates, task.xs, task.ys, refs_per_input=6, seed=1)

    sources = {reference.source for reference in pool.references}
    assert len(sources) == 3, "all three candidates must appear as reference authors"


# ---------------------------------------------------------- the probing task


def test_probing_renders_the_candidate_prompt_against_the_reference_state() -> None:
    llm = ScoringLLM()
    mas = single_agent(llm)
    pool = pool_of(("What is 2+2?", "the answer is 4", 1.0))
    task = ProbingTask(GSM8KTask(SLICE), pool)

    task.evaluate([Prompt("Solve: {query}")], predictor=mas)

    assert llm.scored == [("Solve: What is 2+2?", "the answer is 4")]


def test_probing_decodes_nothing() -> None:
    llm = ScoringLLM()
    mas = single_agent(llm)
    pool = pool_of(("What is 2+2?", "four", 1.0), ("What is 3+3?", "six", 0.0))
    task = ProbingTask(GSM8KTask(SLICE), pool)

    result = task.evaluate([Prompt("Solve: {query}"), Prompt("Answer: {query}")], predictor=mas)

    assert llm.calls == [], "probing must never call generation"
    assert np.all(result.output_tokens == 0.0)
    assert result.scores.shape == (2, 2)
    assert result.agg_scores.shape == (2,)


def test_a_repeated_probe_cell_is_scored_once() -> None:
    """Two references that decoded to the same text under the same prompt are one probe."""
    llm = ScoringLLM()
    mas = single_agent(llm)
    pool = pool_of(("What is 2+2?", "four", 1.0), ("What is 2+2?", "four", 1.0))
    task = ProbingTask(GSM8KTask(SLICE), pool)

    task.evaluate([Prompt("Solve: {query}")], predictor=mas)

    assert len(llm.scored) == 1


def test_a_model_that_cannot_teacher_force_is_named_in_the_error() -> None:
    mas = single_agent(MockLLM())
    task = ProbingTask(GSM8KTask(SLICE), pool_of(("What is 2+2?", "four", 1.0)))

    with pytest.raises(TypeError, match="has no score"):
        task.evaluate([Prompt("Solve: {query}")], predictor=mas)


def test_probing_carries_upstream_state_into_a_multi_agent_terminal_prompt() -> None:
    """The terminal agent's prompt reads {draft}, and the reference supplies it."""
    llm = ScoringLLM()
    mas = MAS(
        [
            Agent("solver", prompt="Solve: {query}", output="draft"),
            Agent("checker", prompt="Check {draft}", inputs=["query", "draft"], output="prediction"),
        ],
        edges=[("solver", "checker")],
        llm=llm,
    )
    pool = ReferencePool(
        [
            Reference(
                x="What is 2+2?",
                inputs={"query": "What is 2+2?", "draft": "maybe 4"},
                completion="yes, 4",
                reward=1.0,
            )
        ]
    )
    task = ProbingTask(GSM8KTask(SLICE), pool)

    task.evaluate([Prompt(mas.encode())], predictor=mas)

    assert llm.scored == [("Check maybe 4", "yes, 4")]


# ------------------------------------------------------------------ P2 itself


def test_the_reference_optimizer_runs_unchanged_on_a_probing_task() -> None:
    """The whole point of the package: no optimizer learns that anything changed.

    The scorer is rigged so that a prompt mentioning "step by step" concentrates
    probability on the rewarded reference, which is what the optimizer should find.
    """

    def logprob(context: str, continuation: str) -> float:
        good = "step by step" in context and continuation == "correct"
        return -1.0 if good else -5.0

    llm = ScoringLLM(logprob)
    mas = single_agent(llm, "Solve: {query}")
    pool = pool_of(
        ("What is 2+2?", "correct", 1.0),
        ("What is 2+2?", "wrong", 0.0),
        ("What is 3+3?", "correct", 1.0),
        ("What is 3+3?", "wrong", 0.0),
    )
    task = ProbingTask(GSM8KTask(SLICE), pool)
    optimizer = PrefixSearchOptimizer(mas, task)

    best = optimizer.optimize(n_steps=1)

    assert len(optimizer.history) == 1
    assert llm.calls == [], "the loop must have run without a single generation"
    assert "step by step" in best.agents["solver"].prompt.lower()


def test_a_probing_task_reuses_the_wrapped_tasks_reward() -> None:
    inner = GSM8KTask(SLICE)
    task = ProbingTask(inner, pool_of(("What is 2+2?", "four", 1.0)))

    scores = task.reward(["What is 2+2?"], ["#### 4"], ["the answer is 4.0"])

    assert scores.tolist() == [1.0], "GSM8K's numeric tolerance, not string equality"


# --------------------------------------------------------------- agreement


def test_rank_agreement_is_one_for_an_identical_ordering() -> None:
    assert rank_agreement([0.1, 0.5, 0.9], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_rank_agreement_is_minus_one_for_a_reversed_ordering() -> None:
    assert rank_agreement([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_rank_agreement_is_undefined_without_variance() -> None:
    assert math.isnan(rank_agreement([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_rank_agreement_averages_tied_ranks() -> None:
    assert rank_agreement([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)


def test_rank_agreement_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="Cannot correlate"):
        rank_agreement([1.0, 2.0], [1.0])


# ------------------------------------------------------------------ live


@requires_gateway
def test_probing_against_the_live_gateway() -> None:
    """Env-gated, and skips today because the gateway cannot teacher-force.

    The capability check runs first and costs nothing: ``/v1/score`` answers 501
    before any upstream call, so this skips without decoding the reference pool it
    would otherwise need. It starts verifying the moment a scoring-capable backend
    is configured, rather than being deleted and forgotten.
    """
    llm = APILLM(model="llama-3.1-8b", max_tokens=256, max_concurrent_calls=4)
    try:
        llm.score("The capital of France is", " Paris")
    except ScoringUnsupportedError as error:
        pytest.skip(f"gateway cannot teacher-force yet: {error}")

    mas = single_agent_live(llm)
    task = GSM8KTask(SLICE)
    pool = ReferencePool.generate(task, [mas], task.xs, task.ys, refs_per_input=1, seed=0)
    probing = ProbingTask(task, pool)

    result = probing.evaluate([Prompt(mas.encode())], predictor=mas)

    assert result.agg_scores.shape == (1,)
    assert np.isfinite(result.agg_scores[0])
    assert np.all(result.output_tokens == 0.0), "probing decodes nothing"


def single_agent_live(llm: APILLM) -> MAS:
    """A one-agent MAS on a live model, which is the exact single-prompt case."""
    return MAS(
        [Agent("solver", prompt="Solve the problem. End with the final number alone.\n\n{query}", output="prediction")],
        llm=llm,
    )


# ---------------------------------------------------------------- readout


def test_the_readout_reports_perfect_agreement_when_probing_tracks_generation() -> None:
    """The scorer is rigged so likelihood mass follows the reward the MAS earns."""
    correct = {"What is 2+2?": "4", "What is 3+3?": "6", "What is 5+5?": "10"}

    def responder(prompt: str) -> str:
        answer = next((value for key, value in correct.items() if key in prompt), "0")
        return answer if prompt.startswith("good") else "0"

    def logprob(context: str, continuation: str) -> float:
        right = continuation in correct.values()
        return -1.0 if (context.startswith("good") and right) else -6.0

    llm = ScoringLLM(logprob, responder)
    mas = single_agent(llm, "good: {query}")
    task = GSM8KTask(SLICE)

    readout = measure_agreement(
        task,
        mas,
        [Prompt("good: {query}"), Prompt("bad: {query}")],
        refs_per_input=1,
        seed=0,
    )

    assert readout.generation[0] > readout.generation[1], "the good prompt decodes better"
    assert readout.probing[0] > readout.probing[1], "and probing must agree"
    assert readout.rho == pytest.approx(1.0)


def test_the_readout_refuses_a_single_candidate() -> None:
    llm = ScoringLLM()
    mas = single_agent(llm)

    with pytest.raises(ValueError, match="at least two candidates"):
        measure_agreement(GSM8KTask(SLICE), mas, [Prompt("Solve: {query}")])
