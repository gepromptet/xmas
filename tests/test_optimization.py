"""The optimization loop: propose, evaluate through the task, select."""

from __future__ import annotations

import pandas as pd
import pytest
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.prompt import Prompt

from xmas import MAS, Agent
from xmas.optimization import MASOptimizer, PrefixSearchOptimizer, PromptUpdater
from xmas.optimization.attribution import UniformAttributor
from xmas.optimization.optimizer import traces_by_position
from xmas.optimization.prefix_search import PREFIXES
from xmas.optimization.updaters import AgentPromptView
from xmas.tasks.gsm8k import GSM8KTask
from xmas.trace import Trace

from .conftest import MockLLM


class AddMagic(BaseOptimizer):
    """A stand-in promptolution optimizer, as unmodified as any real one would be."""

    def _pre_optimization_loop(self) -> None:
        return

    def _step(self) -> list[Prompt]:
        return [Prompt(f"{MAGIC} {prompt.instruction}") for prompt in self.prompts]


MAGIC = "Think step by step."
"""The one prefix the mock model responds to, so improvement is unambiguous."""

ANSWERS = {"What is 17 * 23?": "391", "What is 100 - 42?": "58", "What is 6 * 7?": "42"}


def gsm8k_task() -> GSM8KTask:
    """A three-item GSM8K-shaped slice."""
    return GSM8KTask(
        pd.DataFrame(
            {
                "question": list(ANSWERS),
                "answer": [f"#### {value}" for value in ANSWERS.values()],
            }
        )
    )


def prefix_sensitive_llm() -> MockLLM:
    """Answers correctly only when the magic prefix is present."""

    def respond(prompt: str) -> str:
        # A "Final:" prompt belongs to a downstream checker, which passes its
        # upstream draft through unchanged.
        if "Final:" in prompt:
            return prompt.split("Final:", 1)[1].strip()
        question = next((q for q in ANSWERS if q in prompt), None)
        if question is None:
            return "0"
        return ANSWERS[question] if MAGIC in prompt else "0"

    return MockLLM(respond)


def single_agent(llm: MockLLM) -> MAS:
    return MAS([Agent("solver", prompt="Answer: {query}", output="prediction")], llm=llm)


# --------------------------------------------------------------------- loop


def test_the_reference_optimizer_completes_a_full_loop_and_improves() -> None:
    """The acceptance case: propose, evaluate through the task, select, improve."""
    task = gsm8k_task()
    mas = single_agent(prefix_sensitive_llm())

    before = task.evaluate(Prompt(mas.encode()), predictor=mas).agg_scores[0]
    assert before == 0.0, "the starting system must be genuinely bad"

    optimizer = PrefixSearchOptimizer(mas, task)
    best = optimizer.optimize(n_steps=1)

    assert optimizer.history == [1.0]
    assert MAGIC in best.agents["solver"].prompt
    assert best is not mas, "the incumbent must be replaced by a better candidate"


def test_the_loop_keeps_the_incumbent_when_nothing_is_better() -> None:
    task = gsm8k_task()
    mas = single_agent(MockLLM(lambda prompt: "0"))

    optimizer = PrefixSearchOptimizer(mas, task)
    best = optimizer.optimize(n_steps=1)

    assert best is mas, "a tie at the top must keep the current system"


def test_propose_includes_the_incumbent_and_one_candidate_per_prefix() -> None:
    mas = single_agent(prefix_sensitive_llm())
    candidates = PrefixSearchOptimizer(mas, gsm8k_task()).propose()

    assert candidates[0] is mas
    assert len(candidates) == 1 + len(PREFIXES)


def test_evaluate_batches_same_structure_candidates_through_one_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt candidates share a graph, so they cost one task evaluation, not N."""
    task = gsm8k_task()
    mas = single_agent(prefix_sensitive_llm())
    optimizer = PrefixSearchOptimizer(mas, task)

    calls: list[int] = []
    original = task.evaluate

    def counting_evaluate(prompts, predictor, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(len(prompts) if isinstance(prompts, list) else 1)
        return original(prompts, predictor, **kwargs)

    monkeypatch.setattr(task, "evaluate", counting_evaluate)
    scores = optimizer.evaluate(optimizer.propose())

    assert calls == [1 + len(PREFIXES)], "one call carrying the whole population"
    assert len(scores) == 1 + len(PREFIXES)


def test_evaluate_splits_candidates_that_differ_structurally() -> None:
    """A different graph cannot ride another graph's prompt set, so it gets its own pass."""
    task = gsm8k_task()
    llm = prefix_sensitive_llm()
    one = single_agent(llm)
    two = MAS(
        [
            Agent("solver", prompt="Answer: {query}", output="draft"),
            Agent("checker", prompt="Final: {draft}", inputs=["draft"], output="prediction"),
        ],
        edges=[("solver", "checker")],
        llm=llm,
    )

    class Fixed(MASOptimizer):
        def propose(self) -> list[MAS]:
            return [one, two]

        def select(self, candidates: list[MAS], scores: list[float]) -> MAS:
            return candidates[0]

    scores = Fixed(one, task).evaluate([one, two])
    assert len(scores) == 2


# ---------------------------------------------------------- attributor seam


def test_the_attributor_seam_is_consulted_and_its_credit_is_stored() -> None:
    seen: list[tuple[int, int]] = []

    class RecordingAttributor:
        def attribute(self, traces: list[Trace], rewards: list[float]) -> dict[str, list[float]]:
            seen.append((len(traces), len(rewards)))
            return {"solver": [0.0] * len(rewards)}

    mas = single_agent(prefix_sensitive_llm())
    optimizer = PrefixSearchOptimizer(mas, gsm8k_task(), attributor=RecordingAttributor())
    optimizer.optimize(n_steps=1)

    assert seen, "the attributor must be consulted after evaluation"
    assert optimizer.credit == {"solver": [0.0] * seen[0][1]}


def test_a_fully_cached_evaluation_contributes_no_rollouts_rather_than_stale_ones() -> None:
    # `last_traces` holds whatever ran most recently, so a task that answers
    # entirely from its cache leaves the previous call's rollouts standing there.
    # Pairing those against this call's score grid would hand the attributor every
    # reward attached to the wrong rollout, which is invisible in the result.
    mas = single_agent(prefix_sensitive_llm())
    task = gsm8k_task()
    optimizer = PrefixSearchOptimizer(mas, task, attributor=UniformAttributor())

    optimizer.evaluate([mas])
    assert optimizer.rollouts, "the first pass runs rollouts and records them"
    assert mas.last_traces, "and leaves them where the second pass would find them"

    optimizer.evaluate([mas])  # every cell is cached now, so nothing runs

    assert optimizer.rollouts == []


def test_alignment_refuses_to_pair_a_reconstruction_that_stopped_matching() -> None:
    # `traces_by_position` recomputes which cells missed the task's cache from a snapshot taken
    # before the call, which holds only while its key shape matches promptolution's
    # own. Should that shape ever drift, every cell reads as a miss while only the
    # cells that ran have traces, and pairing them would attach every reward to the
    # wrong rollout. Called directly because the whole point is a state the
    # optimizer's own path cannot currently produce.
    mas = single_agent(prefix_sensitive_llm())
    task = gsm8k_task()
    optimizer = PrefixSearchOptimizer(mas, task, attributor=UniformAttributor())
    optimizer.evaluate([mas])

    encoded = [mas.encode(), mas.encode()]

    with pytest.raises(ValueError, match="zip"):
        traces_by_position(task, mas, encoded, cached=set())


def test_credit_steers_which_agent_is_varied() -> None:
    llm = prefix_sensitive_llm()
    mas = MAS(
        [
            Agent("solver", prompt="Answer: {query}", output="draft"),
            Agent("checker", prompt="Final: {draft}", inputs=["draft"], output="prediction"),
        ],
        edges=[("solver", "checker")],
        llm=llm,
    )
    optimizer = PrefixSearchOptimizer(mas, gsm8k_task())
    optimizer.credit = {"solver": [0.9], "checker": [0.1]}

    assert optimizer.target_agent() == "checker", "the least-credited agent is the one to fix"


# ------------------------------------------------------- promptolution seam


def test_agent_prompt_view_presents_one_agent_as_a_single_prompt_predictor() -> None:
    llm = prefix_sensitive_llm()
    mas = MAS(
        [
            Agent("solver", prompt="Answer: {query}", output="draft"),
            Agent("checker", prompt="Final: {draft}", inputs=["draft"], output="prediction"),
        ],
        edges=[("solver", "checker")],
        llm=llm,
    )
    view = AgentPromptView(mas, "solver")

    preds, _ = view.predict(prompts=[f"{MAGIC} Answer: {{query}}"], xs=["What is 6 * 7?"])

    assert preds == ["42"], "the substituted prompt must reach the named agent"
    assert mas.agents["solver"].prompt == "Answer: {query}", "the view must not mutate the MAS"


def test_agent_prompt_view_rejects_an_unknown_agent() -> None:
    mas = single_agent(prefix_sensitive_llm())
    with pytest.raises(KeyError, match="no agent"):
        AgentPromptView(mas, "nope")


def test_prompt_updater_runs_a_promptolution_optimizer_unmodified() -> None:
    """P4 made executable: a promptolution optimizer is a prompt updater for one node."""
    task = gsm8k_task()
    mas = single_agent(prefix_sensitive_llm())
    updater = PromptUpdater(AddMagic, task=task)

    candidates = updater.propose(mas, credit={}, n=1)

    assert len(candidates) == 1
    assert MAGIC in candidates[0].agents["solver"].prompt
    assert mas.agents["solver"].prompt == "Answer: {query}"


def test_prompt_updater_targets_the_least_credited_agent() -> None:
    task = gsm8k_task()
    mas = MAS(
        [
            Agent("solver", prompt="Answer: {query}", output="draft"),
            Agent("checker", prompt="Final: {draft}", inputs=["draft"], output="prediction"),
        ],
        edges=[("solver", "checker")],
        llm=prefix_sensitive_llm(),
    )
    updater = PromptUpdater(AddMagic, task=task)

    assert updater.target_agent(mas, {"solver": [0.9], "checker": [0.2]}) == "checker"
    assert updater.target_agent(mas, {}) == "solver", "no credit means deterministic order"


# ------------------------------------------------------------------ population


def test_the_population_is_a_knob_because_the_estimator_is_population_relative() -> None:
    """A probing task's weights are bounded by M, so M is part of the measurement."""
    mas = single_agent(prefix_sensitive_llm())

    assert len(PrefixSearchOptimizer(mas, gsm8k_task(), population=16).propose()) == 16
    assert len(PrefixSearchOptimizer(mas, gsm8k_task(), population=4).propose()) == 4


def test_the_default_population_is_every_prefix() -> None:
    mas = single_agent(prefix_sensitive_llm())

    assert len(PrefixSearchOptimizer(mas, gsm8k_task()).propose()) == 1 + len(PREFIXES)


def test_a_population_of_one_leaves_nothing_to_compare() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        PrefixSearchOptimizer(single_agent(prefix_sensitive_llm()), gsm8k_task(), population=1)
