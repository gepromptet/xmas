"""The whole loop on GSM8K: attribution-guided prompts, and topology search.

Both variants run here against mock models, so CI exercises the full path with no
credential and no spend. The live equivalents live in `test_live_gateway.py` and
skip without one.

A mock model has to encode *something* for an optimizer to find, and what it
encodes is the claim being tested. Each mock below states its rule in one
sentence, and neither rule is one the optimizer is told about.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.prompt import Prompt

from xmas import MAS, Agent
from xmas.optimization import (
    CantanteOptimizer,
    EvolutionaryTopologyOptimizer,
    GraphUpdater,
    MechanicalSynthesizer,
    TopologyBudget,
    UniformAttributor,
)
from xmas.tasks.gsm8k import GSM8KTask

from .conftest import MockLLM

SLICE = pd.DataFrame(
    {
        "question": [
            "Natalia sold clips to 48 friends in April, and then she sold half as many "
            "clips in May. How many clips did Natalia sell altogether in April and May?",
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
            "of babysitting. How much did she earn?",
        ],
        "answer": ["#### 72", "#### 10"],
    }
)
"""Two real GSM8K problems, inlined so the suite needs no dataset download."""

ANSWERS = [72.0, 10.0]
PEER = "<<peer>>"
"""Marker every mock agent stamps on its output, so downstream agents are countable."""

QUALITY = "step by step"
"""The instruction the prompt-optimization mock rewards. The optimizer is not told."""


def question_index(prompt: str) -> int:
    """Work out which problem a rendered prompt is about.

    From an upstream agent's stamp where there is one, and from the question text
    otherwise, so an agent deep in the graph can still be scored.
    """
    stamped = re.search(rf"{re.escape(PEER)} q=(\d+)", prompt)
    if stamped:
        return int(stamped.group(1))
    for index, question in enumerate(SLICE["question"]):
        if question[:40] in prompt:
            return index
    return 0


def stamp(index: int, value: float) -> str:
    """Render one mock agent's answer, carrying the problem id for downstream agents."""
    return f"{PEER} q={index} a={value:g}\nThe final answer is {value:g}"


def consensus_model() -> MockLLM:
    """A model that is right only when it can see at least two upstream answers.

    The rule topology search has to discover: one opinion is not enough. Nothing
    about the prompt text matters here, so a prompt optimizer would find nothing.
    """

    def respond(prompt: str) -> str:
        index = question_index(prompt)
        peers = prompt.count(PEER)
        return stamp(index, ANSWERS[index] if peers >= 2 else ANSWERS[index] + 1)

    return MockLLM(respond)


def instructed_model() -> MockLLM:
    """A model that is right only when its prompt asks it to work step by step.

    The rule prompt optimization has to discover. Nothing about the topology
    matters here, so a topology search would find nothing.
    """

    def respond(prompt: str) -> str:
        index = question_index(prompt)
        return stamp(index, ANSWERS[index] if QUALITY in prompt else ANSWERS[index] + 1)

    return MockLLM(respond)


def pipeline(llm: MockLLM) -> MAS:
    """The starting system: one solver, one checker, and nothing clever."""
    return MAS(
        agents=[
            Agent("solver", "Answer this problem.\n\n{query}", output="solver_out"),
            Agent(
                "checker",
                "Give the final number for this working.\n\n{solver_out}",
                output="prediction",
                inputs=["solver_out"],
            ),
        ],
        edges=[("solver", "checker")],
        llm=llm,
    )


def gsm8k(llm: MockLLM) -> GSM8KTask:
    """The task, over the inlined slice."""
    return GSM8KTask(df=SLICE.copy())


class Rewriter(BaseOptimizer):
    """A prompt optimizer that offers one rewrite and keeps whichever scored better.

    Small enough to be deterministic, ordinary enough that the broker cannot tell
    it from a real one. `EvoPromptGA` is exercised against the broker in
    `test_cantante.py`; what this test is about is the loop reaching GSM8K.
    """

    def _pre_optimization_loop(self) -> None:
        return None

    def _step(self) -> list[Prompt]:
        # The rewrite prepends the instruction and leaves the prompt otherwise
        # alone, so it works for any agent and cannot drop a placeholder. A
        # rewrite that hardcoded one agent's variables would silently break the
        # others, which is a mistake a real optimizer can make too.
        offers = [*self.prompts, *(Prompt(f"Work {QUALITY}.\n\n{prompt}") for prompt in self.prompts)]
        scores = self.task.evaluate(offers, self.predictor).agg_scores
        return [offers[index] for index in sorted(range(len(offers)), key=lambda i: -float(scores[i]))]


# ------------------------------------------------------- attribution-guided prompts


def test_attribution_guided_prompt_optimization_improves_a_gsm8k_slice() -> None:
    mas = pipeline(instructed_model())
    task = gsm8k(mas.llm)  # type: ignore[arg-type]
    optimizer = CantanteOptimizer(mas, task, UniformAttributor(), Rewriter, inner_steps=1)

    before = optimizer.evaluate([mas])[0]
    best = optimizer.optimize(n_steps=1)
    after = optimizer.evaluate([best])[0]

    assert before == 0.0
    assert after == 1.0
    assert QUALITY in best.agents["checker"].prompt
    assert optimizer.rounds_served >= 1
    assert set(optimizer.credit) == {"solver", "checker"}


# ------------------------------------------------------------------ topology search


def test_topology_search_improves_a_gsm8k_slice() -> None:
    mas = pipeline(consensus_model())
    task = gsm8k(mas.llm)  # type: ignore[arg-type]
    updater = GraphUpdater(
        MechanicalSynthesizer(),
        TopologyBudget(min_agents=2, max_agents=4, max_edges=8),
        seed=3,
    )
    optimizer = EvolutionaryTopologyOptimizer(mas, task, updater, mutations=4, seed=1)

    before = optimizer.evaluate([mas])[0]
    best = optimizer.optimize(n_steps=3)
    after = optimizer.evaluate([best])[0]

    assert before == 0.0
    assert after > before
    # The system it found feeds its terminal agent from more than one place,
    # which is the rule the mock encoded and nobody told the optimizer.
    terminal = best.agents["prediction" if "prediction" in best.agents else "checker"]
    assert len(terminal.inputs) >= 2
    assert any(entry != "no change" for entry in optimizer.accepted)


def test_the_two_variants_compose_over_one_system() -> None:
    # P4's claim in one test: prompts and topology are separate components on the
    # same loop, so running both is alternating two optimizers rather than
    # reaching for a third framework. One problem in the slice yields to a better
    # prompt and the other to a better shape, so each variant carries half the
    # score and neither can finish alone.
    def respond(prompt: str) -> str:
        index = question_index(prompt)
        good = QUALITY in prompt if index == 0 else prompt.count(PEER) >= 2
        return stamp(index, ANSWERS[index] if good else ANSWERS[index] + 1)

    mas = pipeline(MockLLM(respond))
    task = gsm8k(mas.llm)  # type: ignore[arg-type]

    shape = EvolutionaryTopologyOptimizer(
        mas,
        task,
        GraphUpdater(MechanicalSynthesizer(), TopologyBudget(min_agents=2, max_agents=4, max_edges=8), seed=3),
        mutations=4,
        seed=1,
    )
    widened = shape.optimize(n_steps=3)

    words = CantanteOptimizer(widened, task, UniformAttributor(), Rewriter, inner_steps=1)
    final = words.optimize(n_steps=1)

    assert score(task, mas) == 0.0
    assert score(task, widened) == 0.5  # the shape solved the problem that needed one
    assert score(task, final) == 1.0  # the prompts solved the other
    assert widened.structure != mas.structure  # topology changed
    assert final.structure == widened.structure  # and prompt optimization kept it


def score(task: GSM8KTask, mas: MAS) -> float:
    """Aggregate task score for one system, through the task and nothing else."""
    return float(task.evaluate([Prompt(mas.encode())], predictor=mas).agg_scores[0])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_search_reproduces_from_its_seed(seed: int) -> None:
    def run() -> list[float]:
        mas = pipeline(consensus_model())
        updater = GraphUpdater(MechanicalSynthesizer(), TopologyBudget(max_agents=4, max_edges=8), seed=seed)
        optimizer = EvolutionaryTopologyOptimizer(mas, gsm8k(mas.llm), updater, mutations=3, seed=seed)  # type: ignore[arg-type]
        optimizer.optimize(n_steps=2)
        return optimizer.history

    assert run() == run()
