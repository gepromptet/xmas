"""The CANTANTE loop: one system evaluation, split into per-agent credit.

What is being pinned here is the claim the method rests on. Several agents get
their prompts optimized at the same time; the system is evaluated once per round
rather than once per agent; and each agent's optimizer is handed its own share of
that one result, without knowing it is inside a multi-agent system.
"""

from __future__ import annotations

import math
import time
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.tasks.base_task import EvalResult
from promptolution.utils.prompt import Prompt

from xmas import MAS, Agent, MASTask
from xmas.optimization import broker, cantante
from xmas.optimization.attribution import UNKNOWN, UniformAttributor
from xmas.optimization.cantante import CantanteOptimizer
from xmas.trace import AgentRecord, Trace

from .conftest import MockLLM

GOOD = "GOOD"
"""The token an agent's prompt has to contain for the mock system to be right."""


class Counting(MASTask):
    """Scores one if the terminal output says the answer, zero otherwise."""

    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        return np.asarray([1.0 if str(y) in str(pred) else 0.0 for y, pred in zip(ys, preds, strict=True)])


def task(rows: int = 2) -> Counting:
    """A tiny task whose answer is embedded in the query."""
    frame = pd.DataFrame(
        {"x": [f"question-{index}" for index in range(rows)], "y": [f"answer-{index}" for index in range(rows)]}
    )
    return Counting(df=frame, x_column="x", y_column="y", task_description="say the answer")


def system() -> tuple[MAS, MockLLM]:
    """A two-agent system that only answers correctly once both prompts say GOOD."""

    def respond(prompt: str) -> str:
        index = prompt.rsplit("question-", 1)[-1][:1] if "question-" in prompt else "0"
        return f"answer-{index}" if prompt.count(GOOD) >= 1 else "no idea"

    llm = MockLLM(respond)
    agents = [
        Agent("drafter", "draft {query}", output="draft"),
        Agent("reviewer", "review {draft} for {query}", output="review", inputs=["draft", "query"]),
    ]
    return MAS(agents=agents, edges=[("drafter", "reviewer")], llm=llm), llm


class Scripted(BaseOptimizer):
    """An optimizer that offers a fixed candidate list and keeps the best-scored.

    Stands in for a real promptolution optimizer where the test needs the round
    to be deterministic. It uses the ordinary contract, so what it exercises in
    the broker is exactly what a real optimizer would.
    """

    candidates: ClassVar[dict[str, list[str]]] = {}

    def __init__(self, predictor: object, task: object, initial_prompts: list[str], **kwargs: object) -> None:
        """Initialize with the agent's seed prompts."""
        super().__init__(predictor=predictor, task=task, initial_prompts=initial_prompts)  # type: ignore[arg-type]
        self.agent = getattr(task, "agent", "")

    def _pre_optimization_loop(self) -> None:
        return None

    def _step(self) -> list[Prompt]:
        offers = [Prompt(text) for text in self.candidates.get(self.agent, [])] or list(self.prompts)
        result = self.task.evaluate(offers, self.predictor)
        order = sorted(range(len(offers)), key=lambda index: -float(result.agg_scores[index]))
        return [offers[index] for index in order]


# ------------------------------------------------------------------ attributors


def test_the_uniform_attributor_gives_every_agent_that_ran_the_system_score() -> None:
    traces = [
        Trace(query="q", output="o", records=(AgentRecord("a", 0, "p", "p", {}, {"o": "x"}),)),
        Trace(
            query="q",
            output="o",
            records=(AgentRecord("a", 0, "p", "p", {}, {"o": "x"}), AgentRecord("b", 1, "p", "p", {}, {"o": "y"})),
        ),
    ]

    credit = UniformAttributor().attribute(traces, [1.0, 0.25])

    assert credit["a"] == [1.0, 0.25]
    assert math.isnan(credit["b"][0])  # b did not run in the first rollout
    assert credit["b"][1] == 0.25


def test_an_attributor_refuses_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="parallel"):
        UniformAttributor().attribute([Trace(query="q", output="o")], [1.0, 2.0])


# ----------------------------------------------------------------- the broker


def test_one_round_evaluates_the_system_once_not_once_per_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cost claim. Two agents each offering two candidates become two joint
    # prompt sets scored in a single call, not two separate optimizations of the
    # whole system.
    mas, _ = system()
    Scripted.candidates = {
        "drafter": [f"draft {GOOD} {{query}}", "draft {query}"],
        "reviewer": [f"review {GOOD} {{draft}} for {{query}}", "review {draft} for {query}"],
    }
    subject = task()
    optimizer = CantanteOptimizer(mas, subject, UniformAttributor(), Scripted, inner_steps=1)

    evaluated: list[int] = []
    original = subject.evaluate

    def counting(prompts: Prompt | list[Prompt], **kwargs: Any) -> EvalResult:
        evaluated.append(len(prompts) if isinstance(prompts, list) else 1)
        return original(prompts, **kwargs)

    monkeypatch.setattr(subject, "evaluate", counting)
    optimizer.propose()

    assert optimizer.rounds_served == 1
    assert evaluated == [2]  # one call, two joint prompt sets


def test_a_round_waits_for_the_slow_agent_instead_of_serving_without_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # The same cost claim, under the timing a real inner optimizer produces. CAPO
    # writes few-shot examples before its first evaluate and EvoPromptGA scores an
    # initial population, so agents do not reach the broker at the same instant. A
    # broker that served the first request it saw would spend one whole system
    # evaluation per agent, which is exactly what this seam exists to avoid.
    class Staggered(Scripted):
        def _pre_optimization_loop(self) -> None:
            if self.agent == "reviewer":
                time.sleep(0.2)

    mas, _ = system()
    Scripted.candidates = {
        "drafter": [f"draft {GOOD} {{query}}", "draft {query}"],
        "reviewer": [f"review {GOOD} {{draft}} for {{query}}", "review {draft} for {query}"],
    }
    subject = task()
    optimizer = CantanteOptimizer(mas, subject, UniformAttributor(), Staggered, inner_steps=1)

    evaluated: list[int] = []
    original = subject.evaluate

    def counting(prompts: Prompt | list[Prompt], **kwargs: Any) -> EvalResult:
        evaluated.append(len(prompts) if isinstance(prompts, list) else 1)
        return original(prompts, **kwargs)

    monkeypatch.setattr(subject, "evaluate", counting)
    optimizer.propose()

    assert optimizer.rounds_served == 1
    assert evaluated == [2]


def test_a_round_gives_up_on_an_agent_that_never_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    # The backstop. One agent takes longer than the settle timeout allows, so the
    # round serves the one that is waiting rather than stalling behind it, and the
    # latecomer gets a round of its own.
    class Late(Scripted):
        def _pre_optimization_loop(self) -> None:
            if self.agent == "reviewer":
                time.sleep(0.4)

    monkeypatch.setattr(
        cantante,
        "rounds",
        lambda board, workers: broker.rounds(board, workers, settle_timeout=0.0),
    )
    mas, _ = system()
    Scripted.candidates = {
        "drafter": [f"draft {GOOD} {{query}}", "draft {query}"],
        "reviewer": [f"review {GOOD} {{draft}} for {{query}}", "review {draft} for {query}"],
    }
    optimizer = CantanteOptimizer(mas, task(), UniformAttributor(), Late, inner_steps=1)

    candidates = optimizer.propose()

    assert optimizer.rounds_served == 2  # served separately, but neither was dropped
    assert len(candidates) == 2


def test_each_agent_receives_only_its_own_share_of_the_result() -> None:
    mas, _ = system()
    Scripted.candidates = {
        "drafter": [f"draft {GOOD} {{query}}", "draft {query}"],
        "reviewer": [f"review {GOOD} {{draft}} for {{query}}", "review {draft} for {query}"],
    }
    optimizer = CantanteOptimizer(mas, task(), UniformAttributor(), Scripted, inner_steps=1)

    candidates = optimizer.propose()

    assert len(candidates) == 2  # the incumbent, and what the round settled on
    assert GOOD in candidates[1].agents["drafter"].prompt
    assert set(optimizer.credit) == {"drafter", "reviewer"}


def test_the_loop_improves_the_system_and_records_real_scores_not_credit() -> None:
    mas, _ = system()
    Scripted.candidates = {
        "drafter": ["draft {query}", f"draft {GOOD} {{query}}"],
        "reviewer": ["review {draft} for {query}", f"review {GOOD} {{draft}} for {{query}}"],
    }
    optimizer = CantanteOptimizer(mas, task(), UniformAttributor(), Scripted, inner_steps=1)

    assert optimizer.evaluate([mas]) == [0.0]
    best = optimizer.optimize(n_steps=1)

    assert optimizer.history == [1.0]
    assert GOOD in best.agents["drafter"].prompt
    assert optimizer.evaluate([best]) == [1.0]


def test_an_unmodified_promptolution_optimizer_drives_an_agent() -> None:
    # The payoff of the seam: EvoPromptGA is constructed and run exactly as it
    # would be on a single-prompt task, and never learns it is inside a graph.
    mas, _ = system()
    meta = MockLLM(lambda prompt: f"<prompt>{GOOD}: answer {{query}}</prompt>")
    optimizer = CantanteOptimizer(
        mas,
        task(),
        UniformAttributor(),
        EvoPromptGA,
        inner_steps=1,
        initial_prompts={
            "drafter": ["draft {query}", "please draft {query}"],
            "reviewer": ["review {draft} for {query}", "please review {draft} for {query}"],
        },
        optimizer_kwargs={"meta_llm": meta},
    )

    best = optimizer.optimize(n_steps=1)

    assert optimizer.rounds_served > 0
    assert optimizer.history == [1.0]
    # The meta-model writes the same prompt for whichever agent asks, so which
    # agent ends up carrying it is the optimizer's business, not this test's.
    assert any(GOOD in agent.prompt for agent in best.agents.values())


def test_a_broken_inner_optimizer_does_not_hang_the_round() -> None:
    class Exploding(Scripted):
        def _step(self) -> list[Prompt]:
            raise RuntimeError("inner optimizer failed")

    mas, _ = system()
    optimizer = CantanteOptimizer(mas, task(), UniformAttributor(), Exploding, inner_steps=1)

    # promptolution's own loop swallows a step failure and stops, so the round
    # ends with nothing proposed rather than deadlocking on a parked request.
    assert optimizer.propose() == [mas]
    assert optimizer.rounds_served == 0


def test_unattributed_rollouts_fall_back_to_the_measured_score_and_are_counted() -> None:
    class Silent:
        def attribute(self, traces: list[Trace], rewards: list[float]) -> dict[str, list[float]]:
            return {name: [UNKNOWN] * len(traces) for name in ("drafter", "reviewer")}

    mas, _ = system()
    Scripted.candidates = {"drafter": [f"draft {GOOD} {{query}}", "draft {query}"]}
    optimizer = CantanteOptimizer(mas, task(), Silent(), Scripted, inner_steps=1)

    optimizer.propose()

    assert optimizer.unattributed > 0
    assert all(not math.isnan(value) for values in optimizer.credit.values() for value in values)
