"""End-to-end tests against Tom's gateway.

Every test here is env-gated and skips cleanly without ``XMAS_BASE_URL`` and
``XMAS_API_KEY``, which is how CI stays green with no credential. They are kept
deliberately tiny and pinned to the 8B tier: this is a metered gateway, and a
test suite is a bad reason to spend a budget.
"""

from __future__ import annotations

import pandas as pd
import pytest
from promptolution.utils.prompt import Prompt

from xmas import APILLM, MAS, Agent
from xmas.optimization import PrefixSearchOptimizer
from xmas.tasks.gsm8k import GSM8KTask

from .conftest import requires_gateway

MODEL = "llama-3.1-8b"
"""The student-tier model. Keep live tests on it."""

# Three real GSM8K-style problems, inlined so the suite needs no dataset download.
SLICE = pd.DataFrame(
    {
        "question": [
            "Natalia sold clips to 48 friends in April, and then she sold half as many "
            "clips in May. How many clips did Natalia sell altogether in April and May?",
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
            "of babysitting. How much did she earn?",
            "Betty is saving money for a new wallet which costs $100. Betty has only "
            "half of the money she needs. Her parents decided to give her $15 for that "
            "purpose, and her grandparents twice as much as her parents. How much more "
            "money does Betty need to buy the wallet?",
        ],
        "answer": ["#### 72", "#### 10", "#### 5"],
    }
)


@pytest.fixture
def live_llm() -> APILLM:
    """A model pointed at the gateway, configured entirely from the environment."""
    return APILLM(model=MODEL, max_tokens=512)


@requires_gateway
def test_the_quickstart_runs_against_the_gateway(live_llm: APILLM) -> None:
    """Two agents, one query. The smallest real check that the wiring is live."""
    solver = Agent(
        "solver",
        prompt="Solve this step by step, then state the final number.\n\n{query}",
        output="draft",
    )
    checker = Agent(
        "checker",
        prompt="Given the working below, reply with only the final number.\n\n{draft}",
        inputs=["query", "draft"],
        output="prediction",
    )
    mas = MAS([solver, checker], edges=[("solver", "checker")], llm=live_llm)

    answer = mas("What is 17 * 23?")
    trace = mas.last_trace

    assert "391" in answer
    assert trace is not None
    assert trace.agents == ("solver", "checker")
    assert trace.usage.total_tokens > 0


@requires_gateway
def test_a_single_agent_mas_scores_on_a_live_gsm8k_slice(live_llm: APILLM) -> None:
    """The degenerate case, end to end, against a real model."""
    mas = MAS(
        [
            Agent(
                "solver",
                prompt="Solve the problem. End your reply with the final number alone.\n\n{query}",
                output="prediction",
            )
        ],
        llm=live_llm,
    )
    task = GSM8KTask(SLICE)

    result = task.evaluate(Prompt(mas.encode()), predictor=mas)

    assert result.agg_scores.shape == (1,)
    assert 0.0 <= result.agg_scores[0] <= 1.0


@requires_gateway
def test_the_reference_optimizer_runs_a_live_loop(live_llm: APILLM) -> None:
    """One propose-evaluate-select round on a live slice.

    One step over three problems with four candidates: twelve short 8B calls,
    which is the smallest thing that genuinely exercises the loop.
    """
    mas = MAS(
        [Agent("solver", prompt="Solve: {query}\nFinal number only.", output="prediction")],
        llm=live_llm,
    )
    optimizer = PrefixSearchOptimizer(mas, GSM8KTask(SLICE))

    best = optimizer.optimize(n_steps=1)

    assert len(optimizer.history) == 1
    assert 0.0 <= optimizer.history[0] <= 1.0
    assert best.agents["solver"].prompt
