"""End-to-end tests against Tom's gateway.

Every test here is env-gated and skips cleanly without ``XMAS_BASE_URL`` and
``XMAS_API_KEY``, which is how CI stays green with no credential. They are kept
deliberately tiny and pinned to the 8B tier: this is a metered gateway, and a
test suite is a bad reason to spend a budget.
"""

from __future__ import annotations

import pandas as pd
import pytest
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.utils.prompt import Prompt

from xmas import APILLM, MAS, Agent
from xmas.optimization import (
    CantanteOptimizer,
    ContrastiveAttributor,
    EvolutionaryTopologyOptimizer,
    GraphUpdater,
    LLMPromptSynthesizer,
    PrefixSearchOptimizer,
    TopologyBudget,
    UniformAttributor,
)
from xmas.tasks.gsm8k import GSM8KTask
from xmas.trace import Trace

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


# --------------------------------------------------------------------- WP4 loops

TINY = SLICE.head(2)
"""Two problems. Every WP4 live test runs on this and nothing larger."""

GATEWAY_CONCURRENCY = 4
"""What the gateway allows at once. promptolution defaults to 32.

Exceeding it is answered with a 429, which the client then retries around, so the
run still finishes but only after provoking the limiter and backing off. Staying
under it is the difference between respecting a documented limit and discovering
it repeatedly.
"""


def throttled(max_tokens: int = 512) -> APILLM:
    """A gateway model that stays inside the concurrency the gateway allows."""
    return APILLM(model=MODEL, max_tokens=max_tokens, max_concurrent_calls=GATEWAY_CONCURRENCY)


@requires_gateway
def test_attribution_reads_a_live_rollout(live_llm: APILLM) -> None:
    """One contrastive judgement over two real rollouts of one problem.

    The smallest thing that proves the meta-model can answer the attribution
    prompt as specified. Three 8B calls: two rollouts and one judgement.
    """
    mas = MAS(
        [
            Agent("solver", prompt="Solve: {query}\nFinal number only.", output="draft"),
            Agent(
                "checker", prompt="Reply with only the final number.\n\n{draft}", output="prediction", inputs=["draft"]
            ),
        ],
        edges=[("solver", "checker")],
        llm=live_llm,
    )
    task = GSM8KTask(TINY.head(1))
    variants = [mas, mas.copy(prompts={"solver": "Work step by step, then give the final number.\n\n{query}"})]

    rollouts: list[Trace] = []
    rewards: list[float] = []
    for variant in variants:
        result = task.evaluate(Prompt(variant.encode()), predictor=variant)
        rollouts.extend(variant.last_traces)
        rewards.extend(float(value) for value in result.scores[0])

    attributor = ContrastiveAttributor(throttled(1024), group_size=2)
    credit = attributor.attribute(rollouts, rewards)

    assert set(credit) == {"solver", "checker"}
    assert attributor.failures == 0, "the 8B model did not answer the attribution prompt as specified"
    assert all(-1.0 <= value <= 1.0 for values in credit.values() for value in values)


@requires_gateway
def test_the_cantante_loop_runs_live(live_llm: APILLM) -> None:
    """One attribution-guided round over two problems, on the uniform attributor.

    Uniform rather than contrastive on purpose: this test is about the loop
    reaching a real model, and a judging call per round would double the spend
    for something the test above already covers.
    """
    mas = MAS(
        [
            Agent("solver", prompt="Solve: {query}\nShow your working.", output="draft"),
            Agent(
                "checker", prompt="Reply with only the final number.\n\n{draft}", output="prediction", inputs=["draft"]
            ),
        ],
        edges=[("solver", "checker")],
        llm=live_llm,
    )
    optimizer = CantanteOptimizer(
        mas,
        GSM8KTask(TINY),
        UniformAttributor(),
        EvoPromptGA,
        inner_steps=1,
        initial_prompts={
            "solver": ["Solve: {query}\nShow your working.", "Work step by step.\n\n{query}"],
            "checker": ["Reply with only the final number.\n\n{draft}", "State just the number.\n\n{draft}"],
        },
        optimizer_kwargs={"meta_llm": APILLM(model=MODEL, max_tokens=512)},
    )

    best = optimizer.optimize(n_steps=1)

    assert optimizer.rounds_served >= 1
    assert len(optimizer.history) == 1
    assert 0.0 <= optimizer.history[0] <= 1.0
    assert set(best.agents) == {"solver", "checker"}


@requires_gateway
def test_topology_search_runs_live(live_llm: APILLM) -> None:
    """One topology round over two problems, with a real model writing the prompts.

    This is the only live path where the meta-model writes prompts rather than
    judging, so it is where an unusable rewrite would show up.
    """
    mas = MAS(
        [
            Agent("solver", prompt="Solve: {query}\nShow your working.", output="solver_out"),
            Agent(
                "checker",
                prompt="Reply with only the final number.\n\n{solver_out}",
                output="prediction",
                inputs=["solver_out"],
            ),
        ],
        edges=[("solver", "checker")],
        llm=live_llm,
    )
    synthesizer = LLMPromptSynthesizer(throttled())
    updater = GraphUpdater(synthesizer, TopologyBudget(min_agents=2, max_agents=3, max_edges=6), seed=7)
    optimizer = EvolutionaryTopologyOptimizer(mas, GSM8KTask(TINY), updater, mutations=2, recombinations=0, seed=1)

    best = optimizer.optimize(n_steps=1)

    assert len(optimizer.history) == 1
    assert 0.0 <= optimizer.history[0] <= 1.0
    # Whatever it settled on has to be a system that ran, not just one that built.
    assert best.run(str(TINY["question"].iloc[0])).error is None
