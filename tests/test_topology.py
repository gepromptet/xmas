"""Topology search: the operators, the rewiring, and the loop that drives them.

The property that matters most is not that the operators produce interesting
graphs but that they never produce a broken one. A candidate that fails to build
costs an evaluation slot; a candidate that builds but carries stale inputs is
worse, because it scores like a real proposal while changing nothing.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from xmas import MAS, Agent, MASTask
from xmas.graph import END, START, back_edges
from xmas.optimization.evolution import EvolutionaryTopologyOptimizer
from xmas.optimization.graph_updater import GraphUpdater
from xmas.optimization.synthesis import LLMPromptSynthesizer, MechanicalSynthesizer, Rewrite
from xmas.optimization.topology import (
    TopologyBudget,
    crossover,
    cycles,
    is_valid,
    mas_edges,
    mutate,
    terminals,
    within_budget,
)

from .conftest import MockLLM

SEED_NAMES = ["solver", "checker"]
SEED_EDGES = [("solver", "checker")]


def pipeline(llm: MockLLM | None = None) -> MAS:
    """The two-agent starting system every test in here mutates."""
    return MAS(
        agents=[
            Agent("solver", "Solve {query}", output="solver_out"),
            Agent("checker", "Check {solver_out}", output="checker_out", inputs=["solver_out"]),
        ],
        edges=list(SEED_EDGES),
        llm=llm if llm is not None else MockLLM(lambda prompt: f"[{prompt[:20]}]"),
    )


def updater(**kwargs: object) -> GraphUpdater:
    """A deterministic updater, for tests that must reproduce."""
    budget = kwargs.pop("budget", TopologyBudget(min_agents=2, max_agents=4, max_edges=8))
    return GraphUpdater(MechanicalSynthesizer(), budget, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------- budgets


def test_a_budget_no_topology_could_satisfy_is_refused() -> None:
    with pytest.raises(ValueError, match="max_edges"):
        TopologyBudget(min_agents=2, max_agents=6, max_edges=3)
    with pytest.raises(ValueError, match="min_agents"):
        TopologyBudget(min_agents=4, max_agents=2)


def test_a_second_terminal_is_outside_every_budget() -> None:
    # Two agents feeding END make the system's output ambiguous, so this is a
    # bound the operators may never cross, not a preference.
    budget = TopologyBudget(min_agents=2, max_agents=4, max_edges=8)
    assert terminals(["a", "b", "c"], [("a", "b"), ("a", "c")]) == ["b", "c"]
    assert not within_budget(["a", "b", "c"], [("a", "b"), ("a", "c")], budget)


# ----------------------------------------------------------------- operators


@pytest.mark.parametrize("max_cycles", [0, 2])
def test_every_mutation_and_crossover_builds_a_runnable_system(max_cycles: int) -> None:
    budget = TopologyBudget(min_agents=2, max_agents=5, max_edges=10, max_cycles=max_cycles)
    rng = random.Random(11)
    population = [(list(SEED_NAMES), list(SEED_EDGES))]
    cyclic = 0

    for _ in range(150):
        parent = rng.choice(population)
        names, edges = mutate(*parent, budget, rng, rng.randint(1, 3), "solver")
        assert is_valid(names, edges, budget, "solver"), (names, edges)
        _build(names, edges).run("q")
        cyclic += 1 if cycles(names, edges) else 0

        population.append((names, edges))
        population = population[-10:]

        child = crossover(rng.choice(population), rng.choice(population), budget, rng, "solver")
        assert is_valid(*child, budget, "solver"), child
        _build(*child).run("q")

    assert (cyclic > 0) is (max_cycles > 0), "the cycle budget decides whether cycles appear at all"


def _build(names: list[str], edges: list[tuple[str, str]]) -> MAS:
    """Construct a system from a bare topology, which is the validity predicate."""
    agents = [Agent(name, f"{name} {{query}}", output=f"{name}_out") for name in names]
    return MAS(agents=agents, edges=mas_edges(names, edges, "solver"), llm=MockLLM(lambda prompt: "ok"))


def test_a_retry_loop_gets_both_its_entry_and_its_exit_named() -> None:
    # solver -> checker -> solver is the canonical retry loop and the shape
    # topology search exists to find. Every agent has a predecessor, so nothing
    # would be wired from START; every agent has a successor, so nothing would be
    # wired to END. Left alone it is both unreachable and unfinishable.
    names, edges = ["solver", "checker"], [("solver", "checker"), ("checker", "solver")]

    wiring = mas_edges(names, edges, "solver")

    assert wiring[0] == (START, "solver")
    # The exit is the agent that closes the loop, which is also the agent the
    # engine hands the loop-or-continue decision to.
    assert (("checker", END)) in wiring

    mas = _build(names, edges)
    assert back_edges(mas.graph) == {("checker", "solver")}
    assert mas.run("q").error is None


# ------------------------------------------------------------------ rewiring


def test_a_new_predecessor_reaches_the_agent_that_gained_it() -> None:
    # The failure this guards against: the edge is added, the graph looks
    # different, and the agent goes on rendering exactly the prompt it did before.
    mas = pipeline()
    rebuilt = updater().rebuild(mas, ["solver", "helper", "checker"], [("solver", "checker"), ("helper", "checker")])

    assert set(rebuilt.agents["checker"].inputs) == {"solver_out", "helper_out"}
    trace = rebuilt.run("2+2?")
    rendered = trace["checker"][0].rendered_prompt
    assert "{helper_out}" not in rendered  # substituted, not left standing as a placeholder
    assert trace["helper"][0].raw_response in rendered


def test_an_agent_whose_inputs_did_not_change_keeps_its_prompt() -> None:
    mas = pipeline()
    rebuilt = updater().rebuild(mas, ["solver", "checker"], [("solver", "checker")])

    assert rebuilt.agents["checker"].prompt == mas.agents["checker"].prompt
    assert rebuilt.agents["solver"].prompt == mas.agents["solver"].prompt


def test_candidates_are_distinct_and_every_one_of_them_runs() -> None:
    mas = pipeline()
    candidates = updater(seed=3).propose(mas, {}, n=4)

    shapes = {(tuple(candidate.agents), tuple(sorted(candidate.graph.edges))) for candidate in candidates}
    assert len(shapes) == len(candidates)
    for candidate in candidates:
        assert candidate.run("q").error is None


def test_credit_steers_the_change_toward_the_agent_that_earned_least() -> None:
    mas = pipeline()
    proposer = updater(seed=5)
    candidates = proposer.propose(mas, {"solver": [0.9], "checker": [-0.5]}, n=3)

    before = proposer.edges_of(mas)
    touched = [
        candidate
        for candidate in candidates
        if any("checker" in edge for edge in set(before) ^ set(proposer.edges_of(candidate)))
    ]
    assert touched, "the least-credited agent should be where the first proposals land"


# ------------------------------------------------------------------ synthesis


def test_the_meta_model_writes_the_prompt_when_it_answers_as_asked() -> None:
    meta = MockLLM(lambda prompt: "<prompt>Reconcile {solver_out} and {helper_out} into one answer.</prompt>")
    written = LLMPromptSynthesizer(meta).rewrite(
        [Rewrite(agent="checker", prompt="Check {solver_out}", output="out", inputs=["solver_out", "helper_out"])]
    )

    assert written == ["Reconcile {solver_out} and {helper_out} into one answer."]


def test_a_rewrite_that_drops_a_variable_is_rejected() -> None:
    # A prompt missing one of its inputs renders without it, which would make the
    # topology change inert while looking like it had been handled.
    meta = MockLLM(lambda prompt: "<prompt>Check {solver_out} carefully.</prompt>")
    synthesizer = LLMPromptSynthesizer(meta)

    written = synthesizer.rewrite(
        [Rewrite(agent="checker", prompt="Check {solver_out}", output="out", inputs=["solver_out", "helper_out"])]
    )

    assert synthesizer.fallbacks == 1
    assert "{helper_out}" in written[0]


def test_an_unusable_response_falls_back_rather_than_failing_the_search() -> None:
    synthesizer = LLMPromptSynthesizer(MockLLM(lambda prompt: "sure thing"))
    written = synthesizer.rewrite([Rewrite(agent="a", prompt="Do {x}", output="o", inputs=["x", "y"])])

    assert synthesizer.fallbacks == 1
    assert "{y}" in written[0]


# ----------------------------------------------------------------- the loop


class ShapeTask(MASTask):
    """Rewards a system for having the shape the search is supposed to find."""

    def _evaluate(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        return np.asarray([1.0 if "helper" in str(pred) else 0.0 for pred in preds])


def test_the_search_improves_the_system_and_reports_what_it_accepted() -> None:
    frame = pd.DataFrame({"x": ["q1", "q2"], "y": ["a", "a"]})
    task = ShapeTask(df=frame, x_column="x", y_column="y", task_description="grow a helper")
    # The terminal agent echoes every variable it read, so a system only scores
    # once an agent named `agent_*` exists and feeds it.
    mas = MAS(
        agents=[
            Agent("solver", "Solve {query}", output="solver_out"),
            Agent("checker", "Check {solver_out}", output="checker_out", inputs=["solver_out"]),
        ],
        edges=list(SEED_EDGES),
        llm=MockLLM(lambda prompt: prompt.replace("agent_", "helper_")),
    )
    optimizer = EvolutionaryTopologyOptimizer(mas, task, updater(seed=3), mutations=4, seed=1)

    assert optimizer.evaluate([mas]) == [0.0]
    best = optimizer.optimize(n_steps=2)

    assert optimizer.history[-1] == 1.0
    assert len(best.agents) > 2
    assert any(entry != "no change" for entry in optimizer.accepted)


def test_the_population_keeps_the_best_candidates_as_future_parents() -> None:
    frame = pd.DataFrame({"x": ["q1"], "y": ["a"]})
    task = ShapeTask(df=frame, x_column="x", y_column="y")
    optimizer = EvolutionaryTopologyOptimizer(pipeline(), task, updater(seed=3), mutations=3, population_size=2, seed=1)

    optimizer.optimize(n_steps=1)

    assert len(optimizer.population) == 2
    for member in optimizer.population:
        assert nx.is_directed_acyclic_graph(member.graph)
