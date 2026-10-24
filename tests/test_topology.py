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

from xmas import MAS, Agent, Budget, Edge, MASTask, Step
from xmas.edges import EdgeKind
from xmas.graph import END, START, back_edges
from xmas.health import HealthSettings
from xmas.optimization.evolution import EvolutionaryTopologyOptimizer
from xmas.optimization.graph_updater import GraphUpdater
from xmas.optimization.synthesis import LLMPromptSynthesizer, MechanicalSynthesizer, Rewrite
from xmas.optimization.topology import (
    TopologyBudget,
    add_agent,
    crossover,
    cycles,
    is_valid,
    mas_edges,
    mutate,
    terminals,
    within_budget,
)
from xmas.optimization.updaters import PromptUpdater

from .conftest import MockLLM, count_words

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
        build(names, edges).run("q")
        cyclic += 1 if cycles(names, edges) else 0

        population.append((names, edges))
        population = population[-10:]

        child = crossover(rng.choice(population), rng.choice(population), budget, rng, "solver")
        assert is_valid(*child, budget, "solver"), child
        build(*child).run("q")

    assert (cyclic > 0) is (max_cycles > 0), "the cycle budget decides whether cycles appear at all"


def build(names: list[str], edges: list[tuple[str, str]]) -> MAS:
    """Construct a system from a bare topology, which is the validity predicate."""
    agents = [Agent(name, f"{name} {{query}}", output=f"{name}_out") for name in names]
    return MAS(agents=agents, edges=mas_edges(names, edges, "solver"), llm=MockLLM(lambda prompt: "ok"))


def test_a_one_agent_system_can_grow() -> None:
    # Defect reproduction. `add_agent` used to pick its predecessor from every
    # agent including the terminal, which on a one-agent system closed a cycle
    # between the two. Under an acyclic budget every such mutation was discarded,
    # so the search silently could not grow past its first node. A live run is
    # what surfaced it: three steps of topology search that accepted nothing.
    budget = TopologyBudget(min_agents=1, max_agents=3, max_edges=6)
    rng = random.Random(0)

    assert add_agent(["solver"], [], rng) == (["solver", "agent_1"], [("agent_1", "solver")])

    grown = sum(1 for _ in range(200) if len(mutate(["solver"], [], budget, rng, 1, "solver")[0]) > 1)
    assert grown > 0, "a one-agent system must be able to gain an agent"


@pytest.mark.parametrize(
    ("names", "edges"),
    [
        (["solver"], []),
        (["a", "b"], [("a", "b")]),
        (["a", "b", "c"], [("a", "b"), ("b", "c")]),
    ],
)
def test_a_new_agent_is_never_wired_into_a_cycle_with_the_terminal(
    names: list[str], edges: list[tuple[str, str]]
) -> None:
    rng = random.Random(1)
    grown_names, grown_edges = add_agent(names, edges, rng)
    new = grown_names[-1]
    terminal = terminals(names, edges)[0]

    assert (new, terminal) in grown_edges
    assert (terminal, new) not in grown_edges


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

    mas = build(names, edges)
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


def test_rewired_agents_read_what_their_predecessors_actually_write() -> None:
    # The defect: an agent's inputs were derived from its predecessors' *names*,
    # so a system whose outputs are named for what they hold rather than for the
    # agent that wrote them got rewired to variables nothing writes. Every fixture
    # in this file happens to follow the `{name}_out` convention, which is exactly
    # why that went unseen, so this one deliberately does not.
    mas = MAS(
        agents=[
            Agent("drafter", "Draft an answer to {query}", output="draft"),
            Agent("reviewer", "Review {draft}", output="review", inputs=["draft"]),
        ],
        edges=[("drafter", "reviewer")],
        llm=MockLLM(lambda prompt: f"[{prompt[:20]}]"),
    )

    candidates = updater(seed=1).propose(mas, {}, n=6)

    assert candidates
    for candidate in candidates:
        written = {agent.output for agent in candidate.agents.values()} | {"query"}
        read = {name for agent in candidate.agents.values() for name in agent.inputs}
        assert read <= written, f"{sorted(read - written)} is read but nothing writes it"


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

    def reward(self, xs: list[str], ys: list[str], preds: list[str]) -> np.ndarray:
        """One if a helper's output reached the prediction."""
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


# ----------------------------------------------------------------- version 2


def judged(*, branches: bool = False) -> MAS:
    """Build planner -> writer -> count -> judge: a step, a frozen json judge, health and every runtime field set.

    With ``branches``, the judge decides between ``publish`` and ``apology``,
    which both feed ``final``.
    """
    agents = [
        Agent("judge", "Judge {draft}: {counts}", output="verdict", inputs=["draft", "counts"], output_type="json"),
        Agent("planner", "Plan an answer to {query}", output="plan"),
        Agent("writer", "Write about {query} along {plan}", output="draft", inputs=["query", "plan"], model="strong"),
    ]
    edges: list[tuple[str, str] | Edge] = [("planner", "writer"), ("writer", "count"), ("count", "judge")]
    state = {"draft": {"merge": "last"}}
    if branches:
        agents += [
            Agent("publish", "Publish {draft}", output="published", inputs=["draft"]),
            Agent("apology", "Apologise for {query}", output="published"),
            Agent("final", "Finish with {published}", output="answer", inputs=["published"]),
        ]
        edges += [
            Edge("judge", "publish", EdgeKind.WHEN, when="verdict.fits"),
            Edge("judge", "apology", EdgeKind.ELSE),
            ("publish", "final"),
            ("apology", "final"),
        ]
        state["published"] = {"merge": "last"}

    def answer(prompt: str) -> str:
        return '{"fits": true}' if prompt.startswith("Judge") else "A short draft."

    return MAS(
        agents=agents,
        edges=edges,
        steps=[Step("count", uses="count_words")],
        llm=MockLLM(answer),
        llms={"strong": MockLLM(answer)},
        max_steps=17,
        default_model="base-model",
        tools=[count_words],
        max_tool_rounds=2,
        budget=Budget(max_calls=50),
        inputs={"query": "text"},
        state=state,
        frozen=("judge",),
        health=HealthSettings(min_samples=2),
    )


def test_graph_updater_keeps_kinds_frozen_and_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rebuild must not drop a field of the system, nor turn a step into an
    # agent: steps stay steps, every field reaches the candidate, and a
    # candidate the check refuses (say the writer was removed, so nothing
    # writes the draft) is dropped rather than raised.
    budget = TopologyBudget(min_agents=2, max_agents=7, max_edges=12)
    mas = judged()
    mas.max_visits = 1  # not the default, so a rebuild that drops it shows
    candidates = updater(seed=2, budget=budget).propose(mas, {}, n=6)

    assert candidates
    for candidate in candidates:
        assert candidate.steps == mas.steps
        assert isinstance(candidate.steps["count"], Step)
        assert "count" not in candidate.agents
        assert candidate.agents["judge"] == mas.agents["judge"]
        kept = ("version", "inputs", "merges", "boxes", "frozen", "health")
        assert [getattr(candidate, name) for name in kept] == [getattr(mas, name) for name in kept]
        runtime = ("llm", "llms", "max_steps", "default_model", "tools", "max_tool_rounds", "max_visits", "budget")
        assert [getattr(candidate, name) for name in runtime] == [getattr(mas, name) for name in runtime]
        assert candidate.run("cats").error is None

    # Conditional edges and their endpoints stay exactly as they were, in order.
    branched = judged(branches=True)
    conditions = [edge for edge in branched.edges if edge.kind is not EdgeKind.PLAIN]
    proposer = updater(seed=4, budget=budget)
    candidates = proposer.propose(branched, {}, n=6) + proposer.recombine(branched, branched, n=2)

    assert candidates
    for candidate in candidates:
        assert [edge for edge in candidate.edges if edge.kind is not EdgeKind.PLAIN] == conditions
        for name in ("judge", "publish", "apology"):
            assert candidate.agents[name] == branched.agents[name]
        assert {("count", "judge"), ("publish", "final"), ("apology", "final")} <= set(proposer.edges_of(candidate))
        assert candidate.steps == branched.steps

    # Credit never steers a topology candidate to the frozen judge, which no candidate may rewire.
    credit = {"judge": [-1.0], "writer": [0.5], "planner": [0.2]}
    assert updater(seed=2, budget=budget).weakest_agent(mas, credit) == "planner"
    assert updater(seed=2, budget=budget).weakest_agent(mas, {"judge": [-1.0]}) is None

    # A prompt optimizer never picks the frozen judge, not even when it earned least.
    prompts = PromptUpdater(object, None)  # type: ignore[arg-type]
    assert prompts.target_agent(mas, {}) == "planner"
    assert prompts.target_agent(mas, {"judge": [-1.0], "writer": [0.5], "planner": [0.2]}) == "planner"
    monkeypatch.setattr(mas, "frozen", frozenset(mas.agents))
    with pytest.raises(ValueError, match="frozen"):
        prompts.target_agent(mas, {})
