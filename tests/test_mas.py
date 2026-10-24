"""Behaviour of the MAS: running, tracing, copying and serializing."""

from __future__ import annotations

import re
import time
from pathlib import Path

import openai
import pytest

from xmas import MAS, Agent
from xmas.budget import Budget
from xmas.errors import ErrorKind, RunStatus
from xmas.graph import END, START
from xmas.optimization.attribution import ContrastiveAttributor
from xmas.optimization.attribution import prompt as attribution_prompt

from .conftest import MockLLM, status_error


def build_pair(llm: MockLLM) -> MAS:
    """A two-agent solver and checker."""
    solver = Agent("solver", prompt="Solve step by step.", output="draft")
    checker = Agent("checker", prompt="Verify and answer.", inputs=["query", "draft"], output="prediction")
    return MAS([solver, checker], edges=[("solver", "checker")], llm=llm)


def solve_and_check(llm: MockLLM, budget: Budget | None = None) -> MAS:
    """A solver and a checker whose rendered prompts name the query, so every call is told apart."""
    solver = Agent("solver", prompt="Solve {query}", output="draft")
    checker = Agent("checker", prompt="Check {draft}", inputs=["draft"], output="prediction")
    return MAS([solver, checker], edges=[("solver", "checker")], llm=llm, budget=budget)


def test_quickstart_runs_and_returns_the_terminal_output(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    answer = mas("What is 17 * 23?")
    assert answer.startswith("echo:")
    assert len(mock_llm.calls) == 2


def test_call_records_a_trace_with_one_record_per_agent(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    mas("What is 17 * 23?")
    trace = mas.last_trace

    assert trace is not None
    assert trace.query == "What is 17 * 23?"
    assert trace.agents == ("solver", "checker")
    assert [record.step for record in trace.records] == [0, 1]
    assert trace.usage.total_tokens > 0


def test_agents_read_upstream_outputs(mock_llm: MockLLM) -> None:
    solver = Agent("solver", prompt="Solve {query}", output="draft")
    checker = Agent("checker", prompt="Check draft: {draft}", inputs=["draft"], output="prediction")
    mas = MAS([solver, checker], edges=[("solver", "checker")], llm=mock_llm)
    mas("2+2")

    trace = mas.last_trace
    assert trace is not None
    (checker_record,) = trace["checker"]
    assert checker_record.inputs["draft"] == "echo:Solve 2+2"
    assert "echo:Solve 2+2" in checker_record.rendered_prompt


def test_run_returns_a_trace_without_touching_shared_state(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    trace = mas.run("q")
    assert trace.output
    assert mas.last_trace is None, "run() must not set the single-call convenience slot"


def test_run_batch_assigns_traces_in_order(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    traces = mas.run_batch(["a", "b", "c"])
    assert [trace.query for trace in traces] == ["a", "b", "c"]
    assert [trace.query for trace in mas.last_traces] == ["a", "b", "c"]


def test_implicit_wiring_connects_the_sentinels(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    assert (START, "solver") in mas.graph.edges
    assert ("checker", END) in mas.graph.edges
    assert mas.terminal == "checker"


def test_copy_overrides_prompts_without_mutating_the_original(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    derived = mas.copy(prompts={"solver": "New instruction."})

    assert derived.agents["solver"].prompt == "New instruction."
    assert mas.agents["solver"].prompt == "Solve step by step."
    assert derived.agents["checker"].prompt == mas.agents["checker"].prompt


def test_copy_rejects_unknown_agents(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    with pytest.raises(ValueError, match="unknown agents"):
        mas.copy(prompts={"nope": "x"})


def test_agents_are_mutable_through_the_mas(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    mas.agents["solver"].prompt = "Changed."
    mas("q")
    assert any(call == "Changed." for call in mock_llm.calls)


def test_dict_round_trip_is_exact(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    spec = mas.to_dict()
    rebuilt = MAS.from_dict(dict(spec), llm=mock_llm)
    assert rebuilt.to_dict() == spec


def test_yaml_round_trip_is_exact_and_plain_data(tmp_path: Path, mock_llm: MockLLM) -> None:
    import yaml

    mas = build_pair(mock_llm)
    path = tmp_path / "mas.yaml"
    mas.to_yaml(path)

    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict)
    assert set(raw) == {"version", "agents", "edges", "default_model"}

    rebuilt = MAS.from_yaml(path, llm=mock_llm)
    assert rebuilt.to_dict() == mas.to_dict()
    assert rebuilt("q") == mas("q")


def test_compilation_is_cached_and_survives_a_prompt_edit(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    first = mas.compiled
    assert mas.compiled is first, "compilation must be cached"

    mas.agents["solver"].prompt = "A different prompt."
    assert mas.compiled is first, "a prompt edit is not a structural change"


def test_compilation_is_invalidated_by_a_structural_change(mock_llm: MockLLM) -> None:
    mas = build_pair(mock_llm)
    first = mas.compiled

    mas.agents["solver"].output = "renamed"
    mas.agents["checker"].inputs = ["query", "renamed"]
    assert mas.compiled is not first, "changing wiring must recompile"


def test_a_failed_run_becomes_data_not_an_exception() -> None:
    def explode(prompt: str) -> str:
        raise RuntimeError("upstream is down")

    mas = MAS([Agent("only", prompt="p", output="prediction")], llm=MockLLM(explode))
    trace = mas.run("q")
    assert trace.failed
    assert trace.error is not None
    assert "upstream is down" in trace.error

    # the checker's key is out of money: the solver's finished work stays in the trace, and the kind says why
    def broke_checker(prompt: str) -> str:
        if prompt.startswith("Check"):
            raise status_error(openai.PermissionDeniedError, 403, "Key limit exceeded")
        return f"echo:{prompt}"

    llm = MockLLM(broke_checker)
    trace = solve_and_check(llm).run("2+2")
    assert llm.calls == ["Solve 2+2", "Check echo:Solve 2+2"]
    assert [(record.agent, record.step) for record in trace.records] == [("solver", 0)]
    assert trace.status is RunStatus.FAILED
    assert trace.error_kind is ErrorKind.MONEY
    assert trace.error is not None
    assert trace.error.startswith("PermissionDeniedError: ")
    assert trace.output == ""
    assert all(record.latency_s is not None for record in trace.records)

    # the judge is still told the terminal agent's key, not the last key the failed rollout's records hold
    ok = solve_and_check(MockLLM()).run("2+2")
    template, names = ContrastiveAttributor.ATTRIBUTION_PROMPT, ["solver", "checker"]
    judged = attribution_prompt(template, {"executions": []}, [trace, ok], names)
    assert 'under the key "prediction"' in judged


@pytest.mark.parametrize(
    ("budget", "agents", "calls", "error"),
    [
        (Budget(max_cost_usd=0.0), [], 0, re.escape("BudgetExceeded: cost_usd 0 >= max 0 before call 1")),
        (Budget(max_wall_s=0.0), [], 0, r"BudgetExceeded: wall_s \S+ >= max 0 before call 1"),
        (Budget(max_calls=1), ["solver"], 1, re.escape("BudgetExceeded: calls 1 >= max 1 before call 2")),
        (
            Budget(max_cost_usd=1.0),  # MockLLM reports no cost and has no price
            ["solver"],
            1,
            re.escape(
                "BudgetExceeded: cost_unknown: 1 call reported no cost and has no price,"
                " so max_cost_usd 1 cannot be checked before call 2"
            ),
        ),
    ],
    ids=["zero-cost", "zero-wall", "one-call", "unknown-cost"],
)
def test_a_budget_ends_the_run_over_budget_and_keeps_records(
    budget: Budget, agents: list[str], calls: int, error: str
) -> None:
    llm = MockLLM()
    trace = solve_and_check(llm, budget).run("2+2")
    assert trace.status is RunStatus.OVER_BUDGET
    assert trace.error_kind is ErrorKind.BUDGET
    assert [record.agent for record in trace.records] == agents
    assert llm.calls == ["Solve 2+2"][:calls]  # the checker is never called
    assert trace.error is not None
    assert re.fullmatch(error, trace.error), trace.error
    assert all(record.latency_s is not None for record in trace.records)


def test_run_batch_pays_each_item_once_when_one_fails() -> None:
    def checker_fails_on_b(prompt: str) -> str:
        if prompt == "Check echo:Solve b":
            raise RuntimeError("b broke")
        return f"echo:{prompt}"

    llm = MockLLM(checker_fails_on_b)
    a, b, c = solve_and_check(llm).run_batch(["a", "b", "c"])

    prompts = [f"{step} {item}" for item in "abc" for step in ("Solve", "Check echo:Solve")]
    assert len(llm.calls) == 6
    assert sorted(llm.calls) == sorted(prompts)  # each prompt exactly once: no rerun of the batch
    assert (a.status, b.status, c.status) == (RunStatus.OK, RunStatus.FAILED, RunStatus.OK)
    assert (a.output, c.output) == ("echo:Check echo:Solve a", "echo:Check echo:Solve c")
    assert a.agents == c.agents == ("solver", "checker")
    assert b.error == "RuntimeError: b broke"
    assert [(record.agent, record.raw_response) for record in b.records] == [("solver", "echo:Solve b")]


def test_a_batch_item_does_not_spend_its_wall_budget_waiting_for_a_worker() -> None:
    def slow(prompt: str) -> str:
        time.sleep(0.1)
        return "ok"

    llm = MockLLM(slow)
    mas = MAS([Agent("solver", "Solve {query}", output="answer")], llm=llm, budget=Budget(max_wall_s=0.08))
    traces = mas.run_batch([str(item) for item in range(40)])  # more items than the batch has workers (at most 32)
    assert [trace.error for trace in traces] == [None] * 40  # each run's clock starts with its run, not the batch
    assert len(llm.calls) == 40


def test_ambiguous_output_is_rejected_at_construction(mock_llm: MockLLM) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        MAS(
            [
                Agent("a", prompt="p", output="x"),
                Agent("b", prompt="p", output="y"),
            ],
            llm=mock_llm,
        )
