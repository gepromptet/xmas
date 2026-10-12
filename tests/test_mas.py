"""Behaviour of the MAS: running, tracing, copying and serializing."""

from __future__ import annotations

from pathlib import Path

import pytest

from xmas import MAS, Agent
from xmas.graph import END, START

from .conftest import MockLLM


def build_pair(llm: MockLLM) -> MAS:
    """A two-agent solver and checker."""
    solver = Agent("solver", prompt="Solve step by step.", output="draft")
    checker = Agent("checker", prompt="Verify and answer.", inputs=["query", "draft"], output="prediction")
    return MAS([solver, checker], edges=[("solver", "checker")], llm=llm)


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


def test_ambiguous_output_is_rejected_at_construction(mock_llm: MockLLM) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        MAS(
            [
                Agent("a", prompt="p", output="x"),
                Agent("b", prompt="p", output="y"),
            ],
            llm=mock_llm,
        )
