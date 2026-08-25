"""The signed-off public surface, executed exactly as written.

This mirrors the quickstart line for line. If it needs editing to keep passing,
the public surface changed and the README changed with it.
"""

from __future__ import annotations

from pathlib import Path

from xmas import MAS, Agent

from .conftest import MockLLM


def test_the_quickstart_runs_as_written(tmp_path: Path) -> None:
    llm = MockLLM(lambda prompt: "391" if "Verify" in prompt else "17 * 23 = 391")

    solver = Agent("solver", prompt="Solve step by step.", output="draft")
    checker = Agent(
        "checker",
        prompt="Verify and answer.",
        inputs=["query", "draft"],
        output="prediction",
    )

    mas = MAS([solver, checker], edges=[("solver", "checker")], llm=llm)

    answer = mas("What is 17 * 23?")  # __call__ runs it
    trace = mas.last_trace  # full per-agent record of that call

    assert answer == "391"
    assert trace is not None
    assert trace.agents == ("solver", "checker")
    assert trace["solver"][0].outputs == {"draft": "17 * 23 = 391"}

    path = tmp_path / "gsm8k.yaml"
    mas.to_yaml(path)  # serialization as methods
    mas2 = MAS.from_yaml(path, llm=llm)

    assert mas2.to_dict() == mas.to_dict()
    assert mas2("What is 17 * 23?") == "391"

    new = mas.copy(prompts={"solver": "Solve it carefully."})  # optimizers derive candidates this way

    assert new.agents["solver"].prompt == "Solve it carefully."
    assert mas.agents["solver"].prompt == "Solve step by step."


def test_agents_are_addressable_and_mutable_through_the_mas() -> None:
    llm = MockLLM()
    mas = MAS([Agent("solver", prompt="Solve.", output="prediction")], llm=llm)

    mas.agents["solver"].prompt = "Solve, but better."
    mas("q")

    assert mas.agents["solver"].prompt == "Solve, but better."
    assert llm.calls == ["Solve, but better."]


def test_no_langgraph_type_appears_in_the_public_surface() -> None:
    """The engine is an implementation detail, and the signatures must say so."""
    import inspect

    import xmas

    for name in xmas.__all__:
        member = getattr(xmas, name)
        if not inspect.isclass(member) and not inspect.isfunction(member):
            continue
        for _, method in inspect.getmembers(member, predicate=inspect.isfunction):
            annotations = getattr(method, "__annotations__", {})
            rendered = " ".join(str(value) for value in annotations.values())
            assert "langgraph" not in rendered.lower(), f"{member.__name__}.{method.__name__} leaks langgraph"
