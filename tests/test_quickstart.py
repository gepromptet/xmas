"""The signed-off public surface, executed exactly as written.

This mirrors the quickstart line for line. If it needs editing to keep passing,
the public surface changed and the README changed with it.
"""

from __future__ import annotations

from pathlib import Path

from xmas import MAS, Agent

from .conftest import MockLLM


def test_the_quickstart_runs_as_written(tmp_path: Path) -> None:
    llm = MockLLM(lambda prompt: "Rayleigh scattering." if "as your own" in prompt else "Short waves scatter more.")

    intern = Agent("intern", prompt="Answer this question, showing your working: {query}", output="draft")
    boss = Agent(
        "boss",
        prompt="Question: {query}\nYour intern's draft: {draft}\nGive the final answer in one sentence, as your own.",
        inputs=["query", "draft"],
        output="prediction",
    )

    mas = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)

    assert mas("Why is the sky blue?") == "Rayleigh scattering."
    trace = mas.last_trace
    assert trace is not None
    assert trace.agents == ("intern", "boss")
    assert trace["intern"][0].outputs == {"draft": "Short waves scatter more."}
    assert "Your intern's draft: Short waves scatter more." in llm.calls[-1]

    path = tmp_path / "team.yaml"
    mas.to_yaml(path)
    again = MAS.from_yaml(path, llm=llm)
    assert again.to_dict() == mas.to_dict()

    fairer = mas.copy(prompts={"boss": mas.agents["boss"].prompt + " Credit the intern."})
    assert fairer.agents["boss"].prompt.endswith(" Credit the intern.")
    assert not mas.agents["boss"].prompt.endswith(" Credit the intern.")


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
