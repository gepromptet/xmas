"""The signed-off public surface, executed exactly as written.

This mirrors the quickstart line for line. If it needs editing to keep passing,
the public surface changed and the README changed with it.
"""

from __future__ import annotations

from pathlib import Path

from xmas import MAS, Agent

from .conftest import MockLLM


def test_the_quickstart_runs_as_written(tmp_path: Path) -> None:
    llm = MockLLM(lambda prompt: "A rocket." if "You are Santa" in prompt else "A rocket and a pony.")

    elf = Agent("elf", prompt="You are a workshop elf. Suggest one gift for this wish list: {query}", output="gift")
    santa = Agent(
        "santa",
        prompt="You are Santa. Wish list: {query}\nYour elf suggests: {gift}\n"
        "Give the final gift in one line. No ponies, the sleigh is full.",
        inputs=["query", "gift"],
        output="prediction",
    )

    workshop = MAS([elf, santa], edges=[("elf", "santa")], llm=llm)

    assert workshop("A pony, a rocket and a PhD") == "A rocket."
    trace = workshop.last_trace
    assert trace is not None
    assert trace.agents == ("elf", "santa")
    assert trace["elf"][0].outputs == {"gift": "A rocket and a pony."}
    assert "Your elf suggests: A rocket and a pony." in llm.calls[-1]

    path = tmp_path / "workshop.yaml"
    workshop.to_yaml(path)
    again = MAS.from_yaml(path, llm=llm)
    assert again.to_dict() == workshop.to_dict()

    nicer = workshop.copy(prompts={"santa": workshop.agents["santa"].prompt + " Be jolly."})
    assert nicer.agents["santa"].prompt.endswith(" Be jolly.")
    assert not workshop.agents["santa"].prompt.endswith(" Be jolly.")


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
