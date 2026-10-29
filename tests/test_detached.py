"""``detached``: an attribute of every node kind that keeps the node away from every optimizer."""

from __future__ import annotations

import pytest

from xmas import MAS, Agent, Code, HumanAction, MASModule
from xmas.optimization.updaters import movable_agents

from .conftest import MockLLM, count_words


def system(*, detach: bool) -> MAS:
    """A writer, a code node counting its words, a module shortening the draft and an approval, each detached or not."""
    inner = MAS([Agent("cutter", "Cut {page}", "short", ["page"])], inputs={"page": "text"}, llm=MockLLM())
    return MAS(
        [
            Agent("writer", "Write {query}", "draft"),
            Agent("judge", "Judge {draft} {counts}", "verdict", ["draft", "counts"], detached=detach),
        ],
        [("writer", "count"), ("writer", "shorten"), ("count", "judge"), ("shorten", "judge"), ("judge", "ok")],
        code_nodes=[Code("count", "count_words", detached=detach)],
        modules=[MASModule("shorten", inner, {"page": "draft"}, {"short": "summary"}, detached=detach)],
        human_actions=[HumanAction("ok", "approve", ["verdict"], "approval", detached=detach)],
        state={"draft": {"merge": "last"}},
        tools=[count_words],
        llm=MockLLM(),
        version=2,
    )


def test_every_node_kind_can_be_detached_and_the_spec_keeps_it() -> None:
    mas = system(detach=True)
    assert mas.detached == frozenset({"judge", "count", "shorten", "ok"})
    spec = mas.to_dict()
    assert [entry.get("detached") for section in ("agents", "code_nodes", "modules", "human_actions")
            for entry in spec[section]] == [None, True, True, True, True]  # fmt: skip
    again = MAS.from_dict(spec, llm=MockLLM(), tools=[count_words])
    assert again.detached == mas.detached
    assert again.to_dict() == spec
    assert {node["name"]: node.get("detached", False) for node in mas.describe()["nodes"]} == {
        "writer": False,
        "count": True,
        "shorten": True,
        "judge": True,
        "ok": True,
    }


def test_detached_is_left_out_when_unset_and_changes_the_id_but_not_the_structure() -> None:
    attached, detached = system(detach=False), system(detach=True)
    assert attached.detached == frozenset()
    assert all("detached" not in entry for section in ("agents", "code_nodes") for entry in attached.to_dict()[section])
    assert attached.structure == detached.structure
    assert attached.describe()["id"] != detached.describe()["id"]


def test_a_detached_agent_is_no_optimizer_target_and_makes_the_system_version_2() -> None:
    mas = MAS(
        [Agent("a", "A {query}", "x"), Agent("b", "B {x}", "y", ["x"], detached=True)], [("a", "b")], llm=MockLLM()
    )
    assert mas.version == 2
    assert movable_agents(mas) == ["a"]
    assert mas.copy(prompts={"a": "Again {query}"}).agents["b"].detached
    with pytest.raises(ValueError, match="need version 2"):
        MAS([Agent("b", "B {query}", "y", detached=True)], llm=MockLLM(), version=1)


def test_the_spec_refuses_a_detached_that_is_not_a_bool() -> None:
    spec = system(detach=True).to_dict()
    spec["code_nodes"][0]["detached"] = "yes"
    with pytest.raises(ValueError, match="'detached' to be true or false"):
        MAS.from_dict(spec, llm=MockLLM(), tools=[count_words])
