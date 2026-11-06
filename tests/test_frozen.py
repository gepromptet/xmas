"""``frozen``: an attribute of every node kind that keeps the node away from every optimizer."""

from __future__ import annotations

import pytest

from xmas import MAS, Agent, HumanNode, ScriptNode, SubsystemNode
from xmas.optimization.updaters import movable_agents

from .conftest import MockLLM, count_words


def system(*, freeze: bool) -> MAS:
    """A writer, a script node counting words, a subsystem shortening the draft and an approval, all frozen or not."""
    inner = MAS([Agent("cutter", "Cut {page}", "short", ["page"])], inputs={"page": "text"}, llm=MockLLM())
    return MAS(
        [
            Agent("writer", "Write {query}", "draft"),
            Agent("judge", "Judge {draft} {counts}", "verdict", ["draft", "counts"], frozen=freeze),
        ],
        [("writer", "count"), ("writer", "shorten"), ("count", "judge"), ("shorten", "judge"), ("judge", "ok")],
        script_nodes=[ScriptNode("count", "count_words", frozen=freeze)],
        subsystem_nodes=[SubsystemNode("shorten", inner, {"page": "draft"}, {"short": "summary"}, frozen=freeze)],
        human_nodes=[HumanNode("ok", "approve", ["verdict"], "approval", frozen=freeze)],
        state={"draft": {"merge": "last"}},
        tools=[count_words],
        llm=MockLLM(),
        version=2,
    )


def test_every_node_kind_can_be_frozen_and_the_spec_keeps_it() -> None:
    mas = system(freeze=True)
    assert mas.frozen == frozenset({"judge", "count", "shorten", "ok"})
    spec = mas.to_dict()
    assert [entry.get("frozen") for section in ("agents", "script_nodes", "subsystem_nodes", "human_nodes")
            for entry in spec[section]] == [None, True, True, True, True]  # fmt: skip
    again = MAS.from_dict(spec, llm=MockLLM(), tools=[count_words])
    assert again.frozen == mas.frozen
    assert again.to_dict() == spec
    assert {node["name"]: node.get("frozen", False) for node in mas.describe()["nodes"]} == {
        "writer": False,
        "count": True,
        "shorten": True,
        "judge": True,
        "ok": True,
    }


def test_frozen_is_left_out_when_unset_and_changes_the_id_but_not_the_structure() -> None:
    unfrozen, frozen = system(freeze=False), system(freeze=True)
    assert unfrozen.frozen == frozenset()
    assert all("frozen" not in entry for section in ("agents", "script_nodes") for entry in unfrozen.to_dict()[section])
    assert unfrozen.structure == frozen.structure
    assert unfrozen.describe()["id"] != frozen.describe()["id"]


def test_a_frozen_agent_is_no_optimizer_target_and_makes_the_system_version_2() -> None:
    mas = MAS([Agent("a", "A {query}", "x"), Agent("b", "B {x}", "y", ["x"], frozen=True)], [("a", "b")], llm=MockLLM())
    assert mas.version == 2
    assert movable_agents(mas) == ["a"]
    assert mas.copy(prompts={"a": "Again {query}"}).agents["b"].frozen
    with pytest.raises(ValueError, match="need version 2"):
        MAS([Agent("b", "B {query}", "y", frozen=True)], llm=MockLLM(), version=1)


def test_the_spec_refuses_a_frozen_that_is_not_a_bool() -> None:
    spec = system(freeze=True).to_dict()
    spec["script_nodes"][0]["frozen"] = "yes"
    with pytest.raises(ValueError, match="'frozen' to be true or false"):
        MAS.from_dict(spec, llm=MockLLM(), tools=[count_words])
