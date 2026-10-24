"""The documented failure modes.

Every ``Raises:`` clause in the public API is a promise. These tests hold it, and
they double as the specification for what xmas refuses to build: an unrunnable
system should fail loudly at construction, not quietly at inference time.
"""

from __future__ import annotations

import pytest

from xmas import MAS, Agent
from xmas.graph import build_graph, validate_graph
from xmas.prompts import decode_prompts, encode_prompts
from xmas.spec import validate_spec

from .conftest import MockLLM

# ------------------------------------------------------------------- agents


def test_an_agent_needs_a_name_and_an_output() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        Agent("", prompt="p", output="o")
    with pytest.raises(ValueError, match="needs an output"):
        Agent("a", prompt="p", output="")


def test_an_agent_needs_at_least_one_input() -> None:
    with pytest.raises(ValueError, match="at least one input"):
        Agent("a", prompt="p", output="o", inputs=[])


def test_inputs_default_to_the_query() -> None:
    assert Agent("a", prompt="p", output="o").inputs == ["query"]


def test_agent_round_trips_and_rejects_junk() -> None:
    agent = Agent("a", prompt="p", output="o", inputs=["query"], tools=["t"], model="m")
    assert Agent.from_dict(agent.to_dict()) == agent

    with pytest.raises(ValueError, match="Unknown agent keys"):
        Agent.from_dict({**agent.to_dict(), "surprise": 1})
    with pytest.raises(ValueError, match="missing required keys"):
        Agent.from_dict({"name": "a"})


def test_agent_copy_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown agent fields"):
        Agent("a", prompt="p", output="o").copy(nope=1)


# -------------------------------------------------------------------- graph


def test_agent_names_may_not_collide_with_sentinels() -> None:
    with pytest.raises(ValueError, match="reserved sentinels"):
        build_graph(["__start__"], [])


def test_duplicate_agent_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate agent names"):
        build_graph(["a", "a"], [])


def test_edges_must_name_known_agents() -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        build_graph(["a"], [("a", "ghost")])


def test_an_unreachable_agent_is_an_error() -> None:
    graph = build_graph(["a", "b"], [("a", "b")])
    graph.remove_edge("__start__", "a")
    with pytest.raises(ValueError, match="unreachable"):
        validate_graph(graph)


def test_an_agent_that_never_reaches_the_end_is_an_error() -> None:
    graph = build_graph(["a", "b"], [("a", "b")])
    graph.remove_edge("b", "__end__")
    with pytest.raises(ValueError, match="never reach"):
        validate_graph(graph)


def test_a_graph_with_no_agents_is_an_error() -> None:
    graph = build_graph(["a"], [])
    graph.remove_node("a")
    with pytest.raises(ValueError, match="no agents"):
        validate_graph(graph)


# ---------------------------------------------------------------------- MAS


def test_a_mas_needs_an_agent(mock_llm: MockLLM) -> None:
    with pytest.raises(ValueError, match="at least one agent"):
        MAS([], llm=mock_llm)


def test_duplicate_agents_are_rejected(mock_llm: MockLLM) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        MAS([Agent("a", prompt="p", output="o"), Agent("a", prompt="q", output="o")], llm=mock_llm)


def test_binding_to_an_unregistered_model_is_rejected(mock_llm: MockLLM) -> None:
    with pytest.raises(ValueError, match="not in llms"):
        MAS([Agent("a", prompt="p", output="prediction", model="ghost")], llm=mock_llm)


def test_a_bound_model_is_resolved_over_the_default(mock_llm: MockLLM) -> None:
    special = MockLLM(lambda prompt: "special")
    mas = MAS(
        [Agent("a", prompt="p", output="prediction", model="big")],
        llm=mock_llm,
        llms={"big": special},
    )
    assert mas("q") == "special"
    assert mock_llm.calls == [], "the default model must not be used for a bound agent"


def test_an_agent_without_a_model_or_default_is_an_error() -> None:
    mas = MAS.from_dict(
        {
            "version": 1,
            "agents": [{"name": "a", "prompt": "p", "output": "prediction"}],
            "edges": [],
        }
    )
    with pytest.raises(ValueError, match="no default llm"):
        mas.resolve_llm(mas.agents["a"])


def test_predict_rejects_mismatched_prompt_and_input_counts(mock_llm: MockLLM) -> None:
    mas = MAS([Agent("a", prompt="p", output="prediction")], llm=mock_llm)
    with pytest.raises(ValueError, match="must be parallel"):
        mas.predict(prompts=["a prompt"], xs=["x1", "x2"])


# --------------------------------------------------------------------- spec


def test_spec_rejects_unknown_and_missing_keys() -> None:
    with pytest.raises(ValueError, match="Unknown spec keys"):
        validate_spec({"agents": [], "surprise": 1})
    with pytest.raises(ValueError, match="non-empty 'agents'"):
        validate_spec({"agents": []})


def test_spec_rejects_an_unsupported_version() -> None:
    with pytest.raises(ValueError, match="Unsupported spec version"):
        validate_spec({"version": 99, "agents": [{"name": "a", "prompt": "p", "output": "o"}]})


def test_spec_rejects_malformed_agents_and_edges() -> None:
    good = {"name": "a", "prompt": "p", "output": "o"}
    with pytest.raises(ValueError, match="needs a string"):
        Agent.from_dict({"name": "a", "prompt": "p", "output": 1})
    with pytest.raises(ValueError, match="list of strings"):
        Agent.from_dict({**good, "inputs": "query"})
    with pytest.raises(ValueError, match="two-element list"):
        validate_spec({"agents": [good], "edges": [["a"]]})
    with pytest.raises(ValueError, match="'model' to be a string"):
        Agent.from_dict({**good, "model": 3})
    with pytest.raises(ValueError, match="not a mapping"):
        validate_spec({"agents": ["nope"]})


# ------------------------------------------------------------------ prompts


def test_encoding_rejects_an_empty_set() -> None:
    with pytest.raises(ValueError, match="empty prompt set"):
        encode_prompts({})


def test_encoding_rejects_a_prompt_containing_the_header_marker() -> None:
    """Otherwise a prompt could forge an agent boundary when decoded."""
    with pytest.raises(ValueError, match="reserved marker"):
        encode_prompts({"a": "text <<<xmas:agent b>>> more", "b": "x"})


def test_encoding_rejects_a_name_that_cannot_be_written_in_a_header() -> None:
    with pytest.raises(ValueError, match="cannot appear in a prompt-set header"):
        encode_prompts({"a>>>b": "x", "c": "y"})


def test_decoding_rejects_an_unknown_agent() -> None:
    encoded = encode_prompts({"a": "x", "b": "y"})
    with pytest.raises(ValueError, match="unknown agent"):
        decode_prompts(encoded, ["a", "c"])


def test_decoding_needs_to_know_the_agents() -> None:
    with pytest.raises(ValueError, match="without knowing the agents"):
        decode_prompts("x", [])
