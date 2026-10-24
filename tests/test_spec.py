"""Spec version 2: the worked examples round-trip, the check names every problem, describe is plain data.

The media example's models section round-trips too, and the check refuses every agent whose model cannot read or
write the kinds it does. The humans example's human nodes and ``first of`` race round-trip, the check names every
problem of a human node and of a race, and a human node's questions and answers work without a run.

These run on construction, serialization and ``describe()`` alone; running a
version 2 system is tested in ``test_steps.py``, ``test_operators.py`` and
``test_chains.py``.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from xmas import END, MAS, START, Agent
from xmas.chains import ForEach
from xmas.check import check_system
from xmas.convert import document_text
from xmas.edges import Edge, EdgeKind, dump_edges, parse_edges
from xmas.errors import RunStatus
from xmas.health import HealthSettings, summarize
from xmas.human import ANSWER_FIELDS, FORMS, Human, Pause, Question, ReplayAsker, answer_value, fill, question_id
from xmas.media import TEXT_ONLY, Capabilities, File
from xmas.nodes import Box, Step
from xmas.optimization.graph_updater import GraphUpdater
from xmas.optimization.synthesis import MechanicalSynthesizer
from xmas.optimization.topology import TopologyBudget
from xmas.routes import Join
from xmas.store import Store
from xmas.tools import Tool, ToolSource, tool
from xmas.trace import AgentRecord, Event, Trace
from xmas.values import ValueType

from .conftest import (
    DECK,
    MEDIA,
    MockLLM,
    SeeingLLM,
    add,
    apology_text,
    assemble_deck,
    count_words,
    publish_text,
    render_page,
    render_pages,
)


@tool
def repeat(text: str, times: int) -> str:
    """Repeat a text."""
    return text * times


class RemoteTools(ToolSource):
    """A tool source standing in for an MCP server."""

    name = "tools"

    def tools(self) -> list[Tool]:
        """Offer one remote tool."""
        schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
        return [Tool("web_fetch", "Fetch a page.", schema)]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Answer every call with the same page."""
        return "page"


# The worked example: conditions, a box, frozen and health. The box's inner
# input is `page`, since the type words are reserved value names.
EXAMPLE = """
version: 2
agents:
- name: writer
  model: text-model
  reads: [query, verdict]
  writes: {draft: text}
  tools: [web_fetch]
- name: judge
  reads: [draft, counts]
  writes: {verdict: json}
  schema:
    fits: {type: bool, means: the draft makes one claim in under 300 words}
    action: {type: text, one of: [publish, shorten], means: what should happen next}
steps:
- {name: count, uses: count_words}
- {name: publish, uses: publish_text, reads: {text: draft}}
- {name: apology, uses: apology_text}
boxes:
- name: shorten
  reads: {page: draft}
  writes: {short: draft}
  system:
    version: 2
    inputs: {page: text}
    agents: [{name: cutter, reads: [page], writes: {short: text}}]
    prompts: {cutter: 'Cut this to half its length: {page}'}
edges:
- writer -> count -> judge
- 'judge -> writer': {when: not verdict.fits, max_visits: 3}
- 'judge -> publish': {switch: verdict.action}
- 'judge -> shorten': {switch: verdict.action}
- 'judge -> apology': else
- shorten -> publish
- 'shorten -> apology': {on error: any}
- 'count -> apology': {on error: [timeout, outage]}
state: {draft: {merge: last}}
frozen: [judge]
health: {length_factor: 4}
callables:
  count_words: {kind: python, import: 'tests.conftest:count_words', reads: {draft: text}, writes: {counts: json}}
  publish_text: {kind: python, import: 'tests.conftest:publish_text', reads: {text: text}, writes: {published: text}}
  apology_text: {kind: python, import: 'tests.conftest:apology_text', reads: {query: text}, writes: {published: text}}
  web_fetch: {kind: mcp, server: tools, reads: {url: text}, writes: {web_fetch: text}}
prompts:
  writer: 'Answer {query} in a short draft. The last verdict, empty at first: {verdict}'
  judge: "Judge this draft: {draft}\\nIts word counts: {counts}"
"""


def example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE)
    return data


def load(data: dict[str, Any]) -> MAS:
    return MAS.from_dict(
        copy.deepcopy(data),
        llms={"text-model": MockLLM(turns=[])},
        tools=[count_words, publish_text, apology_text, repeat, RemoteTools()],
    )


def test_the_worked_example_round_trips() -> None:
    data = example()
    mas = load(data)
    assert (mas.version, list(mas.agents), list(mas.steps), list(mas.boxes), sorted(mas.frozen)) == (
        2,
        ["writer", "judge"],
        ["count", "publish", "apology"],
        ["shorten"],
        ["judge"],
    )
    assert mas.agents["judge"].output_type == "json"
    assert mas.steps["publish"] == Step("publish", "publish_text", {"text": "draft"})
    renamed = Step("count", "count_words", writes={"counts": "tally"})
    assert renamed.to_dict() == {"name": "count", "uses": "count_words", "writes": {"counts": "tally"}}
    assert mas.health == HealthSettings(length_factor=4)
    assert mas.while_edges == (("judge", "writer"),)
    box = mas.boxes["shorten"]
    assert (box.reads, box.writes, list(box.system.agents)) == ({"page": "draft"}, {"short": "draft"}, ["cutter"])
    assert mas.edges[0] == Edge("writer", "count")
    assert mas.edges[3] == Edge("judge", "publish", EdgeKind.SWITCH, switch="verdict.action", case="publish")
    assert mas.edges[-1] == Edge("count", "apology", EdgeKind.ERROR, errors=("timeout", "outage"))
    assert (mas.merges, {k: str(v) for k, v in mas.inputs.items()}) == ({"draft": "last"}, {"query": "text"})
    assert mas.info is not None
    assert mas.info.writers["counts"] == ("count",)
    assert str(mas.info.types["verdict"]) == "json"
    assert mas.info.writers["published"] == ("publish", "apology")  # alternatives
    assert mas.info.joins == {
        "publish": Join("merge", (("judge", "shorten"),)),
        "apology": Join("merge", (("judge", "shorten", "count"),)),
        END: Join("merge", (("publish", "apology"),)),
    }
    assert {s: t for s, t in mas.graph.edges if s == START} == {START: "writer"}  # the loop's first node

    spec = mas.to_dict()
    assert spec == data
    assert list(spec) == [
        "version", "agents", "steps", "boxes", "edges", "state", "frozen", "health", "callables", "prompts"
    ]  # fmt: skip
    again = load(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec
    assert again.structure == mas.structure  # idempotent
    copied = mas.copy()
    assert (copied.to_dict(), copied.frozen, copied.health, copied.boxes) == (spec, mas.frozen, mas.health, mas.boxes)
    assert copied.structure == mas.structure

    # A new answer form, or a box's inner system, changes the structure; frozen and health do not.
    reformed = mas.copy()
    reformed.agents["judge"].schema = {"fits": {"type": "bool"}}
    assert reformed.structure != mas.structure
    assert load({**data, "frozen": [], "health": {}}).structure == mas.structure
    inner = example()
    inner["boxes"][0]["system"]["agents"][0]["model"] = "text-model"
    assert load(inner).structure != mas.structure


@tool
def is_long(draft: str) -> bool:
    """Say whether a draft runs past ten words."""
    return len(draft.split()) > 10


def test_a_callable_only_an_edge_condition_calls_counts_as_used() -> None:
    writer = Agent("writer", "Write {query}", output="draft", inputs=["query"])
    done = Agent("done", "Done {draft}", output="final", inputs=["draft"])
    again = Agent("again", "Again {query}", output="final", inputs=["query"])
    edges = [Edge("writer", "done", EdgeKind.WHEN, when="is_long(draft)"), Edge("writer", "again", EdgeKind.ELSE)]
    state = {"final": {"merge": "last"}}
    spec = MAS([writer, done, again], edges, state=state, tools=[is_long], llm=MockLLM(turns=[])).to_dict()
    assert spec["callables"]["is_long"]["import"] == "tests.test_spec:is_long"
    assert MAS.from_dict(spec, llm=MockLLM(turns=[]), tools=[is_long]).to_dict() == spec


def test_a_version_2_system_is_not_saved_as_one_yaml_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.mas bundle"):
        load(example()).to_yaml(tmp_path / "v2.yaml")


def test_the_edge_grammar_parses_and_dumps_back_as_written() -> None:
    written = example()["edges"]
    edges = parse_edges(written)
    assert edges[2] == Edge("judge", "writer", EdgeKind.WHILE, when="not verdict.fits", max_visits=3)
    assert edges[3].case == "publish"
    assert edges[8].errors == ("timeout", "outage")
    assert dump_edges(edges) == written
    assert dump_edges(parse_edges(dump_edges(edges))) == written

    other_forms = [
        {"a -> b": {"switch": "v.k", "case": "go"}},
        {"a -> c": {"when": "v.n > 2"}},
        {"b -> c": {"on error": "outage"}},
    ]
    assert [edge.kind for edge in parse_edges(other_forms)] == [EdgeKind.SWITCH, EdgeKind.WHEN, EdgeKind.ERROR]
    assert dump_edges(parse_edges(other_forms))[2] == {"b -> c": {"on error": ["outage"]}}
    assert dump_edges(parse_edges(["a -> b -> c", "a -> c"])) == ["a -> b -> c", "a -> c"]  # two chains


def test_a_version_1_system_has_no_version_2_parts() -> None:
    v1 = MAS(
        [Agent("solver", "Solve {query}", "answer"), Agent("checker", "Check {answer}", "verdict", inputs=["answer"])],
        [(START, "solver"), ("solver", "checker"), ("checker", "solver"), ("checker", END)],
        llm=MockLLM(),
    )
    assert (v1.version, v1.steps, v1.info) == (1, {}, None)
    assert v1.copy().structure == v1.structure

    described = v1.describe()  # its marker loop is a loop edge
    assert json.loads(json.dumps(described)) == described
    assert described["version"] == 1
    loop = {"from": "checker", "to": "solver", "kind": "loop", "when": "<continue> in output", "max_visits": 3}
    assert loop in described["edges"]
    assert [node["kind"] for node in described["nodes"]] == ["agent", "agent"]


EXAMPLE_DATA = example()
AGENTS, STEPS, EDGES = EXAMPLE_DATA["agents"], EXAMPLE_DATA["steps"], EXAMPLE_DATA["edges"]
PROMPTS, CALLABLES = EXAMPLE_DATA["prompts"], EXAMPLE_DATA["callables"]
WRITER, JUDGE = AGENTS
COUNT, PUBLISH, APOLOGY = STEPS


@pytest.mark.parametrize(
    ("change", "phrases"),
    [
        pytest.param(
            {
                "agents": [WRITER, {**JUDGE, "reads": [*JUDGE["reads"], "notes"]}],
                "prompts": {**PROMPTS, "judge": PROMPTS["judge"] + " {notes}"},
                "steps": [{**COUNT, "writes": {"counts": "len"}}, PUBLISH, APOLOGY],  # a second problem
            },
            [
                r"3 problems",
                r"'judge' reads 'notes', which is neither an input",
                r"'count' writes 'len', which is a reserved name",
            ],
            id="unread input",
        ),
        pytest.param({"state": {}}, [r"value 'draft' has 2 writers \('writer', 'shorten'\)"], id="two writers"),
        pytest.param(
            {"prompts": {**PROMPTS, "writer": PROMPTS["writer"] + " {counts}"}},
            [r"agent 'writer': prompt placeholders \['counts'\] are not among its reads"],
            id="placeholder not a read",
        ),
        pytest.param(
            {
                "steps": [{**COUNT, "uses": "count_letters"}, PUBLISH, APOLOGY],
                "callables": {name: entry for name, entry in CALLABLES.items() if name != "count_words"},
            },
            [r"Step 'count' uses unknown callable 'count_letters'"],
            id="unknown uses",
        ),
        pytest.param(
            {
                "steps": [*STEPS, {"name": "echo", "uses": "repeat", "reads": {"times": "draft"}}],
                "edges": [*EDGES, "publish -> echo"],
            },
            [r"step 'echo' passes text 'draft' to parameter 'times' \(integer\)"],
            id="text into an int parameter",
        ),
        pytest.param(
            {
                "inputs": {"query": "text", "photo": "image"},
                "agents": [{**WRITER, "reads": [*WRITER["reads"], "photo"]}, JUDGE],
                "prompts": {**PROMPTS, "writer": PROMPTS["writer"] + " {photo}"},
            },
            [
                re.escape(
                    "agent 'writer' reads image 'photo', but model 'text-model' accepts only text; bind a model that "
                    "accepts image (models: {text-model: {accepts: [image]}}) or convert it first with a step"
                )
            ],
            id="image",
        ),
        pytest.param(
            {"agents": [{**WRITER, "writes": {"len": "text"}}, JUDGE]},
            [r"'writer' writes 'len', which is a reserved name"],
            id="reserved name",
        ),
        pytest.param(
            {"steps": [COUNT, {**PUBLISH, "reads": {"txt": "draft"}}, APOLOGY]},
            [r"step 'publish': reads renames 'txt', which is not a parameter of 'publish_text'"],
            id="bad rename",
        ),
        pytest.param(
            {"edges": [*EDGES, "publish -> count"]},
            [r"a loop needs a while edge: the edge publish -> count closes a cycle"],
            id="cycle without a while edge",
        ),
        pytest.param(
            {"prompts": {"writer": PROMPTS["writer"]}},
            [r"Agent 'judge' has no prompt in the 'prompts' section"],
            id="missing prompt",
        ),
        pytest.param(
            {"callables": {**CALLABLES, "count_words": {**CALLABLES["count_words"], "writes": {"counts": "text"}}}},
            [r"callable 'count_words' writes \{'counts': 'text'\} in the spec"],
            id="callables mismatch",
        ),
    ],
)
def test_check_names_every_dataflow_problem(change: dict[str, Any], phrases: list[str]) -> None:
    with pytest.raises(ValueError, match=phrases[0]) as info:
        load({**EXAMPLE_DATA, **change})
    for phrase in phrases:
        assert re.search(phrase, str(info.value)), (phrase, str(info.value))


def edge(index: int, replacement: Any) -> dict[str, list[Any]]:
    """The example's edges with the one at ``index`` replaced."""
    return {"edges": [replacement if i == index else item for i, item in enumerate(EDGES)]}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"edges": [*EDGES, {"count -> publish": {"when": "counts", "max_visits": 2}}]},
            r"the while edge count -> publish closes no loop; make it a when edge",
            id="while closes no loop",
        ),
        pytest.param(
            {"edges": [*EDGES, {"writer -> apology": "else"}]},
            r"node 'writer' has 0 when, while or switch edges and 1 else edges",
            id="else without a decision",
        ),
        pytest.param(
            edge(4, "judge -> apology"),
            r"node 'judge' has 3 when, while or switch edges and 0 else edges",
            id="decision without an else",
        ),
        pytest.param(
            edge(3, {"judge -> shorten": {"switch": "verdict.action", "case": "trim"}}),
            r"edge judge -> shorten: case 'trim' is not one of \['publish', 'shorten'\]",
            id="case outside one of",
        ),
        pytest.param(
            edge(3, {"judge -> shorten": {"switch": "verdict.action", "case": "publish"}}),
            r"node 'judge' switches to case 'publish' twice",
            id="case twice",
        ),
        pytest.param(
            edge(2, {"judge -> publish": {"switch": "verdict.action == 'publish'"}}),
            r"edge judge -> publish: switch takes a path such as verdict.action",
            id="switch on no path",
        ),
        pytest.param(
            edge(7, {"count -> apology": {"on error": ["timeout", "budget"]}}),
            r"edge count -> apology: on error 'budget' is never routed; the run stops on it",
            id="budget routed",
        ),
        pytest.param(
            edge(7, {"count -> apology": {"on error": ["flood"]}}),
            r"edge count -> apology: on error 'flood' is no error kind",
            id="unknown error kind",
        ),
        pytest.param(
            edge(7, {"count -> apology": {"on error": ["any", "timeout"]}}),
            r"edge count -> apology: on error 'any' stands alone",
            id="any beside kinds",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fits and published", "max_visits": 3}}),
            r"edge judge -> writer decides on 'published', which is not readable at 'judge'",
            id="root not readable",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fit", "max_visits": 3}}),
            r"edge judge -> writer reads verdict.fit, but 'judge''s answer form has no field 'fit'",
            id="field not in the form",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "draft.size > 3", "max_visits": 3}}),
            r"edge judge -> writer reads the field 'size' of 'draft', which is not json",
            id="field of text",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fits and approved(draft)", "max_visits": 3}}),
            r"edge judge -> writer calls 'approved', which is no registered callable",
            id="unknown function",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fits ==", "max_visits": 3}}),
            r"edge judge -> writer: .* at position 19",
            id="bad predicate",
        ),
        pytest.param(
            {"boxes": [{**EXAMPLE_DATA["boxes"][0], "reads": {"page": "counts"}}]},
            r"box 'shorten' passes json 'counts' to its text input 'page'",
            id="box passes the wrong type",
        ),
        pytest.param({"frozen": ["judge", "ghost"]}, r"frozen names 'ghost', which is no node", id="frozen ghost"),
    ],
)
def test_check_names_every_operator_problem(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message) as info:
        load({**EXAMPLE_DATA, **change})
    assert len(re.findall(message, str(info.value))) == 1, str(info.value)


@pytest.mark.parametrize(
    ("edges", "agents", "message"),
    [
        pytest.param(
            [("a", "b"), ("b", "c"), Edge("b", "f", EdgeKind.ERROR, errors=("timeout",))],
            [
                ("a", "draft", ["query"]),
                ("b", "verdict", ["draft"]),
                ("c", "done", ["verdict"]),
                ("f", "fixed", ["verdict"]),
            ],
            r"'f' runs only when 'b' fails, so it cannot read 'verdict'",
            id="reads a failed node's write",
        ),
        pytest.param(
            [
                ("f", "b"),
                ("f", "x"),
                Edge("x", "d", EdgeKind.WHEN, when="flag"),
                Edge("x", "e", EdgeKind.ELSE),
                ("b", "j"),
                ("d", "j"),
            ],
            [
                ("f", "o1", ["query"]),
                ("b", "o2", ["query"]),
                ("x", "flag", ["query"]),
                ("d", "o3", ["query"]),
                ("e", "o4", ["query"]),
                ("j", "o5", ["query"]),
            ],
            r"node 'j' is neither a merge nor a join: d -> j does not run on every pass that runs b",
            id="neither merge nor join",
        ),
        pytest.param(
            [("a", "b"), ("a", "c")],
            [("a", "o1", ["query"]), ("b", "o2", ["query"]), ("c", "o3", ["query"])],
            r"several nodes feed '__end__' \(\['b', 'c'\]\) on one pass",
            id="parallel terminals",
        ),
        pytest.param(
            [
                ("w", "c"),
                Edge("c", "w", EdgeKind.WHILE, when="flag", max_visits=2),
                Edge("c", "fin", EdgeKind.ELSE),
                ("w", "fin"),
            ],
            [("w", "o1", ["query"]), ("c", "flag", ["query"]), ("fin", "o2", ["query"])],
            r"node 'fin' cannot wait for w -> fin, which leaves the loop w \.\. c on every pass",
            id="leaves a loop on every pass",
        ),
    ],
)
def test_check_names_every_operator_problem_of_a_small_system(
    edges: list[Any], agents: list[tuple[str, str, list[str]]], message: str
) -> None:
    """Agents are ``(name, write, reads)``, and each prompt uses every read."""
    with pytest.raises(ValueError, match=message) as info:
        MAS(
            [Agent(name, " ".join(f"{{{r}}}" for r in reads), write, inputs=reads) for name, write, reads in agents],
            edges=edges,
            llm=MockLLM(),
            version=2,
        )
    assert len(re.findall(message, str(info.value))) == 1, str(info.value)


def test_describe_is_plain_data_that_names_every_node_and_edge() -> None:
    mas = load(example())
    described = mas.describe()
    assert json.loads(json.dumps(described)) == described
    assert set(described) == {"version", "inputs", "nodes", "edges", "joins", "id"}
    assert (described["version"], described["inputs"]) == (2, {"query": "text"})
    assert described["joins"] == {"publish": "merge", "apology": "merge", END: "merge"}
    assert described["id"].startswith("sha256:")
    assert len(described["id"]) == 71

    writer, count, judge, box, publish, apology = described["nodes"]
    assert writer == {
        "name": "writer",
        "kind": "agent",
        "model": "text-model",
        "reads": {"query": "text", "verdict": "json"},
        "writes": {"draft": "text"},
        "tools": ["web_fetch"],
        "frozen": False,
    }
    assert count == {
        "name": "count",
        "kind": "step",
        "uses": "count_words",
        "reads": {"draft": "text"},
        "writes": {"counts": "json"},
    }
    assert judge == {
        "name": "judge",
        "kind": "agent",
        "model": None,
        "reads": {"draft": "text", "counts": "json"},
        "writes": {"verdict": "json"},
        "tools": [],
        "frozen": True,
        "schema": JUDGE["schema"],
    }
    assert (box["name"], box["kind"]) == ("shorten", "box")
    assert (box["reads"], box["writes"]) == ({"draft": "text"}, {"draft": "text"})
    assert box["system"] == mas.boxes["shorten"].system.describe()
    assert box["system"]["inputs"] == {"page": "text"}
    assert box["system"]["nodes"][0]["name"] == "cutter"
    assert [(node["name"], node["kind"]) for node in (publish, apology)] == [("publish", "step"), ("apology", "step")]

    edges = described["edges"]
    assert edges[0] == {"from": START, "to": "writer", "kind": "plain", "implicit": True}
    assert {"from": "writer", "to": "count", "kind": "plain"} in edges
    assert {"from": "judge", "to": "writer", "kind": "while", "when": "not verdict.fits", "max_visits": 3} in edges
    assert {"from": "judge", "to": "publish", "kind": "switch", "switch": "verdict.action", "case": "publish"} in edges
    assert {"from": "judge", "to": "apology", "kind": "else"} in edges
    assert {"from": "count", "to": "apology", "kind": "error", "errors": ["timeout", "outage"]} in edges
    assert {"from": "apology", "to": END, "kind": "plain", "implicit": True} in edges


def test_the_describe_id_follows_wiring_and_frozen_but_not_prompts() -> None:
    mas = load(example())
    described = mas.describe()
    assert load(mas.to_dict()).describe() == described
    assert mas.copy(prompts={"writer": "Answer {query} at length. {verdict}"}).describe()["id"] == described["id"]
    rewired = mas.copy(edges=mas.edges[:-1])  # count's on error edge gone
    assert rewired.describe()["id"] != described["id"]
    unfrozen = load({**example(), "frozen": []}).describe()
    assert unfrozen["id"] != described["id"]
    assert unfrozen["nodes"][2]["frozen"] is False


def test_a_box_maps_its_inner_inputs_and_never_contains_itself() -> None:
    data = example()
    with pytest.raises(ValueError, match=r"reads must map exactly the inner inputs \['page'\], got \['text'\]"):
        load({**data, "boxes": [{**data["boxes"][0], "reads": {"text": "draft"}}]})
    with pytest.raises(ValueError, match=r"writes must map inner writes, some of \['short'\], got \['long'\]"):
        load({**data, "boxes": [{**data["boxes"][0], "writes": {"long": "draft"}}]})
    v1 = MAS([Agent("a", "Do {query}", "x")], llm=MockLLM())
    with pytest.raises(ValueError, match="the system in a box must be version 2"):
        Box("old", v1, {"query": "query"}, {"x": "x"})

    mas = load(data)
    inner = mas.boxes["shorten"].system
    nested = MAS([], boxes=[Box("outer", mas, {"query": "page"}, {"published": "short"})], inputs={"page": "text"})
    assert nested.describe()["nodes"][0]["system"]["nodes"][3]["system"] == inner.describe()  # boxes nest
    inner.boxes["again"] = Box("again", nested, {"page": "page"}, {"short": "page2"})  # a system put inside itself
    with pytest.raises(ValueError, match="box 'again' contains the system it sits in"):
        check_system(inner)


DECK_DATA: dict[str, Any] = yaml.safe_load(DECK)


def load_deck(data: dict[str, Any] | None = None) -> MAS:
    return MAS.from_dict(
        copy.deepcopy(DECK_DATA if data is None else data),
        llms={"writer-model": MockLLM(), "judge-model": MockLLM()},
        tools=[render_page, assemble_deck],
    )


def test_the_deck_example_round_trips() -> None:
    mas = load_deck()
    assert (mas.version, list(mas.each), mas.prompt_names, mas.retry) == (
        2,
        ["drafter", "checker", "captioner", "figure_checker", "render"],
        {"checker": "judge", "figure_checker": "judge"},
        {"render": {"outage": 1}},
    )
    assert mas.each["drafter"] == ForEach("page", "plan.slides", 4, "id", {"cost_usd": 0.05})
    assert mas.agents["drafter"].reads == {"slide": "page", "rules": "house", "verdict": "verdict"}
    assert mas.agents["checker"].prompt == mas.agents["figure_checker"].prompt == DECK_DATA["prompts"]["judge"]
    info = mas.checked()
    drafter, captioner = info.chains["drafter"], info.chains["captioner"]
    assert (sorted(drafter.nodes), drafter.each.root, captioner.nodes) == (
        ["checker", "drafter", "render"],
        "plan",
        ("captioner", "figure_checker"),
    )
    words = {name: str(vtype) for name, vtype in drafter.inputs.items()}
    assert words == {"page": "json", "house": "text"}
    assert set(captioner.inputs) == {"figure", "house"}
    assert {name: str(vtype) for name, vtype in drafter.writes.items()} == {
        "draft": "text",
        "verdict": "json",
        "measure": "json",
    }
    assert {name: str(info.types[name]) for name in ("draft", "verdict", "measure", "caption", "figure_verdict")} == {
        "draft": "list of text",
        "verdict": "list of json",
        "measure": "list of json",
        "caption": "list of text",
        "figure_verdict": "list of json",
    }
    assert Edge("checker", END, EdgeKind.ELSE) in drafter.system.edges  # the exit ends one item's run
    assert drafter.system.resolve_llm(mas.agents["checker"]) is mas.llms["judge-model"]  # through the outer system
    assert info.edges == (Edge("drafter", "assemble"), Edge("captioner", "assemble"))
    assert info.joins == {
        "assemble": Join("join", (("drafter",), ("captioner",)), (None, None)),
        END: Join("join", (("noter",), ("assemble",)), (None, None)),
    }
    assert set(info.graph.successors("checker")) == {"drafter", "assemble"}  # the full topology, for describe

    spec = mas.to_dict()
    assert spec == DECK_DATA
    assert list(spec) == ["version", "inputs", "outputs", "agents", "steps", "edges", "callables", "prompts"]
    assert list(spec["agents"][1]) == ["name", "model", "prompt", "for each", "reads", "writes", "schema"]
    assert list(spec["steps"][0]) == ["name", "uses", "for each", "retry"]
    again = load_deck(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec
    assert again.structure == mas.structure  # idempotent


def test_a_prompt_given_to_one_sharer_reaches_the_others() -> None:
    """A prompt given to one agent of a shared prompt reaches the others; a split made by hand does not dump."""
    mas = load_deck()
    shared = "Is this claim made? {planned} {shown}"
    copied = mas.copy(prompts={"checker": shared})
    assert copied.agents["figure_checker"].prompt == shared
    assert copied.to_dict()["prompts"]["judge"] == shared
    copied.agents["checker"].prompt = "Split {planned} {shown}"
    with pytest.raises(ValueError, match="sharing the prompt 'judge' hold different texts"):
        copied.to_dict()


def test_the_deck_example_describes_its_chains_and_nodes() -> None:
    described = load_deck().describe()
    assert json.loads(json.dumps(described)) == described
    assert described["outputs"] == {"deck": "json", "notes": "text"}
    assert described["groups"] == [
        {
            "name": "drafter",
            "for each": "page in plan.slides",
            "nodes": ["drafter", "render", "checker"],
            "at once": 4,
            "key": "id",
            "item budget": {"cost_usd": 0.05},
        },
        {
            "name": "captioner",
            "for each": "figure in plan.figures",
            "nodes": ["captioner", "figure_checker"],
            "key": "id",
        },
    ]
    nodes = {node["name"]: node for node in described["nodes"]}
    assert nodes["drafter"]["for each"] == "page in plan.slides"
    assert "for each" not in nodes["noter"]
    assert nodes["checker"]["placeholders"] == {"planned": "page", "shown": "measure"}
    assert "placeholders" not in nodes["noter"]
    assert nodes["checker"]["reads"] == {"page": "json", "measure": "json"}  # one item's types
    assert nodes["assemble"]["reads"] == {"draft": "list of text", "verdict": "list of json", "caption": "list of text"}
    assert nodes["render"]["retry"] == {"outage": 1}
    assert described["joins"] == {"assemble": "join", END: "join"}


def test_the_structure_follows_chain_reads_and_retry_settings() -> None:
    mas = load_deck()

    def variant(name: str, **fields: Any) -> MAS:
        data = copy.deepcopy(DECK_DATA)
        for entry in data["agents"] + data["steps"]:
            entry.update(fields if entry["name"] == name else {})
        return load_deck(data)

    head = DECK_DATA["agents"][0]["for each"]
    assert variant("drafter", **{"for each": {**head, "at once": 2}}).structure != mas.structure
    assert variant("drafter", **{"for each": {**head, "item budget": {"wall_s": 60}}}).structure == mas.structure
    assert variant("noter", reads={"plan": "plan", "house": "house"}).structure != mas.structure
    assert variant("render", retry={"timeout": 2}).structure != mas.structure
    assert mas.copy(prompts={"noter": "Notes on {plan}"}).structure == mas.structure


def chain_of(form: Any) -> MAS:
    """A one-agent chain over ``plan.slides`` whose list of drafts is the output."""
    return MAS(
        [Agent("writer", "Write {page}", "draft", inputs=["page"])],
        inputs={"plan": "json"},
        outputs={"draft": "list of text"},
        each={"writer": form},
        llm=MockLLM(),
    )


@pytest.mark.parametrize(
    ("form", "want"),
    [
        pytest.param("page in plan.slides", ForEach("page", "plan.slides"), id="short"),
        pytest.param({"item": "page", "in": "plan.slides"}, ForEach("page", "plan.slides"), id="long"),
        pytest.param(" page \t in  plan.slides ", ForEach("page", "plan.slides"), id="spaces"),
        pytest.param(
            {"item": "page", "in": "plan.slides", "at once": 4, "key": "id", "item budget": {"wall_s": 120}},
            ForEach("page", "plan.slides", 4, "id", {"wall_s": 120}),
            id="options",
        ),
        pytest.param(
            "page in plan.slides[0]", r"'page in plan.slides\[0\]' must read '<item> in <list>'", id="indexing"
        ),
        pytest.param("page plan.slides", r"must read '<item> in <list>'", id="no in"),
        pytest.param("in in plan.slides", r"chain 'writer': the item 'in' is a reserved name", id="reserved item"),
        pytest.param(
            {"item": "page", "in": "plan.slides", "key": "id", "at once": 0},
            r"at once must be a positive integer, got 0",
            id="round trip",
        ),
    ],
)
def test_for_each_grammar(form: Any, want: ForEach | str) -> None:
    if isinstance(want, str):
        with pytest.raises(ValueError, match=want):
            chain_of(form)
        return
    each = ForEach.parse(form)
    assert each == want
    assert (each.root, each.fields) == ("plan", ("slides",))
    assert str(each) == "page in plan.slides"
    assert ForEach.parse(each.to_spec()) == each
    assert chain_of(each).each["writer"] == each
    assert each.to_spec() == (form if each.options else "page in plan.slides")
    if each.options:  # the round trip of a dumped long form, the options left out when none is set
        assert chain_of(form).to_dict()["agents"][0]["for each"] == form
        assert ForEach.parse({"item": "page", "in": "plan.slides", "key": "id"}).to_spec() == {
            "item": "page",
            "in": "plan.slides",
            "key": "id",
        }


def deck(**sections: Any) -> dict[str, Any]:
    """The deck with some top-level sections replaced."""
    return {**copy.deepcopy(DECK_DATA), **sections}


def entries(section: str, name: str, **fields: Any) -> dict[str, list[dict[str, Any]]]:
    """The deck's ``section`` with the fields of entry ``name`` replaced, ``None`` dropping one."""
    changed = []
    for entry in copy.deepcopy(DECK_DATA[section]):
        replaced = {key: value for key, value in {**entry, **fields}.items() if value is not None}
        changed.append(replaced if entry["name"] == name else entry)
    return {section: changed}


def over(path: str, key: str | None = None) -> dict[str, Any]:
    """The deck with the slide chain running over ``path``."""
    head = {"item": "page", "in": path, **({"key": key} if key else {})}
    agents = entries("agents", "drafter", **{"for each": head})["agents"]
    agents[1]["for each"] = f"page in {path}"
    return deck(agents=agents, steps=entries("steps", "render", **{"for each": f"page in {path}"})["steps"])


def boxed_chain() -> MAS:
    """A chain whose one node is a box holding a chain of its own."""
    inner = chain_of("page in plan.slides")
    return MAS(
        [],
        boxes=[Box("pages", inner, {"plan": "deck"}, {"draft": "drafts"})],
        inputs={"decks": "json"},
        outputs={"drafts": "list of list of text"},
        each={"pages": "deck in decks"},
    )


def small(**settings: Any) -> MAS:
    """Two agents, ``a -> b``, with the given settings."""
    agents = [Agent("a", "Do {query}", "x"), Agent("b", "Check {x}", "y", inputs=["x"])]
    return MAS(agents, edges=[("a", "b")], llm=MockLLM(), **settings)


@pytest.mark.parametrize(
    ("build", "phrase"),
    [
        pytest.param(
            lambda: load_deck(deck(edges=[*DECK_DATA["edges"], "noter -> checker"])),
            r"chain \['drafter', 'checker', 'render'\] \(page in plan.slides\) is entered at \['checker', 'drafter'\]",
            id="two heads",
        ),
        pytest.param(
            lambda: load_deck(deck(edges=[*DECK_DATA["edges"], "render -> noter"])),
            r"chain 'drafter' leaves to \['assemble', 'noter'\]; a chain leaves to one node",
            id="exits to two targets",
        ),
        pytest.param(
            lambda: load_deck(deck(edges=[*DECK_DATA["edges"], {"render -> assemble": {"on error": "any"}}])),
            r"chain 'drafter': the error edge render -> assemble leaves it; route it inside the chain",
            id="an error edge leaving",
        ),
        pytest.param(
            boxed_chain, r"chain 'pages': box 'pages' holds a for each chain; chains do not nest", id="nested"
        ),
        pytest.param(
            lambda: load_deck(deck(inputs={"plan": "json", "house": "text", "page": "json"})),
            r"chain 'drafter': the item 'page' is an input or a written value too",
            id="item is an input",
        ),
        pytest.param(
            lambda: load_deck(over("spec.slides")),
            r"'drafter' reads 'spec', which is neither an input nor written by any node",
            id="root unreadable",
        ),
        pytest.param(
            lambda: load_deck(over("house")),
            r"chain 'drafter': for each page in house runs over text 'house'; a chain runs over json or a list",
            id="text root",
        ),
        pytest.param(
            lambda: load_deck(over("titles", key="id") | {"inputs": {**DECK_DATA["inputs"], "titles": "list of text"}}),
            r"chain 'drafter': key 'id' needs json items, not text",
            id="key on text items",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "noter", writes={"draft": "text"}))),
            r"chain 'drafter': 'noter' writes 'draft', which the chain writes; a value written in a chain has",
            id="chain write written outside",
        ),
        pytest.param(
            lambda: load_deck(
                deck(
                    **entries("agents", "checker", **{"for each": {"item": "page", "in": "plan.slides", "at once": 2}})
                )
            ),
            r"chain 'drafter': 'checker' sets at once, which only the head 'drafter' sets",
            id="option off the head",
        ),
        pytest.param(
            lambda: small(each={"a": "q in query"}, version=1),
            r"for each, retry, shared prompts, read maps, conditional edges and typed writes need version 2",
            id="each in version 1",
        ),
        pytest.param(
            lambda: MAS(
                [Agent("writer", "Write {page}", "draft", inputs=["page"])],
                inputs={"plan": "json"},
                each={"writer": "page in plan"},
                llm=MockLLM(),
            ),
            r"chain 'writer': it feeds the end, which needs declared outputs",
            id="chain into the end without outputs",
        ),
        pytest.param(
            lambda: load_deck(deck(outputs={**DECK_DATA["outputs"], "summary": "text"})),
            r"output 'summary' is neither an input nor written by any node",
            id="output not written",
        ),
        pytest.param(
            lambda: load_deck(deck(outputs={"deck": "json", "notes": "json"})),
            r"output 'notes' is declared json but is text",
            id="output type",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("steps", "render", retry={"money": 1}))),
            r"chain 'drafter': 'render' retries on 'money'; a node retries on \['outage', 'rate_limited', 'timeout'\]",
            id="retry money",
        ),
        pytest.param(
            lambda: MAS(
                [],
                boxes=[Box("b", chain_of("page in plan.slides"), {"plan": "plan"}, {"draft": "d"})],
                inputs={"plan": "json"},
                retry={"b": {"outage": 1}},
            ),
            r"box 'b' takes no retry; retry inside its system",
            id="retry on a box",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "checker", prompt="judges"))),
            r"Agent 'checker': prompt 'judges' names no entry of 'prompts'",
            id="prompt names no entry",
        ),
        pytest.param(
            lambda: load_deck().copy(
                prompts={"checker": "A {planned} {shown}", "figure_checker": "B {planned} {shown}"}
            ),
            r"agents \['checker', 'figure_checker'\] share the prompt 'judge' but hold different texts",
            id="shared texts differ",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "noter", reads=["plan", "error"]))),
            r"'noter' reads 'error', which only a node an on error edge reaches can read",
            id="error read without an error edge",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "noter", writes={"run": "text"}))),
            r"'noter' writes 'run', which is a reserved name",
            id="a node writing run",
        ),
    ],
)
def test_check_names_every_chain_problem(build: Any, phrase: str) -> None:
    with pytest.raises(ValueError, match=phrase):
        build()


def test_agent_read_maps_and_dicts() -> None:
    drafter = Agent("drafter", "Draft {slide} in {rules}", "draft", inputs={"slide": "page", "rules": "house"})
    assert (drafter.reads, drafter.inputs, drafter.renamed) == (
        {"slide": "page", "rules": "house"},
        ["page", "house"],
        True,
    )
    data = drafter.to_dict()
    assert data["inputs"] == {"slide": "page", "rules": "house"}
    assert Agent.from_dict(data) == drafter
    assert drafter.copy(inputs=["page"]).reads == {"page": "page"}
    assert drafter.copy(inputs={"slide": "page"}).to_dict()["inputs"] == {"slide": "page"}
    assert Agent("a", "{v} {w}", "x", inputs={"v": "v", "w": "v"}).inputs == ["v"]  # values once each, in order
    assert Agent("a", "{query}", "x", inputs={"query": "query"}).to_dict()["inputs"] == ["query"]  # identity is a list
    with pytest.raises(ValueError, match="Unknown agent keys: \\['reads'\\]"):
        Agent.from_dict({**data, "reads": {"slide": "page"}})
    with pytest.raises(ValueError, match="'inputs' to be a list of strings or a map of names to names"):
        Agent.from_dict({**data, "inputs": {"slide": 3}})
    v1 = Agent("solver", "Solve {query}", "answer")
    assert list(v1.to_dict()) == ["name", "prompt", "inputs", "output", "tools", "model"]
    v1.inputs = ["query", "hint"]
    assert v1.reads == {"query": "query", "hint": "hint"}
    assert not v1.renamed

    # A renamed read makes a system version 2; a placeholder outside the map is a problem.
    reader = Agent("reader", "Read {text}", "said", inputs={"text": "query"})
    with pytest.raises(ValueError, match="read maps"):
        MAS([reader], llm=MockLLM(), version=1)
    with pytest.raises(ValueError, match=r"prompt placeholders \['query'\] are not among its reads \['text'\]"):
        MAS([reader.copy(prompt="Read {query}")], llm=MockLLM())
    with pytest.raises(ValueError, match="takes the prompt 'reader', another agent's name"):
        MAS([reader, Agent("twin", "Read {query}", "heard")], llm=MockLLM(), prompt_names={"twin": "reader"})

    llm = MockLLM(lambda prompt: prompt.upper())
    trace = MAS([reader], llm=llm).run("cats")
    assert trace["reader"][0].inputs == {"text": "cats"}
    assert llm.calls == ["Read cats"]  # keyed by placeholder


# ------------------------------------------------------------------- media

MEDIA_DATA: dict[str, Any] = yaml.safe_load(MEDIA)
BARE_MEDIA = {key: value for key, value in MEDIA_DATA.items() if key != "models"}
PLAN = {"slides": [{"id": "intro"}, {"id": "method"}, {"id": "result"}]}


def load_media(data: dict[str, Any] | None = None, llm: Any = None) -> MAS:
    """The media example, its vision model a :class:`SeeingLLM` unless given."""
    return MAS.from_dict(
        copy.deepcopy(MEDIA_DATA if data is None else data),
        llms={"vision-model": SeeingLLM() if llm is None else llm},
        tools=[document_text, render_pages],
    )


def test_the_media_example_round_trips(png: Path) -> None:
    mas = load_media()
    assert {name: str(vtype) for name, vtype in mas.checked().types.items()} == {
        "plan": "json",
        "house": "document",
        "review": "text",
        "house_rules": "text",
        "slide_png": "list of image",
    }
    assert mas.models == {"vision-model": Capabilities(accepts=("image",))}
    spec = mas.to_dict()
    assert spec == MEDIA_DATA
    assert list(spec) == ["version", "inputs", "outputs", "agents", "steps", "edges", "models", "callables", "prompts"]
    again = load_media(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec
    assert again.structure == mas.structure  # idempotent

    described = mas.describe()
    assert json.loads(json.dumps(described)) == described
    assert described["models"] == {"vision-model": {"accepts": ["image"]}}
    assert described["nodes"][-1]["reads"] == {"slide_png": "list of image", "house_rules": "text", "plan": "json"}
    bare = load_media(BARE_MEDIA)  # its model states accepts itself
    assert (bare.models, bare.to_dict(), bare.structure) == ({}, BARE_MEDIA, mas.structure)
    assert "models" not in bare.describe()
    assert bare.describe()["id"] == described["id"]  # no compiled graph changes
    with pytest.raises(ValueError, match="'models' entry 'vision-model' must map accepts and produces to lists"):
        load_media({**MEDIA_DATA, "models": {"vision-model": {"accepts": "image"}}})

    # the house rules reach the reviewer as text, the three rendered pages as pictures
    seer = SeeingLLM("The deck keeps its plan.")
    house = png.with_name("house.md")
    house.write_text("# House\nOne claim per page.", encoding="utf-8")
    trace = load_media(llm=seer).run_state({"plan": PLAN, "house": house})
    assert trace.error is None
    assert trace.outputs == {"review": "The deck keeps its plan."}
    (parts,) = seer.parts
    assert [type(part).__name__ for part in parts] == ["str", "Ref", "str", "Ref", "str", "Ref"]
    assert parts[0].endswith("Rules: # House\nOne claim per page.\nPages: \n[pages 0] ")
    assert parts[2:5:2] == ["\n[pages 1] ", "\n[pages 2] "]
    assert [part.read_bytes() for part in parts[1::2]] == [b"page intro", b"page method", b"page result"]


def media_system(*agents: Agent, **settings: Any) -> MAS:
    """``agents`` on a text model, ``text-model``, or on the unnamed default; both are scripted."""
    return MAS(list(agents), llms={"text-model": MockLLM(turns=[])}, llm=MockLLM(turns=[]), **settings)


@tool
def shout(text: str) -> str:
    """Shout a text."""
    return text.upper()


@pytest.mark.parametrize(
    ("build", "phrases"),
    [
        pytest.param(
            lambda: media_system(
                Agent("judge", "Look: {png}", "verdict", ["png"], model="text-model"),
                Agent("glance", "Glance at {png}", "note", ["png"]),
                inputs={"png": "image"},
                outputs={"verdict": "text", "note": "text"},
            ),
            [
                (
                    "agent 'judge' reads image 'png', but model 'text-model' accepts only text; bind a model that "
                    "accepts image (models: {text-model: {accepts: [image]}}) or convert it first with a step"
                ),
                (
                    "agent 'glance' reads image 'png', but the default model accepts only text; give it accepts = "
                    '("text", "image") or convert it first with a step'
                ),
            ],
            id="image read on a text model",
        ),
        pytest.param(
            lambda: load_media(BARE_MEDIA, llm=MockLLM()),
            ["agent 'reviewer' reads image 'slide_png', but model 'vision-model' accepts only text"],
            id="list of image",
        ),
        pytest.param(
            lambda: media_system(
                Agent("writer", "Write in the style of {house}", "draft", ["house"]),
                steps=[Step("loud", "shout", {"text": "house"})],
                tools=[shout],
                inputs={"house": "document"},
                outputs={"draft": "text"},
            ),
            [
                (
                    "agent 'writer' reads document 'house'; a model reads a document through a conversion step, "
                    "such as one using xmas.convert.document_text"
                ),
                (
                    "step 'loud' passes document 'house' to parameter 'text' (string); text goes into a string "
                    "parameter only, document into a document parameter (xmas.media.Document) or a file, json into "
                    "any other"
                ),
            ],
            id="document read",
        ),
        pytest.param(
            lambda: media_system(Agent("reader", "Sum up {sheet}", "summary", ["sheet"]), inputs={"sheet": "table"}),
            [
                (
                    "agent 'reader' reads table 'sheet'; a model reads a table through a conversion step, such as one "
                    "using xmas.convert.table_rows"
                )
            ],
            id="table read",
        ),
        pytest.param(
            lambda: media_system(
                Agent("painter", "Paint {query}", "picture", model="text-model", output_type="image"),
                Agent("singer", "Sing {query}", "sound", output_type="audio"),
                outputs={"picture": "image", "sound": "audio"},
            ),
            [
                (
                    "agent 'painter' writes image 'picture', but model 'text-model' produces only text; bind a model "
                    "that produces image (models: {text-model: {produces: [image]}})"
                ),
                "agent 'singer' writes audio 'sound'; a model answers with text, json or image",
            ],
            id="image write on a text model",
        ),
        pytest.param(
            lambda: media_system(
                Agent("painter", "Paint {query}", "picture", tools=["add"], output_type="image"),
                tools=[add],
                models={"text-model": {"produces": ["image"]}},
            ),
            ["agent 'painter' writes image and has tools; an image answer takes none"],
            id="image write with tools",
        ),
        pytest.param(
            lambda: media_system(
                Agent("judge", "Look: {png}", "verdict", ["png"], model="text-model"),
                Agent(
                    "painter",
                    "Paint {verdict}",
                    "picture",
                    ["verdict"],
                    model="text-model",
                    output_type="list of image",
                ),
                steps=[Step("post", "web_fetch", {"url": "png"})],
                tools=[RemoteTools()],
                inputs={"png": "image"},
                outputs={"picture": "list of image"},
                models={"text-model": {"accepts": ["image"], "produces": ["image"]}},
            ),
            [
                (
                    "model 'text-model' accepts image but cannot send content parts; use APILLM, or give it "
                    "generate_parts_with_meta and full_response = True"
                ),
                (
                    "model 'text-model' produces image but cannot answer with images; use APILLM, or give it "
                    "generate_image_with_meta and full_response = True"
                ),
                "step 'post' passes image 'png' to remote tool 'web_fetch'; remote tools read no media",
            ],
            id="a media model without parts",
        ),
        pytest.param(
            lambda: media_system(Agent("a", "Do {query}", "x"), models={"ghost": {"accepts": ["image"]}}),
            ["models names 'ghost', which no agent binds"],
            id="models naming an unbound model",
        ),
        pytest.param(
            lambda: load_media({**MEDIA_DATA, "models": {"vision-model": {"accepts": ["image"], "colour": "red"}}}),
            [
                (
                    "models entry 'vision-model' has key 'colour'; it takes model, accepts, produces and the "
                    "parameters max_tokens, timeout_s, temperature, top_p, seed, reasoning_effort"
                )
            ],
            id="an unknown key in models",
        ),
    ],
)
def test_check_names_every_capability_problem(build: Any, phrases: list[str]) -> None:
    with pytest.raises(ValueError, match=re.escape(phrases[0])) as info:
        build()
    for phrase in phrases:
        assert phrase in str(info.value), (phrase, str(info.value))


class Squinting(MockLLM):
    """A model whose ``accepts`` is no list of words, so it reads text only."""

    accepts = "image"


@tool(writes={"size": "json"})
def weigh(upload: File) -> int:
    """Weigh any file."""
    return upload.size or 0


def test_capabilities_resolve_in_order(png: Path) -> None:
    judge = Agent("judge", "Look: {png}", "verdict", ["png"], model="seer")
    seer = SeeingLLM()

    def judged(**settings: Any) -> MAS:
        return MAS([judge], llms={"seer": seer}, inputs={"png": "image"}, **settings)

    # the models entry over the model's own attributes over text only
    assert judged().capabilities(judge) == Capabilities(("text", "image"))
    declared = judged(models={"seer": {"accepts": ["image", "audio"]}})
    assert declared.capabilities(judge) == Capabilities(("image", "audio"))
    assert declared.version == 2
    with pytest.raises(ValueError, match="model 'seer' accepts only text; bind a model that accepts image"):
        judged(models={"seer": {}})
    plain = Agent("plain", "Read {query}", "note")
    assert MAS([plain], llm=Squinting(), version=2).capabilities(plain) == TEXT_ONLY
    assert MAS([plain], llm=MockLLM()).models == {}  # version 1

    # default_model names the default model
    looker = Agent("looker", "Look: {png}", "verdict", ["png"])
    named = MAS(
        [looker], llm=seer, default_model="eye", inputs={"png": "image"}, models={"eye": {"accepts": ["image"]}}
    )
    assert named.capabilities(looker) == Capabilities(("image",))
    with pytest.raises(
        ValueError, match=re.escape("model 'eye' accepts only text; bind a model that accepts image (models: {eye:")
    ):
        MAS([looker], llm=seer, default_model="eye", inputs={"png": "image"}, models={"eye": {"accepts": []}})

    # each agent's own model must send parts, in any order, also when two models share the name
    other = Agent("other", "Look again: {png}", "second", ["png"], model="eye")
    sees = {"eye": {"accepts": ["image"]}}
    for agents in ([looker, other], [other, looker]):
        with pytest.raises(ValueError, match="model 'eye' accepts image but cannot send content parts"):
            MAS(
                agents,
                llm=seer,
                llms={"eye": MockLLM(meta=True)},
                default_model="eye",
                inputs={"png": "image"},
                models=sees,
            )

    # copy, rebuild and a chain's system keep the models
    assert declared.copy().models == declared.models
    assert declared.copy().to_dict()["models"] == {"seer": {"accepts": ["image", "audio"]}}
    updater = GraphUpdater(MechanicalSynthesizer(), TopologyBudget(max_agents=2, max_edges=2), seed=0)
    assert updater.rebuild(declared, updater.names_of(declared), updater.edges_of(declared)).models == declared.models
    each: dict[str, Any] = {
        "inputs": {"pngs": "list of image"},
        "each": {"judge": "png in pngs"},
        "outputs": {"verdict": "list of text"},
    }
    chained = MAS([judge], llms={"seer": seer}, models={"seer": {"accepts": ["image"]}}, **each)
    assert chained.checked().chains["judge"].system.models == chained.models
    with pytest.raises(
        ValueError, match="chain 'judge': agent 'judge' reads image 'png', but model 'seer' accepts only"
    ):
        MAS([judge], llms={"seer": seer}, models={"seer": {}}, **each)

    # a step reading a file takes an image
    weighed = MAS([], steps=[Step("weigh", "weigh", {"upload": "png"})], inputs={"png": "image"}, tools=[weigh])
    assert weighed.describe()["nodes"][0]["reads"] == {"png": "image"}
    trace = weighed.run_state({"png": png})
    assert trace.error is None
    assert trace.output == "6"


@tool(writes={"deck": "json"})
def assemble_final(final_draft: list[str]) -> dict[str, Any]:
    """Put the final pages together."""
    return {"pages": final_draft}


HUMANS = """\
version: 2
inputs: {plan: json}
outputs: {deck: json, approval: json, notes: text}
agents:
- {name: drafter, model: writer-model, for each: {item: page, in: plan.slides, key: id}, reads: {slide: page},
   writes: {draft: text}}
- {name: outline, model: small-model, reads: [plan], writes: {outline: text}}
- {name: quick_notes, model: small-model, prompt: notes, reads: [outline], writes: {notes: text}}
- {name: careful_notes, model: writer-model, prompt: notes, reads: [outline], writes: {notes: text}}
steps:
- {name: assemble, uses: assemble_final}
humans:
- {name: fix_title, form: edit, for each: page in plan.slides, reads: [page, draft],
   writes: {answer: title_fix, value: final_draft}, question: "Slide {page.id}: fix the title if it does not say the claim.",
   edits: draft}
- {name: approve, form: approve, reads: [deck, plan], writes: approval, question: "Ship '{plan.title}'?",
   labels: {'yes': "Ship all slides of {plan.title}", 'no': "Hold the deck"}, timeout_s: 86400, default: {approved: false}}
edges:
- drafter -> fix_title -> assemble -> approve
- 'outline -> quick_notes': first of
- 'outline -> careful_notes': first of
callables:
  assemble_final: {kind: python, import: 'tests.test_spec:assemble_final', reads: {final_draft: json}, writes: {deck: json}}
prompts:
  drafter: "Draft slide {slide}."
  outline: "Outline speaker notes for {plan}."
  notes: "Write speaker notes from this outline: {outline}"
"""  # noqa: E501
"""An edit per slide in a chain, a race of two note writers, and an approve gate that times out after a day."""

HUMANS_DATA: dict[str, Any] = yaml.safe_load(HUMANS)


def load_humans(data: dict[str, Any] | None = None) -> MAS:
    return MAS.from_dict(
        copy.deepcopy(HUMANS_DATA if data is None else data),
        llms={"writer-model": MockLLM(), "small-model": MockLLM()},
        tools=[assemble_final],
    )


def test_humans_and_races_round_trip() -> None:
    mas = load_humans()
    assert (mas.version, list(mas.humans), list(mas.nodes)[-1]) == (2, ["fix_title", "approve"], "approve")
    approve, fix_title = mas.humans["approve"], mas.humans["fix_title"]
    assert approve.labels == {"yes": "Ship all slides of {plan.title}", "no": "Hold the deck"}
    assert (approve.answer, approve.written, fix_title.written) == (
        "approval",
        ("approval",),
        ("title_fix", "final_draft"),
    )
    assert mas.each["fix_title"] == ForEach("page", "plan.slides")
    info = mas.checked()
    assert (info.humans, info.races) == (("drafter", "approve"), {"outline": ("quick_notes", "careful_notes")})
    assert info.chains["drafter"].system.checked().humans == ("fix_title",)  # the human inside the item's system
    assert {name: str(info.types[name]) for name in ("title_fix", "final_draft", "approval", "notes")} == {
        "title_fix": "list of json",
        "final_draft": "list of text",
        "approval": "json",
        "notes": "text",
    }
    assert info.joins[END] == Join("join", (("quick_notes", "careful_notes"), ("approve",)), ("__in__.__end__.0", None))
    assert mas.edges[-1] == Edge("outline", "careful_notes", EdgeKind.FIRST)

    spec = mas.to_dict()
    assert spec == HUMANS_DATA
    assert list(spec) == [
        "version", "inputs", "outputs", "agents", "steps", "humans", "edges", "callables", "prompts"
    ]  # fmt: skip
    assert list(spec["humans"][0]) == ["name", "form", "for each", "reads", "writes", "question", "edits"]
    assert list(spec["humans"][1]) == [
        "name", "form", "reads", "writes", "question", "labels", "timeout_s", "default"
    ]  # fmt: skip
    again = load_humans(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec
    assert again.structure == mas.structure  # idempotent
    unquoted = yaml.safe_load(HUMANS.replace("'yes'", "yes").replace("'no'", "no"))  # YAML reads them as booleans
    assert load_humans(unquoted).humans["approve"] == approve
    assert dump_edges(parse_edges(spec["edges"])) == spec["edges"]

    # describe names the human nodes and the race; the structure follows every setting but the question text
    described = mas.describe()
    humans = [node for node in described["nodes"] if node["kind"] == "human"]
    assert humans == [
        {
            "name": "fix_title",
            "kind": "human",
            "form": "edit",
            "reads": {"page": "json", "draft": "text"},
            "writes": {"title_fix": "json", "final_draft": "text"},
            "for each": "page in plan.slides",
        },
        {
            "name": "approve",
            "kind": "human",
            "form": "approve",
            "reads": {"deck": "json", "plan": "json"},
            "writes": {"approval": "json"},
            "timeout_s": 86400,
        },
    ]
    assert [edge["kind"] for edge in described["edges"] if edge["from"] == "outline"] == ["first of", "first of"]
    asked = copy.deepcopy(HUMANS_DATA)
    asked["humans"][1]["question"] = "Ship it?"
    assert load_humans(asked).structure == mas.structure
    assert load_humans(asked).describe()["id"] == described["id"]
    relabelled = copy.deepcopy(HUMANS_DATA)
    relabelled["humans"][1]["labels"]["no"] = "Redraw"
    assert load_humans(relabelled).structure != mas.structure
    shorter = copy.deepcopy(HUMANS_DATA)
    shorter["humans"][1]["timeout_s"] = 60
    assert load_humans(shorter).describe()["id"] != described["id"]

    # the asker and the store are runtime: copied by reference, never in the spec
    asker, store = ReplayAsker([]), Store()
    held = load_humans()
    held.asker, held.store = asker, store
    copied = held.copy()
    assert (copied.asker, copied.store, copied.humans) == (asker, store, held.humans)
    assert copied.to_dict() == spec
    assert held.v2_kwargs()["humans"] == tuple(held.humans.values())
    assert held.runs() is store
    assert isinstance(mas.runs(), Store)
    assert mas.runs() is mas.runs()
    assert held.checked().chains["drafter"].system.runs() is store  # a chain's items keep the outer runs


def test_a_human_makes_a_system_version_2() -> None:
    gate = Human("approve", "approve", ["draft"], "approval", question="Ship this? {draft}")
    writer = Agent("writer", "Write {query}", "draft")
    mas = MAS([writer], humans=[gate], edges=[("writer", "approve")], llm=MockLLM())
    assert (mas.version, mas.checked().humans, mas.describe()["nodes"][1]["kind"]) == (2, ("approve",), "human")
    alone = MAS([], humans=[Human("pick", "ask", ["query"], options=["a", "b"])], outputs={"answer": "json"})
    assert alone.to_dict()["humans"] == [{"name": "pick", "form": "ask", "reads": ["query"], "writes": "answer",
                                          "options": ["a", "b"]}]  # fmt: skip
    with pytest.raises(ValueError, match="humans, inputs, outputs"):
        MAS([writer], humans=[gate], edges=[("writer", "approve")], llm=MockLLM(), version=1)
    with pytest.raises(ValueError, match=r"Node names must be unique: \['writer'\]"):
        MAS([writer], humans=[Human("writer", "approve", ["query"])], llm=MockLLM())
    assert mas.copy().structure == mas.structure
    assert "human approve" in mas.structure
    assert Human("r", "judge", {"left": "a", "right": "b"}, pair=["left", "right"]).to_dict()["reads"] == {
        "left": "a",
        "right": "b",
    }
    # an edit's value has the type of what it edits: an input, or its chain's item
    fix = Human("fix", "edit", ["title"], {"answer": "fix", "value": "final"}, edits="title")
    retitled = MAS([], humans=[fix], inputs={"title": "text"}, outputs={"final": "text"})
    per_item = MAS([], humans=[fix], inputs={"titles": "list of text"}, each={"fix": "title in titles"},
                   outputs={"final": "list of text"})  # fmt: skip
    assert str(retitled.checked().types["final"]) == "text"
    assert per_item.checked().humans == ("fix",)
    assert per_item.checked().chains["fix"].writes == {"fix": ValueType("json"), "final": ValueType("text")}
    refused: list[tuple[dict[str, Any], str]] = [
        ({"name": ""}, "needs a name"),
        ({"reads": "draft"}, "reads is a list of texts"),
        ({"reads": {"d": ""}}, "reads must name values"),
        ({"writes": {"answer": 3}}, "writes is a value name"),
        ({"options": [1, 2]}, "options is a list of texts"),
        ({"pair": "ab"}, "pair is a list of texts"),
        ({"labels": ["yes"]}, "labels is a mapping"),
        ({"edits": 3}, "edits names a read"),
        ({"timeout_s": True}, "timeout_s is a number"),
    ]
    for settings, phrase in refused:
        with pytest.raises(ValueError, match=phrase):
            Human(**{"name": "h", "form": "ask", **settings})


HUMAN_BASE = """\
version: 2
inputs: {plan: json}
outputs: {approval: json}
agents:
- {name: writer, reads: [plan], writes: {draft: text}}
humans:
- {name: approve, form: approve, reads: [draft, plan], writes: approval, question: "Ship {plan.title}?"}
edges:
- writer -> approve
prompts: {writer: "Write {plan}."}
"""
"""A writer and an approve gate; each case of the human check changes the gate."""


def gate(**fields: Any) -> MAS:
    """The base system with the gate's fields replaced, ``None`` dropping one; ``spec`` replaces top-level keys."""
    data: dict[str, Any] = yaml.safe_load(HUMAN_BASE)
    data.update(fields.pop("spec", {}))
    entry = {key: value for key, value in {**data["humans"][0], **fields}.items() if value is not None}
    return MAS.from_dict({**data, "humans": [entry]}, llm=MockLLM())


def routed(when: str) -> dict[str, Any]:
    """The base system's spec with a ``ship`` node the gate routes to when ``when`` holds, else to the end."""
    data: dict[str, Any] = yaml.safe_load(HUMAN_BASE)
    return {
        "agents": [*data["agents"], {"name": "ship", "reads": ["draft"], "writes": {"shipped": "text"}}],
        "outputs": {"approval": "json", "shipped": "text"},
        "edges": ["writer -> approve", {"approve -> ship": {"when": when}}, {f"approve -> {END}": "else"}],
        "prompts": {**data["prompts"], "ship": "Ship {draft}."},
    }


def test_the_gate_checks() -> None:
    assert gate().checked().humans == ("approve",)
    assert gate(form="ask", options="plan.choices").checked().humans == ("approve",)
    timed = gate(timeout_s=0, default=None, spec=routed("approval.approved"))  # the else edge takes a timeout
    assert timed.checked().joins[END].kind == "merge"


def test_a_mapping_write_names_the_answer() -> None:
    typed = gate(writes={"approval": "json"})  # {name: type}, as an agent writes
    assert typed.humans["approve"] == gate().humans["approve"]
    assert typed.checked().humans == ("approve",)
    with pytest.raises(ValueError, match=r"writes is a value name, \{name: json\}"):
        gate(writes={"approval": "text"})


@pytest.mark.parametrize(
    ("build", "phrase"),
    [
        pytest.param(lambda: gate(form="vote"), r"human 'approve': form 'vote' is not one of", id="unknown form"),
        pytest.param(
            lambda: gate(options=["a", "b"]),
            r"human 'approve': options belongs to the ask form, not to approve",
            id="field foreign to the form",
        ),
        pytest.param(lambda: gate(form="ask"), r"an ask node needs options", id="ask without options"),
        pytest.param(
            lambda: gate(form="ask", options="draft"),
            r"human 'approve': options 'draft' reads text 'draft'; options read from state are a list of text",
            id="path option not a list of text",
        ),
        pytest.param(
            lambda: gate(form="ask", options="notes"),
            r"options 'notes' is not the path of a read",
            id="path option not a read",
        ),
        pytest.param(
            lambda: gate(form="judge", pair=["draft", "draft"]),
            r"pair names two different reads of \['draft', 'plan'\], got \['draft', 'draft'\]",
            id="judge pair not two reads",
        ),
        pytest.param(lambda: gate(form="judge"), r"a judge node needs a pair", id="judge without a pair"),
        pytest.param(
            lambda: gate(form="edit", edits="title", writes={"answer": "approval", "value": "final"}),
            r"edits names one of its reads \['draft', 'plan'\], got 'title'",
            id="edit edits not a read",
        ),
        pytest.param(
            lambda: gate(form="edit", edits="draft"),
            r"human 'approve': the edit form writes \['answer', 'value'\], got \['answer'\]",
            id="edit without writes value",
        ),
        pytest.param(
            lambda: gate(question="Ship {deck}?"),
            r"placeholder 'deck' is not among its reads \['draft', 'plan'\]",
            id="template placeholder not read",
        ),
        pytest.param(
            lambda: gate(question="Ship {draft.title}?"),
            r"placeholder 'draft.title' reads a field of text 'draft'",
            id="field of a text",
        ),
        pytest.param(
            lambda: gate(labels={"maybe": "Later"}),
            r"labels sets 'maybe'; an approve node labels 'yes' and 'no'",
            id="label key not yes or no",
        ),
        pytest.param(
            lambda: gate(timeout_s=60),
            r"human 'approve' times out after 60 s with no default; give a default or else edge",
            id="timeout without default",
        ),
        pytest.param(
            lambda: gate(default={"pick": "draft"}),
            r"default sets \['pick'\]; the approve form's answer takes \['approved', 'option', 'comment'\]",
            id="default field foreign to the form",
        ),
        pytest.param(
            lambda: gate(default={"approved": "no"}),
            r"default approved is true or false, got 'no'",
            id="default of the wrong type",
        ),
        pytest.param(
            lambda: gate(timeout_s=-1, default={"approved": False}),
            r"timeout_s is -1; give 0 or more seconds",
            id="negative timeout",
        ),
        pytest.param(
            lambda: MAS(
                [Agent("writer", "Write {query}", "draft")],
                [("writer", "approve")],
                humans=[Human("approve", "approve", ["draft"])],
                llm=MockLLM(),
                version=1,
            ),
            r"Steps, boxes, humans, inputs, outputs",
            id="human in a version 1 system",
        ),
        pytest.param(lambda: gate(writes="len"), r"'approve' writes 'len', which is a reserved name", id="reserved"),
        pytest.param(
            lambda: gate(spec={"outputs": {"approval": "text"}}),
            r"output 'approval' is declared text but is json",
            id="writes typed other than json",
        ),
        pytest.param(
            lambda: gate(reads=["draft", "plan", "run"]),
            r"human 'approve' reads 'run'; a human reads values nodes write, not 'error' or 'run'",
            id="run read",
        ),
        pytest.param(
            lambda: gate(reads=["draft", "plan", "notes"]),
            r"'approve' reads 'notes', which is neither an input nor written by any node",
            id="read not written",
        ),
        pytest.param(
            lambda: gate(spec=routed("approval.yes")),
            r"edge approve -> ship reads approval.yes, but 'approve''s answer has the fields \['approved', 'option'",
            id="routing on a field the answer lacks",
        ),
        pytest.param(
            lambda: MAS.from_dict(
                {**yaml.safe_load(HUMAN_BASE), "humans": [{"name": "approve", "form": "approve", "writes": 3}]},
                llm=MockLLM(),
            ),
            r"Human 'approve' needs 'writes' to be a string or a mapping, got 3",
            id="spec shape",
        ),
        pytest.param(
            lambda: MAS(
                [Agent("writer", "Write {plan}", "draft", inputs=["plan"])],
                [("writer", "approve")],
                humans=[Human("approve", "approve", ["draft"], "approval")],
                inputs={"plan": "json"},
                outputs={"approval": "json"},
                retry={"approve": {"outage": 1}},
                llm=MockLLM(),
            ),
            r"human 'approve' takes no retry",
            id="retry on a human",
        ),
        pytest.param(
            lambda: load_humans(
                {
                    **HUMANS_DATA,
                    "humans": [
                        {**HUMANS_DATA["humans"][0], "question": "Slide {page.id}: {title}"},
                        HUMANS_DATA["humans"][1],
                    ],
                }
            ),
            r"chain 'drafter': human 'fix_title': placeholder 'title' is not among its reads",
            id="a human inside a chain",
        ),
    ],
)
def test_check_names_every_human_problem(build: Any, phrase: str) -> None:
    with pytest.raises(ValueError, match=phrase):
        build()


RACE = """\
version: 2
outputs: {notes: text}
agents:
- {name: outline, writes: {outline: text}}
- {name: quick, prompt: notes, reads: [outline], writes: {notes: text}}
- {name: careful, prompt: notes, reads: [outline], writes: {notes: text}}
edges:
- 'outline -> quick': first of
- 'outline -> careful': first of
prompts: {outline: "Outline {query}.", notes: "Notes from {outline}."}
"""
"""A race of two note writers; each case of the race check changes it."""


def race(*agents: dict[str, Any], edges: list[Any] | None = None, **spec: Any) -> MAS:
    """The race with agents added, its edges replaced and top-level keys set."""
    data: dict[str, Any] = yaml.safe_load(RACE)
    data["agents"] += agents
    data["edges"] = data["edges"] if edges is None else edges
    data["prompts"].update({agent["name"]: f"Do {{{agent['reads'][0]}}}." for agent in agents if "prompt" not in agent})
    return MAS.from_dict({**data, **spec}, llm=MockLLM())


FIRST = [{"outline -> quick": "first of"}, {"outline -> careful": "first of"}]


def test_a_race_checks() -> None:
    info = race().checked()
    assert info.races == {"outline": ("quick", "careful")}
    assert info.writers["notes"] == ("quick", "careful")
    assert info.joins[END].kind == "merge"  # the winner's writes arrive
    polish = {"name": "polish", "reads": ["notes"], "writes": {"final": "text"}}
    joined = race(polish, edges=[*FIRST, "quick -> polish", "careful -> polish"], outputs={"final": "text"})
    assert joined.checked().joins["polish"].kind == "merge"


@pytest.mark.parametrize(
    ("build", "phrase"),
    [
        pytest.param(
            lambda: race(edges=[FIRST[0], "outline -> careful"], state={"notes": {"merge": "last"}}),
            r"the first of race from 'outline' has one contestant, 'quick'; a race needs two or more",
            id="one contestant",
        ),
        pytest.param(
            lambda: MAS.from_dict(
                {
                    **yaml.safe_load(RACE),
                    "agents": [
                        {"name": "outline", "writes": {"outline": "text"}},
                        {"name": "quick", "prompt": "notes", "reads": ["outline"], "writes": {"notes": "text"}},
                        {"name": "careful", "prompt": "notes", "reads": ["outline"], "writes": {"long_notes": "text"}},
                    ],
                },
                llm=MockLLM(),
            ),
            r"its contestants write differently \{'quick': \{'notes': 'text'\}, 'careful': \{'long_notes': 'text'\}\}",
            id="unequal writes",
        ),
        pytest.param(
            lambda: MAS(
                [
                    Agent("outline", "Outline {plan}", "outline", inputs=["plan"]),
                    Agent("quick", "Notes {outline}", "notes", inputs=["outline"]),
                    Agent("careful", "Notes {outline} {page}", "notes", inputs=["outline", "page"]),
                ],
                [Edge("outline", "quick", EdgeKind.FIRST), Edge("outline", "careful", EdgeKind.FIRST)],
                inputs={"plan": "json"},
                outputs={"notes": "text"},
                each={"careful": "page in plan"},
                llm=MockLLM(),
            ),
            r"'careful' is a chain head; agents, steps and boxes race",
            id="chain head contestant",
        ),
        pytest.param(
            lambda: MAS(
                [
                    Agent("outline", "Outline {query}", "outline"),
                    Agent("quick", "Notes {outline}", "notes", ["outline"]),
                ],
                [Edge("outline", "quick", EdgeKind.FIRST), Edge("outline", "careful", EdgeKind.FIRST)],
                humans=[Human("careful", "approve", ["outline"], "verdict")],
                outputs={"notes": "text"},
                llm=MockLLM(),
            ),
            r"'careful' is a human node; agents, steps and boxes race",
            id="human contestant",
        ),
        pytest.param(
            lambda: MAS(
                [
                    Agent("outline", "Outline {query}", "outline"),
                    Agent("quick", "Notes {outline}", "notes", ["outline"]),
                ],
                [Edge("outline", "quick", EdgeKind.FIRST), Edge("outline", "careful", EdgeKind.FIRST)],
                boxes=[
                    Box(
                        "careful",
                        MAS(
                            [Agent("w", "Notes {topic}", "draft", ["topic"])],
                            [("w", "ok")],
                            humans=[Human("ok", "edit", ["draft"], {"answer": "okd", "value": "notes"}, edits="draft")],
                            inputs={"topic": "text"},
                            outputs={"notes": "text"},
                            llm=MockLLM(),
                        ),
                        reads={"topic": "outline"},
                        writes={"notes": "notes"},
                    )
                ],
                outputs={"notes": "text"},
                llm=MockLLM(),
            ),
            r"box 'careful' holds a human node; a contestant never waits for a person",
            id="box holding a human contestant",
        ),
        pytest.param(
            lambda: race(
                {"name": "fallback", "reads": ["outline"], "writes": {"apology": "text"}},
                edges=[*FIRST, {"careful -> fallback": {"on error": "any"}}],
                outputs={"notes": "text", "apology": "text"},
            ),
            r"'careful' has an on error edge; a contestant's failure is the race's",
            id="error edge on a contestant",
        ),
        pytest.param(
            lambda: race(
                {"name": "a", "reads": ["notes"], "writes": {"x": "text"}},
                {"name": "b", "reads": ["notes"], "writes": {"y": "text"}},
                edges=[*FIRST, "careful -> a", "careful -> b", "quick -> a"],
                outputs={"x": "text", "y": "text"},
            ),
            r"'careful' has 2 out-edges; a contestant has one plain edge or none",
            id="two out-edges",
        ),
        pytest.param(
            lambda: race(
                {"name": "polish", "reads": ["notes"], "writes": {"final": "text"}},
                edges=[*FIRST, "quick -> polish"],
                outputs={"final": "text", "notes": "text"},
            ),
            r"its contestants go on to different nodes \{'quick': 'polish', 'careful': '__end__'\}",
            id="differing next nodes",
        ),
        pytest.param(
            lambda: race(
                {"name": "second", "reads": ["query"], "writes": {"other": "text"}},
                {"name": "third", "prompt": "notes", "reads": ["outline"], "writes": {"notes": "text"}},
                edges=[*FIRST, "outline -> second", {"second -> careful": "first of"}, {"second -> third": "first of"}],
                outputs={"notes": "text"},
                state={"notes": {"merge": "last"}},
            ),
            r"'careful' runs in 2 first of races; a node races in one",
            id="a contestant in two races",
        ),
        pytest.param(
            lambda: race(
                {"name": "skip", "reads": ["outline"], "writes": {"notes": "text"}},
                edges=[*FIRST, {"outline -> skip": {"when": "true"}}, {"outline -> END": "else"}],
            ),
            r"'outline' also decides with when, switch or else edges; race from a node of its own",
            id="a deciding race source",
        ),
    ],
)
def test_check_names_every_race_problem(build: Any, phrase: str) -> None:
    with pytest.raises(ValueError, match=phrase):
        build()


NOW = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)
GATE_PLAN = {"title": "Deck", "choices": ["short", {"label": "long"}]}


LATER = NOW + timedelta(seconds=30)
"""When the approval gate's question is answered, half way to its timeout."""
GATE_TYPES = {"deck": ValueType("json"), "plan": ValueType("json")}


def approval_human() -> Human:
    """A gate that asks to ship the deck, times out after a minute and then rejects."""
    return Human(
        "approve", "approve", ["deck", "plan"], "approval", question="Ship '{plan.title}'?",
        labels={"yes": "Ship {plan.title}"}, timeout_s=60, default={"approved": False},
    )  # fmt: skip


def approval_question() -> Question:
    """The gate's first question of run ``r1``."""
    return approval_human().ask({"deck": {"pages": 2}, "plan": GATE_PLAN}, GATE_TYPES, run_id="r1", visit=1, now=NOW)


def pick_question() -> Question:
    """A question whose options come from the plan's choices, for item ``intro`` of a chain."""
    pick = Human("pick", "ask", ["plan"], options="plan.choices", question="Which?")
    return pick.ask({"plan": GATE_PLAN}, GATE_TYPES, run_id="r1", visit=1, now=NOW, item=0, item_key="intro")


def test_a_human_asks_a_question_without_a_run() -> None:
    types = GATE_TYPES
    question = approval_question()
    assert (question.id, question.text, question.options) == ("r1/approve/1", "Ship 'Deck'?", ("Ship Deck", "Reject"))
    assert question.shown == {"deck": '{"pages": 2}', "plan": json.dumps(GATE_PLAN)}
    assert question.deadline == (NOW + timedelta(seconds=60)).isoformat()
    assert not question.due(NOW)
    assert question.due(NOW + timedelta(seconds=60))
    assert Question.from_dict(json.loads(json.dumps(question.to_dict()))) == question
    assert Pause("r1", (question,)).to_dict() == {"run_id": "r1", "questions": [question.to_dict()]}
    assert question_id("r1", "fix", 2, item=3) == "r1/fix/3/2"
    assert question_id("r1", "fix", 1, 0, "intro") == "r1/fix/intro/1"
    assert fill("{plan.title} {plan.missing} {other} {{plan}}", {"plan": GATE_PLAN}, types) == "Deck  {other} {{plan}}"


def test_an_answer_is_checked_and_completed() -> None:
    approve, question = approval_human(), approval_question()
    answer = answer_value(question, {"approved": True, "who": "tom"}, now=LATER)
    assert answer == {
        "approved": True,
        "option": "Ship Deck",
        "who": "tom",
        "asked_at": NOW.isoformat(),
        "answered_at": LATER.isoformat(),
        "comment": None,
        "timed_out": False,
    }
    assert answer_value(question, {"option": "Reject", "who": "tom"}, now=LATER)["approved"] is False
    timed_out = answer_value(question, question.default or {}, now=LATER, timed_out=True)
    assert (timed_out["approved"], timed_out["option"], timed_out["who"], timed_out["timed_out"]) == (
        False, "Reject", None, True
    )  # fmt: skip
    assert approve.state_writes(answer) == {"approval": answer}
    with pytest.raises(ValueError, match="'extra' is not a field") as info:
        answer_value(question, {"approved": "yes", "option": "Ship", "extra": 1}, now=LATER)
    for phrase in ("'extra' is not a field", "approved is true or false", "who answered is missing"):
        assert phrase in str(info.value)

    with pytest.raises(
        ValueError, match="comment is a text, got 3; option 'Reject' is not the label of the choice, 'Ship Deck'"
    ):
        answer_value(question, {"approved": True, "option": "Reject", "who": "tom", "comment": 3}, now=LATER)


def test_options_come_from_the_labels_or_from_state() -> None:
    types = GATE_TYPES
    literal = Human("pick", "ask", ["plan"], options=["{plan.title} as it is", "redo"], question="{plan}")
    shown = literal.ask({"plan": GATE_PLAN}, types, run_id="r1", visit=1, now=NOW)
    assert (shown.text, shown.options) == (json.dumps(GATE_PLAN), ("Deck as it is", "redo"))
    pick = Human("pick", "ask", ["plan"], options="plan.choices", question="Which?")
    asked = pick_question()
    assert (asked.id, asked.options, asked.deadline) == ("r1/pick/intro/1", ("short", "long"), None)
    assert answer_value(asked, {"option": "long", "who": "ann"}, now=NOW)["option"] == "long"
    with pytest.raises(ValueError, match=r"option 'medium' is not one of \['short', 'long'\]"):
        answer_value(asked, {"option": "medium", "who": "ann"}, now=NOW)
    with pytest.raises(ValueError, match="option is missing"):
        answer_value(asked, {"who": "ann"}, now=NOW)
    with pytest.raises(ValueError, match="must be a list of text"):
        pick.ask({"plan": {"choices": "short"}}, types, run_id="r1", visit=1, now=NOW)


def test_a_judge_and_an_edit_node_answer() -> None:
    judge = Human("judge", "judge", {"left": "a", "right": "b"}, pair=["left", "right"])
    duel = judge.ask({"a": "one", "b": "two"}, {}, run_id="r1", visit=2, now=NOW)
    assert duel.options == ("left", "right", "tie")
    assert duel.shown == {"left": "one", "right": "two"}
    assert answer_value(duel, {"pick": "tie", "who": "tom", "comment": "both fine"}, now=NOW)["comment"] == "both fine"

    edit = Human("fix", "edit", ["draft"], {"answer": "fix", "value": "final"}, edits="draft")
    page = edit.ask({"draft": "Old title"}, {}, run_id="r1", visit=1, now=NOW)
    changed = answer_value(page, {"value": "New title", "who": "tom"}, now=NOW)
    same = answer_value(page, {"value": "Old title", "who": "tom"}, now=NOW)
    kept = answer_value(page, {"who": "tom"}, now=NOW)
    assert (changed["changed"], same["changed"], kept["changed"], "value" in kept) == (True, False, False, False)
    assert edit.state_writes(changed) == {
        "fix": {k: v for k, v in changed.items() if k != "value"},
        "final": "New title",
    }
    assert edit.state_writes(kept, "Old title")["final"] == "Old title"
    with pytest.raises(ValueError, match="changed is true, but the edited value is missing"):
        answer_value(page, {"changed": True, "who": "tom"}, now=NOW)
    typed = edit.ask({"draft": "Old title"}, {"final": ValueType("text")}, run_id="r1", visit=1, now=NOW)
    assert (page.value_type, typed.value_type) == (None, "text")
    assert Question.from_dict({k: v for k, v in typed.to_dict().items() if k != "value_type"}).value_type is None
    with pytest.raises(ValueError, match="the edited value is not text: expected text, got dict"):
        answer_value(typed, {"value": {"title": "New"}, "who": "tom"}, now=NOW)


def test_a_host_answered_at_is_a_zoned_time_not_before_the_question() -> None:
    """A host's answered_at is an ISO time with its zone, not before the question."""
    question = approval_question()
    given_at = {"approved": True, "who": "tom", "answered_at": LATER.isoformat()}
    assert answer_value(question, given_at, now=NOW)["answered_at"] == LATER.isoformat()
    for stamp, problem in (
        ("yesterday", "answered_at is an ISO time with its zone"),
        ("2026-10-24T09:00:30", "answered_at is an ISO time with its zone"),
        (5, "answered_at is an ISO time with its zone"),
        ((NOW - timedelta(seconds=1)).isoformat(), "comes before the question was asked"),
    ):
        with pytest.raises(ValueError, match=problem):
            answer_value(question, {"approved": True, "who": "tom", "answered_at": stamp}, now=LATER)
    assert set(ANSWER_FIELDS) == set(FORMS)


def test_an_optimizer_replays_recorded_answers_and_asks_nobody() -> None:
    question, asked = approval_question(), pick_question()
    answer = answer_value(question, {"approved": True, "who": "tom"}, now=LATER)
    timed_out = answer_value(question, question.default or {}, now=LATER, timed_out=True)
    record = AgentRecord(
        "approve", 0, "", "", {}, {}, kind="human", raw_response=json.dumps(answer), visit=1, wait_s=30.0
    )
    unanswered = AgentRecord("approve", 1, "", "", {}, {}, kind="human", raw_response=json.dumps(timed_out), visit=2)
    replay = ReplayAsker(Trace("q", "", (record, unanswered)))
    given = {"approved": True, "option": "Ship Deck", "who": "tom", "comment": None}
    assert replay.ask(question) == given  # answered when asked again: the recorded time stays behind
    assert replay.ask(Question(**{**question.to_dict(), "visit": 2, "options": ("a", "b")})) is None  # a default
    assert replay.ask(asked) is None
    assert answer_value(question, replay.ask(question) or {}, now=LATER) == answer


def test_records_events_and_health_carry_the_waits() -> None:
    answer = answer_value(approval_question(), {"approved": True, "who": "tom"}, now=LATER)
    record = AgentRecord(
        "approve", 0, "", "", {}, {}, kind="human", raw_response=json.dumps(answer), visit=1, wait_s=30.0
    )
    assert record == AgentRecord("approve", 0, "", "", {}, {}, kind="human", raw_response=json.dumps(answer), visit=1)
    assert record.to_dict()["wait_s"] == 30.0
    assert "lost" not in record.to_dict()
    lost = AgentRecord("quick", 2, "p", "p", {}, {}, lost=True)
    assert lost.to_dict()["lost"] is True
    assert summarize([record, lost]).budget == {"waited_s": {"spent": 30.0}}
    assert summarize([lost]).budget == {}
    events = [
        Event("pause", "approve", "r1", 0.1, question="r1/approve/1"),
        Event("resume", "approve", "r1", 30.1, question="r1/approve/1", wait_s=30.0),
        Event("stop", "", "r1", 31.0, reason="stopped"),
        Event("end", "quick", "r1", 1.0, status="lost"),
    ]
    assert [event.to_dict() for event in events][2] == {"kind": "stop", "node": "", "run_id": "r1", "at_s": 31.0,
                                                        "reason": "stopped"}  # fmt: skip
    assert Trace("q", "").pause is None
    assert "pause" not in Trace("q", "").to_dict()


def test_a_paused_trace_says_so() -> None:
    question = Human("a", "approve", ["query"]).ask({"query": "q"}, {}, run_id="r", visit=1, now=NOW)
    paused = Trace("q", "", pause=Pause("r", (question,)))
    assert paused.status is RunStatus.PAUSED
    assert paused == Trace("q", "")
    assert paused.to_dict()["status"] == "paused"
    assert paused.to_dict()["pause"]["questions"][0]["id"] == "r/a/1"
