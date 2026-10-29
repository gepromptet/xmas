"""Verify: checksums against the lock, the dry check, warm-up examples under a budget, and cases from traces.

Every model is scripted, and every call costs 0.01 USD unless its model reports
no cost. Bundles are saved with ``MAS.save`` into temporary folders.
"""

from __future__ import annotations

import dataclasses
import json
import re
import warnings
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import xmas.verify
from tests.conftest import (
    DECK,
    LONG_PAGE,
    MEDIA,
    PLAN,
    PNG,
    MockLLM,
    SeeingLLM,
    assemble_deck,
    deck,
    draft,
    judge,
    render_page,
    render_pages,
)
from xmas import MAS, Agent, Budget, ErrorKind, HumanAction, MASModule
from xmas.bundle import Lock
from xmas.config import Config
from xmas.convert import document_text
from xmas.examples import Check, parse_examples, run_checks, save_examples
from xmas.runtime import run_node
from xmas.trace import AgentRecord
from xmas.verify import VerifyFailed, VerifyReport, VerifyWarning, verify

NAMES = Config(catalogue=("writer-model", "judge-model", "vision-model"))
"""A configured catalogue holding every model name the test systems bind."""


def examples_file(folder: Path, *cases: dict[str, Any], **top: Any) -> Path:
    """Write ``folder/examples/warmup.yaml`` with ``cases``; ``top`` adds keys such as ``budget``."""
    path = folder / "examples" / "warmup.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"version": 1, **top, "examples": list(cases)}, sort_keys=False), encoding="utf-8")
    return path


def calls(mas: MAS) -> list[list[str]]:
    """Return the prompts each of a system's scripted models was sent."""
    return [cast("MockLLM", llm).calls for llm in mas.llms.values()]


def check(mas: MAS, level: int, *, config: Config = NAMES, **settings: Any) -> VerifyReport:
    """Verify a system that was never saved: level 0 finds no lock, so the report warns."""
    with pytest.warns(VerifyWarning, match="level 0: not saved: no lock"):
        return verify(mas, level, config=config, **settings)


def gated() -> MAS:
    """A writer, then a person who approves the draft; nobody answers."""
    return MAS(
        [Agent("writer", "Write {query}", "draft")],
        [("writer", "approve")],
        human_actions=[HumanAction("approve", "approve", ["draft"], "approval", question="Ship {draft}?")],
        outputs={"approval": "json"},
        llm=MockLLM(meta=True),
    )


CHECKER: dict[str, Any] = {
    "name": "checker-sees-the-overflow",
    "node": "checker",
    "inputs": {"page": {"id": "intro", "claim": "the intro slide"}, "measure": {"words": 13, "overflow": True}},
    "recorded": {"fits": False, "fix": "cut it"},
    "must": [{"form": "valid"}, {"path": "verdict.fits", "equals": False}, {"path": "verdict.fix", "not empty": True}],
}
RENDER: dict[str, Any] = {
    "name": "render-measures",
    "node": "render",
    "inputs": {"draft": LONG_PAGE},
    "recorded": {"words": 13, "overflow": True},
}
SMALL = {"slides": [{"id": "a", "claim": "x"}], "figures": [{"id": "f1", "shows": "the setup"}]}
RUN: dict[str, Any] = {
    "name": "the-whole-deck",
    "inputs": {"plan": SMALL, "house": "plain"},
    "expect": {
        "deck": [{"path": "deck.fits", "len": 1}, {"path": "deck.fits[0]", "equals": True}],
        "render": [{"equals": {"words": 2, "overflow": False}}],
        "noter": [{"equals": "Notes."}],
    },
}


BAD_FILES = [
    ("- a list\n", "an examples file is a mapping with version, budget, examples"),
    ("version: 2\nexamples: []\n", "version is 1, got 2"),
    ("version: 1\nexamples: {}\n", "examples is a list of cases"),
    (
        "version: 1\ncorpus: /x\nexamples: []\n",
        "the file takes version, budget, examples; unknown: corpus (a note goes",
    ),
    ("version: 1\nbudget: 0.03\nexamples: []\n", "budget is a mapping such as {cost_usd: 0.03}, got 0.03"),
    ("version: 1\nbudget: {usd: 1}\nexamples: []\n", "budget takes cost_usd, wall_s, per example; unknown: usd"),
    ("version: 1\nbudget: {per example: 1}\nexamples: []\n", "budget.per example is a mapping"),
    ("version: 1\nbudget: {cost_usd: -1}\nexamples: []\n", "Budget.max_cost_usd must be >= 0, got -1"),
    ("version: 1\nbudget: {cost_usd: x}\nexamples: []\n", "not supported between instances of 'str' and 'int'"),
    ("version: 1\nexamples: [x]\n", "example 1: a case is a mapping"),
    ("version: 1\nexamples: [{name: 'a b'}]\n", "example 1: name is a word of letters"),
    ("version: 1\nexamples: [{name: a, node: 1}]\n", "example 'a': node is a node's name, got 1"),
    ("version: 1\nexamples: [{name: a, inputs: [1]}]\n", "example 'a': inputs map value names to values"),
    ("version: 1\nexamples: [{name: a, from: 1}]\n", "example 'a': from is a text"),
    ("version: 1\nexamples: [{name: a, recorded: 1}]\n", "example 'a': recorded belongs to a node case"),
    ("version: 1\nexamples: [{name: a, node: b, expect: {}}]\n", "example 'a': expect belongs to a run case"),
    ("version: 1\nexamples: [{name: a, node: b, must: {}}]\n", "example 'a': must is a list of checks"),
    ("version: 1\nexamples: [{name: a, expect: {x: 1}}]\n", "example 'a': expect maps an output or a node to a list"),
    ("version: 1\nexamples: [{name: a, node: b, must: [1]}]\n", "example 'a': must 1: a check is a mapping"),
    ("version: 1\nexamples: [{name: a, node: b, must: [{near: 1, within: -1}]}]\n", "within is a number of at least 0"),
    ("version: 1\nexamples: [{name: a, node: b, must: [{len: x}]}]\n", "must 1: len takes a count, got 'x'"),
    (
        "version: 1\nexamples: [{name: a, node: b, must: [{matches: '('}]}]\n",
        "must 1: matches takes a regular expression (",
    ),
    (
        "version: 1\nexamples: [{name: a, inputs: {p: {location: /x.png, media_type: image/png}}}]\n",
        "not a path inside",
    ),
    (
        "version: 1\nexamples: [{name: a, inputs: {p: {location: examples/media/b.png, media_type: image/png}}}]\n",
        "example 'a': media examples/media/b.png is not in the bundle",
    ),
    (
        (
            "version: 1\nexamples: [{name: a, inputs: {p: {location: examples/media/a.png, media_type: image/png,"
            " sha256: f0}}}]\n"
        ),
        "example 'a': media examples/media/a.png changed since the case was written (sha256)",
    ),
    ("version: 1\nexamples: [{name: a}, {name: a}]\n", "example 'a' is named in examples/warmup.yaml too"),
]
"""Examples files that do not read, with the problem each names."""


def test_level_0_unchanged_since_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = deck()
    folder = d.save(tmp_path / "deck.mas")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = verify(d, 0)
    lock = Lock.from_yaml((folder / "xmas.lock").read_text(encoding="utf-8"))
    assert report.ok
    assert report.lines() == [
        f"level 0: unchanged since saved (bundle {lock.bundle[:16]}, {len(lock.files)} files)",
        "ok: level 0",
    ]
    assert report.to_dict()["lock"] == {"changed": [], "added": [], "missing": [], "code_changed": []}

    (folder / "prompts" / "noter.md").write_text("Write notes, by hand: {plan}\n", encoding="utf-8")
    (folder / "prompts" / "captioner.md").unlink()
    examples_file(folder)
    monkeypatch.setattr(xmas.verify, "code_hashes", lambda mas: {**lock.code, "render_page": "b"})
    with pytest.warns(VerifyWarning, match="level 0 differs"):
        report = verify(d, 0)
    assert report.lines() == [
        "level 0: changed: prompts/noter.md",
        "level 0: added: examples/warmup.yaml",
        "level 0: missing: prompts/captioner.md",
        "level 0: code changed: render_page",
        "level 0: in memory, not saved: prompts/captioner.md, prompts/noter.md",
        "not ok: level 0 differs",
    ]
    monkeypatch.undo()

    edited = deck()
    edited.save(tmp_path / "edited.mas")
    edited.agents["noter"].prompt = "Write short notes for this plan: {plan}"
    with pytest.warns(VerifyWarning):
        report = verify(edited, 0)
    assert report.lock is not None
    assert report.lock.clean
    assert report.unsaved == ("prompts/noter.md",)

    with pytest.warns(VerifyWarning, match="not saved: no lock"):
        report = verify(deck(), 0)
    assert (report.lock, report.lines()) == (None, ["level 0: not saved: no lock", "not ok: level 0 differs"])
    (tmp_path / "edited.mas" / "xmas.lock").write_text("files: [\n", encoding="utf-8")  # a lock edited by hand
    with pytest.warns(VerifyWarning):
        report = verify(edited, 0)
    assert report.lines()[0] == "level 0: changed: xmas.lock"
    assert not report.ok
    with pytest.raises(ValueError, match="levels are 0, 1 and 2, got 3"):
        verify(deck(), 3)


def test_level_1_is_dry(tmp_path: Path, png: Path) -> None:
    d = deck()
    report = check(d, 1)
    assert report.problems == ()
    assert report.lines()[1] == "level 1: no problems; model names checked against the configured catalogue"
    assert check(d, 1, config=Config()).lines()[1] == "level 1: no problems; model names not checked (no endpoint)"

    seer = SeeingLLM()
    media = MAS.from_dict(yaml.safe_load(MEDIA), llms={"vision-model": seer}, tools=[document_text, render_pages])
    assert check(media, 1).problems == ()  # the pages render as pictures, the house rules as text

    d.agents["noter"].prompt = "Write speaker notes for this plan: {plan}, and for {nowhere}"
    d.agents["checker"].schema = {"fits": {"type": "maybe"}}
    file = examples_file(
        tmp_path,
        {"name": "lost", "node": "nobody", "inputs": {}},
        {"name": "mistyped", "node": "render", "inputs": {"draft": 3}},
        {"name": "bad-check", "node": "render", "inputs": {"draft": "x"}, "must": [{"looks": "fine"}]},
    )
    report = check(d, 1, examples=file, config=Config(catalogue=("writer-model-2", "judge-model")))
    lines = list(report.problems)
    assert len(lines) == 6, lines
    assert lines[0] == "agent 'noter': prompt placeholders ['nowhere'] are not among its reads ['plan']"
    assert lines[1].startswith("chain 'drafter': Answer form field 'fits': unknown type 'maybe'")  # named once
    assert lines[2].startswith(
        "examples/warmup.yaml: example 'bad-check': must 1: unknown check 'looks'; the kinds are"
    )
    assert lines[3].startswith("example 'lost': 'nobody' is no node of the system; its nodes: ['assemble', 'captioner'")
    assert lines[4] == "example 'mistyped': read 'draft' is not text: expected text, got int"
    assert lines[5] == (
        "model 'writer-model' (agents drafter, captioner, noter) is not in the configured catalogue "
        "(XMAS_MODELS or endpoint.catalogue); closest: writer-model-2, judge-model"
    )
    assert report.misses == {"writer-model": ("writer-model-2", "judge-model")}
    assert report.lines()[1] == "level 1: 6 problems; model names checked against the configured catalogue"
    assert calls(d) == [[], []]
    assert seer.calls == []
    assert seer.parts == []

    for text, problem in BAD_FILES:
        with pytest.raises(ValueError, match=f"^examples/warmup.yaml: .*{re.escape(problem)}"):
            parse_examples({"examples/warmup.yaml": text.encode(), "examples/media/a.png": PNG})
    asked = gated()
    people = examples_file(
        tmp_path / "people",
        {"name": "ask", "node": "approve", "inputs": {"draft": "x"}},
        {
            "name": "run",
            "inputs": {"query": "x"},
            "expect": {
                "nobody": [{"exists": True}],
                "approval": [{"form": "valid"}],
                "approve": [{"path": "approval.approved", "exists": True}],
            },
        },
    )
    assert check(asked, 1, examples=people).problems == (
        "example 'ask': node 'approve' is a human action: it asks a person and takes no example",
        "example 'run': expect names 'nobody', which is no output or node of the system",
        "example 'run': form valid: form judges a node's answer, and 'approval' is an output",
    )
    plain = MAS([Agent("solver", "Solve {query}", "answer")], llm=MockLLM())
    assert check(plain, 1).lines()[1] == "level 1: no problems; model names not checked (no model is named)"
    with pytest.raises(ValueError, match="no examples at"):
        verify(plain, 1, examples=tmp_path / "nowhere")
    linked = tmp_path / "linked"
    (linked / "examples" / "media").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("not for a model", encoding="utf-8")
    (linked / "examples" / "media" / "a.png").symlink_to(secret)
    with pytest.raises(ValueError, match=r"a.png is or sits in a symlink; examples are plain files"):
        verify(plain, 1, examples=linked)
    case = (
        "version: 1\nexamples:\n- {name: c, inputs: {query: {location: examples/media/a.png, media_type: image/png}}}\n"
    )
    with pytest.raises(ValueError, match=r"media examples/media/a.png is or sits in a symlink"):
        parse_examples({"examples/warmup.yaml": case.encode()}, linked)


def test_level_2_node_cases(tmp_path: Path) -> None:
    d = deck()
    graph = d.run_state({"plan": PLAN, "house": "plain"})
    first = next(r for r in graph.records if r.agent == "drafter" and r.item == 0 and r.visit == 1)
    alone = run_node(d, "drafter", {"page": PLAN["slides"][0], "house": "plain"})  # no verdict yet, as at that visit
    assert alone.records[0].rendered_prompt == first.rendered_prompt
    assert (alone.outputs, alone.output, alone.cost_usd) == ({"draft": LONG_PAGE}, LONG_PAGE, 0.01)

    file = examples_file(
        tmp_path,
        CHECKER,
        {**CHECKER, "name": "checker-wrong", "must": [{"path": "verdict.fits", "equals": True}]},
        RENDER,
        {**RENDER, "name": "render-wrong", "recorded": {"words": 12, "overflow": True}},
    )
    report = check(d, 2, examples=file)
    assert [(r.name, r.node, r.status, r.cost_usd) for r in report.examples] == [
        ("checker-sees-the-overflow", "checker", "pass", 0.01),
        ("checker-wrong", "checker", "fail", 0.01),
        ("render-measures", "render", "pass", 0.0),
        ("render-wrong", "render", "fail", 0.0),
    ]
    rows = {r.name: r for r in report.examples}
    assert rows["checker-wrong"].failures == ("verdict.fits equals true: got false",)
    assert rows["render-wrong"].failures == (
        'equals {"words": 12, "overflow": true}: got {"words": 13, "overflow": true}',
    )
    assert all(r.latency_s > 0 for r in report.examples)
    assert report.cost_usd == pytest.approx(0.02)
    assert report.lines()[2].startswith("level 2: checker-sees-the-overflow  pass  0.0100 USD  ")
    assert report.lines()[4] == "  - verdict.fits equals true: got false"
    assert report.lines()[-1] == "not ok: level 0 differs; 2 examples passed, 2 fail, 0.0200 USD"

    with pytest.raises(ValueError, match=r"- 'page' is no read of 'render', which reads \['draft'\]"):
        run_node(d, "render", {"page": "x"})
    stubborn = MAS.from_dict(
        yaml.safe_load(DECK),
        llms={"writer-model": MockLLM(draft, meta=True), "judge-model": MockLLM(lambda p: "no JSON", meta=True)},
        tools=[render_page, assemble_deck],
    )
    failed = run_node(stubborn, "checker", CHECKER["inputs"])
    assert (failed.error_kind, failed.cost_usd, len(failed.records)) == (ErrorKind.BAD_FORMAT, 0.02, 1)

    inner = MAS(
        [Agent("cutter", "Cut {page}", output="short", inputs=["page"])], inputs={"page": "text"}, llm=MockLLM()
    )
    boxed = MAS(
        [Agent("writer", "Write about {query}", output="draft")],
        [("writer", "shorten")],
        modules=[MASModule("shorten", inner, reads={"page": "draft"}, writes={"short": "short"})],
        llm=MockLLM(),
    )
    with pytest.raises(
        ValueError, match="node 'cutter' sits inside module 'shorten': give the example to the module's system"
    ):
        run_node(boxed, "cutter", {"page": "x"})
    whole = run_node(boxed, "shorten", {"draft": "A draft."})  # the module itself is one node
    assert whole.outputs == {"short": "echo:Cut A draft."}
    assert whole.records[0].kind == "module"
    with pytest.raises(ValueError, match=r"'nobody' is no node of the system; its nodes: \['shorten', 'writer'\]"):
        run_node(boxed, "nobody", {})
    asked = gated()
    with pytest.raises(ValueError, match="'approve' is a human action: it asks a person and takes no example"):
        run_node(asked, "approve", {"draft": "x"})


def test_level_2_run_cases_and_budgets(tmp_path: Path) -> None:
    wrong = {
        **RUN,
        "name": "wrong",
        "expect": {"checker": [{"path": "verdict.fits", "equals": False}], "deck": [{"path": "deck.pages", "len": 3}]},
    }
    report = check(deck(), 2, examples=examples_file(tmp_path, RUN, wrong), budget=0.2)
    assert [(r.status, r.failures) for r in report.examples] == [
        ("pass", ()),
        ("fail", ("checker item 0: verdict.fits equals false: got true", "deck: deck.pages len 3: has length 1")),
    ]
    assert report.examples[0].cost_usd == pytest.approx(0.07)  # two drafts, two checks, a caption, its check, notes

    report = check(deck(), 2, examples=examples_file(tmp_path, CHECKER, {**CHECKER, "name": "again"}), budget=0.01)
    assert [(r.status, r.notes) for r in report.examples] == [
        ("pass", ()),
        ("skipped", ("budget: nothing left of 0.0100 USD",)),
    ]
    capped = examples_file(tmp_path, RUN, budget={"cost_usd": 0.5, "per example": {"cost_usd": 0.02, "wall_s": 60}})
    (row,) = check(deck(), 2, examples=capped).examples
    assert row.status == "fail"
    assert row.failures[0].startswith("budget: BudgetExceeded: cost_usd")
    walled = examples_file(tmp_path, RUN, budget={"cost_usd": 0.5})
    for given in (None, Budget(max_wall_s=60)):  # a budget without money keeps the 0.02 USD per example
        (row,) = check(deck(), 2, examples=walled, budget=given).examples
        assert row.status == "fail", given
        assert row.failures[0].startswith("budget: BudgetExceeded: cost_usd"), given

    d = deck()
    report = check(d, 2, examples=examples_file(tmp_path, RENDER, CHECKER), budget=0)
    assert [r.status for r in report.examples] == ["skipped", "skipped"]
    assert report.cost_usd == 0.0
    assert calls(d) == [[], []]

    unpriced = MAS.from_dict(
        yaml.safe_load(DECK),
        llms={"writer-model": MockLLM(draft, meta=True), "judge-model": MockLLM(judge)},
        tools=[render_page, assemble_deck],
    )
    report = check(unpriced, 2, examples=examples_file(tmp_path, CHECKER, RENDER))
    assert [(r.status, r.cost_usd) for r in report.examples] == [("fail", None), ("skipped", 0.0)]
    assert report.examples[0].failures == (
        "cost_unknown: a call reported no cost and its model has no price, so the cap cannot bind",
    )
    assert report.examples[1].notes == ("budget: an earlier example's cost is unknown",)
    assert report.cost_usd is None
    assert report.lines()[-1].endswith("0 examples passed, 1 fail, 1 skipped, cost unknown")

    asked = gated()
    (row,) = check(
        asked, 2, examples=examples_file(tmp_path / "people", {"name": "run", "inputs": {"query": "x"}})
    ).examples
    assert (row.status, row.failures) == ("fail", ("the run paused on a question: examples run without people",))

    d = deck()
    report = check(d, 2, examples=examples_file(tmp_path, CHECKER), config=Config(catalogue=("judge-model",)))
    assert (report.examples, report.notes) == ((), ("not run: level 1 found problems",))
    assert "level 2: not run: level 1 found problems" in report.lines()
    assert calls(d) == [[], []]


RECORD = AgentRecord("checker", 0, "", "", {}, {})
VALUES = {
    "verdict": {"fits": False, "fix": "cut it", "orders": [{"kind": "slides", "ids": ["S13"]}], "reads": []},
    "draft": "layout: section\ntitle: Our Line of Research",
    "score": 0.82,
}


@pytest.mark.parametrize(
    ("spec", "failure"),
    [
        ({"form": "valid"}, None),
        ({"form": "valid", "path": "verdict"}, "verdict form valid: BadFormat: no JSON twice"),
        ({"path": "verdict.fits", "equals": False}, None),
        ({"path": "verdict.fits", "equals": 0}, "verdict.fits equals 0: got false"),
        ({"path": "verdict.fix", "contains": "cut"}, None),
        ({"path": "verdict.orders[0].ids", "contains": "S14"}, 'verdict.orders[0].ids contains "S14": got ["S13"]'),
        ({"path": "draft", "not contains": "```"}, None),
        (
            {"path": "draft", "not contains": "Research"},
            'draft not contains "Research": got "layout: section\\ntitle: Our Line of Research"',
        ),
        ({"path": "draft", "matches": "^title: Our"}, None),
        (
            {"path": "draft", "matches": "^layout: slide"},
            'draft matches "^layout: slide": got "layout: section\\ntitle: Our Line of Research"',
        ),
        ({"path": "draft", "line": "title: Our Line of Research"}, None),
        (
            {"path": "draft", "line": "title: Our Line"},
            'draft line "title: Our Line": no such line in "layout: section\\ntitle: Our Line of Research"',
        ),
        ({"path": "verdict.orders", "len": 1}, None),
        ({"path": "verdict.orders", "len": 2}, "verdict.orders len 2: has length 1"),
        ({"path": "verdict.fix", "not empty": True}, None),
        ({"path": "verdict.reads", "not empty": True}, "verdict.reads not empty: empty"),
        ({"path": "verdict.orders[0].kind", "exists": True}, None),
        (
            {"path": "verdict.orders[1].kind", "exists": True},
            "verdict.orders[1].kind exists: nothing at verdict.orders[1].kind",
        ),
        ({"path": "score", "near": 0.8, "within": 0.05}, None),
        ({"path": "score", "near": 0.7, "within": 0.05}, "score near 0.7 within 0.05: got 0.82, off by 0.12"),
        ({"path": "draft", "near": "layout: section\ntitle: Our line of research", "within": 0.1}, None),
        ({"path": "draft", "near": "something else", "within": 0.1}, "draft near"),
        ({"path": "verdict", "near": 1, "within": 0.5}, "verdict near 1 within 0.5: got {"),
        ({"path": "score", "contains": "8"}, 'score contains "8": got 0.82, not a text or a list'),
        ({"path": "score", "len": 1}, "score len 1: got 0.82, which has no length"),
        ({"path": "score", "line": "0.82"}, 'score line "0.82": got 0.82, not a text'),
        ({"path": "score", "near": "0.82", "within": 0.1}, 'score near "0.82" within 0.1: got 0.82, not a text'),
        (
            {"path": "nothing", "equals": 1},
            "nothing equals 1: no value 'nothing'; there are ['draft', 'score', 'verdict']",
        ),
        ({"near": {"a": 1}, "within": 0.1}, "refused: near takes a number or a text"),
        ({"equals": 1, "contains": 1}, "refused: a check holds one kind, got ['equals', 'contains']"),
        ({"looks": "fine"}, "refused: unknown check 'looks'"),
        ({"near": 1}, "refused: 'within' goes with 'near', and 'near' needs it"),
        ({"path": "verdict..fits", "equals": 1}, "refused: path is a value name, then .field or [i] steps"),
    ],
)
def test_every_check_kind(spec: dict[str, Any], failure: str | None) -> None:
    if failure is not None and failure.startswith("refused: "):
        with pytest.raises(ValueError, match=f"^here: {failure.removeprefix('refused: ').replace('[', r'\[')}"):
            Check.parse(spec, "here")
        return
    check = Check.parse(spec, "here")
    assert Check.parse(check.to_spec(), "again") == check
    record = (
        dataclasses.replace(RECORD, error="BadFormat: no JSON twice") if "path" in spec and "form" in spec else RECORD
    )
    found = run_checks([check], VALUES, record)
    assert found == [] if failure is None else len(found) == 1 and found[0].startswith(failure), found


def test_warning_strict_and_only(tmp_path: Path) -> None:
    d = deck()
    folder = d.save(tmp_path / "deck.mas")
    examples_file(folder, CHECKER, {**CHECKER, "name": "checker-wrong", "must": [{"equals": {"fits": True}}]}, RENDER)
    d.save(folder)  # the lock now holds the examples too
    with pytest.warns(VerifyWarning) as caught:
        report = verify(d, 2, config=NAMES)
    assert len(caught) == 1
    assert str(caught[0].message) == str(report)
    assert report.lines()[0].startswith("level 0: unchanged since saved")
    with pytest.raises(VerifyFailed) as failed:
        verify(d, 2, strict=True, config=NAMES)
    assert [r.status for r in failed.value.report.examples] == ["pass", "fail", "pass"]
    assert str(failed.value) == str(failed.value.report)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_name = verify(d, 2, only=["checker-sees-the-overflow"], config=NAMES)
        by_node = verify(d, 2, only=["render"], config=NAMES)
    assert [r.name for r in by_name.examples] == ["checker-sees-the-overflow"]
    assert by_name.lines()[-1] == "ok: 1 example passed, 0.0100 USD"
    assert [r.name for r in by_node.examples] == ["render-measures"]
    data = by_name.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert (data["ok"], data["examples"][0]["status"]) == (True, "pass")
    with pytest.warns(VerifyWarning, match="1 problem"):
        unknown = verify(d, 1, only=["nobody"], config=NAMES)
    assert unknown.problems == ("only names no example or node: 'nobody'",)


def test_save_examples_from_a_trace(tmp_path: Path, png: Path) -> None:
    d = deck()
    folder = d.save(tmp_path / "deck.mas")
    trace = d.run_state({"plan": PLAN, "house": "plain"})
    names = save_examples(d, trace)
    assert sorted(names) == ["assemble", "captioner", "checker", "drafter", "figure_checker", "noter", "render"]
    written = yaml.safe_load((folder / "examples" / "warmup.yaml").read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in written["examples"]}
    assert cases["drafter"]["inputs"] == {"page": PLAN["slides"][0], "house": "plain"}  # no verdict at its first visit
    assert cases["drafter"]["must"] == [{"not empty": True}]
    assert cases["render"] == {
        "name": "render",
        "node": "render",
        "from": cases["render"]["from"],
        "inputs": {"draft": LONG_PAGE},
        "recorded": {"words": 13, "overflow": True},
    }
    assert cases["checker"]["recorded"] == {"fits": False, "fix": "cut it"}
    assert cases["checker"]["must"] == [
        {"form": "valid"},
        {"path": "verdict.fits", "exists": True},
        {"path": "verdict.fix", "exists": True},
    ]
    assert trace.run_id is not None
    assert cases["checker"]["from"].startswith(f"trace {trace.run_id[:8]}, record ")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = verify(d, 2, budget=0.1, config=NAMES)
    assert report.lines()[0].startswith("level 0: unchanged since saved")  # the lock was refreshed
    assert [r.status for r in report.examples] == ["pass"] * 7
    assert report.cost_usd == pytest.approx(0.05)

    assert save_examples(d, trace, nodes=["checker"]) == ["checker"]  # replaces the case of its name
    assert len(yaml.safe_load((folder / "examples" / "warmup.yaml").read_text(encoding="utf-8"))["examples"]) == 7
    (folder / "prompts" / "noter.md").write_text("Notes, by hand: {plan}\n", encoding="utf-8")
    save_examples(d, trace, nodes=["noter"])
    with pytest.warns(VerifyWarning):
        assert verify(d, 0).lines()[0] == "level 0: changed: prompts/noter.md"  # a changed bundle keeps its lock
    (folder / "xmas.lock").write_text("files: [\n", encoding="utf-8")
    save_examples(d, trace, nodes=["noter"])
    assert (folder / "xmas.lock").read_text(encoding="utf-8") == "files: [\n"  # a lock edited by hand stays
    with pytest.raises(ValueError, match="save the system first, or give path="):
        save_examples(deck(), trace)
    with pytest.raises(ValueError, match="node 'nobody' finished no first visit in the trace"):
        save_examples(d, trace, nodes=["nobody"])


def test_save_examples_keeps_media_in_the_bundle(tmp_path: Path, png: Path) -> None:
    seer = SeeingLLM("The deck keeps its plan.")
    media = MAS.from_dict(yaml.safe_load(MEDIA), llms={"vision-model": seer}, tools=[document_text, render_pages])
    house = png.with_name("house.md")
    house.write_text("# House\nOne claim per page.", encoding="utf-8")
    shown = media.run_state({"plan": {"slides": [{"id": "intro"}]}, "house": house})
    other = tmp_path / "media.mas"
    assert save_examples(media, shown, nodes=["reviewer", "render"], path=other) == ["reviewer", "render"]
    reviewer, render = yaml.safe_load((other / "examples" / "warmup.yaml").read_text(encoding="utf-8"))["examples"]
    (page,) = reviewer["inputs"]["slide_png"]
    assert page["location"].startswith("examples/media/")
    assert (other / page["location"]).read_bytes() == b"page intro"
    assert render["recorded"] == [page]  # the code node's pictures, compared by their bytes
    report = check(media, 2, examples=other)
    assert [r.passed for r in report.examples] == [True, True]
    assert len(seer.parts) == 2
    with pytest.raises(ValueError, match=re.escape("named by a plain word, got '../x'")):
        save_examples(media, shown, nodes=["reviewer"], path=other, file="../x")
    zipped = tmp_path / "zipped.mas"
    zipped.write_bytes(b"PK")
    with pytest.raises(ValueError, match="is a zipped bundle: save_examples writes into a bundle folder"):
        save_examples(media, shown, path=zipped)
