"""The README's version 2 blocks, executed as written.

Each test mirrors one code block of the README line for line, with a scripted
``MockLLM`` in place of ``APILLM``, and checks the prints the README shows. If a
test needs editing to keep passing, the README changed with it. The quickstart's
mirror is ``test_quickstart.py``; the guide's pages are mirrored in
``test_docs_guide.py``. The last tests pin the README's claims that have no block
of their own: the tour, the components, the configuration, the citation and the
rulings.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from xmas import MAS, Agent, Step
from xmas.bundle import read
from xmas.tools import tool

from .conftest import MockLLM
from .test_docs_guide import DRAFT, GUIDE, QUERY, configured_endpoint, quickstart_agents, scripted

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
FITS = '{"fits": true}'


def section(title: str) -> str:
    """Return the README section under the heading ``## title``, up to the next one."""
    return README.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def block(fence: str, start: str) -> str:
    """Return the README code block that opens with ``fence`` and whose text starts with ``start``."""
    return start + README.split(f"```{fence}\n{start}", 1)[1].split("```", 1)[0]


# ------------------------------------------------------------ the README's definitions


@tool
def count_words(draft: str) -> int:
    """Count the words of a draft."""
    return len(draft.split())


def version_2_system(llm: MockLLM) -> MAS:
    """The system of 'Version 2: a step and a JSON judge'."""
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    judge = Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
        inputs=["query", "draft", "count_words"],
        output="verdict",
        output_type="json",
        schema={"fits": {"type": "bool", "means": "the draft answers the question in under 50 words"}},
    )
    return MAS(
        [writer, judge],
        steps=[Step("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )


# ------------------------------------------------- Version 2: a step and a JSON judge


def test_the_version_2_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(lambda: FITS)

    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    judge = Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
        inputs=["query", "draft", "count_words"],
        output="verdict",
        output_type="json",
        schema={"fits": {"type": "bool", "means": "the draft answers the question in under 50 words"}},
    )
    mas = MAS(
        [writer, judge],
        steps=[Step("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )

    trace = mas.run("Why is the sky blue?")
    print(trace.output)  # {"fits": true}
    print([(record.agent, record.kind) for record in trace.records])
    # [('writer', 'agent'), ('count', 'step'), ('judge', 'agent')]

    assert capsys.readouterr().out.splitlines() == [
        '{"fits": true}',
        "[('writer', 'agent'), ('count', 'step'), ('judge', 'agent')]",
    ]
    shown = trace["judge"][0].rendered_prompt
    assert shown.startswith(f"Question: {QUERY}\nDraft, 4 words: {DRAFT}\nJudge the draft.\n\nAnswer with one JSON")
    assert trace["count"][0].outputs == {"count_words": "4"}  # a JSON value, kept as its text
    assert mas.describe()["id"] == version_2_system(llm).describe()["id"]


# --------------------------------------------------------------- Save, load and verify


def test_the_save_block_runs_as_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    endpoint = configured_endpoint(monkeypatch, tmp_path)
    endpoint.responder = lambda prompt: DRAFT if prompt.startswith("Write") else FITS
    mas = version_2_system(endpoint)

    mas.save("team.mas")
    team = MAS.load("team.mas")  # models from your configuration, functions already imported
    print(team.describe()["id"] == mas.describe()["id"])  # True
    print(team.verify())  # checksums and a dry check; no model is called

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "True"
    assert re.fullmatch(r"level 0: unchanged since saved \(bundle [0-9a-f]{16}, 4 files\)", out[1])
    assert out[2:] == ["level 1: no problems; model names checked against the endpoint catalogue", "ok: level 1"]
    assert endpoint.calls == []
    listed = [line.split()[0] for line in block("text", "team.mas/").splitlines()[1:]]
    assert sorted(read("team.mas")) == sorted(
        f"{name}{prompt}" if name == "prompts/" else name
        for name in listed
        for prompt in (["judge.md", "writer.md"] if name == "prompts/" else [""])
    )
    assert not any(b"://" in data or b"key" in data for data in read("team.mas").values())
    assert team.run(QUERY).output == FITS


# ------------------------------------------------- claims without a block of their own


def test_per_agent_models_bind_by_name_and_an_unknown_name_is_refused() -> None:
    fast, slow = scripted(), scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft", model="fast")
    judge = Agent("judge", prompt="{draft}", inputs=["draft"], output="review", model="slow")
    mas = MAS([writer, judge], edges=[("writer", "judge")], llms={"fast": fast, "slow": slow})
    mas.run(QUERY)
    assert (len(fast.calls), len(slow.calls)) == (1, 1)

    with pytest.raises(ValueError, match="not in llms"):
        MAS([writer, judge], edges=[("writer", "judge")], llms={"fast": fast})


def test_the_tour_links_every_guide_page() -> None:
    tour = section("A tour of version 2")
    linked = re.findall(r"\]\(docs/guide/([a-z-]+\.md)\)", tour)
    assert len(linked) == len(set(linked))
    assert set(linked) == {path.name for path in GUIDE.glob("*.md")}
    for target in re.findall(r"\]\((docs/[^)#]+)", README):
        assert (ROOT / target).is_file(), target


def test_every_component_the_readme_lists_exists() -> None:
    import xmas.optimization

    components = README.split("## 🏗 Components", 1)[1].split("\n## ", 1)[0]
    names = [line.split("`")[1] for line in components.splitlines() if line.startswith("- **`")]
    assert len(names) == 19
    for name in names:
        assert hasattr(xmas, name) or hasattr(xmas.optimization, name), name


def test_the_configuration_names_real_variables() -> None:
    table = section("Configuration").split("| Variable | Meaning |", 1)[1].split("\n\n", 1)[0]
    named = re.findall(r"^\| `(XMAS_[A-Z_]+)`", table, flags=re.MULTILINE)
    assert named == ["XMAS_BASE_URL", "XMAS_API_KEY", "XMAS_RUNS", "XMAS_STORE"]
    source = (ROOT / "xmas" / "config.py").read_text(encoding="utf-8")
    for name in named:
        assert f'"{name}"' in source, name


def test_the_citation_matches_the_readme() -> None:
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["authors"] == [{"family-names": "Zehle", "given-names": "Tom"}]
    bibtex = block("bibtex", "@software{")
    assert f"title={{{cff['title']}}}" in bibtex
    assert "author={Zehle, Tom}" in bibtex
    assert "year={2026}" in bibtex
    assert f"url={{{cff['repository-code']}}}" in bibtex


def test_the_readme_keeps_the_rulings() -> None:
    assert ".xmas" not in README
    assert "langgraph" not in README.lower()
    assert "—" not in README.replace(section("📚 Scientific Publications"), "")
    assert "> **Status: beta.** Everything in this README is built." in README
    assert "Learning from human picks is\n> planned and not in the code yet" in README
    for code in README.split("```yaml")[1:]:
        assert "http" not in code.split("```", 1)[0]
        assert "key" not in code.split("```", 1)[0]
    publications = section("📚 Scientific Publications")
    assert "**CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution** — Zehle, 2026." in publications
    assert publications.count("\n- **") == 3


def test_a_system_of_agents_and_plain_edges_is_version_1() -> None:
    intern, boss = quickstart_agents()
    assert MAS([intern, boss], edges=[("intern", "boss")], llm=scripted()).version == 1
