"""The ``.mas`` bundle: save and load as a folder or a zip, prompt files, nested boxes, callables, the lock.

A bundle names models and tool servers and holds no address or key; loading it
checks the model names against the configured catalogue and refuses runs while
one is missing.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import stat
import sys
import tomllib
import warnings
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from promptolution.llms.base_llm import BaseLLM

import xmas.mas
from xmas import MAS, Agent
from xmas import bundle as mas_bundle
from xmas.bundle import LOCK, Lock, compare, dump_prompt, lock_of, parse_prompt, read
from xmas.cli import main
from xmas.config import (
    Config,
    ModelEntry,
    ModelNameWarning,
    ToolServer,
    Unconfigured,
    build_llms,
    read_catalogue,
)
from xmas.llms.api_llm import APILLM
from xmas.llms.meta import TokenPrice
from xmas.media import Capabilities
from xmas.nodes import Step
from xmas.tools import Tool, ToolSource, tool
from xmas.tools.mcp import MCPTools
from xmas.verify import VerifyWarning

from .conftest import DECK, PLAN, MockLLM, assemble_deck, deck, deck_llms, render_page
from .test_spec import EXAMPLE, RemoteTools

NO_CATALOGUE = Config()
"""A configuration with no endpoint and no catalogue: model names are not checked, and nothing goes to the network."""

DECK_FILES = [
    "callables.yaml",
    "prompts/captioner.md",
    "prompts/drafter.md",
    "prompts/judge.md",
    "prompts/noter.md",
    "xmas.lock",
    "xmas.yaml",
]


def records(mas: MAS) -> list[tuple[str, str, str]]:
    trace = mas.run_state({"plan": PLAN, "house": "plain"})
    assert trace.error is None
    return sorted((r.agent, r.rendered_prompt, r.raw_response) for r in trace.records)


def mtimes(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def example_mas() -> MAS:
    return MAS.from_dict(
        yaml.safe_load(EXAMPLE),
        llms={"text-model": MockLLM(turns=[])},
        tools=[*conftest_tools(), RemoteTools()],
    )


def conftest_tools() -> list[Tool]:
    from .conftest import apology_text, count_words, publish_text

    return [count_words, publish_text, apology_text]


def test_the_deck_saves_and_loads_as_a_bundle(tmp_path: Path) -> None:
    made = deck()
    path = made.save(tmp_path / "deck.mas")
    assert path == made.bundle == tmp_path / "deck.mas"
    assert sorted(read(path)) == DECK_FILES
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    spec = made.to_dict()
    assert "prompts" not in wiring
    assert "callables" not in wiring
    assert wiring == {key: value for key, value in spec.items() if key not in ("prompts", "callables")}
    assert yaml.safe_load((path / "callables.yaml").read_text(encoding="utf-8")) == spec["callables"]
    assert (path / "prompts/judge.md").read_text(encoding="utf-8") == spec["prompts"]["judge"] + "\n"  # no header

    loaded = MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)  # callables imported by their paths
    assert loaded.bundle == path
    assert loaded.unresolved == {}
    assert loaded.to_dict() == spec
    assert loaded.describe()["id"] == made.describe()["id"]
    assert records(loaded) == records(deck())

    lock, before = (path / LOCK).read_bytes(), mtimes(path)
    assert loaded.save(path) == path
    assert (mtimes(path), (path / LOCK).read_bytes()) == (before, lock)  # nothing rewritten
    parsed = Lock.from_yaml(lock.decode("utf-8"))
    assert sorted(parsed.files) == [name for name in DECK_FILES if name != LOCK]
    assert sorted(parsed.code) == ["assemble_deck", "render_page"]
    assert lock.startswith(b"# written by MAS.save; do not edit\n")


def test_a_zipped_bundle_is_the_same_bundle(tmp_path: Path) -> None:
    first = deck().save(tmp_path / "a.mas", zipped=True)
    second = deck().save(tmp_path / "b.mas", zipped=True)
    assert first.read_bytes() == second.read_bytes()  # byte-reproducible
    folder = deck().save(tmp_path / "c.mas")
    assert read(first) == read(folder)
    loaded = MAS.load(first, llms=deck_llms(), config=NO_CATALOGUE)
    assert loaded.to_dict() == deck().to_dict()
    assert records(loaded) == records(deck())
    before = first.stat().st_mtime_ns
    loaded.save(first, zipped=True)
    assert first.stat().st_mtime_ns == before  # same bytes, not rewritten

    nested = tmp_path / "nested.mas"
    with zipfile.ZipFile(nested, "w") as archive:
        for name, data in read(folder).items():
            archive.writestr(f"deck.mas/{name}", data)
    assert read(nested) == read(folder)  # entries under one top folder
    with zipfile.ZipFile(first, "a") as archive:
        archive.writestr("examples/warmup.yaml", b"examples: []\n")
    deck().save(first, zipped=True)
    assert read(first)["examples/warmup.yaml"] == b"examples: []\n"  # carried over, and locked
    assert "examples/warmup.yaml" in Lock.from_yaml(read(first)[LOCK].decode("utf-8")).files

    with pytest.raises(ValueError, match=r"is a zipped bundle; save\(path, zipped=True\)"):
        deck().save(first)
    with pytest.raises(ValueError, match=r"is a bundle folder"):
        deck().save(folder, zipped=True)
    (tmp_path / "junk.mas").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match=r"neither a bundle folder nor a zipped bundle"):
        read(tmp_path / "junk.mas")
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "absent.mas")


def zip_with(path: Path, name: str, data: bytes = b"x", mode: int = stat.S_IFREG | 0o644) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xmas.yaml", b"version: 1\n")
        info = zipfile.ZipInfo(name)
        info.external_attr = mode << 16
        archive.writestr(info, data)
    return path


@pytest.mark.parametrize(
    ("name", "mode", "message"),
    [
        ("../escape.md", stat.S_IFREG | 0o644, "points outside the bundle"),
        ("/etc/passwd", stat.S_IFREG | 0o644, "points outside the bundle"),
        ("prompts/link.md", stat.S_IFLNK | 0o777, "is a symlink"),
        ("xmas.yaml", stat.S_IFREG | 0o644, "the entry 'xmas.yaml' is there twice"),
    ],
)
def test_a_zip_entry_that_leaves_the_bundle_is_refused(tmp_path: Path, name: str, mode: int, message: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns of the duplicate name it writes
        bad = zip_with(tmp_path / "bad.mas", name, mode=mode)
    with pytest.raises(ValueError, match=message):
        read(bad)


def test_a_zip_that_understates_a_size_is_refused(tmp_path: Path) -> None:
    data = bytearray(zip_with(tmp_path / "bomb.mas", "examples/pad.txt", b"A" * 4096).read_bytes())
    for header, offset in ((b"PK\x01\x02", 24), (b"PK\x03\x04", 22)):  # the stated size of the last entry
        at = data.rfind(header) + offset
        data[at : at + 4] = (10).to_bytes(4, "little")
    (tmp_path / "bomb.mas").write_bytes(bytes(data))
    with pytest.raises(ValueError, match=r"bomb.mas is a broken zip: Bad CRC-32"):
        read(tmp_path / "bomb.mas")


def test_too_big_and_symlinked_bundles_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = deck().save(tmp_path / "deck.mas")
    zipped = deck().save(tmp_path / "deck.zip.mas", zipped=True)
    monkeypatch.setattr(mas_bundle, "MAX_BYTES", 100)
    for path in (folder, zipped):
        with pytest.raises(ValueError, match=r"holds more than 100 bytes"):
            read(path)
    monkeypatch.undo()
    (folder / "examples").mkdir()
    (folder / "examples" / "link.yaml").symlink_to(folder / "xmas.yaml")
    with pytest.raises(ValueError, match=r"examples/link.yaml is a symlink"):
        read(folder)


@pytest.mark.parametrize(
    ("text", "schema", "file"),
    [
        ("Judge {draft}", {"fits": {"type": "bool"}}, "---\nschema:\n  fits:\n    type: bool\n---\nJudge {draft}\n"),
        ("---\n---\nA rule line", None, "---\n---\n---\n---\nA rule line\n"),
        ("Answer {query}\n", None, "Answer {query}\n\n"),
        ("", None, "\n"),
        ("--- not a header", None, "--- not a header\n"),
        ("---\r\nA rule line", None, "---\n---\n---\r\nA rule line\n"),
    ],
)
def test_a_prompt_file_is_its_text_and_one_newline(text: str, schema: dict[str, Any] | None, file: str) -> None:
    assert dump_prompt(text, schema) == file
    assert parse_prompt(file, "prompts/p.md") == (text, schema)


def test_a_prompt_file_without_a_final_newline_is_read_as_is() -> None:
    assert parse_prompt("Answer {query}", "prompts/p.md") == ("Answer {query}", None)
    windows = "---\r\nschema:\r\n  fits: {type: bool}\r\n---\r\nJudge {draft}\r\n"  # saved by a Windows editor
    assert parse_prompt(windows, "prompts/p.md") == ("Judge {draft}", {"fits": {"type": "bool"}})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("---\nreads: [x]\nschema: {a: {type: text}}\n---\nHi", r"holds only 'schema', got \['reads', 'schema'\]"),
        ("---\nschema: {a: 1}\nHi\n", "opened on line 1 is never closed by a '---' line"),
        ("---\nschema: [a]\n---\nHi\n", "'schema' maps answer fields to their forms"),
        ("---\n- a\n---\nHi\n", "holds only 'schema', got list"),
        ("---\nschema: {a\n---\nHi\n", "the front matter is not YAML"),
    ],
)
def test_bad_front_matter_is_refused_naming_the_file(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message) as info:
        parse_prompt(raw, "prompts/judge.md")
    assert str(info.value).startswith("prompts/judge.md: ")


def shared_judges(fix: dict[str, Any]) -> MAS:
    spec = yaml.safe_load(DECK)
    spec["agents"][3]["schema"] = copy.deepcopy(spec["agents"][1]["schema"]) | fix
    return MAS.from_dict(spec, llms=dict(deck_llms()), tools=[render_page, assemble_deck])


def test_a_shared_schema_moves_into_the_prompt_file_only_when_every_sharer_agrees(tmp_path: Path) -> None:
    agreeing = shared_judges({})
    path = agreeing.save(tmp_path / "agree.mas")
    text, schema = parse_prompt((path / "prompts/judge.md").read_text(encoding="utf-8"), "judge")
    assert schema == agreeing.agents["checker"].schema
    assert text == agreeing.prompts()["checker"]
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    assert all("schema" not in agent for agent in wiring["agents"])
    assert MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE).to_dict() == agreeing.to_dict()

    wiring["agents"][1]["schema"] = {"fits": {"type": "bool"}}
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Agent 'checker' has a schema in both prompts/judge.md and xmas.yaml"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)

    differing = shared_judges({"fix": {"type": "text", "means": "another"}})
    path = differing.save(tmp_path / "differ.mas")
    assert parse_prompt((path / "prompts/judge.md").read_text(encoding="utf-8"), "judge")[1] is None
    assert MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE).to_dict() == differing.to_dict()


def test_prompt_files_are_checked_against_the_agents(tmp_path: Path) -> None:
    path = deck().save(tmp_path / "deck.mas")
    (path / "prompts/noter.md").write_text("Write notes for {plan} and {budget}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"prompts/noter.md: prompt placeholders \['budget'\] are not among"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "prompts/noter.md").write_text("Write notes for {plan}\n", encoding="utf-8")
    (path / "prompts/ghost.md").write_text("Boo\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"prompts/ghost.md belongs to no agent"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "prompts/ghost.md").unlink()
    (path / "prompts/noter.md").unlink()
    with pytest.raises(ValueError, match=r"Agent 'noter' has no prompt file prompts/noter.md"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    for text, message in [
        ("version: 2\nprompts: {noter: x}\n", "xmas.yaml holds prompts"),
        ("- a list\n", "xmas.yaml must hold a mapping"),
        ("version: [2\n", "xmas.yaml is not YAML"),
    ]:
        (path / "xmas.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "xmas.yaml").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match=r"xmas.yaml is not UTF-8 text"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "xmas.yaml").unlink()
    with pytest.raises(ValueError, match=r"The bundle has no xmas.yaml"):
        mas_bundle.spec_of(read(path))


class Offered(ToolSource):
    """A tool source with no name that offers ``web_fetch``."""

    def tools(self) -> list[Tool]:
        """Offer the remote tool."""
        return RemoteTools().tools()

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Answer with a page."""
        return "page"


def test_boxes_nest_and_mcp_names_its_server(tmp_path: Path) -> None:
    made = example_mas()
    path = made.save(tmp_path / "example.mas")
    names = sorted(read(path))
    assert {"boxes/shorten.mas/xmas.yaml", "boxes/shorten.mas/prompts/cutter.md"} <= set(names)
    assert not any(name.startswith("boxes/") and name.endswith(LOCK) for name in names)
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    assert wiring["boxes"][0]["system"] == "boxes/shorten.mas"
    callables = yaml.safe_load((path / "callables.yaml").read_text(encoding="utf-8"))
    assert callables["web_fetch"] == {
        "kind": "mcp",
        "server": "tools",
        "reads": {"url": "text"},
        "writes": {"web_fetch": "text"},
    }
    llms: dict[str, BaseLLM] = {"text-model": MockLLM(turns=[])}
    loaded = MAS.load(path, llms=llms, tools=[RemoteTools()], config=NO_CATALOGUE)
    assert loaded.to_dict() == made.to_dict()
    assert loaded.describe()["id"] == made.describe()["id"]


def test_a_named_server_is_built_from_the_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made = example_mas()
    path = made.save(tmp_path / "example.mas")
    callables = yaml.safe_load((path / "callables.yaml").read_text(encoding="utf-8"))
    llms: dict[str, BaseLLM] = {"text-model": MockLLM(turns=[])}
    made_servers: list[tuple[str, str | None, str | None, dict[str, Any]]] = []

    class Served(Offered):
        """A fake MCPTools that records how it was made."""

        def __init__(self, url: str, key: str | None = None, *, writes: Any = None, name: str | None = None) -> None:
            """Record the address, key, name and writes."""
            self.name = name
            made_servers.append((url, key, name, dict(writes)))

    monkeypatch.setattr("xmas.tools.mcp.MCPTools", Served)
    served = Config(tools={"tools": ToolServer("https://tools.example.com/mcp", "tool-key")})
    assert MAS.load(path, llms=llms, config=served).to_dict() == made.to_dict()
    assert made_servers == [
        ("https://tools.example.com/mcp", "tool-key", "tools", {"web_fetch": {"web_fetch": "text"}})
    ]
    with pytest.raises(ValueError, match=r"add \[tools.tools\] url to .*, or pass tools=\[...\] offering them"):
        MAS.load(path, llms=llms, config=NO_CATALOGUE)
    del callables["web_fetch"]["server"]
    (path / "callables.yaml").write_text(yaml.safe_dump(callables), encoding="utf-8")
    with pytest.raises(ValueError, match=r"remote tool with no server: pass tools=.*configure \[tools.<name>\] url"):
        MAS.load(path, llms=llms, config=NO_CATALOGUE)


def test_box_paths_and_server_addresses_are_refused(tmp_path: Path) -> None:
    made = example_mas()
    path = made.save(tmp_path / "example.mas")
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    llms: dict[str, BaseLLM] = {"text-model": MockLLM(turns=[])}
    wiring["boxes"][0]["system"] = "../shorten.mas"
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring), encoding="utf-8")
    with pytest.raises(ValueError, match=r"'../shorten.mas' is outside the bundle"):
        MAS.load(path, llms=llms, tools=[RemoteTools()], config=NO_CATALOGUE)
    spec = made.to_dict()
    spec["boxes"][0]["system"] = "boxes/shorten.mas"
    with pytest.raises(ValueError, match=r"names its system 'boxes/shorten.mas': MAS.load reads a bundle's boxes"):
        MAS.from_dict(spec, llms=llms, tools=[*conftest_tools(), RemoteTools()])
    spec = made.to_dict()
    spec["callables"]["web_fetch"]["server"] = "https://tools.example.com/mcp"
    with pytest.raises(ValueError, match=r"its address goes to the configuration"):
        MAS.from_dict(spec, llms=llms, tools=[*conftest_tools(), RemoteTools()])


def test_mcp_tools_have_a_name_and_not_an_address() -> None:
    llms: dict[str, BaseLLM] = {"text-model": MockLLM(turns=[])}
    plain = MAS.from_dict(yaml.safe_load(EXAMPLE) | {"callables": {}}, llms=llms, tools=[*conftest_tools(), Offered()])
    assert plain.to_dict()["callables"]["web_fetch"] == {
        "kind": "mcp",
        "reads": {"url": "text"},
        "writes": {"web_fetch": "text"},
    }
    named = MCPTools("https://tools.example.com/mcp", "k", name="web")
    assert (named.name, repr(named)) == ("web", "MCPTools('https://tools.example.com/mcp', name='web')")
    with pytest.raises(ValueError, match=r"is a name such as 'web'"):
        MCPTools("https://tools.example.com/mcp", "k", name="https://tools.example.com")


def test_a_box_runs_on_its_own_default_model(tmp_path: Path) -> None:
    made = example_mas()
    llms: dict[str, BaseLLM] = {"text-model": MockLLM(turns=[])}
    spec = made.to_dict() | {"default_model": "text-model"}
    spec["boxes"][0]["system"]["default_model"] = "cut-model"
    two = MAS.from_dict(spec, llms=llms, tools=[*conftest_tools(), RemoteTools()])
    boxed = MAS.load(two.save(tmp_path / "two.mas"), tools=[RemoteTools()], config=NO_CATALOGUE)
    inner = boxed.boxes["shorten"].system
    built = {name: llm.model for name, llm in boxed.llms.items() if isinstance(llm, Unconfigured)}
    assert built == {"text-model": "text-model", "cut-model": "cut-model"}  # the box's own default is built
    assert inner.resolve_llm(inner.agents["cutter"]) is boxed.llms["cut-model"]  # and the box runs on it


class OfflineMCP(MCPTools):
    """MCP tools that list one tool without connecting."""

    def tools(self) -> list[Tool]:
        """Offer one remote tool."""
        schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
        return [Tool("web_fetch", "Fetch a page.", schema)]


def test_no_url_or_key_in_any_bundle_file(tmp_path: Path) -> None:
    remote = OfflineMCP("https://tools.example.com/mcp", "sk-test", name="tools")
    gateway = APILLM(model="text-model", base_url="https://gateway.example/v1", api_key="sk-test")
    fetcher = Agent("fetcher", prompt="Read {query}", output="page", tools=["web_fetch"], model="text-model")
    online = MAS([fetcher], llms={"text-model": gateway}, tools=[remote], version=2)
    saved = [
        deck().save(tmp_path / "deck.mas"),
        example_mas().save(tmp_path / "example.mas"),
        online.save(tmp_path / "online.mas"),
    ]
    zipped = online.save(tmp_path / "online.zip.mas", zipped=True)
    assert online.to_dict()["callables"]["web_fetch"]["server"] == "tools"
    for path in [*saved, zipped]:
        for name, data in read(path).items():
            for word in (b"http", b"://", b"api_key", b"sk-test"):
                assert word not in data, (path.name, name, word)


def test_version_1_and_the_one_agent_case(tmp_path: Path) -> None:
    def solver_llm() -> MockLLM:
        return MockLLM(lambda prompt: f"solved {len(prompt)}")

    made = MAS([Agent("solver", prompt="Solve {query}\nShow the working.", output="prediction")], llm=solver_llm())
    path = made.save(tmp_path / "solver.mas")
    assert sorted(read(path)) == ["prompts/solver.md", "xmas.lock", "xmas.yaml"]
    loaded = MAS.load(path, llm=solver_llm(), config=NO_CATALOGUE)
    assert (loaded.version, loaded.to_dict(), loaded.encode()) == (1, made.to_dict(), made.encode())
    assert loaded.predict(made.encode(), ["a", "b"]) == made.predict(made.encode(), ["a", "b"])
    assert loaded.run("q").records == made.run("q").records

    with pytest.raises(ValueError, match=r"named name.mas, not 'solver.yaml'; to_yaml\(path\) writes"):
        made.save(tmp_path / "solver.yaml")
    made.to_yaml(tmp_path / "solver.yaml")
    assert MAS.from_yaml(tmp_path / "solver.yaml", llm=solver_llm()).to_dict() == made.to_dict()
    with pytest.raises(ValueError, match=r"\.mas bundle \(name\.mas/\): save\(path\) writes it"):
        deck().to_yaml(tmp_path / "deck.yaml")

    MAS(list(made.agents.values()), llm=solver_llm(), default_model="solver-model").save(path)
    by_name = MAS.load(path, llms={"solver-model": solver_llm()}, config=NO_CATALOGUE)
    assert by_name.run("q").output == made.run("q").output

    (path / "prompts/solver.md").write_text("---\nschema: {a: {type: text}}\n---\nSolve {query}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a version 1 agent has no answer form"):
        MAS.load(path, llm=solver_llm(), config=NO_CATALOGUE)


def test_save_rewrites_only_what_changed(tmp_path: Path) -> None:
    path = deck().save(tmp_path / "deck.mas")
    callables = (path / "callables.yaml").read_text(encoding="utf-8")
    (path / "callables.yaml").write_text("# the deck's code\n" + callables, encoding="utf-8")
    (path / "examples").mkdir()
    (path / "examples/warmup.yaml").write_text("version: 1\nexamples: []\n", encoding="utf-8")
    (path / "optimize.yaml").write_text("tune: [prompts]  # only the prompts\n", encoding="utf-8")
    deck().save(path)  # the lock takes the new files in
    before = mtimes(path)
    assert sorted(Lock.from_yaml((path / LOCK).read_text(encoding="utf-8")).files) == sorted(set(before) - {LOCK})

    loaded = MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    loaded.copy(prompts={"noter": "Write short notes for {plan}"}).save(path)
    after = mtimes(path)
    assert sorted(name for name in after if after[name] != before[name]) == ["prompts/noter.md", "xmas.lock"]
    assert (path / "prompts/noter.md").read_text(encoding="utf-8") == "Write short notes for {plan}\n"
    assert (path / "callables.yaml").read_text(encoding="utf-8").startswith("# the deck's code\n")
    assert (path / "optimize.yaml").read_text(encoding="utf-8") == "tune: [prompts]  # only the prompts\n"

    spec = yaml.safe_load(DECK)
    spec["agents"] = [agent for agent in spec["agents"] if agent["name"] != "noter"]
    del spec["prompts"]["noter"], spec["outputs"]["notes"]
    MAS.from_dict(spec, llms=dict(deck_llms()), tools=[render_page, assemble_deck]).save(path)
    assert not (path / "prompts/noter.md").exists()
    assert (path / "examples/warmup.yaml").exists()

    (path / "callables.yaml").write_text("render_page: [unclosed\n", encoding="utf-8")
    example_mas().save(path)  # another system: the deck's files go, a box comes
    assert {"boxes/shorten.mas/xmas.yaml", "callables.yaml"} <= set(read(path))
    assert not (path / "prompts/drafter.md").exists()
    deck().save(path)
    assert not (path / "boxes").exists()
    assert "render_page:" in (path / "callables.yaml").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"Prompt name '\.\./evil' cannot be a file name in a bundle"):
        MAS([Agent("noter", "Notes on {query}", "notes")], llm=MockLLM(), prompt_names={"noter": "../evil"}).save(path)

    foreign = tmp_path / "foreign.mas"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match=r"holds no xmas.yaml; a save writes only into a bundle"):
        deck().save(foreign)

    (path / "optimize.yaml").write_text("tune: [weights]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"optimize.yaml tune is a list of \['prompts', 'models', 'edges'\]"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)


@pytest.mark.parametrize(
    ("optimize", "message"),
    [
        ("seed: 3\n", r"unknown keys \['seed'\]"),
        ("task: {examples: warmup, import: 'a:b'}\n", "task is {examples: <file>} or {import"),
        ("optimizer: {steps: 3}\n", "optimizer is a mapping with a 'name'"),
    ],
)
def test_optimize_yaml_is_checked(optimize: str, message: str) -> None:
    files = {"optimize.yaml": optimize.encode()}
    with pytest.raises(ValueError, match=message):
        mas_bundle.read_optimize(files)
    good = {"task": {"examples": "warmup"}, "tune": ["prompts", "models"], "optimizer": {"name": "capo"}}
    assert mas_bundle.read_optimize({"optimize.yaml": yaml.safe_dump(good).encode()}) == good
    assert mas_bundle.read_optimize({}) == mas_bundle.read_optimize({"optimize.yaml": b""}) == {}


STEPS_MODULE = '''
from xmas.tools import tool


@tool
def shout(text: str) -> str:
    """Shout a text."""
    return text.upper()


def pad(text: str, width: int) -> str:
    """Pad a text to a width with dots."""
    return text.ljust(width, ".")
'''

STEPS_BUNDLE = {
    "xmas.yaml": """\
version: 2
outputs: {padded: text}
steps:
- {name: loud, uses: shout, reads: {text: query}}
- {name: wide, uses: pad, reads: {text: shout}}
edges: [loud -> wide]
""",
    "callables.yaml": """\
shout: {kind: python, import: 'stepsmod:shout', reads: {text: text}, writes: {shout: text}}
pad:   # padded to ten
  kind: python
  import: 'stepsmod:pad'
  params: {width: 10}
  reads: {text: text}
  writes: {padded: text}
""",
}


def steps_bundle(root: Path, monkeypatch: pytest.MonkeyPatch, callables: str | None = None) -> Path:
    (root / "stepsmod.py").write_text(STEPS_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    path = root / "steps.mas"
    path.mkdir(exist_ok=True)
    (path / "xmas.yaml").write_text(STEPS_BUNDLE["xmas.yaml"], encoding="utf-8")
    (path / "callables.yaml").write_text(callables or STEPS_BUNDLE["callables.yaml"], encoding="utf-8")
    return path


def test_callables_load_by_import_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = steps_bundle(tmp_path, monkeypatch)
    monkeypatch.delitem(sys.modules, "stepsmod", raising=False)
    with pytest.raises(ValueError, match=r"'stepsmod:shout' is in a module not imported yet; .*trust=True"):
        MAS.load(path, config=NO_CATALOGUE)  # an untrusted bundle imports nothing
    assert "stepsmod" not in sys.modules
    mas = MAS.load(path, config=NO_CATALOGUE, trust=True)
    assert mas.run_state({"query": "hi"}).outputs == {"padded": "HI........"}
    with pytest.raises(ValueError, match=r"'stepsmod:pad' is not a tool made with @tool"):
        MAS.load(path, config=NO_CATALOGUE)  # imported, but a plain function is wrapped only when trusted
    shell = "shell: {kind: python, import: 'os:system', params: {command: 'echo hi'}, writes: {code: json}}\n"
    (path / "callables.yaml").write_text(shell, encoding="utf-8")
    (path / "xmas.yaml").write_text("version: 2\nsteps:\n- {name: go, uses: shell, reads: {}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'os:system' is not a tool made with @tool"):
        MAS.load(path, config=NO_CATALOGUE)
    path = steps_bundle(tmp_path, monkeypatch)
    item, _ = mas.resolve_callable(mas.steps["wide"])
    assert (item.params, sorted(item.reads), item(text="a")) == ({"width": 10}, ["text"], "a.........")
    assert mas.to_dict()["callables"]["pad"]["params"] == {"width": 10}
    assert mas.to_dict()["callables"]["pad"]["import"] == "stepsmod:pad"

    @tool(name="shout")
    def quiet(text: str) -> str:
        """Say a text quietly."""
        return text.lower()

    supplied = MAS.load(path, tools=[quiet], config=NO_CATALOGUE, trust=True)
    assert supplied.run_state({"query": "Hi"}).outputs == {"padded": "hi........"}

    mismatch = STEPS_BUNDLE["callables.yaml"].replace(
        "reads: {text: text}, writes: {shout", "reads: {text: json}, writes: {shout"
    )
    steps_bundle(tmp_path, monkeypatch, mismatch)
    with pytest.raises(ValueError, match=r"callable 'shout' reads {'text': 'json'} in the spec"):
        MAS.load(path, config=NO_CATALOGUE, trust=True)
    for callables, message in [
        (
            STEPS_BUNDLE["callables.yaml"] + "  note: wraps the old pad\n",
            "unknown keys \\['note'\\].*use a YAML comment",
        ),
        (
            STEPS_BUNDLE["callables.yaml"].replace("stepsmod:shout", "stepsmod:yell"),
            "'stepsmod:yell' cannot be imported",
        ),
        (
            STEPS_BUNDLE["callables.yaml"].replace("'stepsmod:shout'", "stepsmod"),
            "needs import: 'package.module:function'",
        ),
        (STEPS_BUNDLE["callables.yaml"].replace("{width: 10}", "{height: 10}"), r"has no parameters \['height'\]"),
        (
            STEPS_BUNDLE["callables.yaml"].replace(
                "kind: python\n  import: 'stepsmod:pad'", "kind: mcp\n  server: web"
            ),
            "binds params, which only a python callable takes",
        ),
    ]:
        steps_bundle(tmp_path, monkeypatch, callables)
        with pytest.raises(ValueError, match=message):
            MAS.load(path, config=NO_CATALOGUE, trust=True)
    (path / "xmas.yaml").write_text(STEPS_BUNDLE["xmas.yaml"] + "callables: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"The callables are in both callables.yaml and xmas.yaml"):
        MAS.load(path, config=NO_CATALOGUE)


def test_a_bound_tool_keeps_its_params_out_of_the_model_view() -> None:
    @tool
    def scale(value: int, factor: int) -> int:
        """Scale a value."""
        return value * factor

    bound = scale.bind(factor=3)
    assert (bound.parameters["required"], sorted(bound.reads), bound(value=2)) == (["value"], ["value"], 6)
    assert bound.validate({"value": "4"}) == {"value": 4, "factor": 3}
    foreign = Tool("look", "Look.", {"type": "object", "properties": {"a": {}, "b": {}}, "additionalProperties": False})
    assert foreign.bind(b=1).validate({"a": 2}) == {"a": 2, "b": 1}
    assert scale.renamed("grow").name == "grow"


def test_lock_compare_and_code_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = deck().save(tmp_path / "deck.mas")
    lock = Lock.from_yaml((path / LOCK).read_text(encoding="utf-8"))
    assert compare(lock, read(path), mas_bundle.code_hashes(deck())).clean
    (path / "prompts/noter.md").write_text("Other notes for {plan}\n", encoding="utf-8")
    (path / "examples").mkdir()
    (path / "examples/warmup.yaml").write_text("examples: []\n", encoding="utf-8")
    (path / "prompts/captioner.md").unlink()
    diff = compare(lock, read(path))
    assert (diff.changed, diff.added, diff.missing, diff.code_changed) == (
        ("prompts/noter.md",),
        ("examples/warmup.yaml",),
        ("prompts/captioner.md",),
        (),
    )
    assert not diff.clean
    assert lock_of(read(path), {}).bundle != lock.bundle
    edited = lock.to_yaml().replace(lock.files["prompts/noter.md"], lock_of(read(path), {}).files["prompts/noter.md"])
    for text in ("version: 2\n", "[unclosed\n", "version: 1\nbundle: x\nfiles: []\n", edited):
        with pytest.raises(ValueError, match=r"is not a lock written by MAS.save"):
            Lock.from_yaml(text)  # the last: a line changed to match an edited file, the digest left as it was

    steps = steps_bundle(tmp_path, monkeypatch)
    mas = MAS.load(steps, config=NO_CATALOGUE, trust=True)
    mas.save(steps)
    saved = Lock.from_yaml((steps / LOCK).read_text(encoding="utf-8"))
    assert sorted(saved.code) == ["pad", "shout"]
    assert all(saved.code.values())
    module = tmp_path / "stepsmod.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace("text.upper()", "text.upper() + '!'"), encoding="utf-8"
    )
    import importlib

    importlib.reload(sys.modules["stepsmod"])
    changed = MAS.load(steps, config=NO_CATALOGUE, trust=True)
    assert compare(saved, read(steps), mas_bundle.code_hashes(changed)).code_changed == ("shout",)

    namespace: dict[str, Any] = {}
    exec("def echo(query):\n    return query\n", namespace)  # no source file to read
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    echo = Tool("echo", "Echo.", schema, fn=namespace["echo"], writes={"echo": "text"})
    unread = MAS([], steps=[Step("e", "echo")], tools=[echo], llm=MockLLM())
    assert mas_bundle.code_hashes(unread) == {"echo": None}
    assert compare(Lock("x", {}, {"echo": "old"}), {}, {"echo": None}).code_changed == ()  # no source: skipped


def test_model_names_warn_and_refuse_runs(tmp_path: Path) -> None:
    path = deck().save(tmp_path / "deck.mas")
    near = Config(catalogue=("writer-model", "judge-models"))
    llms = deck_llms()
    with pytest.warns(
        ModelNameWarning, match=r"model 'judge-model' \(agents checker, figure_checker\) is not in"
    ) as caught:
        mas = MAS.load(path, llms=llms, config=near)
    assert [(w.message.name, w.message.suggestions) for w in caught if isinstance(w.message, ModelNameWarning)] == [
        ("judge-model", ("judge-models", "writer-model"))
    ]
    assert mas.unresolved == {"judge-model": ("judge-models", "writer-model")}
    for call in (
        lambda: mas.run_state({"plan": PLAN, "house": "plain"}),
        lambda: mas.run("q"),
        lambda: mas.run_batch(["q"]),
        lambda: mas.predict("p", ["q"]),
        lambda: mas("q"),
        lambda: mas.resume("r1", {}),
    ):
        with pytest.raises(
            ValueError, match=r"'judge-model' \(closest: judge-models, writer-model\)\. Set the name in models:"
        ):
            call()
    assert all(not model.calls for model in llms.values())
    assert MAS.load(path, llms=deck_llms(), config=near, resolve=False).unresolved == {}
    assert MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE).unresolved == {}

    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    wiring["models"] = {"judge-model": {"model": "judge-models", "accepts": []}}
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring, sort_keys=False), encoding="utf-8")
    entry = MAS.load(path, llms=deck_llms(), config=near)
    assert entry.unresolved == {}
    assert entry.models == {"judge-model": ModelEntry(model="judge-models")}
    assert entry.to_dict()["models"] == {"judge-model": {"model": "judge-models"}}
    assert records(entry) == records(deck())

    rebound = MAS.load(path, llms=deck_llms(), config=near, overlay={"noter": "judge-model"})
    assert rebound.agents["noter"].model == "judge-model"
    with pytest.raises(ValueError, match=r"The overlay names unknown agents \['ghost'\]"):
        MAS.load(path, llms=deck_llms(), config=near, overlay={"ghost": "judge-model"})
    offline = MAS.load(path, llms={"writer-model": MockLLM()}, config=near)  # no endpoint: built, fails on a call
    assert isinstance(offline.llms["judge-model"], Unconfigured)
    for bad in ("", "https://gateway.example/v1/kimi", 3):
        with pytest.raises(ValueError, match=r"model is a catalogue id such as 'kimi-k3'"):
            ModelEntry.parse({"model": bad}, "models entry 'x'")
    wiring["models"] = {"judge-model": {"model": ["judge-models"]}}
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"needs 'model' to be a catalogue id"):
        MAS.load(path, llms=deck_llms(), config=near)


# ------------------------------------------------------------ configuration, model names, the command line

CONFIG_FILE = """\
[endpoint]
base_url = "http://file.example/v1"
api_key = "sk-file"
catalogue = ["writer-model", "judge-model"]

[models."judge-model"]
price = {input_per_m = 1, output_per_m = 2}
stream = true
"""

ENVIRONMENT = ("XMAS_BASE_URL", "XMAS_API_KEY", "XMAS_TOOLS_KEY", "XMAS_MODELS", "XMAS_CONFIG")


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No xmas variable set, no configuration file, and a cache of this test's own."""
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return monkeypatch


def test_configuration_order(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    path = deck().save(tmp_path / "deck.mas")
    file = tmp_path / "config.toml"
    file.write_text(CONFIG_FILE, encoding="utf-8")
    file.chmod(0o600)
    clean_env.setenv("XMAS_CONFIG", str(file))
    tools = [render_page, assemble_deck]

    from_file = MAS.load(path, tools=tools)
    judge, writer = from_file.llms["judge-model"], from_file.llms["writer-model"]
    assert isinstance(judge, APILLM)
    assert isinstance(writer, APILLM)
    assert (judge.api_url, judge.api_key, judge.stream, judge.price) == (
        "http://file.example/v1",
        "sk-file",
        True,
        TokenPrice(1.0, 2.0),
    )
    assert (writer.stream, writer.price) == (False, None)
    clean_env.setenv("XMAS_BASE_URL", "http://env.example/v1")
    assert MAS.load(path, tools=tools).llms["judge-model"].api_url == "http://env.example/v1"  # type: ignore[attr-defined]
    given = Config.load(base_url="http://given.example/v1")
    built = MAS.load(path, tools=tools, config=given).llms["judge-model"]
    assert built.api_url == "http://given.example/v1"  # type: ignore[attr-defined]

    assert "sk-file" not in repr(from_file) + repr(given) + repr(Config.load())
    again = from_file.save(tmp_path / "again.mas")
    assert not any(b"sk-file" in data or b"example" in data for data in read(again).values())
    file.chmod(0o644)
    with pytest.warns(UserWarning, match="holds a key and others can read it"):
        MAS.load(path, tools=tools)


def test_model_names_resolve_with_suggestions(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    path = deck().save(tmp_path / "deck.mas")
    clean_env.setenv("XMAS_MODELS", "writer-model,judge-models")
    with pytest.warns(ModelNameWarning, match=r"model 'judge-model' \(agents checker, figure_checker\).*judge-models"):
        mas = MAS.load(path, llms=deck_llms())
    with pytest.raises(ValueError, match="Set the name in models: or fix the catalogue"):
        mas.run_state({"plan": PLAN, "house": "plain"})
    assert all(not model.calls for model in mas.llms.values())  # type: ignore[attr-defined]
    with pytest.warns(VerifyWarning, match="1 problem"):
        report = mas.verify(1)
    assert report.misses == {"judge-model": ("judge-models", "writer-model")}

    clean_env.delenv("XMAS_MODELS")  # an endpoint that cannot be reached: names are not checked, runs go on
    clean_env.setenv("XMAS_BASE_URL", "http://127.0.0.1:9/v1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelNameWarning)
        offline = MAS.load(path, llms=deck_llms())
    assert offline.unresolved == {}
    assert records(offline) == records(deck())

    listed = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "writer-model"}]}))
    first = Config(base_url="http://127.0.0.1:9/v1", api_key="sk-first")
    assert read_catalogue(first, transport=listed).source == "endpoint"  # kept for this endpoint and key
    with pytest.warns(ModelNameWarning, match="'judge-model'"):
        assert MAS.load(path, llms=deck_llms(), config=first).unresolved == {"judge-model": ("writer-model",)}
    second = Config(base_url="http://127.0.0.1:9/v1", api_key="sk-second")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelNameWarning)
        assert MAS.load(path, llms=deck_llms(), config=second).unresolved == {}
    kept = [item.read_text(encoding="utf-8") for item in (tmp_path / "cache").rglob("*.json")]
    assert len(kept) == 1
    assert "sk-first" not in kept[0]


def test_models_entries_and_the_overlay(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    path = deck().save(tmp_path / "deck.mas")
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    models = {
        "judge-model": {
            "model": "judge-x",
            "max_tokens": 900,
            "timeout_s": 30,
            "temperature": 0.2,
            "seed": 7,
            "system_prompt": "Judge.",
        },
        "writer-model": {"reasoning_effort": "low"},
        "editor": {"model": "judge-x", "accepts": ["image"], "top_p": 0.9},
    }

    def write(entries: dict[str, Any]) -> None:
        (path / "xmas.yaml").write_text(yaml.safe_dump({**wiring, "models": entries}, sort_keys=False), "utf-8")

    write(models)
    config = Config(base_url="http://gateway.example/v1", api_key="sk-test", catalogue=("judge-x", "writer-model"))
    tools = [render_page, assemble_deck]
    mas = MAS.load(path, tools=tools, config=config, overlay={"noter": "editor"})
    judge, writer, editor = (mas.llms[name] for name in ("judge-model", "writer-model", "editor"))
    assert isinstance(judge, APILLM)
    assert isinstance(writer, APILLM)
    assert isinstance(editor, APILLM)
    assert (judge.model_id, judge.call_timeout_s, judge.call_params()) == (
        "judge-x",
        30,
        {"max_tokens": 900, "temperature": 0.2, "seed": 7, "system_prompt": "Judge."},
    )
    assert judge.system_prompt == "Judge."
    assert writer.call_params() == {"max_tokens": 4096, "reasoning_effort": "low"}
    assert writer.system_prompt == "You are a helpful assistant."  # promptolution's default
    assert (mas.agents["noter"].model, editor.model_id, editor.accepts) == ("editor", "judge-x", ("image",))
    judged = ModelEntry(
        model="judge-x",
        params=(("max_tokens", 900), ("timeout_s", 30), ("temperature", 0.2), ("seed", 7), ("system_prompt", "Judge.")),
    )
    assert mas.models["judge-model"] == judged
    # temperature, seed and the system prompt are the constructor's own keywords; top_p and reasoning_effort travel
    # as call_kwargs
    assert judged.llm_kwargs() == {
        "max_tokens": 900,
        "call_timeout_s": 30,
        "temperature": 0.2,
        "seed": 7,
        "system_prompt": "Judge.",
    }
    assert ModelEntry(params=(("top_p", 0.9), ("seed", 7))).llm_kwargs() == {"seed": 7, "call_kwargs": {"top_p": 0.9}}
    assert mas.to_dict()["models"] == models
    assert ModelEntry.parse({"accepts": ["image"]}, "x") == Capabilities(accepts=("image",))

    for entry, message in (
        ({"colour": "red"}, "has key 'colour'; it takes model, accepts, produces and the parameters max_tokens"),
        ({"temperature": "hot"}, "temperature is a number, not 'hot'"),
        ({"max_tokens": 1.5}, "max_tokens is a whole number, not 1.5"),
        ({"seed": True}, "seed is a whole number, not True"),
        ({"reasoning_effort": 2}, "reasoning_effort is a text, not 2"),
        ({"system_prompt": 2}, "system_prompt is a text, not 2"),
    ):
        write({"judge-model": entry})
        with pytest.raises(ValueError, match=re.escape(f"models entry 'judge-model' {message}"[:60])):
            MAS.load(path, tools=tools, config=config)

    keyless = build_llms(["local-model"], {}, Config(base_url="http://localhost:8080/v1"))["local-model"]
    assert isinstance(keyless, APILLM)
    assert keyless.model_id == "local-model"  # an endpoint that needs no key

    write({"judge-model": {"model": "judge-x"}})
    unconfigured = MAS.load(path, tools=tools, config=Config(catalogue=("judge-x", "writer-model")))
    offline = unconfigured.llms["judge-model"]
    assert isinstance(offline, Unconfigured)
    assert offline.model == "judge-x"
    with pytest.raises(ValueError, match=r"model 'judge-x' has no endpoint: set XMAS_BASE_URL, or base_url"):
        offline.get_response("x")
    trace = unconfigured.run_state({"plan": PLAN, "house": "plain"})
    assert trace.failed
    assert "has no endpoint" in str(trace.error)


class CommandLine:
    """Runs the xmas command, checks its exit code and keeps all it printed."""

    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Read the output through ``capsys``."""
        self.capsys = capsys
        self.printed: list[str] = []

    def __call__(self, code: int, *args: str) -> str:
        """Run ``xmas args``, expect exit ``code`` and return what it printed."""
        assert main(list(args)) == code, args
        captured = self.capsys.readouterr()
        self.printed.append(captured.out + captured.err)
        return captured.out + captured.err

    def leaked(self, secret: str) -> bool:
        """Whether any command printed ``secret``."""
        return any(secret in text for text in self.printed)


@pytest.fixture
def out(tmp_path: Path, clean_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> CommandLine:
    """The command line, run in ``tmp_path`` with a key set that it must never print."""
    clean_env.chdir(tmp_path)
    clean_env.setenv("XMAS_API_KEY", "sk-secret")
    return CommandLine(capsys)


def test_the_command_line_inits_checks_and_shows(tmp_path: Path, out: CommandLine) -> None:
    hello = tmp_path / "hello.mas"
    assert "wrote" in out(0, "init", str(hello))
    assert sorted(read(hello)) == ["examples/warmup.yaml", "optimize.yaml", "prompts/assistant.md", LOCK, "xmas.yaml"]
    assert "exists" in out(2, "init", str(hello))
    assert "names at least one agent" in out(2, "init", str(tmp_path / "none.mas"), "--agents", ",")
    assert "a bundle is named name.mas, not 'hello'" in out(2, "init", str(tmp_path / "hello"))
    assert "no bundle at" in out(2, "check", str(tmp_path / "none.mas"))
    assert out(0, "check", str(hello)) == "ok: 1 node, 0 edges, 0 chains\n"
    pair = tmp_path / "pair.mas"
    out(0, "init", str(pair), "--agents", "writer,editor", "--model", "small-model")
    assert "editor  agent  small-model  query text, writer_draft text -> answer text" in out(0, "show", str(pair))

    path = deck().save(tmp_path / "deck.mas")
    assert out(0, "check", str(path)) == "ok: 7 nodes, 6 edges, 2 chains\n"
    shown = out(0, "show", str(path))
    assert "  checker -> drafter  while not verdict.fits (max 2)" in shown
    assert "  render          step   render_page  draft text -> measure json  for each page in plan.slides" in shown
    assert json.loads(out(0, "show", "--json", str(path))) == deck().describe()
    (path / "examples").mkdir()
    (path / "examples" / "warmup.yaml").write_text("version: 1\nexamples: [{name: a, nodes: x}]\n", "utf-8")
    assert "unknown: nodes" in out(1, "check", str(path))
    assert not out.leaked("sk-secret")


def test_the_command_line_imports_code_beside_a_bundle_only_when_trusted(
    tmp_path: Path, clean_env: pytest.MonkeyPatch, out: CommandLine
) -> None:
    (tmp_path / "cwdtools.py").write_text(STEPS_MODULE, encoding="utf-8")  # code beside the bundle
    local = tmp_path / "local.mas"
    local.mkdir()
    (local / "xmas.yaml").write_text(STEPS_BUNDLE["xmas.yaml"], encoding="utf-8")
    (local / "callables.yaml").write_text(STEPS_BUNDLE["callables.yaml"].replace("stepsmod", "cwdtools"), "utf-8")
    before = list(sys.path)
    assert "'cwdtools:shout' is in a module not imported yet" in out(1, "check", str(local))
    assert "cwdtools" not in sys.modules
    assert sys.path == before
    assert out(0, "check", "--trust", str(local)) == "ok: 2 nodes, 1 edge, 0 chains\n"
    assert sys.path == before  # the current folder joined it for that command only
    clean_env.delitem(sys.modules, "cwdtools")
    assert not out.leaked("sk-secret")


def test_the_command_line_diffs_bundles(tmp_path: Path, out: CommandLine) -> None:
    hello = tmp_path / "hello.mas"
    out(0, "init", str(hello))
    pair = tmp_path / "pair.mas"
    out(0, "init", str(pair), "--agents", "writer,editor", "--model", "small-model")
    path = deck().save(tmp_path / "deck.mas")
    other = tmp_path / "other.mas"
    shutil.copytree(path, other)
    (other / "prompts" / "noter.md").write_text("Notes, by hand: {plan}\n", encoding="utf-8")
    changed = out(1, "diff", str(path), str(other))
    assert "-Write" in changed
    assert "+Notes, by hand: {plan}" in changed
    assert out(0, "diff", str(path), str(path)) == "same files\n"
    assert out(1, "diff", str(other)) == "changed: prompts/noter.md\n"
    assert json.loads(out(1, "diff", "--json", str(other))) == {
        "changed": ["prompts/noter.md"],
        "added": [],
        "missing": [],
    }
    assert out(0, "diff", str(path)) == "unchanged since saved (xmas.lock)\n"
    assert json.loads(out(1, "diff", "--json", str(path), str(hello)))["removed"][0] == "callables.yaml"
    assert "no bundle at" in out(2, "diff", str(tmp_path / "none.mas"))
    (pair / LOCK).unlink()
    assert "has no xmas.lock; give a second bundle" in out(1, "diff", str(pair))
    assert not out.leaked("sk-secret")


def test_the_command_line_runs(tmp_path: Path, clean_env: pytest.MonkeyPatch, out: CommandLine) -> None:
    hello = tmp_path / "hello.mas"
    out(0, "init", str(hello))
    path = deck().save(tmp_path / "deck.mas")
    assert "no endpoint: set XMAS_BASE_URL" in out(1, "run", str(hello), "--query", "Why?")
    clean_env.setenv("XMAS_BASE_URL", "http://127.0.0.1:9/v1")
    clean_env.setenv("XMAS_MODELS", "text-model,writer-model,judge-model")
    scripted = {**deck_llms(), "text-model": MockLLM(lambda p: "Rayleigh.", meta=True)}
    clean_env.setattr(xmas.mas, "build_llms", lambda names, models, config: scripted)
    answered = out(0, "run", str(hello), "--query", "Why?", "--budget", "1")
    assert re.fullmatch(r"Rayleigh\.\nok: 1 model calls, 0\.0100 USD, \d+\.\d s\n", answered)
    ran = out(0, "run", str(path), "--input", f"plan={json.dumps(PLAN)}", "--input", "house=plain")
    assert json.loads(ran.split("\nok: ")[0])["notes"] == "Notes."
    assert "--input takes NAME=VALUE" in out(2, "run", str(path), "--input", "nope=1")
    assert "a json input is JSON" in out(2, "run", str(path), "--input", "plan={", "--input", "house=plain")
    clean_env.setattr(xmas.mas, "build_llms", lambda names, models, config: {"text-model": MockLLM(answers=[])})
    assert "IndexError" in out(1, "run", str(hello), "--query", "Why?").splitlines()[-1]
    clean_env.setenv("XMAS_MODELS", "writer-model")
    refused = out(1, "run", str(hello), "--query", "Why?")
    assert "warning: model 'text-model' (agent assistant)" in refused
    assert "Model names not in the catalogue" in refused
    assert not out.leaked("sk-secret")


def test_the_command_line_verifies(tmp_path: Path, clean_env: pytest.MonkeyPatch, out: CommandLine) -> None:
    hello = tmp_path / "hello.mas"
    out(0, "init", str(hello))
    scripted = {**deck_llms(), "text-model": MockLLM(lambda p: "Rayleigh.", meta=True)}
    clean_env.setattr(xmas.mas, "build_llms", lambda names, models, config: scripted)
    assert out(0, "verify", str(hello)).splitlines()[-1] == "ok: level 1"
    assert (
        out(0, "verify", str(hello), "--level", "2", "--strict").splitlines()[-1] == "ok: 1 example passed, 0.0100 USD"
    )
    assert json.loads(out(0, "verify", "--json", str(hello), "--level", "0"))["ok"] is True
    (hello / "prompts" / "assistant.md").write_text("Answer briefly: {query}\n", encoding="utf-8")
    assert "level 0: changed: prompts/assistant.md" in out(0, "verify", str(hello), "--level", "0")
    assert out(1, "verify", str(hello), "--level", "0", "--strict").splitlines()[-1] == "not ok: level 0 differs"
    assert not out.leaked("sk-secret")


def test_the_command_is_installed_as_xmas() -> None:
    scripts = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert scripts["scripts"] == {"xmas": "xmas.cli:main"}


def test_the_run_log_commands(capsys: pytest.CaptureFixture[str]) -> None:
    def out(code: int, *args: str) -> str:
        assert main(list(args)) == code, args
        captured = capsys.readouterr()
        return captured.out + captured.err

    assert "no runs in Store(" in out(0, "runs")
    assert json.loads(out(0, "runs", "--json")) == []
    mas = deck()
    first = mas.run_state({"plan": PLAN, "house": "plain"})
    broken = MAS([Agent("b", "Answer {query}", "answer")], llm=MockLLM(answers=[RuntimeError("boom")]))
    broken.run("q")
    MAS([Agent("a", "Answer {query}", "answer")], llm=MockLLM()).run("q")  # a model that reports no cost
    lines = out(0, "runs").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith(f"{first.run_id}  ok  ")
    assert lines[0].endswith("  0.2500 USD")
    assert f"  {mas.describe()['id']}  " in lines[0]
    assert lines[2].endswith("cost unknown")
    failed = json.loads(out(0, "runs", "--json", "--status", "failed"))
    assert [row["status"] for row in failed] == ["failed"]
    assert failed[0]["cost_usd"] == 0.0
    mine = json.loads(out(0, "runs", "--json", "--system", mas.describe()["id"]))
    assert [row["run_id"] for row in mine] == [first.run_id]
    assert mine[0]["cost_usd"] == pytest.approx(0.25)
    assert out(0, "runs", "--limit", "2").startswith(failed[0]["run_id"])

    table = out(0, "trace", str(first.run_id)).splitlines()
    assert table[0].split() == ["step", "node", "kind", "item", "cost_usd", "latency_s", "finish"]
    assert table[-1] == f"ok: {len(first.records)} records, 0.2500 USD"
    assert table[6].split()[:5] == ["5", "drafter", "agent", "intro", "0.0100"]
    whole = json.loads(out(0, "trace", "--json", str(first.run_id)))
    assert whole["records"] == first.to_dict()["records"]
    assert whole["outputs"] == first.to_dict()["outputs"]
    assert out(0, "trace", failed[0]["run_id"]).splitlines()[1:] == [
        "failed: 0 records, cost unknown",
        "unknown: RuntimeError: boom",
    ]
    assert "xmas trace: no run 'nope' in Store(" in out(1, "trace", "nope")
