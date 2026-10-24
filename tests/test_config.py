"""Configuration, the model catalogue and name resolution, all offline."""

from __future__ import annotations

import warnings
from pathlib import Path

import httpx
import pytest

from xmas.config import (
    CATALOGUE_TTL_S,
    Catalogue,
    Config,
    ModelNameWarning,
    ToolServer,
    closest,
    config_path,
    misses,
    read_catalogue,
    resolve_names,
)
from xmas.llms.meta import TokenPrice

KEY = "sk-secret-0042"
BASE = "https://gateway.example/v1"
# the gateway's list as recorded on 11 Oct 2026 (33 ids)
GATEWAY = [
    "claude-haiku-5.5",
    "claude-opus-5.5",
    "claude-sonnet-5",
    "claude-sonnet-5.5",
    "deepseek-v4-pro",
    "deepseek-v4.1-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "glm-4.6",
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.3-flashx",
    "gpt-5.6-luna",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-sol",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "grok-4.7",
    "kimi-k2.6",
    "kimi-k3",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "mimo-v2.6-flash",
    "mimo-v2.6-pro",
    "minimax-m3",
    "mistral-large-4",
    "mistral-small-2603",
    "qwen-3.8",
    "qwen3.7-flash",
    "qwen3.8-max",
    "step-5-preview",
]


@pytest.fixture(autouse=True)
def cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def write(tmp_path: Path, text: str, mode: int = 0o600) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


FILE = """
[endpoint]
base_url = "http://file.example/v1/"
api_key_env = "GATEWAY_KEY"
stream = true

[models."kimi-k3"]
price = {input_per_m = 0.6, output_per_m = 2, cached_input_per_m = 0.1}
stream = false

[models."glm-5.3-flash"]
stream = true

[tools.web]
url = "https://tools.example.com/mcp"
key_env = "WEB_KEY"

[tools.local]
url = "http://127.0.0.1:9000/mcp"

[store]
media = "~/media"
runs = "/var/runs.sqlite"
"""


def test_configuration_order(tmp_path: Path) -> None:
    path = write(tmp_path, FILE)
    env = {"XMAS_CONFIG": str(path), "GATEWAY_KEY": "file-key", "WEB_KEY": "web-key"}
    from_file = Config.load(environ=env)
    assert from_file.source == path
    assert (from_file.base_url, from_file.api_key, from_file.catalogue) == ("http://file.example/v1", "file-key", None)
    assert from_file.prices == {"kimi-k3": TokenPrice(0.6, 2.0, 0.1)}
    assert from_file.streams == {"kimi-k3": False, "glm-5.3-flash": True}
    assert (from_file.stream_of("kimi-k3"), from_file.stream_of("qwen-3.8")) == (False, True)
    assert from_file.tools == {
        "web": ToolServer("https://tools.example.com/mcp", "web-key"),
        "local": ToolServer("http://127.0.0.1:9000/mcp"),
    }
    assert (from_file.media_store, from_file.run_store) == (Path("~/media").expanduser(), Path("/var/runs.sqlite"))

    env |= {
        "XMAS_BASE_URL": "http://env.example/v1",
        "XMAS_API_KEY": "env-key",
        "XMAS_MODELS": " kimi-k3, glm-5.3-flash ,",
        "XMAS_STORE": "/env/media",
        "XMAS_RUNS": "",
    }
    from_env = Config.load(environ=env)
    assert (from_env.base_url, from_env.api_key, from_env.catalogue) == (
        "http://env.example/v1",
        "env-key",
        ("kimi-k3", "glm-5.3-flash"),
    )
    assert (from_env.media_store, from_env.run_store) == (Path("/env/media"), Path("/var/runs.sqlite"))

    given = Config.load(environ=env, base_url="http://arg.example/v1/", api_key=None, catalogue=["x"])
    assert (given.base_url, given.api_key, given.catalogue) == ("http://arg.example/v1", "env-key", ("x",))
    assert Config.load(tmp_path / "other.toml", environ=env).source is None  # the argument path wins, and may be absent
    with pytest.raises(TypeError, match="no argument 'source'"):
        Config.load(environ=env, source=path)
    with pytest.raises(TypeError, match="no argument 'endpoint'"):
        Config.load(environ=env, endpoint="x")


def test_the_file_is_found_and_a_missing_one_is_fine(tmp_path: Path) -> None:
    assert config_path({"XDG_CONFIG_HOME": str(tmp_path)}) == tmp_path / "xmas" / "config.toml"
    assert config_path({"XMAS_CONFIG": "~/x.toml"}) == Path.home() / "x.toml"
    assert config_path({}) == Path.home() / ".config" / "xmas" / "config.toml"
    nothing = Config.load(environ={"XDG_CONFIG_HOME": str(tmp_path)})
    assert nothing == Config()
    (tmp_path / "xmas").mkdir()
    write(tmp_path / "xmas", '[endpoint]\napi_key = "k"\ncatalogue = ["a", "b"]\n')
    found = Config.load(environ={"XDG_CONFIG_HOME": str(tmp_path)})
    assert (found.source, found.api_key, found.catalogue) == (tmp_path / "xmas" / "config.toml", "k", ("a", "b"))
    assert Config.load(environ={"XDG_CONFIG_HOME": str(tmp_path), "XMAS_MODELS": "c"}).catalogue == ("c",)
    named_but_unset = write(tmp_path, '[endpoint]\napi_key_env = "NOT_SET"\n')
    assert Config.load(named_but_unset, environ={}).api_key is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[endpoint\n", r"config.toml: not a TOML file"),
        ("[gateway]\n", r"unknown table 'gateway'; the file takes endpoint, models, tools, store"),
        ("[endpoint]\nbase = 'x'\n", r"unknown key 'endpoint.base'; endpoint takes base_url, api_key"),
        ("[endpoint]\nbase_url = 8000\n", r"endpoint.base_url is a string, not int"),
        ("[endpoint]\nstream = 'yes'\n", r"endpoint.stream is true or false, not str"),
        ("[endpoint]\ncatalogue = ['a', 1]\n", r"endpoint.catalogue is a list of model ids, not list"),
        ("[endpoint]\napi_key = 'a'\napi_key_env = 'B'\n", r"endpoint takes api_key or api_key_env, not both"),
        ("endpoint = 1\n", r"endpoint is a table, not int"),
        ("models = 1\n", r"models is a table, not int"),
        ("[models]\nkimi = 1\n", r'models."kimi" is a table, not int'),
        ("[models.kimi]\ntemperature = 0.2\n", r"unknown key 'models.\"kimi\".temperature'; models.\"kimi\" takes"),
        ("[models.kimi]\nprice = {input_per_m = 1}\n", r'models."kimi".price needs output_per_m'),
        ("[models.kimi]\nprice = {input_per_m = true, output_per_m = 1}\n", r"input_per_m is a number, not bool"),
        ("[models.kimi]\nprice = 1\n", r'models."kimi".price is a table, not int'),
        ("[tools.web]\nkey = 'k'\n", r"tools.web needs url"),
        ("[tools.web]\nurl = 'u'\nkey = 'k'\nkey_env = 'K'\n", r"tools.web takes key or key_env, not both"),
        ("[store]\ncache = '/x'\n", r"unknown key 'store.cache'; store takes media, runs"),
    ],
)
def test_a_malformed_file_names_the_key(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Config.load(write(tmp_path, text), environ={})


def test_no_key_in_a_repr_or_a_message(tmp_path: Path) -> None:
    path = write(tmp_path, f'[endpoint]\napi_key = "{KEY}"\n[tools.web]\nurl = "u"\nkey = "{KEY}"\n', mode=0o644)
    with pytest.warns(UserWarning, match=r"holds a key and others can read it; chmod 600") as caught:
        config = Config.load(path, environ={})
    assert caught[0].filename == __file__
    assert (config.api_key, config.tools["web"].key) == (KEY, KEY)
    assert KEY not in repr(config) + repr(config.tools["web"]) + str(caught[0].message)
    in_urls = Config(base_url=f"https://me:{KEY}@gw.example/v1?key={KEY}", tools={"web": ToolServer(f"u?token={KEY}")})
    assert KEY not in repr(in_urls)
    assert "base_url='https://gw.example/v1'" in repr(in_urls)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Config.load(write(tmp_path, f'[endpoint]\napi_key = "{KEY}"\n', mode=0o600), environ={})
        Config.load(write(tmp_path, '[endpoint]\napi_key_env = "K"\n', mode=0o644), environ={"K": KEY})
    with pytest.raises(ValueError, match="api_key is a string, not list") as refused:
        Config.load(write(tmp_path, f'[endpoint]\napi_key = ["{KEY}"]\n'), environ={})
    assert KEY not in str(refused.value)

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"bad key {KEY}"})

    keyed = Config(base_url=f"https://user:{KEY}@gateway.example/v1", api_key=KEY)
    note = read_catalogue(keyed, transport=httpx.MockTransport(refuse)).note
    assert note == "the catalogue of https://gateway.example/v1 could not be read (HTTP 401); model names not checked"
    found = resolve_names(["kimi"], keyed, catalogue=Catalogue(("kimi-k3",), "endpoint"))
    assert KEY not in str(found.warnings[0])


class Gateway:
    """A mock ``/models`` endpoint that counts its calls and can go offline."""

    def __init__(self, ids: tuple[str, ...] = ("claude-sonnet-5.5", "kimi-k3")) -> None:  # noqa: D107
        self.ids, self.calls, self.offline, self.seen = ids, 0, False, list[httpx.Request]()

    def __call__(self, request: httpx.Request) -> httpx.Response:  # noqa: D102
        if self.offline:
            raise httpx.ConnectError("no route", request=request)
        self.calls += 1
        self.seen.append(request)
        return httpx.Response(200, json={"object": "list", "data": [{"id": name} for name in self.ids]})


def test_the_catalogue_is_kept_for_a_while_and_stands_in_offline(cache_home: Path) -> None:
    gateway, now = Gateway(), [1_792_800_000.0]  # 2026-10-24 00:00 UTC
    transport = httpx.MockTransport(gateway)

    def read(config: Config, **flags: bool) -> Catalogue:
        return read_catalogue(config, transport=transport, clock=lambda: now[0], **flags)

    config = Config(base_url=BASE, api_key=KEY)
    assert read(config) == Catalogue(("claude-sonnet-5.5", "kimi-k3"), "endpoint")
    assert str(gateway.seen[0].url) == f"{BASE}/models"
    assert gateway.seen[0].headers["authorization"] == f"Bearer {KEY}"
    now[0] += CATALOGUE_TTL_S - 1
    assert read(config) == Catalogue(("claude-sonnet-5.5", "kimi-k3"), "cache")
    assert gateway.calls == 1
    assert read(config, refresh=True).source == "endpoint"
    assert gateway.calls == 2
    now[0] += CATALOGUE_TTL_S
    gateway.ids = ("kimi-k3",)
    assert read(config) == Catalogue(("kimi-k3",), "endpoint")
    (kept,) = cache_home.joinpath("xmas").iterdir()
    assert KEY not in kept.read_text(encoding="utf-8")

    now[0] += 86_400
    gateway.offline = True
    assert read(config) == Catalogue(("kimi-k3",), "cache", "cache of 2026-10-24 00:09 UTC (ConnectError)")
    other_key = Config(base_url=BASE, api_key="sk-other")
    assert read(other_key) == Catalogue(
        None, "none", f"the catalogue of {BASE} could not be read (ConnectError); model names not checked"
    )
    gateway.offline = False
    assert read(Config(base_url=BASE)).source == "endpoint"
    assert "authorization" not in gateway.seen[-1].headers
    assert len(list(cache_home.joinpath("xmas").iterdir())) == 2

    kept.write_text("not json", encoding="utf-8")
    gateway.offline = True
    assert read(config).ids is None


def test_an_unwritable_cache_still_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "file").write_text("", encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "file"))
    gateway = Gateway()
    for _ in range(2):
        assert read_catalogue(Config(base_url=BASE), transport=httpx.MockTransport(gateway)).source == "endpoint"
    assert gateway.calls == 2


def test_a_configured_list_or_no_endpoint_asks_nobody() -> None:
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no call")

    nobody = httpx.MockTransport(never)
    assert read_catalogue(Config(base_url=BASE, catalogue=("a",)), transport=nobody) == Catalogue(("a",), "configured")
    assert read_catalogue(Config(), transport=nobody) == Catalogue(None, "none", "no endpoint; model names not checked")


@pytest.mark.parametrize(
    "answer",
    [httpx.Response(200, text="<html>"), httpx.Response(200, json={"models": []}), httpx.Response(503, json={})],
    ids=["not json", "no data", "unavailable"],
)
def test_an_odd_answer_is_no_catalogue(answer: httpx.Response) -> None:
    found = read_catalogue(Config(base_url=BASE), transport=httpx.MockTransport(lambda request: answer))
    assert found.ids is None
    assert found.note.endswith("; model names not checked")


@pytest.mark.parametrize(
    ("name", "near"),
    [
        ("claude-sonnet-5-5", ["claude-sonnet-5.5", "claude-sonnet-5", "claude-opus-5.5"]),
        ("kimi-k2", ["kimi-k2.6", "kimi-k3"]),
        ("gemini-3.8-flash-lite", ["gemini-3.5-flash-lite", "gemini-3.8-flash"]),
        ("moonshotai/kimi-k3", ["kimi-k3"]),
        ("Kimi-K3", ["kimi-k3"]),
        ("gpt6-sol", ["gpt-6-sol", "gpt-6.1-sol", "gpt-6-luna"]),
        ("sonnet", []),
        ("text-model", []),
    ],
)
def test_closest_names_from_the_gateway_list(name: str, near: list[str]) -> None:
    assert closest(name, GATEWAY) == near


def test_closest_keeps_vendor_ids_and_the_limit() -> None:
    assert closest("kimi-k3", ["moonshotai/kimi-k3", "other/kimi-k3"], n=1) == ["moonshotai/kimi-k3"]
    assert closest("glm-5.3", ["glm-5.3-flash", "glm-5.3-flashx", "glm-5.2"], n=2) == ["glm-5.2", "glm-5.3-flash"]


def test_names_resolve_with_suggestions_and_never_refuse_unchecked() -> None:
    config = Config(base_url="http://127.0.0.1:8000/v1", api_key=KEY)
    gateway = Catalogue(tuple(GATEWAY), "endpoint")
    found = resolve_names(
        {"claude-sonnet-5-5": "agent editor", "kimi-k3": "agent noter", "sonnet": ""}, config, catalogue=gateway
    )
    assert found.checked
    assert found.missing == ("claude-sonnet-5-5", "sonnet")
    assert found.suggestions == {
        "claude-sonnet-5-5": ("claude-sonnet-5.5", "claude-sonnet-5", "claude-opus-5.5"),
        "sonnet": (),
    }
    assert [str(warning) for warning in found.warnings] == [
        (
            "model 'claude-sonnet-5-5' (agent editor) is not in the catalogue of http://127.0.0.1:8000/v1; "
            "closest: claude-sonnet-5.5, claude-sonnet-5, claude-opus-5.5"
        ),
        "model 'sonnet' is not in the catalogue of http://127.0.0.1:8000/v1; nothing close",
    ]
    first = found.warnings[0]
    assert isinstance(first, UserWarning)
    assert (first.name, first.suggestions) == ("claude-sonnet-5-5", found.suggestions["claude-sonnet-5-5"])
    with pytest.warns(ModelNameWarning, match="closest: claude-sonnet-5.5"):
        warnings.warn(first, stacklevel=1)

    listed = resolve_names(["kimi-k2", "kimi-k2"], Config(catalogue=("kimi-k3",)))
    assert listed.suggestions == {"kimi-k2": ("kimi-k3",)}
    assert str(listed.warnings[0]) == (
        "model 'kimi-k2' is not in the configured catalogue (XMAS_MODELS or endpoint.catalogue); closest: kimi-k3"
    )
    unchecked = resolve_names(["anything"], Config())
    assert (unchecked.checked, unchecked.missing, unchecked.warnings) == (False, (), ())
    assert unchecked.catalogue.note == "no endpoint; model names not checked"
    assert misses(["a"], Catalogue(None, "none")) == {}
    assert misses(["a", "b"], Catalogue(("a",), "configured")) == {"b": ()}


def test_the_documented_file_loads(tmp_path: Path) -> None:
    import xmas.config

    block = (xmas.config.__doc__ or "").split(".. code-block:: toml", 1)[1].split("The environment variables", 1)[0]
    path = write(tmp_path, "\n".join(line[4:] for line in block.splitlines()))
    config = Config.load(path, environ={"GATEWAY_KEY": "k", "WEB_TOOLS_KEY": "w"})
    assert (config.base_url, config.api_key, config.catalogue) == ("http://localhost:8000/v1", "k", ("llama-3.1-8b",))
    assert (config.stream_of("llama-3.1-8b"), config.tools["web"].key) == (True, "w")
