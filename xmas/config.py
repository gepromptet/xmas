"""Configuration: where models and tools are served, and which model names exist there.

A saved system names its models and tool servers; it never holds an address or a key.
Those come from here, first from the arguments a caller passes, then from the
environment, then from a TOML file, ``$XMAS_CONFIG`` or else
``${XDG_CONFIG_HOME:-~/.config}/xmas/config.toml``. A missing file is no error.

.. code-block:: toml

    [endpoint]
    base_url = "http://localhost:8000/v1"
    api_key_env = "GATEWAY_KEY"        # or api_key = "..."
    catalogue = ["llama-3.1-8b"]       # optional; else GET {base_url}/models
    stream = false

    [models."llama-3.1-8b"]
    price = {input_per_m = 0.6, output_per_m = 2.5}
    stream = true

    [tools.web]
    url = "https://tools.example.com/mcp"
    key_env = "WEB_TOOLS_KEY"          # or key = "..."

    [store]
    media = "~/.cache/xmas/store"
    runs = "~/.cache/xmas/runs.sqlite"

The environment variables are ``XMAS_BASE_URL``, ``XMAS_API_KEY``, ``XMAS_MODELS`` (a
comma list that stands for the catalogue, so names check offline), ``XMAS_STORE``
(media) and ``XMAS_RUNS`` (the run store, else :func:`runs_path`). A tool server
without a key leaves it to ``XMAS_TOOLS_KEY``, which :class:`~xmas.tools.mcp.MCPTools`
reads itself.

The catalogue is the list of model ids the endpoint serves. :func:`read_catalogue`
takes a configured list, else asks the endpoint and keeps its answer for
:data:`CATALOGUE_TTL_S` per endpoint and key; offline, a kept answer of any age
stands in. :func:`resolve_names` checks names against it: a miss is a
:class:`ModelNameWarning` naming the closest ids, never a substitute. No key appears
in a ``repr``, a message or the kept answer, and nothing here runs on import.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import stat
import time
import tomllib
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import urlsplit, urlunsplit

import httpx
from promptolution.llms.base_llm import BaseLLM

from xmas.llms import APILLM
from xmas.llms.meta import TokenPrice
from xmas.media import TEXT_ONLY, Capabilities

if TYPE_CHECKING:
    from xmas.tools import ToolSource

__all__ = [
    "CATALOGUE_TTL_S",
    "CONFIG_ENV",
    "FETCH_TIMEOUT_S",
    "MODELS_ENV",
    "PARAMS",
    "Catalogue",
    "Config",
    "ModelEntry",
    "ModelNameWarning",
    "Resolution",
    "ToolServer",
    "Unconfigured",
    "build_llms",
    "closest",
    "config_path",
    "misses",
    "read_catalogue",
    "resolve_names",
    "runs_path",
    "tool_sources",
]


CONFIG_ENV = "XMAS_CONFIG"
MODELS_ENV = "XMAS_MODELS"
CATALOGUE_TTL_S = 300.0
FETCH_TIMEOUT_S = 5.0

SCALARS: dict[str, dict[str, type]] = {
    "endpoint": {"base_url": str, "api_key": str, "api_key_env": str, "catalogue": list, "stream": bool},
    "model": {"price": dict, "stream": bool},
    "price": {"input_per_m": float, "output_per_m": float, "cached_input_per_m": float},
    "tool": {"url": str, "key": str, "key_env": str},
    "store": {"media": str, "runs": str},
}
TABLES = ("endpoint", "models", "tools", "store")
SHAPES = {str: "a string", bool: "true or false", float: "a number", dict: "a table", list: "a list of model ids"}
PARAM_WORDS: dict[tuple[type, ...], str] = {(str,): "a text", (int,): "a whole number"}


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the configuration file's path: ``$XMAS_CONFIG``, else ``config.toml`` under the user's config folder."""
    environ = os.environ if environ is None else environ
    if environ.get(CONFIG_ENV):
        return Path(environ[CONFIG_ENV]).expanduser()
    return Path(environ.get("XDG_CONFIG_HOME") or Path.home() / ".config").expanduser() / "xmas" / "config.toml"


def runs_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the run store's default file: ``xmas/runs.sqlite`` under ``${XDG_DATA_HOME:-~/.local/share}``."""
    environ = os.environ if environ is None else environ
    return Path(environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share").expanduser() / "xmas" / "runs.sqlite"


@dataclass(frozen=True)
class ToolServer:
    """A tool server a callable names by ``server:``; its key and the credentials in its address never show.

    Attributes:
        url: The server's address.
        key: Its bearer key, or ``None`` to leave it to ``XMAS_TOOLS_KEY``.
    """

    url: str
    key: str | None = None

    def __repr__(self) -> str:
        """Show the address without credentials or query, never the key."""
        return f"ToolServer(url={redact_url(self.url)!r})"


@dataclass(frozen=True)
class Config:
    """Endpoint, key, catalogue, prices, streaming, tool servers and stores, as :meth:`load` resolved them.

    Attributes:
        base_url: The OpenAI-compatible endpoint, or ``None`` when none is configured.
        api_key: Its key; never shown in a ``repr``.
        catalogue: Model ids configured to stand for the endpoint's list, or ``None`` to ask the endpoint.
        stream: Whether calls stream, for models without their own setting.
        streams: Per model id, whether its calls stream.
        prices: Per model id, the price used when a response reports no cost.
        tools: Tool servers by the name callables use.
        media_store: Folder for stored media, or ``None`` for the default.
        run_store: File of the run store, or ``None`` for :func:`runs_path`;
            ``:memory:`` for one in memory, which logs no run.
        source: The file that was read, or ``None`` when there was none.
    """

    base_url: str | None = None
    api_key: str | None = None
    catalogue: tuple[str, ...] | None = None
    stream: bool = False
    streams: Mapping[str, bool] = field(default_factory=dict)
    prices: Mapping[str, TokenPrice] = field(default_factory=dict)
    tools: Mapping[str, ToolServer] = field(default_factory=dict)
    media_store: Path | None = None
    run_store: Path | None = None
    source: Path | None = None

    @classmethod
    def load(
        cls, path: str | os.PathLike[str] | None = None, environ: Mapping[str, str] | None = None, **given: Any
    ) -> Config:
        """Resolve the configuration: ``given`` fields that are not ``None``, then the environment, then the file.

        Args:
            path: The file to read, instead of :func:`config_path`.
            environ: The environment to read, instead of ``os.environ``.
            **given: Field values that win over everything else.

        Raises:
            TypeError: For a ``given`` name that is not a field.
            ValueError: For a file that is not TOML, or holds a key or a value it does not take; the message names it.
        """
        environ = os.environ if environ is None else environ
        for name in given:
            if name == "source" or name not in {f.name for f in fields(cls)}:
                raise TypeError(f"Config.load takes no argument {name!r}")
        file = Path(path).expanduser() if path is not None else config_path(environ)
        values = read_file(file, environ) if file.is_file() else {}
        listed = [name.strip() for name in environ.get(MODELS_ENV, "").split(",") if name.strip()]
        from_env = {
            "base_url": environ.get("XMAS_BASE_URL"),
            "api_key": environ.get("XMAS_API_KEY"),
            "catalogue": listed or None,
            "media_store": environ.get("XMAS_STORE"),
            "run_store": environ.get("XMAS_RUNS"),
        }
        for layer in (from_env, given):
            values.update({key: value for key, value in layer.items() if value is not None and value != ""})
        if values.get("base_url"):
            values["base_url"] = str(values["base_url"]).rstrip("/")
        if values.get("catalogue") is not None:
            values["catalogue"] = tuple(values["catalogue"])
        for key in ("media_store", "run_store"):
            if key in values:
                values[key] = Path(values[key]).expanduser()
        return cls(**values, source=file if file.is_file() else None)

    def __repr__(self) -> str:
        """Show every field but the key; the endpoint without credentials or query."""
        shown = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "api_key"}
        shown["base_url"] = None if self.base_url is None else redact_url(self.base_url)
        return f"Config({', '.join(f'{name}={value!r}' for name, value in shown.items())})"

    def stream_of(self, model: str) -> bool:
        """Whether calls to ``model`` stream: its own setting, else :attr:`stream`."""
        return self.streams.get(model, self.stream)


def read_file(path: Path, environ: Mapping[str, str]) -> dict[str, Any]:
    """Parse the file into :class:`Config` fields, naming the key of any problem."""
    data = parse_toml(path)
    for key in data:
        if key not in TABLES:
            raise ValueError(f"{path}: unknown table '{key}'; the file takes {', '.join(TABLES)}")
    endpoint = checked_table(path, data.get("endpoint", {}), "endpoint", "endpoint")
    store = checked_table(path, data.get("store", {}), "store", "store")
    values: dict[str, Any] = {
        "base_url": endpoint.get("base_url"),
        "api_key": secret(endpoint, "api_key", environ),
        "catalogue": endpoint.get("catalogue"),
        "stream": endpoint.get("stream", False),
        "media_store": store.get("media"),
        "run_store": store.get("runs"),
    }
    prices, streams = read_models(path, data)
    tools = read_tools(path, data, environ)
    warn_if_others_read_a_key(path, data, endpoint)
    return {**{k: v for k, v in values.items() if v is not None}, "prices": prices, "streams": streams, "tools": tools}


def parse_toml(path: Path) -> dict[str, Any]:
    """Return the TOML content of ``path``; a file that is not TOML is a ``ValueError`` naming it."""
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: not a TOML file ({exc})") from None


def checked_table(path: Path, table: Any, kind: str | None, where: str) -> dict[str, Any]:
    """Return ``table`` once it is a table whose keys and values fit ``kind``; ``None`` for a table of tables."""
    if not isinstance(table, dict):
        raise ValueError(f"{path}: {where} is a table, not {type(table).__name__}")
    if kind is None:  # a table of tables, such as [models."<id>"]
        return table
    allowed = SCALARS[kind]
    for key, value in table.items():
        check_scalar(path, allowed, where, key, value)
    for key, alias in (("api_key", "api_key_env"), ("key", "key_env")):
        if key in table and alias in table:
            raise ValueError(f"{path}: {where} takes {key} or {alias}, not both")
    return table


def check_scalar(path: Path, allowed: Mapping[str, type], where: str, key: str, value: Any) -> None:
    """Refuse a ``key`` the table does not take, or a ``value`` not of the shape it takes."""
    if key not in allowed:
        raise ValueError(f"{path}: unknown key '{where}.{key}'; {where} takes {', '.join(allowed)}")
    want = allowed[key]
    fits = isinstance(value, (int, float)) if want is float else isinstance(value, want)
    words = key != "catalogue" or all(isinstance(item, str) for item in value)
    if isinstance(value, bool) is not (want is bool) or not fits or not words:
        raise ValueError(f"{path}: {where}.{key} is {SHAPES[want]}, not {type(value).__name__}")


def read_models(path: Path, data: Mapping[str, Any]) -> tuple[dict[str, TokenPrice], dict[str, bool]]:
    """Return the ``[models."<id>"]`` tables as the prices and the stream settings they state."""
    prices, streams = {}, {}
    for model, entry in checked_table(path, data.get("models", {}), None, "models").items():
        where = f"models.{json.dumps(model)}"
        checked_table(path, entry, "model", where)
        if "stream" in entry:
            streams[model] = entry["stream"]
        if "price" in entry:
            prices[model] = read_price(path, entry["price"], f"{where}.price")
    return prices, streams


def read_price(path: Path, table: Any, where: str) -> TokenPrice:
    """Return a model's ``price`` table as a :class:`TokenPrice`; both rates are needed."""
    price = checked_table(path, table, "price", where)
    missing = [key for key in ("input_per_m", "output_per_m") if key not in price]
    if missing:
        raise ValueError(f"{path}: {where} needs {' and '.join(missing)}")
    return TokenPrice(**{key: float(value) for key, value in price.items()})


def read_tools(path: Path, data: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, ToolServer]:
    """Return the ``[tools.<name>]`` tables as tool servers, each with its url and key."""
    tools = {}
    for name, entry in checked_table(path, data.get("tools", {}), None, "tools").items():
        where = f"tools.{name}"
        checked_table(path, entry, "tool", where)
        if "url" not in entry:
            raise ValueError(f"{path}: {where} needs url")
        tools[name] = ToolServer(entry["url"], secret(entry, "key", environ))
    return tools


def warn_if_others_read_a_key(path: Path, data: Mapping[str, Any], endpoint: Mapping[str, Any]) -> None:
    """Warn when the file states a key and its mode lets the group or others read it."""
    keyed = "api_key" in endpoint or any("key" in entry for entry in data.get("tools", {}).values())
    if keyed and path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH):
        warnings.warn(f"{path} holds a key and others can read it; chmod 600 {path}", UserWarning, stacklevel=4)


def secret(table: Mapping[str, Any], key: str, environ: Mapping[str, str]) -> str | None:
    """Return the ``key`` a table states, else the environment variable its ``<key>_env`` names."""
    return table.get(key) or (environ.get(table[f"{key}_env"]) if f"{key}_env" in table else None)


def redact_url(url: str) -> str:
    """Return ``url`` without credentials, query or fragment, fit for a message."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


@dataclass(frozen=True)
class Catalogue:
    """The model ids an endpoint serves, and where they came from.

    Attributes:
        ids: The ids, or ``None`` when no list could be had and names go unchecked.
        source: ``"configured"``, ``"endpoint"``, ``"cache"`` or ``"none"``.
        note: Why the list is old or missing, for a report line.
    """

    ids: tuple[str, ...] | None
    source: str
    note: str = ""


def cache_file(base_url: str, api_key: str | None) -> Path:
    """Return the file that keeps the answer of ``base_url`` for ``api_key``; the key enters only hashed."""
    folder = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser() / "xmas"
    keyed = hashlib.sha256((api_key or "").encode()).hexdigest()
    return folder / f"catalogue-{hashlib.sha256(f'{base_url}\n{keyed}'.encode()).hexdigest()[:16]}.json"


def read_cached(file: Path) -> tuple[float, tuple[str, ...]] | None:
    try:
        kept = json.loads(file.read_text(encoding="utf-8"))
        return float(kept["fetched"]), tuple(str(name) for name in kept["ids"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_catalogue(
    config: Config,
    *,
    refresh: bool = False,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], float] = time.time,
) -> Catalogue:
    """Return the catalogue: the configured list, else a fresh kept answer, else ``GET {base_url}/models``.

    Args:
        config: Where to look.
        refresh: Ask the endpoint even when a kept answer is fresh.
        transport: The HTTP transport, for tests.
        clock: The time in seconds since the epoch, for tests.

    Returns:
        The ids; when the endpoint cannot be reached, a kept answer of any age, else ``ids=None`` with a note.
    """
    if config.catalogue is not None:
        return Catalogue(tuple(config.catalogue), "configured")
    if not config.base_url:
        return Catalogue(None, "none", "no endpoint; model names not checked")
    file = cache_file(config.base_url, config.api_key)
    kept = read_cached(file)
    now = clock()
    if kept is not None and not refresh and 0 <= now - kept[0] < CATALOGUE_TTL_S:
        return Catalogue(kept[1], "cache")
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    try:
        with httpx.Client(transport=transport, timeout=FETCH_TIMEOUT_S) as client:
            response = client.get(f"{config.base_url}/models", headers=headers)
            response.raise_for_status()
            ids = tuple(str(entry["id"]) for entry in response.json()["data"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        why = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
        if kept is not None:
            date = datetime.fromtimestamp(kept[0], UTC).strftime("%Y-%m-%d %H:%M UTC")
            return Catalogue(kept[1], "cache", f"cache of {date} ({why})")
        return Catalogue(
            None,
            "none",
            f"the catalogue of {redact_url(config.base_url)} could not be read ({why}); model names not checked",
        )
    with contextlib.suppress(OSError):  # an unwritable cache only costs the next call a fetch
        file.parent.mkdir(parents=True, exist_ok=True)
        temp = file.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps({"fetched": now, "ids": list(ids)}), encoding="utf-8")
        temp.replace(file)
    return Catalogue(ids, "endpoint")


def bare_id(name: str) -> str:
    return name.rsplit("/", 1)[-1].lower()


def closest(name: str, ids: Iterable[str], n: int = 3) -> list[str]:
    """Return up to ``n`` ids nearest ``name``: equal but for case and vendor prefix, else difflib's closest."""
    by_bare: dict[str, list[str]] = {}
    for candidate in ids:
        by_bare.setdefault(bare_id(candidate), []).append(candidate)
    if bare_id(name) in by_bare:
        return by_bare[bare_id(name)][:n]
    picks = difflib.get_close_matches(bare_id(name), list(by_bare), n=n, cutoff=0.6)
    return [candidate for bare in picks for candidate in by_bare[bare]][:n]


def misses(names: Iterable[str], catalogue: Catalogue) -> dict[str, tuple[str, ...]]:
    """Return each name not in the catalogue with its closest ids; nothing when the catalogue is unknown."""
    if catalogue.ids is None:
        return {}
    known = set(catalogue.ids)
    return {name: tuple(closest(name, catalogue.ids)) for name in dict.fromkeys(names) if name not in known}


class ModelNameWarning(UserWarning):
    """A model name the catalogue does not hold; ``name`` and ``suggestions`` say which and what is near."""

    def __init__(self, message: str, name: str, suggestions: tuple[str, ...]) -> None:
        """Keep the message, the name and the closest ids."""
        super().__init__(message)
        self.name = name
        self.suggestions = suggestions


@dataclass(frozen=True)
class Resolution:
    """The outcome of :func:`resolve_names`.

    Attributes:
        catalogue: The catalogue the names were checked against.
        suggestions: Each missing name with its closest ids, possibly none.
        warnings: One :class:`ModelNameWarning` per missing name, for the caller to issue.
    """

    catalogue: Catalogue
    suggestions: Mapping[str, tuple[str, ...]]
    warnings: tuple[ModelNameWarning, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        """The names not in the catalogue."""
        return tuple(self.suggestions)

    @property
    def checked(self) -> bool:
        """Whether there was a catalogue to check against."""
        return self.catalogue.ids is not None


def resolve_names(
    names: Iterable[str] | Mapping[str, str], config: Config, *, catalogue: Catalogue | None = None
) -> Resolution:
    """Check model ids against the catalogue, reading it unless one is given.

    Args:
        names: Model ids, or a mapping of ids to who binds them (``"agent editor"``), named in the warning.
        config: Where the catalogue comes from.
        catalogue: A catalogue already read.

    Returns:
        The missing names with suggestions and their warnings; nothing missing when the catalogue is unknown.
    """
    catalogue = read_catalogue(config) if catalogue is None else catalogue
    found = misses(names, catalogue)
    labels = names if isinstance(names, Mapping) else {}
    where = (
        f"the catalogue of {redact_url(config.base_url)}"
        if catalogue.source != "configured" and config.base_url
        else f"the configured catalogue ({MODELS_ENV} or endpoint.catalogue)"
    )
    issued = []
    for name, near in found.items():
        who = f" ({labels[name]})" if labels.get(name) else ""
        tail = f"closest: {', '.join(near)}" if near else "nothing close"
        issued.append(ModelNameWarning(f"model {name!r}{who} is not in {where}; {tail}", name, near))
    return Resolution(catalogue, found, tuple(issued))


PARAMS: dict[str, tuple[type, ...]] = {
    "max_tokens": (int,),
    "timeout_s": (int, float),
    "temperature": (int, float),
    "top_p": (int, float),
    "seed": (int,),
    "reasoning_effort": (str,),
}
"""The model parameters a ``models:`` entry may set, with the types each takes."""

LLM_PARAMS = {"max_tokens": "max_tokens", "timeout_s": "call_timeout_s"}  # the others go out with each request


@dataclass(frozen=True)
class ModelEntry(Capabilities):
    """A ``models:`` entry that names a catalogue id or sets parameters: ``editor: {model: kimi-k3, temperature: 0.2}``.

    One id may stand under two names with different parameters. The
    parameters apply to the model objects :meth:`~xmas.MAS.load` builds; a
    model object passed in keeps its own.

    Attributes:
        model: The catalogue id the bound name stands for, or ``None`` for the name itself.
        params: Parameters from :data:`PARAMS`, in the order the entry states them.
    """

    model: str | None = None
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def parse(cls, entry: Any, where: str) -> Capabilities:
        """Read a ``models:`` entry: a :class:`ModelEntry` when it names a model or a parameter, else capabilities.

        Raises:
            ValueError: As :meth:`Capabilities.parse`, for another key, for a ``model`` that is not a catalogue id,
                or for a parameter of the wrong type.
        """
        if not isinstance(entry, Mapping):
            return Capabilities.parse(entry, where)
        for key in entry:
            if key not in ("accepts", "produces", "model", *PARAMS):
                raise ValueError(
                    f"{where} has key {key!r}; it takes model, accepts, produces and the parameters {', '.join(PARAMS)}"
                )
        stated = Capabilities.parse({key: entry[key] for key in ("accepts", "produces") if key in entry}, where)
        model = entry.get("model")
        if "model" in entry and (not isinstance(model, str) or not model or "://" in model):
            raise ValueError(f"{where} model is a catalogue id such as 'kimi-k3', not {model!r}")
        params = []
        for key, value in entry.items():
            if key in PARAMS:
                if isinstance(value, bool) or not isinstance(value, PARAMS[key]):
                    raise ValueError(f"{where} {key} is {PARAM_WORDS.get(PARAMS[key], 'a number')}, not {value!r}")
                params.append((key, value))
        if model is None and not params:
            return stated
        return cls(stated.accepts, stated.produces, model, tuple(params))

    def to_spec(self) -> dict[str, Any]:
        """Return the ``models:`` entry: the id, the stated capabilities, then the parameters."""
        return {**({"model": self.model} if self.model else {}), **super().to_spec(), **dict(self.params)}

    def llm_kwargs(self) -> dict[str, Any]:
        """Return the parameters as :class:`~xmas.llms.APILLM` takes them; ``timeout_s`` is ``call_timeout_s``."""
        kwargs: dict[str, Any] = {LLM_PARAMS[key]: value for key, value in self.params if key in LLM_PARAMS}
        sent = {key: value for key, value in self.params if key not in LLM_PARAMS}
        return {**kwargs, "call_kwargs": sent} if sent else kwargs


class Unconfigured(BaseLLM):
    """A bound model name with no endpoint configured: the system builds and checks, and its first call fails."""

    full_response = True

    def __init__(self, model: str, capabilities: Capabilities = TEXT_ONLY) -> None:
        """Keep the model id and what the ``models:`` entry says it accepts and produces."""
        super().__init__()
        self.model = model
        self.accepts, self.produces = capabilities.accepts, capabilities.produces

    def fail(self, *args: Any, **kwargs: Any) -> NoReturn:  # noqa: ARG002 (stands in for every call method)
        """Refuse the call: the model has no endpoint to send it to."""
        raise ValueError(
            f"model {self.model!r} has no endpoint: set XMAS_BASE_URL, or base_url under [endpoint] in {config_path()}"
        )

    _get_response = generate_with_meta = get_tool_response = get_tool_response_meta = fail
    generate_parts_with_meta = generate_image_with_meta = fail


def build_llms(names: Iterable[str], models: Mapping[str, Capabilities], config: Config) -> dict[str, BaseLLM]:
    """Return a model object for each bound name, on the configured endpoint.

    Each is an :class:`~xmas.llms.APILLM` for the name's catalogue id (its
    ``models:`` entry's ``model``, else the name) with the entry's
    capabilities and parameters and the configuration's key, price and
    streaming; without an endpoint, an :class:`Unconfigured` model.
    """
    built: dict[str, BaseLLM] = {}
    for name in names:
        entry = models.get(name, TEXT_ONLY)
        ident = entry.model if isinstance(entry, ModelEntry) and entry.model else name
        if not config.base_url:
            built[name] = Unconfigured(ident, entry)
            continue
        built[name] = APILLM(
            model=ident,
            base_url=config.base_url,
            api_key=config.api_key or "none",  # the client needs a key; an endpoint without one ignores it
            price=config.prices.get(ident),
            stream=config.stream_of(ident),
            accepts=entry.accepts or ("text",),
            produces=entry.produces or ("text",),
            **(entry.llm_kwargs() if isinstance(entry, ModelEntry) else {}),
        )
    return built


def tool_sources(servers: Mapping[str, Mapping[str, Mapping[str, str]]], config: Config) -> list[ToolSource]:
    """Return one MCP tool source per named server, at its configured address.

    Args:
        servers: Per server name, the callables it serves with what each writes.
        config: Where the servers' addresses and keys come from.

    Raises:
        ValueError: For a server the configuration does not list; the message names both remedies.
    """
    from xmas.tools.mcp import MCPTools  # noqa: PLC0415  here, not at the top: MCP is an extra

    sources: list[ToolSource] = []
    for server, writes in servers.items():
        found = config.tools.get(server)
        if found is None:
            raise ValueError(
                f"Callables {sorted(writes)} name the server {server!r}, which the configuration does not list: "
                f"add [tools.{server}] url to {config_path()}, or pass tools=[...] offering them."
            )
        sources.append(MCPTools(found.url, found.key, writes=dict(writes), name=server))
    return sources
