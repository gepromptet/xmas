"""The ``.mas`` bundle: a system saved as a folder of plain files, or the same files zipped into one.

.. code-block:: text

    deck.mas/
      xmas.yaml          the spec, MAS.to_dict(), without its prompts and callables; boxes as paths
      callables.yaml     the callables section, when there is one
      prompts/<name>.md  one prompt per agent, or per shared prompt name, as the model sees it
      boxes/<box>.mas/   each box's system, a bundle of its own (no lock)
      optimize.yaml      optional settings for the optimizers; never written by save
      examples/*.yaml    optional warm-up cases; never written by save
      xmas.lock          the sha256 of every other file and of each python callable's source

A prompt file holds the text and one final newline. Its optional front matter, a
``---`` line, YAML and a ``---`` line, holds only ``schema``, the answer form,
written when every agent using the prompt has the same one; otherwise each agent
keeps its ``schema`` in ``xmas.yaml``. A bundle names its models and tool servers
and never holds an address or a key: those come from :mod:`xmas.config`.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import os
import stat
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

from xmas.prompts import extract_placeholders
from xmas.spec import V2_KEYS, VERSION_2

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator, Mapping

    from xmas.mas import MAS

__all__ = [
    "CALLABLES",
    "LOCK",
    "MAX_BYTES",
    "OPTIMIZE",
    "SUFFIX",
    "WIRING",
    "Lock",
    "LockDiff",
    "check_lock",
    "code_hashes",
    "compare",
    "dump_prompt",
    "dump_yaml",
    "files_of",
    "lock_of",
    "outside",
    "owned",
    "parse_prompt",
    "put",
    "read",
    "read_optimize",
    "read_yaml",
    "relock",
    "same",
    "spec_of",
    "systems",
    "write",
]


SUFFIX = ".mas"
WIRING = "xmas.yaml"
CALLABLES = "callables.yaml"
OPTIMIZE = "optimize.yaml"
LOCK = "xmas.lock"
MAX_BYTES = 64 * 2**20

PROMPTS = "prompts/"
BOXES = "boxes/"
LOCK_NOTE = "# written by MAS.save; do not edit\n"
OPTIMIZE_KEYS = ("optimizer", "task", "tune")
TUNE = ("prompts", "models", "edges")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def file_name(name: str, what: str) -> str:
    if not name or name[0] in ".-" or not all(c.isascii() and (c.isalnum() or c in "_.-") for c in name):
        raise ValueError(f"{what} name {name!r} cannot be a file name in a bundle: use letters, digits, _ . and -.")
    return name


def dump_yaml(data: Any) -> bytes:
    """Dump ``data`` as a bundle file holds YAML: key order kept, unicode as is."""
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120).encode("utf-8")


def file_text(files: Mapping[str, bytes], path: str) -> str:
    try:
        return files[path].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not UTF-8 text: {error}") from None


def read_yaml(files: Mapping[str, bytes], path: str) -> Any:
    """Return the YAML file at ``path``; a file that is not UTF-8 or not YAML is a ValueError naming it."""
    try:
        return yaml.safe_load(file_text(files, path))
    except yaml.YAMLError as error:
        raise ValueError(f"{path} is not YAML: {error}") from None


def mapping(files: Mapping[str, bytes], path: str) -> dict[str, Any]:
    data = read_yaml(files, path)
    if not isinstance(data, dict | None):
        raise ValueError(f"{path} must hold a mapping, not {type(data).__name__}.")
    return data or {}


def dump_prompt(text: str, schema: Mapping[str, Any] | None) -> str:
    """Return a prompt file: the front matter when there is a schema, the text, one final newline.

    A text whose first line is ``---`` gets an empty front matter, so it is never
    read as one.
    """
    if schema is not None:
        head = "---\n" + yaml.safe_dump({"schema": dict(schema)}, sort_keys=False, allow_unicode=True) + "---\n"
    elif text.split("\n", 1)[0] in ("---", "---\r"):
        head = "---\n---\n"
    else:
        head = ""
    return f"{head}{text}\n"


def parse_prompt(raw: str, where: str) -> tuple[str, dict[str, Any] | None]:
    """Return a prompt file's text and its front matter's schema, ``None`` when it has none.

    One final newline is the file's and is dropped; ``where`` names the file in errors. A file
    whose front matter has Windows line ends is read with plain newlines throughout.

    Raises:
        ValueError: For front matter that is never closed, is not YAML, or holds anything but a ``schema`` mapping.
    """
    if raw.startswith("---\r\n"):
        raw = raw.replace("\r\n", "\n")
    text, schema = raw, None
    lines = raw.split("\n")
    if lines[0] == "---":
        try:
            end = lines.index("---", 1)
        except ValueError:
            raise ValueError(f"{where}: the front matter opened on line 1 is never closed by a '---' line") from None
        try:
            head = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as error:
            raise ValueError(f"{where}: the front matter is not YAML: {error}") from None
        head = {} if head is None else head
        if not isinstance(head, dict) or set(head) - {"schema"}:
            got = sorted(head) if isinstance(head, dict) else type(head).__name__
            raise ValueError(f"{where}: the front matter holds only 'schema', got {got}")
        if "schema" in head:
            if not isinstance(head["schema"], dict):
                raise ValueError(f"{where}: 'schema' maps answer fields to their forms, got {head['schema']!r}")
            schema = head["schema"]
        text = "\n".join(lines[end + 1 :])
    return text.removesuffix("\n"), schema


def files_of(mas: MAS) -> dict[str, bytes]:
    """Return the files a save writes for ``mas``, by bundle path: every owned file but the lock."""
    return dict(sorted(bundle_files(mas.to_dict(), "").items()))


def bundle_files(spec: dict[str, Any], prefix: str) -> dict[str, bytes]:
    spec = copy.deepcopy(spec)
    files: dict[str, bytes] = {}
    if spec.get("version") == VERSION_2:
        agents = spec.get("agents", [])
        for key, text in spec.pop("prompts", {}).items():
            sharers = [agent for agent in agents if agent.get("prompt", agent["name"]) == key]
            schemas = [agent.get("schema") for agent in sharers]
            schema = schemas[0] if schemas[0] is not None and all(s == schemas[0] for s in schemas) else None
            if schema is not None:
                for agent in sharers:
                    del agent["schema"]
            files[f"{prefix}{PROMPTS}{file_name(key, 'Prompt')}.md"] = dump_prompt(text, schema).encode("utf-8")
        for box in spec.get("boxes", []):
            path = f"{BOXES}{file_name(box['name'], 'Box')}{SUFFIX}"
            files.update(bundle_files(box["system"], f"{prefix}{path}/"))
            box["system"] = path
        callables = spec.pop("callables", None)
        if callables:
            files[prefix + CALLABLES] = dump_yaml(callables)
    else:
        for agent in spec["agents"]:
            text = agent.pop("prompt")
            files[f"{prefix}{PROMPTS}{file_name(agent['name'], 'Agent')}.md"] = dump_prompt(text, None).encode("utf-8")
    files[prefix + WIRING] = dump_yaml(spec)
    return files


def spec_of(files: Mapping[str, bytes], prefix: str = "") -> dict[str, Any]:
    """Return the spec :meth:`xmas.MAS.from_dict` reads, from a bundle's files (the one under ``prefix``).

    Raises:
        ValueError: For a missing file, a file that is not what its place says, a
            schema both in a prompt's front matter and in ``xmas.yaml``, the
            callables in both files, a prompt file no agent uses, a placeholder
            that is not a read, or a box path outside the bundle.
    """
    where = prefix + WIRING
    spec = wiring_of(files, prefix)
    start = prefix + PROMPTS
    texts = prompt_files(files, start)
    v2 = spec.get("version") == VERSION_2
    agents = prompt_keys(spec, v2=v2)
    stray = sorted(set(texts) - {key for _, key in agents})
    if stray:
        raise ValueError(f"{start}{stray[0]}.md belongs to no agent.")
    for agent, key in agents:
        path = f"{start}{key}.md"
        if key not in texts:
            raise ValueError(f"Agent {agent.get('name')!r} has no prompt file {path}.")
        text, schema = texts[key]
        if v2:
            attach_v2_prompt(agent, schema, text, path, where)
        else:
            attach_v1_prompt(agent, schema, text, path)
    if not v2:
        return spec
    spec["prompts"] = {key: text for key, (text, _) in texts.items()}
    for box in spec.get("boxes") or []:
        if isinstance(box, dict) and isinstance(box.get("system"), str):
            box["system"] = spec_of(files, inner(prefix, box["system"]))
    return {**{key: spec[key] for key in V2_KEYS if key in spec}, **{k: v for k, v in spec.items() if k not in V2_KEYS}}


def wiring_of(files: Mapping[str, bytes], prefix: str) -> dict[str, Any]:
    """Return ``xmas.yaml`` under ``prefix`` with the callables of ``callables.yaml`` put in; it holds no prompts."""
    where = prefix + WIRING
    if where not in files:
        raise ValueError(f"The bundle has no {where}.")
    spec = mapping(files, where)
    if prefix + CALLABLES in files:
        if "callables" in spec:
            raise ValueError(f"The callables are in both {prefix}{CALLABLES} and {where}; keep one.")
        spec["callables"] = mapping(files, prefix + CALLABLES)
    if "prompts" in spec:
        raise ValueError(f"{where} holds prompts; each prompt is a file {prefix}{PROMPTS}<name>.md.")
    return spec


def prompt_files(files: Mapping[str, bytes], start: str) -> dict[str, tuple[str, dict[str, Any] | None]]:
    """Return the text and schema of each prompt file directly under ``start``, by its name without ``.md``."""
    return {
        path[len(start) : -len(".md")]: parse_prompt(file_text(files, path), path)
        for path in sorted(files)
        if path.startswith(start) and path.endswith(".md") and "/" not in path[len(start) :]
    }


def prompt_keys(spec: dict[str, Any], *, v2: bool) -> list[tuple[dict[str, Any], Any]]:
    """Pair each agent with the prompt file name it reads: its shared ``prompt`` name in version 2, else its name."""
    return [
        (agent, agent.get("prompt", agent.get("name")) if v2 else agent.get("name"))
        for agent in spec.get("agents") or []
        if isinstance(agent, dict)
    ]


def attach_v1_prompt(agent: dict[str, Any], schema: dict[str, Any] | None, text: str, path: str) -> None:
    """Put a version 1 agent's prompt text in second place; its file may carry no schema."""
    if schema is not None:
        raise ValueError(f"{path}: a version 1 agent has no answer form, so its file has no schema.")
    insert(agent, 1, "prompt", text)


def attach_v2_prompt(agent: dict[str, Any], schema: dict[str, Any] | None, text: str, path: str, where: str) -> None:
    """Put the file's schema into a version 2 agent (before ``retry``) and check its placeholders are reads."""
    if schema is not None:
        if "schema" in agent:
            raise ValueError(f"Agent {agent.get('name')!r} has a schema in both {path} and {where}; keep one.")
        insert(agent, len(agent) - ("retry" in agent), "schema", copy.deepcopy(schema))
    reads = agent.get("reads", ["query"])
    if isinstance(reads, list | dict):
        missing = sorted(extract_placeholders(text) - set(reads))
        if missing:
            raise ValueError(
                f"{path}: prompt placeholders {missing} are not among the reads of agent "
                f"{agent.get('name')!r} {list(reads)}."
            )


def insert(entry: dict[str, Any], index: int, key: str, value: Any) -> None:
    items = list(entry.items())
    items.insert(index, (key, value))
    entry.clear()
    entry.update(items)


def outside(path: str) -> bool:
    """Whether a posix ``path`` leaves the folder it is relative to: absolute, with ``..``, a drive or a backslash."""
    parts = PurePosixPath(path).parts
    if not parts:
        return True
    absolute = path.startswith("/") or ":" in parts[0]
    climbs = ".." in parts
    windows_separator = "\\" in path
    return absolute or climbs or windows_separator


def inner(prefix: str, path: str) -> str:
    if outside(path):
        raise ValueError(f"A box's system {path!r} is outside the bundle; it is a path such as 'boxes/name.mas'.")
    return f"{prefix}{'/'.join(PurePosixPath(path).parts)}/"


def read_optimize(files: Mapping[str, bytes]) -> dict[str, Any]:
    """Return ``optimize.yaml``, checked, or an empty mapping when the bundle has none.

    Raises:
        ValueError: For a key other than task, tune and optimizer; a task that is
            not one ``examples`` file or one ``import``; a word in ``tune`` other
            than prompts, models and edges; an optimizer without a ``name``.
    """
    if OPTIMIZE not in files:
        return {}
    data = mapping(files, OPTIMIZE)
    unknown = sorted(set(data) - set(OPTIMIZE_KEYS))
    if unknown:
        raise ValueError(f"{OPTIMIZE} has unknown keys {unknown}; it takes {list(OPTIMIZE_KEYS)}.")
    task = data.get("task", {"examples": ""})
    if not isinstance(task, dict) or len(task) != 1 or not isinstance(task.get("examples", task.get("import")), str):
        raise ValueError(f"{OPTIMIZE} task is {{examples: <file>}} or {{import: 'module:function'}}, not {task!r}.")
    tune = data.get("tune", [])
    if not isinstance(tune, list) or any(word not in TUNE for word in tune):
        raise ValueError(f"{OPTIMIZE} tune is a list of {list(TUNE)}, not {tune!r}.")
    optimizer = data.get("optimizer", {"name": ""})
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("name"), str):
        raise ValueError(f"{OPTIMIZE} optimizer is a mapping with a 'name', not {optimizer!r}.")
    return data


@dataclass(frozen=True)
class Lock:
    """The checksums a save writes into ``xmas.lock``.

    Attributes:
        bundle: The sha256 over every file's ``"path sha256"`` line, sorted: the bundle's version id.
        files: The sha256 of every file but the lock, by bundle path.
        code: The sha256 of each python callable's source, ``None`` where it has none to read.
    """

    bundle: str
    files: dict[str, str]
    code: dict[str, str | None]

    def to_yaml(self) -> str:
        """Return the lock file's text."""
        data = {"version": 1, "bundle": self.bundle, "files": self.files, "code": self.code}
        return LOCK_NOTE + yaml.safe_dump(data, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> Lock:
        """Read a lock file's text.

        Raises:
            ValueError: If it is not a lock :meth:`to_yaml` wrote, or its bundle digest is not that of its files.
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = None
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("bundle"), str)
            or not isinstance(data.get("files"), dict)
            or not isinstance(data.get("code") or {}, dict)
            or data["bundle"] != digest(data["files"])
        ):
            raise ValueError(f"{LOCK} is not a lock written by MAS.save.")
        return cls(data["bundle"], dict(data["files"]), dict(data.get("code") or {}))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(digests: Mapping[str, Any]) -> str:
    return sha("".join(f"{path} {sha}\n" for path, sha in sorted(digests.items())).encode("utf-8"))


def lock_of(files: Mapping[str, bytes], code: Mapping[str, str | None]) -> Lock:
    """Return the lock of a bundle's files (the lock itself left out) and its callables' source hashes."""
    digests = {path: sha(data) for path, data in sorted(files.items()) if path != LOCK}
    return Lock(digest(digests), digests, dict(sorted(code.items())))


def relock(folder: Path, code: Mapping[str, str | None]) -> None:
    """Write a bundle folder's lock for the files it holds now."""
    put(folder / LOCK, lock_of(read(folder), code).to_yaml().encode("utf-8"))


def systems(mas: MAS, prefix: str = "") -> Iterator[tuple[str, MAS]]:
    """Yield the system and each box's system at every depth, with the prefix its callables are named under."""
    yield prefix, mas
    for name, box in mas.boxes.items():
        yield from systems(box.system, f"{prefix}{name}/")


def code_hashes(mas: MAS) -> dict[str, str | None]:
    """Return the sha256 of each python callable's source, a box's under ``box/name``; ``None`` without source."""
    hashes: dict[str, str | None] = {}
    for prefix, system in systems(mas):
        used = [system.resolve_callable(step)[0] for step in system.steps.values()]
        used += [
            item for agent in system.agents.values() if agent.tools for item, _ in system.resolve_tools(agent).values()
        ]
        for item in used:
            if item.fn is not None:
                try:
                    hashes[prefix + item.name] = sha(inspect.getsource(item.fn).encode("utf-8"))
                except (OSError, TypeError):
                    hashes[prefix + item.name] = None
    return dict(sorted(hashes.items()))


@dataclass(frozen=True)
class LockDiff:
    """How a bundle's files and code differ from its lock.

    Attributes:
        changed: Files whose content changed.
        added: Files the lock does not list.
        missing: Files the lock lists that are gone.
        code_changed: Python callables whose source changed; one without source on either side is skipped.
    """

    changed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    code_changed: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """Whether nothing differs."""
        return not (self.changed or self.added or self.missing or self.code_changed)


def compare(lock: Lock, files: Mapping[str, bytes], code: Mapping[str, str | None] | None = None) -> LockDiff:
    """Return how ``files`` (a :func:`read`, the lock among them or not) and ``code`` differ from ``lock``."""
    now = lock_of(files, {}).files
    return LockDiff(
        changed=tuple(sorted(path for path, sha in now.items() if lock.files.get(path, sha) != sha)),
        added=tuple(sorted(set(now) - set(lock.files))),
        missing=tuple(sorted(set(lock.files) - set(now))),
        code_changed=tuple(
            sorted(
                name
                for name, sha in (code or {}).items()
                if sha is not None and lock.code.get(name) is not None and lock.code[name] != sha
            )
        ),
    )


def check_lock(files: Mapping[str, bytes], code: Mapping[str, str | None]) -> tuple[Lock | None, LockDiff]:
    """Return the lock among a bundle's files and how the files and ``code`` differ from it.

    A lock edited by hand is ``None``, and the lock file itself is what changed.
    """
    try:
        lock = Lock.from_yaml(files[LOCK].decode("utf-8", "replace"))
    except ValueError:
        return None, LockDiff(changed=(LOCK,))
    return lock, compare(lock, files, code)


def read(path: str | os.PathLike[str]) -> dict[str, bytes]:
    """Return every file of a bundle by its posix path inside it: a folder, or a zip (its entries under one folder too).

    Raises:
        FileNotFoundError: If nothing is at ``path``.
        ValueError: For a symlink, an entry that leaves the bundle or is there
            twice, more than :data:`MAX_BYTES`, or a file that is no zip or a
            broken one.
    """
    source = Path(path)
    if source.is_dir():
        return read_folder(source)
    if source.is_file():
        return read_zip(source)
    raise FileNotFoundError(f"There is no bundle at {source}.")


def read_folder(source: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    total = 0
    for item in sorted(source.rglob("*")):
        name = item.relative_to(source).as_posix()
        if item.is_symlink():
            raise ValueError(f"{source}: {name} is a symlink; a bundle holds plain files.")
        if item.is_file():
            total += item.stat().st_size
            if total > MAX_BYTES:
                raise ValueError(f"{source} holds more than {MAX_BYTES} bytes.")
            files[name] = item.read_bytes()
    return files


def read_zip(source: Path) -> dict[str, bytes]:
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile:
        raise ValueError(f"{source} is neither a bundle folder nor a zipped bundle.") from None
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        seen: set[str] = set()
        for info in infos:
            name = info.filename
            if name in seen:
                raise ValueError(f"{source}: the entry {name!r} is there twice.")
            seen.add(name)
            if outside(name):
                raise ValueError(f"{source}: the entry {name!r} points outside the bundle.")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"{source}: the entry {name!r} is a symlink; a bundle holds plain files.")
        if sum(info.file_size for info in infos) > MAX_BYTES:
            raise ValueError(f"{source} holds more than {MAX_BYTES} bytes.")
        try:
            files = {info.filename: archive.read(info) for info in infos}
        except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError, NotImplementedError) as error:
            raise ValueError(f"{source} is a broken zip: {error}") from None
    tops = {name.split("/", 1)[0] for name in files}
    if WIRING not in files and len(tops) == 1 and all("/" in name for name in files):
        (top,) = tops
        files = {name[len(top) + 1 :]: data for name, data in files.items()}
    return files


def owned(path: str) -> bool:
    """Whether a save writes, rewrites and deletes the bundle file at ``path``."""
    return path in (WIRING, CALLABLES) or path.startswith((PROMPTS, BOXES))


def canonical(path: str, data: bytes) -> Any:
    text = data.decode("utf-8")
    if path.endswith(".md"):
        return parse_prompt(text, path)
    return json.dumps(yaml.safe_load(text), sort_keys=True, default=str)


def same(path: str, old: bytes | None, new: bytes | None) -> bool:
    """Whether two versions of a bundle file hold the same parsed content: YAML as loaded, prompts as text and form."""
    if old is None or new is None or old == new:
        return old == new
    try:
        return bool(canonical(path, old) == canonical(path, new))
    except (ValueError, yaml.YAMLError):  # UnicodeDecodeError is a ValueError
        return False


def put(path: Path, data: bytes) -> None:
    """Write a file, unless it holds ``data`` already, through a temporary file beside it and a rename."""
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        Path(temp).chmod(0o644)
        Path(temp).replace(path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def write(
    path: str | os.PathLike[str],
    files: Mapping[str, bytes],
    code: Mapping[str, str | None],
    *,
    zipped: bool = False,
) -> Path:
    """Write a bundle's owned ``files`` and its lock at ``path``, a folder or (``zipped``) one zip file.

    A folder keeps every file whose parsed content is unchanged, deletes owned
    files no longer given, and never touches the others (``examples/``,
    ``optimize.yaml``). A zip carries those others over and is byte-reproducible.

    Raises:
        ValueError: If the name does not end in ``.mas``, a folder is to be
            zipped or a zip unzipped, or the folder is not empty and holds no
            ``xmas.yaml``.
    """
    target = Path(path)
    if target.suffix != SUFFIX:
        raise ValueError(
            f"A bundle is named name{SUFFIX}, not {target.name!r}; to_yaml(path) writes a version 1 system as one file."
        )
    if zipped and target.is_dir():
        raise ValueError(f"{target} is a bundle folder; save(path) without zipped=True rewrites it.")
    if not zipped and target.is_file():
        raise ValueError(f"{target} is a zipped bundle; save(path, zipped=True) rewrites it.")
    return write_zip(target, files, code) if zipped else write_folder(target, files, code)


def write_folder(target: Path, files: Mapping[str, bytes], code: Mapping[str, str | None]) -> Path:
    old: dict[str, bytes] = {}
    if target.is_dir():
        if any(target.iterdir()) and not (target / WIRING).is_file():
            raise ValueError(f"{target} is a folder that holds no {WIRING}; a save writes only into a bundle.")
        old = read(target)
    for name, data in files.items():
        if not same(name, old.get(name), data):
            put(target / name, data)
    for name in old:
        if owned(name) and name not in files:
            (target / name).unlink()
            for parent in (target / name).parents:
                if parent == target or any(parent.iterdir()):
                    break
                parent.rmdir()
    relock(target, code)
    return target


def write_zip(target: Path, files: Mapping[str, bytes], code: Mapping[str, str | None]) -> Path:
    """Write sorted entries with a fixed date and mode, deflated; the entries a save does not own carried over."""
    kept = {name: data for name, data in read(target).items() if not owned(name)} if target.is_file() else {}
    entries = {**kept, **files}
    entries[LOCK] = lock_of(entries, code).to_yaml().encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, entries[name])
    put(target, buffer.getvalue())
    return target
