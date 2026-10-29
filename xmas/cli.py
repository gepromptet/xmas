"""The ``xmas`` command: make, check, show, run, compare and verify ``.mas`` bundles, and read the run log.

``xmas init``, ``check``, ``show``, ``run``, ``diff`` and ``verify`` each take a
bundle, a folder or a zip; ``xmas runs`` and ``xmas trace`` read the run log
(:func:`~xmas.store.open_store`). Exit status 0 means ok, 1 a problem (or, with
``--strict``, a failed verify), 2 a usage error or a bundle that is not there.
Model endpoints and keys come from the configuration (:mod:`xmas.config`) and
are never printed. A bundle's python callables are imported only with
``--trust``; then, as with ``python -m``, the current folder joins
``sys.path`` for the command, so they import from it too.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
import warnings
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xmas.agent import Agent
from xmas.budget import Budget
from xmas.bundle import LOCK, Lock, compare, read
from xmas.config import Config, Unconfigured, config_path
from xmas.examples import parse_examples
from xmas.mas import MAS
from xmas.store import open_store
from xmas.values import TEXT, ValueType, plain, ref_kind
from xmas.verify import VerifyWarning

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

__all__ = [
    "main",
]


EXAMPLES = """\
version: 1
# Warm-up cases: verify(2) and `xmas verify --level 2` run them under a small budget.
examples:
- name: first-answer
  inputs: {query: "Why is the sky blue?"}
  expect:
    answer: [{not empty: true}]
"""

OPTIMIZE = """\
# Optional: what an optimizer may tune, for example
# task: {examples: examples/warmup.yaml}
# tune: [prompts, models]
# optimizer: {name: cantante}
"""


class UsageError(Exception):
    """A command line that cannot be carried out: exit status 2."""


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmas", description="Make, check, run and verify .mas bundles; read the run log."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="write a new bundle to edit")
    init.add_argument("bundle", help="the new bundle, name.mas")
    init.add_argument("--agents", default="assistant", help="agent names, comma separated, run in that order")
    init.add_argument("--model", default="text-model", help="the model name the agents bind")
    check = commands.add_parser("check", help="load a bundle and check it, with no model call")
    check.add_argument("bundle")
    show = commands.add_parser("show", help="the system's nodes and edges")
    show.add_argument("bundle")
    show.add_argument("--json", action="store_true", help="describe() as JSON")
    run = commands.add_parser("run", help="run the system on the configured endpoint")
    run.add_argument("bundle")
    run.add_argument("--query", help="the text of the input query")
    run.add_argument("--input", action="append", default=[], metavar="NAME=VALUE", help="an input value")
    run.add_argument("--budget", type=float, metavar="USD", help="the most the run may spend")
    diff = commands.add_parser("diff", help="file differences of two bundles, or of one bundle and its lock")
    diff.add_argument("bundle")
    diff.add_argument("other", nargs="?")
    diff.add_argument("--json", action="store_true")
    verify = commands.add_parser("verify", help="checksums, the dry check and the warm-up examples")
    verify.add_argument("bundle")
    verify.add_argument("--level", type=int, choices=(0, 1, 2), default=1)
    verify.add_argument("--strict", action="store_true", help="exit 1 when the report is not ok")
    verify.add_argument("--budget", type=float, metavar="USD", help="the most level 2 may spend")
    verify.add_argument("--only", action="append", default=[], metavar="NAME", help="an example or node to run")
    verify.add_argument("--json", action="store_true")
    for command in (check, show, run, verify):
        command.add_argument("--trust", action="store_true", help="import the code the bundle names")
    runs = commands.add_parser("runs", help="the runs in the run log, oldest first")
    runs.add_argument("--json", action="store_true")
    runs.add_argument("--status", help="only the runs with this status, such as ok, failed or paused")
    runs.add_argument("--system", metavar="ID", help="only the runs of the system with this describe() id")
    runs.add_argument("--limit", type=int, metavar="N", help="only the latest N runs")
    trace = commands.add_parser("trace", help="one run's records, from the run log")
    trace.add_argument("run_id")
    trace.add_argument("--json", action="store_true", help="the whole trace as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``xmas`` command and return its exit status."""
    args = parser().parse_args(argv)
    here = str(Path.cwd())
    joined = getattr(args, "trust", False) and here not in sys.path
    if joined:
        sys.path.insert(0, here)
    try:
        with warnings_as_lines():
            return COMMANDS[args.command](args)
    except (UsageError, ValueError) as error:
        print(f"xmas {args.command}: {error}", file=sys.stderr)
        return 2 if isinstance(error, UsageError) else 1
    finally:
        if joined:
            sys.path.remove(here)


@contextmanager
def warnings_as_lines() -> Iterator[None]:
    """Print each warning as one ``warning:`` line; a verify report already holds its own."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for item in caught:
                if not issubclass(item.category, VerifyWarning):
                    print(f"warning: {item.message}", file=sys.stderr)


def load(args: argparse.Namespace, **options: Any) -> MAS:
    if not Path(args.bundle).exists():
        raise UsageError(f"no bundle at {args.bundle}")
    return MAS.load(args.bundle, trust=args.trust, **options)


def init_command(args: argparse.Namespace) -> int:
    path = Path(args.bundle)
    if path.exists():
        raise UsageError(f"{path} exists; init writes a new bundle")
    if path.suffix != ".mas":
        raise UsageError(f"a bundle is named name.mas, not {path.name!r}")
    names = [name.strip() for name in args.agents.split(",") if name.strip()]
    if not names:
        raise UsageError("--agents names at least one agent")
    agents, previous = [], None
    for index, name in enumerate(names):
        output = "answer" if index == len(names) - 1 else f"{name}_draft"
        if previous is None:
            agents.append(Agent(name, "Answer the question: {query}", output, inputs=["query"], model=args.model))
        else:
            prompt = f"Question: {{query}}\nDraft: {{{previous}}}\nImprove the draft."
            agents.append(Agent(name, prompt, output, inputs=["query", previous], model=args.model))
        previous = output
    mas = MAS(
        agents,
        list(pairwise(names)),
        outputs={"answer": "text"},
        llms={args.model: Unconfigured(args.model)},
        version=2,
    )
    mas.save(path)
    (path / "examples").mkdir()
    (path / "examples" / "warmup.yaml").write_text(EXAMPLES, encoding="utf-8")
    (path / "optimize.yaml").write_text(OPTIMIZE, encoding="utf-8")
    mas.save(path)  # the lock now covers the examples
    print(f"wrote {path}: edit its prompts/ and the models in xmas.yaml, then run: xmas check {path}")
    return 0


def summary(description: Mapping[str, Any]) -> str:
    edges = [edge for edge in description["edges"] if not edge.get("implicit")]
    chains = [group for group in description.get("groups", []) if "for each" in group]
    counts = {"node": len(description["nodes"]), "edge": len(edges), "chain": len(chains)}
    return ", ".join(f"{n} {word}{'s' * (n != 1)}" for word, n in counts.items())


def check_command(args: argparse.Namespace) -> int:
    mas = load(args, resolve=False)
    problems: list[str] = []
    parse_examples(read(args.bundle), Path(args.bundle) if Path(args.bundle).is_dir() else None, problems=problems)
    for problem in problems:
        print(f"- {problem}")
    print(f"{len(problems)} problem(s)" if problems else f"ok: {summary(mas.describe())}")
    return 1 if problems else 0


def words(values: Mapping[str, str]) -> str:
    return ", ".join(f"{name} {vtype}" for name, vtype in values.items())


def show_command(args: argparse.Namespace) -> int:
    description = load(args, resolve=False).describe()
    if args.json:
        print(json.dumps(description, indent=2))
        return 0
    lines = [f"{Path(args.bundle).name}: version {description['version']}, {summary(description)}"]
    lines.append(f"inputs: {words(description['inputs'])}")
    if description.get("outputs"):
        lines.append(f"outputs: {words(description['outputs'])}")
    nodes = description["nodes"]
    width = max(len(node["name"]) for node in nodes)
    lines.append("nodes:")
    for node in nodes:
        what = node.get("model") or node.get("uses") or node.get("form") or ""
        row = f"  {node['name']:<{width}}  {node['kind']:<5}  {what}  {words(node.get('reads', {}))}"
        row += f" -> {words(node.get('writes', {}))}"
        if node.get("for each"):
            row += f"  for each {node['for each']}"
        lines.append(row)
    lines.append("edges:")
    for edge in description["edges"]:
        if edge.get("implicit"):
            continue
        rest = [str(value) for key, value in edge.items() if key not in ("from", "to", "kind", "max_visits")]
        how = " ".join([*([edge["kind"]] if edge["kind"] != "plain" else []), *rest])
        cap = f" (max {edge['max_visits']})" if "max_visits" in edge else ""
        lines.append(f"  {edge['from']} -> {edge['to']}" + (f"  {how}{cap}" if how or cap else ""))
    print("\n".join(lines))
    return 0


def value(text: str, vtype: ValueType) -> Any:
    if vtype == TEXT:
        return text
    if vtype.item is None and ref_kind(vtype) is not None:
        return Path(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise UsageError(f"a {vtype} input is JSON, not {text!r}") from None
    return [Path(item) for item in parsed] if ref_kind(vtype) is not None and isinstance(parsed, list) else parsed


def run_command(args: argparse.Namespace) -> int:
    config = Config.load()
    if not config.base_url:
        raise ValueError(f"no endpoint: set XMAS_BASE_URL, or base_url under [endpoint] in {config_path()}")
    mas = load(args, config=config)
    values: dict[str, Any] = {} if args.query is None else {"query": args.query}
    for pair in args.input:
        name, sep, text = pair.partition("=")
        if not sep or name not in mas.inputs:
            raise UsageError(f"--input takes NAME=VALUE with NAME one of {', '.join(mas.inputs)}, not {pair!r}")
        values[name] = value(text, mas.inputs[name])
    if args.budget is not None:
        mas.budget = Budget(max_cost_usd=args.budget)
    started = time.perf_counter()
    trace = mas.run_state(values)
    took = time.perf_counter() - started
    outputs = trace.outputs or {"output": trace.output}
    if len(outputs) == 1 and isinstance(next(iter(outputs.values())), str):
        print(next(iter(outputs.values())))
    else:
        print(json.dumps(plain(outputs), indent=2, ensure_ascii=False))
    cost = "cost unknown" if trace.cost_usd is None else f"{trace.cost_usd:.4f} USD"
    calls = sum(record.kind == "agent" for record in trace.records)
    print(f"{trace.status}: {calls} model calls, {cost}, {took:.1f} s", file=sys.stderr)
    if trace.failed:
        print(f"{trace.error_kind or 'error'}: {trace.error}", file=sys.stderr)
    return 1 if trace.failed else 0


def diff_command(args: argparse.Namespace) -> int:
    if not Path(args.bundle).exists() or (args.other and not Path(args.other).exists()):
        raise UsageError(f"no bundle at {args.bundle if not Path(args.bundle).exists() else args.other}")
    left = read(args.bundle)
    if args.other is None:
        if LOCK not in left:
            raise ValueError(f"{args.bundle} has no {LOCK}; give a second bundle to compare with")
        lock = Lock.from_yaml(left.pop(LOCK).decode("utf-8"))
        found = compare(lock, left)
        result = {key: list(getattr(found, key)) for key in ("changed", "added", "missing")}
        lines = [f"{label}: {path}" for label, paths in result.items() for path in paths]
        print(json.dumps(result, indent=2) if args.json else "\n".join(lines) or f"unchanged since saved ({LOCK})")
        return 0 if found.clean else 1
    right = read(args.other)
    left.pop(LOCK, None)
    right.pop(LOCK, None)
    changed = {
        path: "".join(
            difflib.unified_diff(
                left[path].decode("utf-8", "replace").splitlines(keepends=True),
                right[path].decode("utf-8", "replace").splitlines(keepends=True),
                f"{args.bundle}/{path}",
                f"{args.other}/{path}",
            )
        )
        for path in sorted(set(left) & set(right))
        if left[path] != right[path]
    }
    report = {"added": sorted(set(right) - set(left)), "removed": sorted(set(left) - set(right)), "changed": changed}
    lines = [f"{label}: {path}" for label in ("added", "removed") for path in report[label]]
    lines += [text.rstrip("\n") for text in changed.values()]
    print(json.dumps(report, indent=2) if args.json else "\n".join(lines) or "same files")
    return 1 if report["added"] or report["removed"] or changed else 0


def verify_command(args: argparse.Namespace) -> int:
    mas = load(args, resolve=False)
    report = mas.verify(args.level, budget=args.budget, only=args.only)
    print(json.dumps(report.to_dict(), indent=2) if args.json else str(report))
    return 1 if args.strict and not report.ok else 0


def run_cost(totals: Mapping[str, Any] | None) -> float | None:
    return None if not totals or totals["uncosted_calls"] else float(totals["cost_usd"])


def runs_command(args: argparse.Namespace) -> int:
    store = open_store()
    rows = store.runs(args.status, args.system, args.limit)
    if args.json:
        keys = ("run_id", "status", "system_id", "created_at", "ended_at")
        print(
            json.dumps([{**{key: getattr(row, key) for key in keys}, "cost_usd": run_cost(row.totals)} for row in rows])
        )
        return 0
    for row in rows:
        cost = run_cost(row.totals)
        shown = "cost unknown" if cost is None else f"{cost:.4f} USD"
        print(f"{row.run_id}  {row.status:<11}  {row.system_id}  {row.created_at[:19]}  {shown}")
    if not rows:
        print(f"no runs in {store!r}", file=sys.stderr)
    return 0


def trace_command(args: argparse.Namespace) -> int:
    store = open_store()
    try:
        trace = store.trace(args.run_id)
    except KeyError:
        raise ValueError(f"no run {args.run_id!r} in {store!r}") from None
    if args.json:
        print(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False))
        return 0
    rows = [("step", "node", "kind", "item", "cost_usd", "latency_s", "finish")]
    for record in trace.records:
        item = "" if record.item is None else str(record.item) if record.item_key is None else record.item_key
        cost = "" if record.cost_usd is None else f"{record.cost_usd:.4f}"
        latency = "" if record.latency_s is None else f"{record.latency_s:.2f}"
        finish = record.error_kind or record.finish_reason or ("error" if record.error else "")
        rows.append((str(record.step), record.agent, record.kind, item, cost, latency, str(finish)))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
    cost = "cost unknown" if trace.cost_usd is None else f"{trace.cost_usd:.4f} USD"
    print(f"{trace.status}: {len(trace.records)} records, {cost}", file=sys.stderr)
    if trace.failed:
        print(f"{trace.error_kind or 'error'}: {trace.error}", file=sys.stderr)
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": init_command,
    "check": check_command,
    "show": show_command,
    "run": run_command,
    "diff": diff_command,
    "verify": verify_command,
    "runs": runs_command,
    "trace": trace_command,
}
"""Each command's function, by the name typed after ``xmas``."""


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
