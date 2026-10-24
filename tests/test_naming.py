"""Nothing in xmas or its tests is named with a leading underscore, private or not.

Each module under ``xmas/`` says what it offers in ``__all__``; that and the docs are the public surface.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOOKS = frozenset(
    {
        "_ainvoke_once",
        "_ainvoke_with_retries",
        "_evaluate",
        "_extract_preds",
        "_get_response",
        "_pre_optimization_loop",
        "_step",
    }
)
"""promptolution's base classes call these, so a subclass has to override them under these names."""


def offends(name: str) -> bool:
    """Whether ``name`` starts with an underscore without being ``_`` or a ``__dunder__``."""
    dunder = len(name) > 4 and name.startswith("__") and name.endswith("__")
    return name.startswith("_") and name != "_" and not dunder and name not in HOOKS


def defined(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Yield the line and name of each thing ``tree`` defines: defs, classes, targets, parameters, type variables."""
    for node in ast.walk(tree):
        match node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                yield node.lineno, name
            case ast.Name(ctx=ast.Store(), id=name) | ast.Attribute(ctx=ast.Store(), attr=name):
                yield node.lineno, name
            case ast.arg(arg=name) | ast.alias(asname=str(name)) | ast.ExceptHandler(name=str(name)):
                yield node.lineno, name
            case ast.TypeVar(name=name) | ast.ParamSpec(name=name) | ast.TypeVarTuple(name=name):
                yield node.lineno, name
            case ast.MatchAs(name=str(name)) | ast.MatchStar(name=str(name)) | ast.MatchMapping(rest=str(name)):
                yield node.lineno, name
            case ast.Call(func=ast.Name(id="TypeVar"), args=[ast.Constant(value=str(name)), *_]):
                yield node.lineno, name


def sources() -> list[Path]:
    return sorted([*(ROOT / "xmas").rglob("*.py"), *(ROOT / "tests").rglob("*.py")])


def test_no_name_starts_with_an_underscore() -> None:
    found = [
        f"{path.relative_to(ROOT)}:{line} {name}"
        for path in sources()
        for line, name in defined(ast.parse(path.read_text(encoding="utf-8")))
        if offends(name)
    ]
    assert not found, "rename these, without the leading underscore:\n" + "\n".join(found)


def test_every_module_says_what_it_offers() -> None:
    missing = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "xmas").rglob("*.py"))
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.Assign | ast.AnnAssign)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    assert not missing, "these modules have no __all__:\n" + "\n".join(missing)
