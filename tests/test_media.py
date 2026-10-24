"""Media values: refs, the store, the media kinds' checks and renders, and the built-in conversions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import pytest

from xmas.convert import document_text, table_rows
from xmas.errors import XmasError
from xmas.media import (
    TEXT_ONLY,
    Capabilities,
    Document,
    FileStore,
    Image,
    Kind,
    Ref,
    Table,
    fits_kind,
    kind_of,
    store,
)
from xmas.tools import tool
from xmas.values import (
    FILE,
    JSON,
    MEDIA,
    TEXT,
    check_value,
    coerce_media,
    fits,
    media_kind,
    parse_type,
    plain,
    render_value,
    type_of_hint,
)

PNG = b"\x89PNG-a"


def stored(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def png_sha() -> str:
    return hashlib.sha256(PNG).hexdigest()


def test_refs_and_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "store"
    monkeypatch.setenv("XMAS_STORE", str(root))
    first, again = Ref.from_bytes(PNG, "image/png", "a.png"), Ref.from_bytes(PNG, "image/png", "a.png")
    sha = first.sha256
    assert sha == png_sha()
    assert first == again == Ref(str(root / sha[:2] / f"{sha}.png"), "image/png", 6, sha)
    assert stored(root) == [f"{sha[:2]}/{sha}.png"]
    assert (first.name, first.kind, first.remote, first.read_bytes()) == (f"{sha}.png", "image", False, PNG)
    assert Ref.from_bytes(b"plain", "text/markdown").location.endswith(".md")
    with pytest.raises(TypeError, match="keeps bytes, not str"):
        FileStore(root).put("text", "text/plain")  # type: ignore[arg-type]


def test_path_refs_notice_a_changed_file(tmp_path: Path) -> None:
    picture = tmp_path / "slide-1.png"
    picture.write_bytes(PNG)
    ref = Ref.from_path(picture)
    assert ref == Ref(str(picture.resolve()), "image/png", 6, png_sha())
    assert Ref.from_path(picture, "image/x-custom").media_type == "image/x-custom"
    odd = tmp_path / "notes.unknownext"
    odd.write_text("x")
    assert Ref.from_path(odd).media_type == "application/octet-stream"
    picture.write_bytes(b"\x89PNG-b")
    with pytest.raises(ValueError, match=r"slide-1\.png changed since it was referenced"):
        ref.read_bytes()
    with pytest.raises(ValueError, match="is not a file"):
        Ref.from_path(tmp_path)


def test_remote_refs_and_the_dict_round_trip(tmp_path: Path) -> None:
    remote = Ref("https://example.org/deck/page-2.png?size=large", "image/png")
    assert (remote.name, remote.remote) == ("page-2.png", True)
    with pytest.raises(ValueError, match=r"page-2\.png is remote"):
        remote.read_bytes()

    picture = tmp_path / "slide-1.png"
    picture.write_bytes(PNG)
    ref = Ref.from_path(picture)
    assert remote.to_dict() == {"location": remote.location, "media_type": "image/png"}
    assert list(ref.to_dict()) == ["location", "media_type", "size", "sha256"]
    assert Ref.from_dict(ref.to_dict()) == ref
    assert Ref.from_dict(remote.to_dict()) == remote
    for bad, words in (
        ({**ref.to_dict(), "url": "x"}, "unknown: url"),
        ({"location": "/a.png"}, "needs media_type"),
        ({"location": "/a.png", "media_type": "image/png", "size": True}, "size is a byte count"),
        ({"location": "/a.png", "media_type": "image/png", "sha256": 7}, "sha256 is a hex string"),
    ):
        with pytest.raises(ValueError, match=words):
            Ref.from_dict(bad)


def test_the_store_survives_a_crowd_and_a_full_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crowd = tmp_path / "crowd"
    threads = [threading.Thread(target=FileStore(crowd).put, args=(PNG * 1000, "image/png")) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(stored(crowd)) == 1
    assert stored(crowd)[0].endswith(".png")

    def full_disk(self: Path, target: Path) -> Path:
        raise OSError("no space left")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", full_disk)
        with pytest.raises(OSError, match="no space left"):
            FileStore(tmp_path / "full").put(PNG, "image/png")
    assert stored(tmp_path / "full") == []


def test_store_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = png_sha()
    # a relative root is taken from the working directory; a stored file gets the mode open() gives it
    monkeypatch.chdir(tmp_path)
    near = FileStore("near").put(PNG, "image/png")
    assert near.location == str(tmp_path.resolve() / "near" / sha[:2] / f"{sha}.png")
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(Path(near.location).stat().st_mode) == 0o666 & ~umask

    monkeypatch.delenv("XMAS_STORE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert repr(store()) == f"FileStore({str(tmp_path / 'home' / '.cache' / 'xmas' / 'store')!r})"


def case_kinds(tmp_path: Path) -> None:
    assert [kind_of(t) for t in ("image/png", "audio/x-wav", "application/pdf", "text/html; charset=utf-8")] == [
        "image",
        "audio",
        "document",
        "document",
    ]
    assert [kind_of(t) for t in ("text/csv", "application/vnd.ms-excel", "application/zip", "text/x-python")] == [
        "table",
        "table",
        "file",
        "file",
    ]
    csv = Ref("/t.csv", "text/csv")
    assert (fits_kind(csv, "table"), fits_kind(csv, "file"), fits_kind(csv, "image")) == (True, True, False)
    assert [str(vtype) for vtype in MEDIA] == ["image", "audio", "document", "table", "file"]
    assert [media_kind(parse_type(t)) for t in ("image", "list of list of audio", "text", "list of json")] == [
        "image",
        "audio",
        None,
        None,
    ]
    with pytest.raises(ValueError, match="Kind takes one of"):
        Kind("video")


def case_checks(tmp_path: Path) -> None:
    png, csv = Ref("/a.png", "image/png"), Ref("/t.csv", "text/csv")
    image, images = parse_type("image"), parse_type("list of image")
    assert [check_value(value, image) for value in (png, csv, "a.png", b"\x89PNG")] == [
        None,
        "expected image, got a Ref of text/csv",
        "expected image, got str",
        "expected image, got bytes; return a Path or a Ref, Ref.from_bytes stores bytes",
    ]
    assert check_value([png, None, png], images) is None
    assert check_value([png, None, "a.png"], images) == "expected image at 2, got str"
    assert check_value(csv, FILE) is None
    assert check_value(csv, parse_type("table")) is None
    assert check_value(Ref("/a.wav", "audio/wav"), parse_type("audio")) is None
    assert check_value(Ref("/a.md", "text/markdown"), parse_type("document")) is None
    assert check_value({"page": png}, JSON) == "not JSON at page: Ref"


def case_renders(tmp_path: Path) -> None:
    png = Ref("/a.png", "image/png", 6, "ab12")
    line = '{"location": "/a.png", "media_type": "image/png", "size": 6, "sha256": "ab12"}'
    assert render_value(png, parse_type("image")) == line
    assert render_value([png, None], parse_type("list of image")) == f"[{line}, null]"
    assert render_value(None, parse_type("image")) == ""
    assert render_value(png, FILE, present=False) == ""
    assert json.loads(render_value([], parse_type("list of file"))) == []


def case_plain(tmp_path: Path) -> None:
    png = Ref("/a.png", "image/png")
    as_dict = {"location": "/a.png", "media_type": "image/png"}
    assert plain({"pages": [png, None, (png,)], "n": 1}) == {"pages": [as_dict, None, [as_dict]], "n": 1}
    assert plain("text") == "text"


def case_hints(tmp_path: Path) -> None:
    assert [
        str(type_of_hint(hint)) for hint in (Image, Document, list[Image], Ref, list[Ref], Annotated[Ref, "x"])
    ] == [
        "image",
        "document",
        "list of image",
        "file",
        "list of file",
        "file",
    ]
    assert [type_of_hint(hint) for hint in (Annotated[str, "x"], list[str], int, Capabilities)] == [
        TEXT,
        JSON,
        JSON,
        JSON,
    ]

    @tool
    def paint(
        slide: Image,
        pages: list[Table],
        maybe: Document | None = None,
        note: str | None = None,
        drafts: list[Image | None] | None = None,
    ) -> str:
        """Paint a slide.

        Args:
            slide: The picture.
            pages: The tables.
            maybe: A document, if any.
            note: A remark.
            drafts: Earlier pictures, ``None`` where one failed.
        """
        return slide.name

    assert {name: str(vtype) for name, vtype in paint.reads.items()} == {
        "slide": "image",
        "pages": "list of table",
        "maybe": "document",
        "note": "text",
        "drafts": "list of image",
    }
    assert paint.parameters["properties"]["slide"]["x-xmas-kind"] == "image"
    assert "$defs" not in paint.parameters
    png = Ref("/a.png", "image/png")
    assert paint.validate({"slide": png, "pages": []})["slide"] is png
    assert paint.validate({"slide": png.to_dict(), "pages": []})["slide"] == png
    with pytest.raises(XmasError, match="expected a media reference, got str"):
        paint.validate({"slide": "/a.png", "pages": []})
    with pytest.raises(XmasError, match="unknown: url"):
        paint.validate({"slide": {"url": "x"}, "pages": []})
    assert paint.arguments_model is not None
    dumped = paint.arguments_model.model_validate({"slide": png, "pages": []}).model_dump(mode="json", by_alias=True)
    assert dumped["slide"] == png.to_dict()
    assert type_of_hint({"type": "array", "items": {"type": "string"}}) == JSON
    assert type_of_hint({"anyOf": [{"x-xmas-kind": "audio"}, {"type": "integer"}]}) == JSON


def case_fits(tmp_path: Path) -> None:
    image, images = parse_type("image"), parse_type("list of image")
    assert (fits(image, FILE), fits(parse_type("table"), FILE), fits(image, image)) == (True, True, True)
    assert (fits(images, parse_type("list of file")), fits(FILE, image), fits(TEXT, FILE)) == (False, False, False)
    assert fits(images, JSON)


def case_coerce(tmp_path: Path) -> None:
    picture = tmp_path / "a.png"
    picture.write_bytes(PNG)
    image, images = parse_type("image"), parse_type("list of image")
    by_reference = coerce_media(picture, image)
    assert by_reference == Ref.from_path(picture)
    shelf = FileStore(tmp_path / "shelf")
    copied = coerce_media([picture, None], images, store=shelf)
    assert copied[1] is None
    assert copied[0].location.startswith(str(shelf.root))
    assert copied[0].media_type == "image/png"
    picture.write_bytes(b"\x89PNG-b")
    assert copied[0].read_bytes() == PNG
    assert coerce_media(picture, TEXT) is picture
    assert coerce_media("a.png", image) == "a.png"
    assert coerce_media(by_reference, image) is by_reference
    assert coerce_media(picture, images) is picture


def case_capabilities(tmp_path: Path) -> None:
    vision = Capabilities.parse({"accepts": ["image", "audio"]}, "models entry 'vision'")
    assert vision == Capabilities(accepts=("image", "audio"))
    assert (vision.takes("image"), vision.takes("json"), vision.gives("image"), TEXT_ONLY.takes("audio")) == (
        True,
        True,
        False,
        False,
    )
    painter = Capabilities.parse({"accepts": ["text"], "produces": ("image",)}, "models entry 'p'")
    assert painter.gives("image")
    assert (painter.to_spec(), vision.to_spec(), TEXT_ONLY.to_spec()) == (
        {"accepts": ["text"], "produces": ["image"]},
        {"accepts": ["image", "audio"]},
        {},
    )
    for entry, words in (
        ({"temperature": 0.2}, r"models entry 'x' has key 'temperature'; it takes accepts and produces$"),
        ({"accepts": ["video"]}, r"models entry 'x' accepts 'video'; a model may accept image, audio$"),
        ({"produces": ["audio"]}, r"models entry 'x' produces 'audio'; a model may produce image$"),
        ({"accepts": "image"}, r"models entry 'x' accepts is a list of words"),
        (["image"], r"models entry 'x' is a mapping with accepts and produces, not list"),
    ):
        with pytest.raises(ValueError, match=words):
            Capabilities.parse(entry, "models entry 'x'")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case",
    [case_kinds, case_checks, case_renders, case_plain, case_hints, case_fits, case_coerce, case_capabilities],
    ids=lambda case: case.__name__.removeprefix("case_"),
)
def test_media_kinds_check_render_and_hints(case: Callable[[Path], None], tmp_path: Path) -> None:
    case(tmp_path)


def file(tmp_path: Path, name: str, data: str, media_type: str | None = None) -> Ref:
    path = tmp_path / name
    path.write_text(data, encoding="utf-8")
    return Ref.from_path(path, media_type)


@pytest.mark.parametrize(
    ("name", "files", "expected"),
    [
        (
            "document_text",
            [
                ("deck.md", "# Rules\n\nOne idea per slide.\n", None),
                (
                    "rules.html",
                    (
                        "<html><head><style>p {}</style><script>go()</script></head>"
                        "<body><h1>Rules</h1><p>One <b>idea</b> &amp;\n  one   claim.</p>"
                        "<ul><li>a</li><li>b</li></ul></body></html>"
                    ),
                    None,
                ),
                ("notes.txt", "plain\n", "text/plain; charset=utf-8"),
            ],
            ["# Rules\n\nOne idea per slide.\n", "Rules\nOne idea & one claim.\na\nb", "plain\n"],
        ),
        (
            "document_text",
            [("deck.pdf", "%PDF-1.7", None)],
            "document_text reads text, markdown and html; convert application/pdf with a step of your own",
        ),
        (
            "table_rows",
            [("costs.csv", "﻿item,usd\ntea,2\n\ncake,3.5\n", None), ("people.tsv", "name\trole\nana\tlead\n", None)],
            [[{"item": "tea", "usd": "2"}, {"item": "cake", "usd": "3.5"}], [{"name": "ana", "role": "lead"}]],
        ),
        (
            "table_rows",
            [("sheet.xlsx", "PK", None)],
            "table_rows reads csv and tsv; convert application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    ],
    ids=["text", "pdf-refused", "rows", "xlsx-refused"],
)
def test_conversions(name: str, files: list[tuple[str, str, str | None]], expected: Any, tmp_path: Path) -> None:
    convert = {"document_text": document_text, "table_rows": table_rows}[name]
    refs = [file(tmp_path, *entry) for entry in files]
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            convert(refs[0])
        return
    assert [convert(ref) for ref in refs] == expected
    assert (dict(document_text.reads), dict(table_rows.reads)) == (
        {"source": parse_type("document")},
        {"table": parse_type("table")},
    )
    assert (dict(document_text.writes), dict(table_rows.writes)) == ({"content": TEXT}, {"rows": JSON})
    ragged = file(tmp_path, "ragged.csv", "a,b\n1,2,3\n")
    with pytest.raises(ValueError, match=r"line 2 of ragged\.csv does not have the header's 2 fields"):
        table_rows(ragged)
