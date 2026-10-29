"""Built-in conversions: a document to text and a table to rows.

A model reads a document or a table only through a conversion code node, so what
it was shown is a visible node with a record. These two read the plain formats
with the standard library; a pdf or a spreadsheet needs a step of your own.
Pass them like any callable: ``MAS(..., tools=[document_text])``, and in a
spec ``uses: document_text``.
"""

from __future__ import annotations

import csv
import io
import re
from html.parser import HTMLParser

from xmas.media import Document, Table, base_type
from xmas.tools import tool

__all__ = [
    "document_text",
    "table_rows",
]


BLOCKS = frozenset({
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "footer", "h1", "h2",
    "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "td", "th",
    "title", "tr", "ul",
})  # fmt: skip
HIDDEN = frozenset({"script", "style"})
DELIMITERS = {"text/csv": ",", "text/tab-separated-values": "\t"}


class HTMLText(HTMLParser):
    """Collect an html page's text, one line per block, without scripts and styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pieces: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002
        if tag in HIDDEN:
            self.hidden += 1
        elif tag in BLOCKS:
            self.pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in HIDDEN:
            self.hidden = max(0, self.hidden - 1)
        elif tag in BLOCKS:
            self.pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.pieces.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.pieces).split("\n"))
        return "\n".join(line for line in lines if line)


@tool(writes={"content": "text"})
def document_text(source: Document) -> str:
    """Read a text, markdown or html document as text.

    Text and markdown come back as written; html loses its tags, scripts and
    styles, one line per block.

    Args:
        source: The document to read.
    """
    media_type = base_type(source.media_type)
    if media_type in ("text/plain", "text/markdown"):
        return source.read_bytes().decode("utf-8")
    if media_type == "text/html":
        parser = HTMLText()
        parser.feed(source.read_bytes().decode("utf-8"))
        parser.close()
        return parser.text()
    raise ValueError(
        f"document_text reads text, markdown and html; convert {source.media_type} with a step of your own"
    )


@tool(writes={"rows": "json"})
def table_rows(table: Table) -> list[dict[str, str]]:
    """Read a csv or tsv table as one mapping per row, keyed by its header line.

    Args:
        table: The table to read.
    """
    delimiter = DELIMITERS.get(base_type(table.media_type))
    if delimiter is None:
        raise ValueError(f"table_rows reads csv and tsv; convert {table.media_type} with a step of your own")
    reader = csv.DictReader(io.StringIO(table.read_bytes().decode("utf-8-sig"), newline=""), delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for row in reader:
        if None in row or None in row.values():
            raise ValueError(
                f"table_rows: line {reader.line_num} of {table.name} does not have the header's "
                f"{len(reader.fieldnames or ())} fields"
            )
        rows.append(row)
    return rows
