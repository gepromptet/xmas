r"""A prompt with pictures or sound in it: text and media refs in order, and the content a request sends.

:func:`render_parts` substitutes a template exactly as
:func:`~xmas.prompts.safe_format` does, with each media placeholder held by a
sentinel until the text is split around it. A list of refs becomes one labelled
ref per item, ``\n[pages 0] `` before the first (0-based, as ``item``), and
``(none)`` for an item that failed. :func:`to_content` turns the parts into
chat-completions content; bytes are read only there, so a record, a trace or a
call log shows :func:`marker` lines, never base64.
"""

from __future__ import annotations

import base64
import re
import uuid
from typing import TYPE_CHECKING, Any

from xmas.media import MAX_PART_BYTES, Ref
from xmas.prompts import safe_format

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "Part",
    "content_text",
    "has_media",
    "marker",
    "prompt_messages",
    "record_text",
    "render_parts",
    "to_content",
]


Part = str | Ref
"""One piece of a prompt: text, or a media ref sent as a picture or a sound."""

AUDIO_FORMATS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3"}

KILOBYTE = 1024
"""Bytes in the ``KB`` a record shows."""

MEGABYTE = KILOBYTE * KILOBYTE
"""Bytes in the ``MB`` a record shows."""


def render_parts(
    template: str, text: Mapping[str, str], media: Mapping[str, Ref | Sequence[Any] | None]
) -> tuple[list[Part], str]:
    r"""Substitute ``text`` and ``media`` into ``template``; return the parts and the record text.

    Text values substitute exactly as :func:`~xmas.prompts.safe_format` substitutes them. A single ref is one part;
    a list gives ``\n[<placeholder> <i>] `` then the ref, or ``(none)`` for a ``None`` item, and a list of lists
    labels its items ``<i>.<j>``; a missing value or an empty list gives ``""``. Adjacent text merges and empty
    text is dropped. The record text is the parts with :func:`marker` in place of each ref.
    """
    nonce = uuid.uuid4().hex
    names = list(media)
    held = {name: f"\x00{nonce}{index}\x00" for index, name in enumerate(names)}
    rendered = safe_format(template, {**text, **held})
    pieces = re.split(f"\x00{nonce}(\\d+)\x00", rendered)
    parts: list[Part] = []
    for position, piece in enumerate(pieces):
        if position % 2 == 0:
            append(parts, piece)
            continue
        name = names[int(piece)]
        value = media[name]
        if isinstance(value, Ref):
            parts.append(value)
        elif value is not None:
            add_items(parts, name, "", value)
    return parts, record_text(parts)


def add_items(parts: list[Part], name: str, label: str, items: Sequence[Any]) -> None:
    """Add a list's refs, each after its label; a nested list labels its items ``1.0``, ``1.1``."""
    for index, item in enumerate(items):
        here = f"{label}.{index}" if label else str(index)
        if isinstance(item, list):
            add_items(parts, name, here, item)
            continue
        append(parts, f"\n[{name} {here}] ")
        if item is None:
            append(parts, "(none)")
        else:
            parts.append(item)


def append(parts: list[Part], piece: str) -> None:
    """Add text, merged into a text part before it; empty text is dropped."""
    if not piece:
        return
    if parts and isinstance(parts[-1], str):
        parts[-1] += piece
    else:
        parts.append(piece)


def record_text(parts: Sequence[Part]) -> str:
    """Return the parts as one text with :func:`marker` in place of each ref: what a record shows."""
    return "".join(part if isinstance(part, str) else marker(part) for part in parts)


def marker(ref: Ref) -> str:
    """Return the line a record shows for a ref, such as ``[image slide-3.png, image/png, 182 KB]``."""
    fields = [f"{ref.kind} {ref.name}", ref.media_type]
    if ref.size is not None:
        fields.append(size(ref.size))
    return f"[{', '.join(fields)}]"


def size(count: int) -> str:
    """Return a byte count as ``6 B``, ``182 KB`` or ``3.4 MB``."""
    if count < KILOBYTE:
        return f"{count} B"
    if count < MEGABYTE:
        return f"{round(count / KILOBYTE)} KB"
    return f"{count / MEGABYTE:.1f} MB"


def has_media(parts: Sequence[Part]) -> bool:
    """Whether any part is a ref."""
    return any(isinstance(part, Ref) for part in parts)


def to_content(parts: Sequence[Part]) -> list[dict[str, Any]]:
    """Return the chat-completions content of ``parts``.

    Text becomes a ``text`` part. An image becomes an ``image_url`` part: a
    remote ref's URL as is, else a ``data:`` URI of its bytes. Audio becomes an
    ``input_audio`` part (wav or mp3).

    Raises:
        ValueError: For a part over :data:`~xmas.media.MAX_PART_BYTES`, a remote
            audio ref, an audio type other than wav and mp3, or a document,
            table or file ref, which a model reads through a conversion code node.
    """
    content: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
        elif part.kind == "image":
            url = part.location if part.remote else f"data:{part.media_type};base64,{encoded(part)}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        elif part.kind == "audio":
            audio_format = AUDIO_FORMATS.get(part.media_type)
            if audio_format is None:
                raise ValueError(f"{marker(part)}: a model takes wav or mp3 audio, not {part.media_type}")
            if part.remote:
                raise ValueError(f"{marker(part)}: audio is sent as bytes, and xmas reads no network")
            content.append({"type": "input_audio", "input_audio": {"data": encoded(part), "format": audio_format}})
        else:
            raise ValueError(
                f"{marker(part)}: a model reads a {part.kind} through a conversion code node, never as a content part"
            )
    return content


def encoded(ref: Ref) -> str:
    """Return a ref's bytes as base64, refusing one over the part limit before or after reading it."""
    too_big = f"{marker(ref)} is over the {size(MAX_PART_BYTES)} a content part may carry"
    if ref.size is not None and ref.size > MAX_PART_BYTES:
        raise ValueError(too_big)
    data = ref.read_bytes()
    if len(data) > MAX_PART_BYTES:
        raise ValueError(too_big)
    return base64.b64encode(data).decode("ascii")


def prompt_messages(prompt: str | Sequence[Part], system_prompt: str) -> list[dict[str, Any]]:
    """Return a request's messages: the system prompt, then the prompt as text, or as content with media in it."""
    content = prompt if isinstance(prompt, str) else to_content(prompt) if has_media(prompt) else record_text(prompt)
    return [{"role": "system", "content": str(system_prompt)}, {"role": "user", "content": content}]


def content_text(content: str | Sequence[Mapping[str, Any]]) -> str:
    """Return a message content's text: itself for a string, else its text parts joined."""
    if isinstance(content, str):
        return content
    return "".join(str(part.get("text") or "") for part in content if part.get("type") == "text")
