"""Tools served over MCP (streamable HTTP), usable from xmas's synchronous engine.

Needs the ``mcp`` extra: ``pip install 'xmas[mcp]'``.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from mcp.types import AudioContent, BlobResourceContents, CallToolResult, EmbeddedResource, ImageContent, TextContent

from xmas.errors import XmasError
from xmas.tools import Tool, ToolSource
from xmas.tools.mcp_session import McpSession
from xmas.tools.schema import valid_tool_name

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

__all__ = [
    "MCPTools",
    "result_text",
]


logger = logging.getLogger(__name__)


class MCPTools(ToolSource):
    """Tools served over MCP (streamable HTTP), as a :class:`~xmas.tools.ToolSource`.

    Use it as a context manager, or call :meth:`close` when done::

        with MCPTools("http://host:8300/mcp") as remote:
            mas = MAS(agents=[...], llm=llm, tools=[remote])

    The session opens on the first :meth:`tools` or :meth:`call`. If the server
    forgets it (a restart), the call is resent once on a new session; the tool
    did not run. Any other transport failure ends the run and is not retried,
    because a tool call need not be idempotent.
    """

    def __init__(  # noqa: PLR0913  the public constructor, kept for version 1 callers
        self,
        url: str,
        key: str | None = None,
        *,
        timeout_s: float = 900.0,
        max_concurrent: int = 4,
        writes: Mapping[str, Mapping[str, str]] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the source. Nothing connects until the first use.

        Args:
            url: The server's MCP endpoint, e.g. ``http://host:8300/mcp``.
            key: Bearer key. Defaults to the ``XMAS_TOOLS_KEY`` environment variable.
            timeout_s: Per-call wait. The default covers serving-xmas's 600 s
                heavy-queue wait plus the tool's own wall time.
            max_concurrent: Calls in flight at once. Matches the server's per-key
                default, so a batched run queues here instead of collecting
                ``too_many_concurrent`` refusals.
            writes: What a tool writes when it runs as a code node, by tool name, as
                ``{tool: {write: type}}``; a tool not listed writes ``{tool: text}``.
                A media write, such as ``{"paint": {"png": "image"}}``, is read
                from the result's media blocks (:meth:`call_media`).
            name: The server's name in the configuration (``[tools.<name>]``);
                a saved system's callables name it as their ``server``, never
                the address.

        Raises:
            ValueError: If no key is given or set in the environment, or the
                name holds ``://``.
        """
        key = key or os.environ.get("XMAS_TOOLS_KEY")
        if not key:
            raise ValueError("MCPTools needs a key, or XMAS_TOOLS_KEY set in the environment.")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1.")
        if name is not None and (not name or "://" in name):
            raise ValueError(f"MCPTools name {name!r} is a name such as 'web'; the address is the url.")
        self.name = name
        self.url = url
        self.timeout_s = timeout_s
        self.max_concurrent = max_concurrent
        self.writes = {name: dict(item) for name, item in (writes or {}).items()}
        self.session = McpSession(url, key, timeout_s, max_concurrent)
        self.cached: list[Tool] | None = None

    def tools(self) -> list[Tool]:
        """Every tool the server offers this key (all ``tools/list`` pages), cached after the first call.

        Tools whose names the OpenAI ``tools`` payload would refuse are skipped
        with a warning.
        """
        if self.cached is None:
            found = []
            for item in self.session.list_tools():
                if not valid_tool_name(item.name):
                    logger.warning("MCPTools: skipping tool %r, whose name OpenAI would refuse", item.name)
                    continue
                schema = dict(item.input_schema or {"type": "object", "properties": {}})
                found.append(Tool(item.name, item.description or "", schema, writes=self.writes.get(item.name)))
            self.cached = found
        return list(self.cached)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool on the server and return the text the model sees (:func:`result_text`).

        Thread safe; at most ``max_concurrent`` calls are in flight.

        Raises:
            XmasError: The tool failed (fed back to the model), or the server
                refused the key (``status`` 401, 403 or 429; the run ends).
        """
        return result_text(self.session.call_tool(name, arguments))

    def call_media(self, name: str, arguments: dict[str, Any]) -> list[tuple[bytes, str]]:
        """Run one tool on the server and return its image, audio and blob blocks, decoded, in order.

        Raises:
            XmasError: As :meth:`call` (the same failures and refusals), or for
                a block whose base64 is invalid.
        """
        result = self.session.call_tool(name, arguments)
        result_text(result)  # raises for a failed or refused call, as call does
        return media_blocks(result)

    def close(self) -> None:
        """Close the session and stop the background loop. Idempotent."""
        self.session.close()

    def __enter__(self) -> MCPTools:
        """Return self; the session opens on first use."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the session."""
        self.close()

    def __repr__(self) -> str:
        """Name the endpoint and the name, never the key."""
        named = "" if self.name is None else f", name={self.name!r}"
        return f"MCPTools({self.url!r}{named})"


def media_blocks(result: CallToolResult) -> list[tuple[bytes, str]]:
    """Return a result's image, audio and embedded blob blocks as ``(bytes, media type)``."""
    encoded: list[tuple[str, str]] = []
    for block in result.content:
        if isinstance(block, ImageContent | AudioContent):
            encoded.append((block.data, block.mime_type))
        elif isinstance(block, EmbeddedResource) and isinstance(block.resource, BlobResourceContents):
            encoded.append((block.resource.blob, block.resource.mime_type or "application/octet-stream"))
    try:
        return [(base64.b64decode(data, validate=True), media_type) for data, media_type in encoded]
    except binascii.Error as exc:
        raise XmasError(f"mcp: a media block's base64 is invalid: {exc}") from None


def result_text(result: CallToolResult) -> str:
    """Map one ``tools/call`` result to the text the model sees, or raise.

    serving-xmas answers with either an Envelope (``exit_code``, ``stdout``, ...)
    or an error body (``{"error", "status", "message"}``); any other MCP server
    answers with plain content, whose non-text blocks are named, never inlined.
    The text is ``stdout`` (plus truncation and file-name notes) for a
    successful Envelope, or the joined text of any other successful result.

    Raises:
        XmasError: The tool failed or was called wrongly; or, with ``status``
            401, 403 or 429, the key is unauthorized, out of scope, out of
            budget or over its concurrency limit.
    """
    parts = [
        block.text if isinstance(block, TextContent) else f"[{block.type} content omitted]" for block in result.content
    ]
    text = "\n".join(parts)
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if isinstance(body, dict) and "exit_code" in body:
        exit_code = body.get("exit_code")
        killed = body.get("killed", "none")
        stdout = str(body.get("stdout") or "")
        stderr = str(body.get("stderr") or "")
        if result.is_error or exit_code != 0 or killed != "none":
            raise XmasError(f"exit_code={exit_code} killed={killed}\n{stderr or stdout}".rstrip("\n"))
        out = stdout
        if body.get("truncated"):
            out += "\n[truncated]"
        files = body.get("files") or {}
        if files:
            out += f"\n[files: {', '.join(sorted(files))}]"
        return out
    if isinstance(body, dict) and "error" in body and "status" in body:
        code = str(body["error"])
        status = body["status"]
        message = str(body.get("message") or body["error"])
        if code == "insufficient_budget":  # arrives with 429 already; the run ends either way
            raise XmasError(f"mcp: {code} ({status}): {message}", status=429)
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            raw = body.get("retry_after_s")
            retry_after = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
            raise XmasError(f"mcp: {code} ({status}): {message}", status=429, retry_after=retry_after)
        raise XmasError(f"mcp: {code} ({status}): {message}", status=status if isinstance(status, int) else None)
    if result.is_error:
        raise XmasError(text or "tool failed")
    return text
