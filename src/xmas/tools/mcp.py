"""Tools served over MCP (streamable HTTP), usable from xmas's synchronous engine.

Needs the ``mcp`` extra: ``pip install 'xmas[mcp]'``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.types import CallToolResult, TextContent

from xmas.errors import XmasError
from xmas.tools import Tool, ToolSource
from xmas.tools.mcp_session import McpSession
from xmas.tools.schema import valid_tool_name

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

    def __init__(self, url: str, key: str | None = None, *, timeout_s: float = 900.0, max_concurrent: int = 4) -> None:
        """Initialize the source. Nothing connects until the first use.

        Args:
            url: The server's MCP endpoint, e.g. ``http://host:8300/mcp``.
            key: Bearer key. Defaults to the ``XMAS_TOOLS_KEY`` environment variable.
            timeout_s: Per-call wait. The default covers serving-xmas's 600 s
                heavy-queue wait plus the tool's own wall time.
            max_concurrent: Calls in flight at once. Matches the server's per-key
                default, so a batched run queues here instead of collecting
                ``too_many_concurrent`` refusals.

        Raises:
            ValueError: If no key is given or set in the environment.
        """
        key = key or os.environ.get("XMAS_TOOLS_KEY")
        if not key:
            raise ValueError("MCPTools needs a key, or XMAS_TOOLS_KEY set in the environment.")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1.")
        self.url = url
        self.timeout_s = timeout_s
        self.max_concurrent = max_concurrent
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
                found.append(Tool(item.name, item.description or "", schema))
            self.cached = found
        return list(self.cached)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool on the server and return the text the model sees.

        Thread safe; at most ``max_concurrent`` calls are in flight.

        Args:
            name: The tool.
            arguments: The decoded arguments object.

        Returns:
            The rendered result (see :func:`result_text`).

        Raises:
            XmasError: The tool failed (fed back to the model), or the server
                refused the key (``status`` 401, 403 or 429; the run ends).
        """
        return result_text(self.session.call_tool(name, arguments))

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
        """Name the endpoint, never the key."""
        return f"MCPTools({self.url!r})"


def result_text(result: CallToolResult) -> str:
    """Map one ``tools/call`` result to the text the model sees, or raise.

    serving-xmas answers with either an Envelope (``exit_code``, ``stdout``, ...)
    or an error body (``{"error", "status", "message"}``); any other MCP server
    answers with plain content, whose non-text blocks are named, never inlined.

    Args:
        result: The call's result.

    Returns:
        ``stdout`` (plus truncation and file-name notes) for a successful
        Envelope, or the joined text of any other successful result.

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
        if status == 429:
            raw = body.get("retry_after_s")
            retry_after = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
            raise XmasError(f"mcp: {code} ({status}): {message}", status=429, retry_after=retry_after)
        raise XmasError(f"mcp: {code} ({status}): {message}", status=status if isinstance(status, int) else None)
    if result.is_error:
        raise XmasError(text or "tool failed")
    return text
