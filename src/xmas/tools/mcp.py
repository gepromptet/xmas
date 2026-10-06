"""Tools served over MCP (streamable HTTP), usable from xmas's synchronous engine.

Needs the ``mcp`` extra: ``pip install 'xmas[mcp]'``.

``MCPTools`` holds one MCP session on a private background event loop, the
pattern promptolution's ``APILLM`` uses for its client, so concurrent graph nodes
(``run_batch`` runs them on worker threads) share one session and one handshake.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx2
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CONNECTION_CLOSED,
    INVALID_REQUEST,
    REQUEST_TIMEOUT,
    CallToolResult,
    PaginatedRequestParams,
    TextContent,
)

from xmas.llms.api_llm import RateLimitedError
from xmas.tools import TOOL_NAME, Tool, ToolAccessError, ToolError

__all__ = ["ENV_TOOLS_KEY", "MCPTools"]

ENV_TOOLS_KEY = "XMAS_TOOLS_KEY"
"""Environment variable holding the tool server key."""

SESSION_LOST = frozenset({"Session terminated", "Session not found"})
"""Messages of the ``INVALID_REQUEST`` error a call fails with when the server no longer knows
the session (HTTP 404 on an established session): the SDK server's own JSON-RPC error body,
which the client passes on verbatim, or the client's stand-in when the 404 carries none."""

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Refused:
    """An HTTP-level refusal seen by the response hook, which the SDK would hang on or group."""

    status: int
    retry_after: float | None


def _refusal_error(refused: _Refused) -> Exception:
    """Map an HTTP refusal to the error a run ends with."""
    if refused.status == httpx2.codes.TOO_MANY_REQUESTS:
        return RateLimitedError("rate limited (HTTP 429)", retry_after=refused.retry_after)
    if refused.status == httpx2.codes.UNAUTHORIZED:
        return ToolAccessError("unauthorized (HTTP 401): check XMAS_TOOLS_KEY")
    return ToolAccessError(f"forbidden (HTTP {refused.status})")


def _header_seconds(value: str | None) -> float | None:
    try:
        return None if value is None else max(0.0, float(value))
    except ValueError:
        return None


def _text_of(result: CallToolResult) -> str:
    """Join a result's text blocks; other content kinds are named, never inlined."""
    parts = [
        block.text if isinstance(block, TextContent) else f"[{block.type} content omitted]" for block in result.content
    ]
    return "\n".join(parts)


def result_text(result: CallToolResult) -> str:
    """Map one ``tools/call`` result to the text the model sees, or raise.

    serving-xmas answers with either an Envelope (``exit_code``, ``stdout``, ...)
    or an error body (``{"error", "status", "message"}``); any other MCP server
    answers with plain content.

    Args:
        result: The call's result.

    Returns:
        ``stdout`` (plus truncation and file-name notes) for a successful
        Envelope, or the joined text of any other successful result.

    Raises:
        ToolError: The tool failed or was called wrongly; fed back to the model.
        ToolAccessError: Unauthorized, out of scope, or the key's budget is spent.
        RateLimitedError: Too many concurrent calls for the key.
    """
    text = _text_of(result)
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if isinstance(body, dict) and "exit_code" in body:
        exit_code, killed = body.get("exit_code"), body.get("killed", "none")
        stdout, stderr = str(body.get("stdout") or ""), str(body.get("stderr") or "")
        if result.is_error or exit_code != 0 or killed != "none":
            raise ToolError(f"exit_code={exit_code} killed={killed}\n{stderr or stdout}".rstrip("\n"))
        out = stdout
        if body.get("truncated"):
            out += "\n[truncated]"
        files = body.get("files") or {}
        if files:
            out += f"\n[files: {', '.join(sorted(files))}]"
        return out
    if isinstance(body, dict) and "error" in body and "status" in body:
        code, status, message = str(body["error"]), body["status"], str(body.get("message") or body["error"])
        if status in (httpx2.codes.UNAUTHORIZED, httpx2.codes.FORBIDDEN) or code == "insufficient_budget":
            raise ToolAccessError(f"{code}: {message}")
        if status == httpx2.codes.TOO_MANY_REQUESTS:
            raw = body.get("retry_after_s")
            retry_after = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
            raise RateLimitedError(f"{code}: {message}", retry_after=retry_after)
        raise ToolError(f"{code}: {message}")
    if result.is_error:
        raise ToolError(text or "tool failed")
    return text


class MCPTools:
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
        key = key or os.environ.get(ENV_TOOLS_KEY)
        if not key:
            raise ValueError(f"MCPTools needs a key, or {ENV_TOOLS_KEY} set in the environment.")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1.")
        self.url = url
        self.timeout_s = timeout_s
        self.max_concurrent = max_concurrent
        self._key = key
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._tools: list[Tool] | None = None
        # Loop-side state, touched only from the background loop.
        self._sem: asyncio.Semaphore | None = None
        self._connecting: asyncio.Lock | None = None
        self._session: ClientSession | None = None
        self._owner: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._refused: _Refused | None = None

    # ------------------------------------------------------------- sync API

    def tools(self) -> list[Tool]:
        """Every tool the server offers this key (all ``tools/list`` pages), cached after the first call.

        Tools whose names the OpenAI ``tools`` payload would refuse are skipped
        with a warning.
        """
        if self._tools is None:
            self._tools = self._run(self._alist())
        return list(self._tools)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool on the server and return the text the model sees.

        Thread safe; at most ``max_concurrent`` calls are in flight.

        Args:
            name: The tool.
            arguments: The decoded arguments object.

        Returns:
            The rendered result (see :func:`result_text`).

        Raises:
            ToolError: The tool failed; fed back to the model.
            ToolAccessError: The server refused the key.
            RateLimitedError: Too many concurrent calls for the key.
        """
        return self._run(self._acall(name, arguments))

    def close(self) -> None:
        """Close the session and stop the background loop. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._adisconnect(), loop).result(timeout=30.0)
        except Exception:  # closing is best effort; the loop stops regardless
            logger.warning("MCPTools: error while closing the session", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=30.0)
        loop.close()

    def __enter__(self) -> MCPTools:
        """Return self; the session opens on first use."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the session."""
        self.close()

    def __repr__(self) -> str:
        """Name the endpoint, never the key."""
        return f"MCPTools({self.url!r})"

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run a coroutine on the background loop and wait for it."""
        with self._lock:
            if self._closed:
                coro.close()
                raise RuntimeError("MCPTools is closed.")
            if self._loop is None:
                loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=loop.run_forever, name="MCPToolsLoop", daemon=True)
                self._thread.start()
                self._loop = loop
            loop = self._loop
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=self.timeout_s + 60.0)

    # ------------------------------------------------------------ loop side

    async def _on_response(self, response: httpx2.Response) -> None:
        """Record an HTTP refusal, which the SDK would otherwise surface inconsistently."""
        if response.status_code in (httpx2.codes.UNAUTHORIZED, httpx2.codes.FORBIDDEN, httpx2.codes.TOO_MANY_REQUESTS):
            self._refused = _Refused(response.status_code, _header_seconds(response.headers.get("Retry-After")))

    async def _own(self, ready: asyncio.Future[ClientSession], stop: asyncio.Event) -> None:
        """Own one session: enter and leave the SDK's contexts in this one task, as it requires."""
        try:
            async with (
                httpx2.AsyncClient(
                    headers={"Authorization": f"Bearer {self._key}"},
                    timeout=httpx2.Timeout(30.0, read=self.timeout_s),
                    event_hooks={"response": [self._on_response]},
                ) as client,
                streamable_http_client(self.url, http_client=client) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                ready.set_result(session)
                await stop.wait()
        except BaseException as exc:  # an ExceptionGroup from the transport, or a cancellation
            if not ready.done():
                ready.set_exception(_refusal_error(self._refused) if self._refused else _flatten(exc))
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            if not ready.done():
                ready.set_exception(ConnectionError("MCP session closed before it was ready"))

    async def _connect(self) -> ClientSession:
        """Return the live session, opening one if there is none (one handshake for concurrent callers)."""
        if self._connecting is None:
            self._connecting = asyncio.Lock()
        async with self._connecting:
            if self._session is not None and self._owner is not None and not self._owner.done():
                return self._session
            await self._adisconnect()
            self._refused = None
            ready: asyncio.Future[ClientSession] = asyncio.get_running_loop().create_future()
            self._stop = asyncio.Event()
            self._owner = asyncio.create_task(self._own(ready, self._stop))
            try:
                self._session = await asyncio.wait_for(ready, timeout=self.timeout_s)
            except Exception:
                await self._adisconnect()
                raise
            return self._session

    async def _adisconnect(self) -> None:
        """Close the current session, if any, and wait for its owner to leave the SDK's contexts."""
        owner, stop = self._owner, self._stop
        self._session, self._owner, self._stop = None, None, None
        if owner is None or stop is None:
            return
        stop.set()
        try:
            await asyncio.wait_for(owner, timeout=10.0)
        except (TimeoutError, asyncio.CancelledError):
            owner.cancel()

    async def _alist(self) -> list[Tool]:
        session = await self._connect()
        found: list[Tool] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
            for item in page.tools:
                if not TOOL_NAME.match(item.name):
                    logger.warning("MCPTools: skipping tool %r, whose name OpenAI would refuse", item.name)
                    continue
                schema = dict(item.input_schema or {"type": "object", "properties": {}})
                found.append(Tool(item.name, item.description or "", schema))
            cursor = page.next_cursor
            if not cursor:
                return found

    async def _acall(self, name: str, arguments: dict[str, Any]) -> str:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
        async with self._sem:
            for attempt in range(2):
                session = await self._connect()
                try:
                    result = await session.call_tool(name, arguments, read_timeout_seconds=self.timeout_s)
                except MCPError as exc:
                    if self._refused is not None:
                        refused = _refusal_error(self._refused)
                        await self._adisconnect()
                        raise refused from exc
                    code = exc.code
                    terminated = code == INVALID_REQUEST and exc.message in SESSION_LOST
                    if terminated and attempt == 0:  # the server forgot the session; nothing ran
                        await self._adisconnect()
                        continue
                    if terminated or code == CONNECTION_CLOSED:
                        await self._adisconnect()
                        raise ConnectionError(f"MCP connection lost during {name}: {exc}") from exc
                    if code == REQUEST_TIMEOUT:
                        raise TimeoutError(f"MCP call {name} timed out after {self.timeout_s}s") from exc
                    raise ToolError(f"mcp error {code}: {exc.message}") from exc
                return result_text(result)
        raise AssertionError("unreachable")  # pragma: no cover


def _flatten(exc: BaseException) -> Exception:
    """Return the first leaf of an exception group, or the exception itself."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc if isinstance(exc, Exception) else ConnectionError(f"MCP session failed: {exc!r}")
