"""One MCP session on a private background event loop, callable from synchronous code.

The pattern promptolution's ``APILLM`` uses for its client: concurrent graph nodes
(``run_batch`` runs them on worker threads) share one session and one handshake.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from http import HTTPStatus
from typing import Any

import httpx2
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CONNECTION_CLOSED, INVALID_REQUEST, REQUEST_TIMEOUT, CallToolResult, PaginatedRequestParams
from mcp.types import Tool as McpTool

from xmas.errors import XmasError
from xmas.utils import retry_after_seconds

__all__ = [
    "McpSession",
    "first_leaf",
]


logger = logging.getLogger(__name__)


class McpSession:
    """Owns the background loop and the live session; every public method is synchronous."""

    def __init__(self, url: str, key: str, timeout_s: float, max_concurrent: int) -> None:
        """Initialize the session owner. Nothing connects until the first use."""
        self.url = url
        self.key = key
        self.timeout_s = timeout_s
        self.max_concurrent = max_concurrent
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.closed = False
        # Loop-side state, touched only from the background loop.
        self.semaphore: asyncio.Semaphore | None = None
        self.connecting: asyncio.Lock | None = None
        self.live: ClientSession | None = None
        self.owner: asyncio.Task[None] | None = None
        self.stop: asyncio.Event | None = None
        self.refusal: XmasError | None = None

    def run(self, coro: Any) -> Any:
        """Run a coroutine on the background loop and wait for it."""
        with self.lock:
            if self.closed:
                coro.close()
                raise RuntimeError("MCPTools is closed.")
            if self.loop is None:
                loop = asyncio.new_event_loop()
                self.thread = threading.Thread(target=loop.run_forever, name="MCPToolsLoop", daemon=True)
                self.thread.start()
                self.loop = loop
            loop = self.loop
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=self.timeout_s + 60.0)

    def list_tools(self) -> list[McpTool]:
        """Every tool the server offers this key, all ``tools/list`` pages."""
        return self.run(self.list_all())  # type: ignore[no-any-return]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Run one tool; a session the server forgot is reopened and the call resent once."""
        return self.run(self.call_limited(name, arguments))  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the session and stop the background loop. Idempotent."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            loop = self.loop
            thread = self.thread
        if loop is None or thread is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.disconnect(), loop).result(timeout=30.0)
        except Exception:  # closing is best effort; the loop stops regardless
            logger.warning("MCPTools: error while closing the session", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=30.0)
        loop.close()

    async def record_refusal(self, response: httpx2.Response) -> None:
        """Turn an HTTP refusal into the error the run ends with; the SDK would hang on it or group it."""
        status = response.status_code
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            retry_after = retry_after_seconds(response.headers)
            self.refusal = XmasError(f"mcp: rate limited (429), retry after {retry_after}s", status, retry_after)
        elif status == HTTPStatus.UNAUTHORIZED:
            self.refusal = XmasError("mcp: unauthorized (401): check XMAS_TOOLS_KEY", status)
        elif status == HTTPStatus.FORBIDDEN:
            self.refusal = XmasError("mcp: forbidden (403)", status)

    async def own_session(self, ready: asyncio.Future[ClientSession], stop: asyncio.Event) -> None:
        """Own one session: enter and leave the SDK's contexts in this one task, as it requires."""
        try:
            async with (
                httpx2.AsyncClient(
                    headers={"Authorization": f"Bearer {self.key}"},
                    timeout=httpx2.Timeout(30.0, read=self.timeout_s),
                    event_hooks={"response": [self.record_refusal]},
                ) as client,
                streamable_http_client(self.url, http_client=client) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                ready.set_result(session)
                await stop.wait()
        except BaseException as exc:  # an ExceptionGroup from the transport, or a cancellation
            if not ready.done():
                ready.set_exception(self.refusal or first_leaf(exc))
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            if not ready.done():
                ready.set_exception(ConnectionError("mcp: session closed before it was ready"))

    async def connect(self) -> ClientSession:
        """Return the live session, opening one if there is none (one handshake for concurrent callers)."""
        if self.connecting is None:
            self.connecting = asyncio.Lock()
        async with self.connecting:
            if self.live is not None and self.owner is not None and not self.owner.done():
                return self.live
            await self.disconnect()
            self.refusal = None
            ready: asyncio.Future[ClientSession] = asyncio.get_running_loop().create_future()
            self.stop = asyncio.Event()
            self.owner = asyncio.create_task(self.own_session(ready, self.stop))
            try:
                self.live = await asyncio.wait_for(ready, timeout=self.timeout_s)
            except Exception:
                await self.disconnect()
                raise
            return self.live

    async def disconnect(self) -> None:
        """Close the current session, if any, and wait for its owner to leave the SDK's contexts."""
        owner = self.owner
        stop = self.stop
        self.live = None
        self.owner = None
        self.stop = None
        if owner is None or stop is None:
            return
        stop.set()
        try:
            await asyncio.wait_for(owner, timeout=10.0)
        except (TimeoutError, asyncio.CancelledError):
            owner.cancel()

    async def list_all(self) -> list[McpTool]:
        """Collect every ``tools/list`` page."""
        session = await self.connect()
        found: list[McpTool] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
            found.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                return found

    async def call_limited(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call under the concurrency cap."""
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
        async with self.semaphore:
            return await self.call_once(name, arguments, resend=True)

    async def call_once(self, name: str, arguments: dict[str, Any], *, resend: bool) -> CallToolResult:
        """One ``tools/call``; protocol errors become the errors a run sees."""
        session = await self.connect()
        try:
            return await session.call_tool(name, arguments, read_timeout_seconds=self.timeout_s)
        except MCPError as exc:
            if self.refusal is not None:
                refusal = self.refusal
                await self.disconnect()
                raise refusal from exc
            code = exc.code
            # Server body vs the client's stand-in for a 404 on an established session.
            lost = code == INVALID_REQUEST and exc.message in ("Session terminated", "Session not found")
            if lost or code == CONNECTION_CLOSED:
                await self.disconnect()
            if lost and resend:  # the server forgot the session; nothing ran
                return await self.call_once(name, arguments, resend=False)
            if lost or code == CONNECTION_CLOSED:
                raise ConnectionError(f"mcp: connection lost during {name} ({code}): {exc}") from exc
            if code == REQUEST_TIMEOUT:
                raise TimeoutError(f"mcp: {name} timed out after {self.timeout_s}s ({code})") from exc
            raise XmasError(f"mcp: error ({code}): {exc.message}") from exc


def first_leaf(exc: BaseException) -> Exception:
    """Return the first leaf of an exception group, or the exception itself."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc if isinstance(exc, Exception) else ConnectionError(f"mcp: session failed: {exc!r}")
