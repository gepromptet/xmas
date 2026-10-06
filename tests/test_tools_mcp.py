"""MCPTools against an in-process MCPServer over streamable HTTP (127.0.0.1, ephemeral port).

The fake server speaks serving-xmas's result shapes (Envelope JSON, error bodies)
and sits behind a tiny bearer-auth middleware, so HTTP-level refusals are real.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

pytest.importorskip("mcp")

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent
from starlette.types import ASGIApp, Receive, Scope, Send

from xmas import MAS, Agent, ToolCall
from xmas.errors import XmasError
from xmas.tools.mcp import MCPTools, result_text

from .test_tools import ToolMockLLM, req

KEY = "sk-sx-good"


def envelope(exit_code: int = 0, stdout: str = "", stderr: str = "", **extra: Any) -> dict[str, Any]:
    body = {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "truncated": False, "files": {}}
    body.update({"wall_s": 0.1, "killed": "none", "runtime": "inproc", "image": None})
    body.update(extra)
    return body


def as_result(body: dict[str, Any], is_error: bool) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(body))], is_error=is_error)


def error_body(code: str, status: int, message: str = "no", **extra: Any) -> CallToolResult:
    return as_result({"error": code, "status": status, "message": message, **extra}, is_error=True)


# ------------------------------------------------------------------ server


class Server:
    """MCPServer app, auth middleware and bookkeeping for one test module."""

    def __init__(self) -> None:
        """Build the app; the fixture serves it."""
        self.in_flight = 0
        self.peak = 0
        self.sessions: set[str] = set()
        self.initializes = 0
        mcp = MCPServer("fake-serving-xmas")
        self.mcp = mcp

        @mcp.tool()
        def run_python(code: str) -> CallToolResult:
            """Run Python code."""
            if code == "raise":
                return as_result(envelope(1, stdout="partial", stderr="Traceback: boom"), is_error=True)
            if code == "files":
                return as_result(
                    envelope(0, stdout="wrote", truncated=True, files={"b.txt": "YQ==", "a.txt": "Yg=="}), False
                )
            return as_result(envelope(0, stdout=f"ran {code}"), is_error=False)

        @mcp.tool()
        def refuse(code: str, status: int, retry_after_s: float | None = None) -> CallToolResult:
            """Answer with a serving-xmas error body."""
            extra = {} if retry_after_s is None else {"retry_after_s": retry_after_s}
            return error_body(code, status, f"{code} happened", **extra)

        @mcp.tool()
        async def slow(ms: int) -> str:
            """Sleep, counting concurrent calls."""
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(ms / 1000)
            self.in_flight -= 1
            return "slept"

        @mcp.tool(name="bad.name")
        def bad() -> str:
            """A name OpenAI would refuse."""
            return ""

        self.app = self._auth(mcp.streamable_http_app())

    def _auth(self, app: ASGIApp) -> ASGIApp:
        async def guarded(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await app(scope, receive, send)
                return
            headers = dict(scope["headers"])
            auth = headers.get(b"authorization", b"").decode()
            status = {f"Bearer {KEY}": 0, "Bearer sk-forbidden": 403, "Bearer sk-busy": 429}.get(auth, 401)
            if status:
                extra = [(b"retry-after", b"7")] if status == 429 else []
                await send({"type": "http.response.start", "status": status, "headers": extra})
                await send({"type": "http.response.body", "body": b'{"error": "refused"}'})
                return
            if session := headers.get(b"mcp-session-id"):
                self.sessions.add(session.decode())
            else:
                self.initializes += 1
            await app(scope, receive, send)

        return guarded

    def forget_sessions(self) -> None:
        """Make the server forget every session, as a restart would."""
        self.mcp.session_manager._server_instances.clear()


@pytest.fixture(scope="module")
def server() -> Iterator[tuple[Server, str]]:
    fake = Server()
    config = uvicorn.Config(fake.app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uv.started:
        assert time.monotonic() < deadline, "fake MCP server did not start"
        time.sleep(0.01)
    port = uv.servers[0].sockets[0].getsockname()[1]
    yield fake, f"http://127.0.0.1:{port}/mcp"
    uv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def remote(server: tuple[Server, str]) -> Iterator[MCPTools]:
    with MCPTools(server[1], KEY, timeout_s=20) as tools:
        yield tools


# ---------------------------------------------------------------- over HTTP


def test_lists_tools_once_and_skips_names_openai_would_refuse(remote: MCPTools) -> None:
    names = [t.name for t in remote.tools()]
    assert names == ["run_python", "refuse", "slow"]
    run_python = remote.tools()[0]
    assert run_python.fn is None
    assert run_python.description == "Run Python code."
    assert run_python.parameters["required"] == ["code"]


def test_envelope_results_become_text_or_tool_errors(remote: MCPTools) -> None:
    assert remote.call("run_python", {"code": "1+1"}) == "ran 1+1"
    assert remote.call("run_python", {"code": "files"}) == "wrote\n[truncated]\n[files: a.txt, b.txt]"
    with pytest.raises(XmasError) as failed:
        remote.call("run_python", {"code": "raise"})
    assert str(failed.value) == "exit_code=1 killed=none\nTraceback: boom"


@pytest.mark.parametrize(
    ("code", "status", "retry", "ends_run"),
    [
        ("tool_not_in_scope", 403, None, True),
        ("unauthorized", 401, None, True),
        ("insufficient_budget", 429, None, True),
        ("too_many_concurrent", 429, 1, True),
        ("invalid_arguments", 400, None, False),
        ("unknown_tool", 404, None, False),
        ("internal_error", 500, None, False),
        ("queued_heavy", 503, None, False),
        ("container_unavailable", 503, None, False),
    ],
)
def test_error_bodies_map_to_the_right_error(
    remote: MCPTools, code: str, status: int, retry: int | None, ends_run: bool
) -> None:
    arguments: dict[str, Any] = {"code": code, "status": status}
    if retry is not None:
        arguments["retry_after_s"] = retry
    with pytest.raises(XmasError, match=rf"mcp: {code} \({status}\): {code} happened") as raised:
        remote.call("refuse", arguments)
    assert (raised.value.status in (401, 403, 429)) is ends_run
    assert raised.value.retry_after == (1.0 if retry else None)


def test_one_session_is_shared_and_concurrency_is_capped(server: tuple[Server, str]) -> None:
    fake, url = server
    before = fake.initializes
    fake.peak = 0
    with MCPTools(url, KEY, max_concurrent=2) as tools, ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: tools.call("slow", {"ms": 50}), range(8)))
    assert results == ["slept"] * 8
    assert fake.peak == 2
    assert fake.initializes - before == 1


def test_a_forgotten_session_is_reopened_and_the_call_resent_once(server: tuple[Server, str]) -> None:
    fake, url = server
    with MCPTools(url, KEY) as tools:
        assert tools.call("run_python", {"code": "a"}) == "ran a"
        before = fake.initializes
        fake.forget_sessions()
        assert tools.call("run_python", {"code": "b"}) == "ran b"
        assert fake.initializes - before == 1


@pytest.mark.parametrize(
    ("key", "status", "match"),
    [
        ("sk-wrong", 401, "unauthorized"),
        ("sk-forbidden", 403, "forbidden"),
        ("sk-busy", 429, "429"),
    ],
)
def test_http_refusals(server: tuple[Server, str], key: str, status: int, match: str) -> None:
    with MCPTools(server[1], key, timeout_s=20) as tools:
        started = time.monotonic()
        with pytest.raises(XmasError, match=match) as raised:
            tools.tools()
        with pytest.raises(XmasError, match=match):
            tools.call("run_python", {"code": "x"})
        assert time.monotonic() - started < 10, "a refusal must not wait for the read timeout"
    assert raised.value.status == status
    if status == 429:
        assert raised.value.retry_after == 7.0


def test_a_mas_runs_remote_tools_and_ends_on_a_refusal(remote: MCPTools) -> None:
    agent = Agent("coder", "Compute {query}", output="answer", tools=["run_python", "refuse"])
    llm = ToolMockLLM([("", [req("run_python", '{"code": "2**10"}')]), ("1024", [])])
    mas = MAS(agents=[agent], llm=llm, tools=[remote])
    trace = mas.run("2**10")
    assert trace.error is None
    assert trace.output == "1024"
    assert trace["coder"][0].tool_calls == (ToolCall("run_python", {"code": "2**10"}, "ran 2**10"),)

    llm.turns = [("", [req("refuse", '{"code": "too_many_concurrent", "status": 429, "retry_after_s": 1}')])]
    trace = mas.run("again")
    assert trace.error == "XmasError: mcp: too_many_concurrent (429): too_many_concurrent happened"


# ------------------------------------------------------------- construction


def test_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XMAS_TOOLS_KEY", "sk-env")
    assert MCPTools("http://127.0.0.1:1/mcp").session.key == "sk-env"
    monkeypatch.delenv("XMAS_TOOLS_KEY")
    with pytest.raises(ValueError, match="XMAS_TOOLS_KEY"):
        MCPTools("http://127.0.0.1:1/mcp")


def test_close_is_idempotent_and_final(remote: MCPTools) -> None:
    remote.tools()
    remote.close()
    remote.close()
    with pytest.raises(RuntimeError, match="closed"):
        remote.call("run_python", {"code": "x"})
    MCPTools("http://127.0.0.1:1/mcp", KEY).close()  # never connected


def test_repr_never_shows_the_key() -> None:
    assert KEY not in repr(MCPTools("http://h/mcp", KEY))


# ----------------------------------------------------------- result mapping


def test_result_mapping_for_foreign_servers() -> None:
    plain = CallToolResult(content=[TextContent(type="text", text="a"), TextContent(type="text", text="b")])
    assert result_text(plain) == "a\nb"
    image = CallToolResult(content=[ImageContent(type="image", data="AA==", mime_type="image/png")])
    assert result_text(image) == "[image content omitted]"
    with pytest.raises(XmasError, match="went wrong"):
        result_text(CallToolResult(content=[TextContent(type="text", text="went wrong")], is_error=True))
    with pytest.raises(XmasError, match="tool failed"):
        result_text(CallToolResult(content=[], is_error=True))
    assert result_text(as_result({"other": 1}, is_error=False)) == '{"other": 1}'


def test_result_mapping_for_killed_runs_and_missing_retry_hint() -> None:
    with pytest.raises(XmasError, match="exit_code=0 killed=timeout\nout"):
        result_text(as_result(envelope(0, stdout="out", killed="timeout"), is_error=True))
    with pytest.raises(XmasError) as raised:
        result_text(error_body("too_many_concurrent", 429))
    assert raised.value.retry_after is None
