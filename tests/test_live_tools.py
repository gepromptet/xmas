"""Tools end to end: ``MCPTools`` against a real serving-xmas dev server.

The ``serving_xmas`` fixture starts ``serving serve`` from a serving-xmas checkout
(``uv run --project <checkout>``; the checkout defaults to ``~/serving-xmas`` and is
overridden with ``XMAS_SERVING_XMAS_DIR``) on 127.0.0.1 with an ephemeral port and
every piece of state under ``tmp_path``, waits for ``/healthz``, and mints a key
scoped to ``run_python`` with the server's own CLI. The tests read that key from
the fixture, never from ``XMAS_TOOLS_KEY``, so they need no tool credential.

Both tests need a reachable docker daemon (the server runs each tool in a
container) and skip without one. The fake-model test then runs in the default
suite; the live one additionally needs Tom's gateway (``requires_gateway``).

The live test's model is ``XMAS_LIVE_TOOLS_MODEL``, defaulting to the pinned
``MODEL`` of ``test_live_gateway``. Not every gateway route serves tools: when the
gateway has no tool-capable endpoint for the chosen model, the test fails with a
message naming the model and the variable, so pick one whose route does.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

pytest.importorskip("mcp")

from xmas import APILLM, MAS, Agent
from xmas.tools.mcp import MCPTools

from .conftest import requires_gateway
from .test_live_gateway import GATEWAY_CONCURRENCY, MODEL
from .test_tools import ToolMockLLM, req

SERVING_XMAS_DIR = Path(os.environ.get("XMAS_SERVING_XMAS_DIR", Path.home() / "serving-xmas"))
"""The serving-xmas checkout the dev server is started from."""

RESERVED_PORTS = {8000, 8100, 8200, 8300, 8400}
"""Ports the host's long-running services own. The dev server never takes one."""

JOB_LABEL = "serving-xmas.job"
"""The label serving-xmas puts on every job container."""

QUESTION = "what is 2**20?"
ANSWER = "1048576"

LIVE_TOOLS_MODEL_ENV = "XMAS_LIVE_TOOLS_MODEL"
"""Overrides the live test's model; unset, it is ``test_live_gateway.MODEL``."""


def _docker_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


requires_docker = pytest.mark.skipif(
    not _docker_reachable(), reason="docker is not reachable; the serving-xmas dev server needs it"
)


def _free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])
        if port not in RESERVED_PORTS:
            return port


def _job_containers() -> set[str]:
    listed = subprocess.run(
        ["docker", "ps", "-aq", "--no-trunc", "--filter", f"label={JOB_LABEL}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return set(listed.stdout.split())


@dataclass(frozen=True)
class DevServer:
    """A running serving-xmas dev server and a key it accepts."""

    url: str
    """The MCP endpoint, ``http://127.0.0.1:<port>/mcp``."""
    key: str
    """A key scoped to ``run_python``."""


@pytest.fixture(scope="module")
def serving_xmas(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DevServer]:
    """Start a serving-xmas dev server, yield its URL and a run_python key, then stop it.

    Teardown stops the whole process group and checks that no job container this
    server started is left behind.
    """
    uv = shutil.which("uv")
    if uv is None or not (SERVING_XMAS_DIR / "pyproject.toml").is_file():
        pytest.skip(f"needs uv and a serving-xmas checkout at {SERVING_XMAS_DIR} (XMAS_SERVING_XMAS_DIR)")

    root = tmp_path_factory.mktemp("serving-xmas")
    port = _free_port()
    # serving-xmas needs none of xmas's own settings (gateway URL and key included).
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV" and not k.startswith("XMAS_")}
    env.update(
        {
            "SX_BIND_HOST": "127.0.0.1",
            "SX_PORT": str(port),
            "SX_STATE_DIR": str(root / "state"),
            "SX_SANDBOX_ROOT": str(root / "sandbox"),
            "SX_CACHE_DB": str(root / "sandbox" / "cache" / "cache.db"),
            "SX_WORKSPACE_ROOT": str(root / "workspace"),
            "SX_POLICY_PATH": str(root / "policy.json"),
            "SX_MAX_CONTAINERS": "2",
            "SX_LOG_LEVEL": "warning",
        }
    )
    (root / "workspace").mkdir()
    serving = [uv, "run", "--frozen", "--project", str(SERVING_XMAS_DIR), "serving"]

    issued = subprocess.run(
        [*serving, "keys", "issue", "--label", "xmas-live-tools", "--scopes", "run_python"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    key = issued.stdout.splitlines()[0].strip()
    assert key.startswith("sk-sx-"), f"unexpected `keys issue` output: {issued.stdout!r}"

    before = _job_containers()
    log_path = root / "server.log"
    with log_path.open("ab") as log:
        proc = subprocess.Popen([*serving, "serve"], env=env, stdout=log, stderr=log, start_new_session=True)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        with httpx.Client(timeout=1) as client:
            while True:
                if proc.poll() is not None:
                    pytest.fail(f"serving-xmas exited with {proc.returncode}:\n{log_path.read_text()}")
                with contextlib.suppress(httpx.HTTPError):
                    if client.get(f"{base}/healthz").status_code == 200:
                        break
                if time.monotonic() > deadline:
                    pytest.fail(f"serving-xmas did not answer /healthz:\n{log_path.read_text()}")
                time.sleep(0.05)
        yield DevServer(f"{base}/mcp", key)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    deadline = time.monotonic() + 30
    while (left := _job_containers() - before) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not left, f"job containers left behind: {sorted(left)}"


def coder() -> Agent:
    return Agent(
        "coder",
        "Use the run_python tool to compute the answer, then reply with the number only.\n\n{query}",
        output="answer",
        tools=["run_python"],
    )


@requires_docker
def test_a_scripted_model_runs_python_on_the_dev_server(serving_xmas: DevServer) -> None:
    """The server plumbing without a gateway: a scripted model, a real sandboxed run."""
    llm = ToolMockLLM([("", [req("run_python", '{"code": "print(2**20)"}')]), (ANSWER, [])])
    with MCPTools(serving_xmas.url, serving_xmas.key, timeout_s=120) as remote:
        assert [t.name for t in remote.tools()] == ["run_python"]
        trace = MAS(agents=[coder()], llm=llm, tools=[remote]).run(QUESTION)

    assert trace.error is None
    (call,) = trace["coder"][0].tool_calls
    assert call.tool == "run_python"
    assert not call.failed, call.result
    # stdout first; serving-xmas then lists the files the run left (main.py itself).
    assert call.result.splitlines()[0] == ANSWER
    assert trace.output == ANSWER
    # The model saw the sandbox's stdout as the tool message.
    history = llm.seen[-1][0]
    assert history[-1]["role"] == "tool"
    assert ANSWER in history[-1]["content"]


@requires_gateway
@requires_docker
def test_a_gateway_model_runs_python_on_the_dev_server(serving_xmas: DevServer) -> None:
    """Live: a gateway model (via APILLM) calls run_python on the dev server.

    Needs ``XMAS_BASE_URL`` and ``XMAS_API_KEY`` for the gateway and a reachable
    docker. The tool key comes from the ``serving_xmas`` fixture, not from
    ``XMAS_TOOLS_KEY``, so nothing else has to be set. The model is
    ``XMAS_LIVE_TOOLS_MODEL`` (default ``test_live_gateway.MODEL``).
    """
    model = os.environ.get(LIVE_TOOLS_MODEL_ENV) or MODEL
    llm = APILLM(model=model, max_tokens=256, max_concurrent_calls=GATEWAY_CONCURRENCY)
    with MCPTools(serving_xmas.url, serving_xmas.key, timeout_s=120) as remote:
        trace = MAS(agents=[coder()], llm=llm, tools=[remote]).run(QUESTION)

    if trace.error is not None and "No endpoints found" in trace.error and "Tool Compatibility" in trace.error:
        pytest.fail(
            f"the gateway has no tool-capable route for {model!r}; "
            f"set {LIVE_TOOLS_MODEL_ENV} to a model that serves tools",
            pytrace=False,
        )
    assert trace.error is None, trace.error
    calls = trace["coder"][0].tool_calls
    assert calls, "the model answered without calling run_python"
    assert any(c.tool == "run_python" and not c.failed for c in calls), [c.to_dict() for c in calls]
    assert ANSWER in trace.output.replace(",", "")
