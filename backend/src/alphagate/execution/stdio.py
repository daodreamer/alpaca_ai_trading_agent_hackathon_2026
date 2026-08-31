"""The live MCP transport — adr/0002 D4.

**The only module in the codebase that imports `fastmcp`.** Everything else
depends on the `McpSession` protocol in `session.py`, which is why the test
suite runs offline and the backtest never opens a subprocess.

The awkward part is threading, and it is worth explaining rather than hiding.
`fastmcp`'s client is async and holds a stdio subprocess open across calls; the
agent loop is synchronous and wants one call at a time. Running `asyncio.run`
per call would tear down and respawn the server on every order, which is both
slow and a good way to lose an in-flight response. So this class owns one event
loop on one daemon thread, keeps the client open on it, and marshals each call
across with `run_coroutine_threadsafe`.

Credentials are read from the environment by the *server subprocess*, never
handled or logged here. Nothing in this module formats a key into a string —
specs/06 D4, and the demo video shows this terminal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import suppress
from types import TracebackType
from typing import Any, Final

from alphagate.execution.errors import (
    MalformedToolOutput,
    ToolTimeout,
    TransportFailure,
)
from alphagate.execution.session import ToolArgument, ToolResult, unwrap

__all__ = [
    "DEFAULT_TIMEOUT",
    "FASTMCP_TOOLS_LOGGER",
    "StdioSession",
    "UnstructuredOutputFilter",
    "quiet_unstructured_output",
]

FASTMCP_TOOLS_LOGGER: Final = "fastmcp.client.mixins.tools"
"""Where the client logs its post-call schema validation."""

DEFAULT_TIMEOUT: Final = 30.0
"""Seconds to wait for one tool call. Generous: a cold `uvx` start is slow, and
a timeout here costs a read-back round trip, not a duplicate order."""

_LAUNCH: Final = ("uvx", "--from", "alpaca-mcp-server", "alpaca-mcp-server")


class StdioSession:
    """A live `McpSession` backed by the Alpaca MCP server over stdio."""

    def __init__(
        self,
        *,
        command: Sequence[str] = _LAUNCH,
        env: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._command = tuple(command)
        self._env = dict(env) if env is not None else None
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def __enter__(self) -> StdioSession:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        quiet_unstructured_output()

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name="mcp-stdio", daemon=True)
        thread.start()
        self._loop, self._thread = loop, thread

        transport = StdioTransport(
            command=self._command[0],
            args=list(self._command[1:]),
            env={**os.environ, **(self._env or {})},
        )
        client = Client(transport)
        # fastmcp ships no annotations for its async context manager.
        self._run(client.__aenter__(), timeout=120.0)  # type: ignore[no-untyped-call]
        self._client = client

    def close(self) -> None:
        if self._client is not None and self._loop is not None:
            # A server that will not shut down cleanly is not our problem now;
            # the subprocess is a daemon and dies with this process.
            with suppress(ToolTimeout, TransportFailure):
                self._run(self._client.__aexit__(None, None, None), timeout=15.0)
        self._client = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._loop, self._thread = None, None

    # ------------------------------------------------------------------ #
    # The seam
    # ------------------------------------------------------------------ #

    def call(self, tool: str, arguments: Mapping[str, ToolArgument]) -> ToolResult:
        if self._client is None:
            raise TransportFailure("session is not open; call open() first")
        result = self._run(self._client.call_tool(tool, dict(arguments)), self._timeout)
        return unwrap(tool, _text_of(tool, result))

    def list_tools(self) -> list[str]:
        """Operational, not on the critical path. Used by the pre-open check."""
        if self._client is None:
            raise TransportFailure("session is not open; call open() first")
        tools = self._run(self._client.list_tools(), self._timeout)
        return sorted(tool.name for tool in tools)

    # ------------------------------------------------------------------ #

    def _run(self, coro: Any, timeout: float) -> Any:
        if self._loop is None:  # pragma: no cover - guarded by callers
            raise TransportFailure("session has no event loop")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            future.cancel()
            # Deliberately ToolTimeout, not TransportFailure: the call may have
            # landed. specs/04 D4 — resolve by reading back, never by resending.
            raise ToolTimeout(f"no answer within {timeout}s") from exc
        except Exception as exc:
            raise TransportFailure(f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------- #
# One line of the server's output that is not ours to fix
# ---------------------------------------------------------------------- #


class UnstructuredOutputFilter(logging.Filter):
    """Drops the client's complaint that `structuredContent` was null.

    `alpaca-mcp-server` declares an output schema of `dict[str, Any]` on several
    tools and then answers with `structuredContent: null`. fastmcp's client
    validates the second against the first, fails, and logs at ERROR — once per
    call. A thirty-second heartbeat therefore paints the terminal red all
    session while every order goes through, which is worse than useless: it
    teaches the operator that red means nothing on the one screen where red has
    to mean something.

    We never read the field it is complaining about. `_text_of` takes the
    content block, which is present and correct and carries the trust envelope.

    A filter and not a log level, and the predicate is narrow on purpose:
    structured content that is *present and wrong* is a real server bug and
    still gets through. Only `input_value=None` — the field was simply absent —
    is dropped.
    """

    _MESSAGE: Final = "Error parsing structured content"
    _ABSENT: Final = "input_value=None"

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.ERROR:
            return True
        message = record.getMessage()
        return not (self._MESSAGE in message and self._ABSENT in message)


def quiet_unstructured_output() -> None:
    """Install the filter once, idempotently.

    Called from `open()` rather than at import: a logging side effect at import
    time would reach any process that merely touched this module, and this is a
    workaround for one server's behaviour, not a property of the package.
    """
    logger = logging.getLogger(FASTMCP_TOOLS_LOGGER)
    if not any(isinstance(f, UnstructuredOutputFilter) for f in logger.filters):
        logger.addFilter(UnstructuredOutputFilter())


def _text_of(tool: str, result: Any) -> str:
    """Pull the JSON text out of an MCP `CallToolResult`.

    The server answers with a list of content blocks. We want the first textual
    one, verbatim — it is what the journal stores and what carries the
    `_alpaca_mcp_security` envelope (specs/04 D7).
    """
    content = getattr(result, "content", None)
    if content:
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                return text
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return json.dumps(structured)
    data = getattr(result, "data", None)
    if isinstance(data, str):
        return data
    raise MalformedToolOutput(f"{tool} returned no text content: {result!r}"[:300])
