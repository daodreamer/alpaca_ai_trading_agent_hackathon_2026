"""The transport seam and the trust boundary — specs/04 D6 and D7.

`alphagate.execution` depends on a `Protocol`, never on `fastmcp`
(adr/0002 D4). Two things fall out of that, and both are worth more than the
indirection costs:

* the suite runs offline and deterministically, against `RecordedSession`, and
  never touches a live account;
* the one module that imports the MCP client library is small enough to read in
  a sitting.

**Tool output is untrusted (D7).** Every MCP response arrives wrapped in an
`_alpaca_mcp_security` envelope that says so. This module unwraps `data` for the
caller and *keeps the envelope attached to it*, so the journal can answer a
question most agent projects cannot: which bytes in this decision came from
outside the trust boundary?

The unwrapping happens here, in the session, rather than in `submit` — so a
recorded response and a live response travel the identical code path, and a
future second session implementation cannot forget to do it.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from alphagate.execution.errors import MalformedToolOutput

__all__ = [
    "ENVELOPE_KEY",
    "McpSession",
    "RecordedSession",
    "SecurityEnvelope",
    "ToolArgument",
    "ToolResult",
    "unwrap",
]

ENVELOPE_KEY: Final = "_alpaca_mcp_security"
UNTRUSTED: Final = "untrusted_tool_output"

type ToolArgument = str | int | list[dict[str, str]]
"""What may be sent as a tool argument. Everything Alpaca's schema takes is a
string; `int` and `list` are here because the wire protocol permits them and
pretending otherwise would push a cast onto every call site."""


@dataclass(frozen=True, slots=True)
class SecurityEnvelope:
    """The marker the MCP server attaches to its own output.

    Carried, never acted on. Its whole job is to travel: into the `ToolResult`,
    into the journal record (specs/06 D5), and — if any of this text ever reaches
    a prompt — alongside the bytes it describes.
    """

    trust: str
    instructions: str
    tool_name: str = ""
    risk: str = ""
    """`api_structured` or `external_text`. The server classifies its own tools;
    `external_text` is the one carrying a full prompt-injection warning, and it
    is the one the agent must never splice into a prompt unmarked."""

    @property
    def is_untrusted(self) -> bool:
        return self.trust == UNTRUSTED


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One tool response: the payload, its provenance, and the original bytes."""

    tool: str
    data: Mapping[str, Any]
    envelope: SecurityEnvelope | None
    raw: str
    """The response exactly as it arrived. The journal stores this verbatim —
    a paraphrase of a rejection reason is not a rejection reason."""

    def require(self, key: str) -> Any:
        """Read a field that must be present.

        Raises rather than returning `None`. A missing `status` defaulted to
        something plausible is how an order nobody knows about turns up at
        reconciliation.
        """
        if key not in self.data:
            raise MalformedToolOutput(
                f"{self.tool} response has no {key!r}; got keys {sorted(self.data)}"
            )
        return self.data[key]


@runtime_checkable
class McpSession(Protocol):
    """The seam. `fastmcp` lives behind this and nowhere else."""

    def call(self, tool: str, arguments: Mapping[str, ToolArgument]) -> ToolResult: ...


def unwrap(tool: str, payload: str) -> ToolResult:
    """Parse one MCP response, separating the data from its trust marker.

    An unwrapped payload — one with no envelope — is accepted and recorded with
    `envelope=None` rather than rejected. Not every tool wraps its output, and
    refusing to read the ones that do not would make the adapter depend on a
    detail of the server's implementation. What must never happen is the reverse:
    an envelope that is present being dropped on the floor.
    """
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MalformedToolOutput(f"{tool} returned text that is not JSON: {payload[:200]!r}")\
            from exc

    if not isinstance(parsed, dict):
        raise MalformedToolOutput(
            f"{tool} returned a {type(parsed).__name__}, expected an object"
        )

    marker = parsed.get(ENVELOPE_KEY)
    envelope: SecurityEnvelope | None = None
    if isinstance(marker, dict):
        envelope = SecurityEnvelope(
            trust=str(marker.get("trust", "")),
            instructions=str(marker.get("instructions", "")),
            tool_name=str(marker.get("tool_name", "")),
            risk=str(marker.get("risk", "")),
        )

    data = parsed.get("data", parsed) if envelope is not None else parsed
    if not isinstance(data, dict):
        raise MalformedToolOutput(f"{tool} wrapped a {type(data).__name__}, expected an object")

    return ToolResult(tool=tool, data=data, envelope=envelope, raw=payload)


@dataclass
class RecordedSession:
    """Replays captured tool responses. The only session the test suite uses.

    Responses are queued per tool, so a test can script a sequence — a timeout
    followed by a successful read-back is the case specs/04 D4 turns on. A queued
    `Exception` is raised instead of returned, which is how transport failures
    and timeouts are scripted without a network.

    Every call is recorded in `calls`, because most of what these tests assert is
    not the return value but *what was sent*: the sign of a limit price, the
    stability of a client order id, and the absence of a second submission after
    a timeout.
    """

    responses: dict[str, deque[str | Exception]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, ToolArgument]]] = field(default_factory=list)

    @classmethod
    def scripted(cls, **by_tool: str | Exception | Sequence[str | Exception]) -> RecordedSession:
        """Build a session from `tool_name=response` keyword arguments."""
        queues: dict[str, deque[str | Exception]] = {}
        for tool, scripted in by_tool.items():
            items: Iterable[str | Exception] = (
                [scripted] if isinstance(scripted, (str, Exception)) else scripted
            )
            queues[tool] = deque(items)
        return cls(responses=queues)

    def call(self, tool: str, arguments: Mapping[str, ToolArgument]) -> ToolResult:
        self.calls.append((tool, dict(arguments)))
        queue = self.responses.get(tool)
        if not queue:
            raise AssertionError(
                f"RecordedSession has no scripted response for {tool!r}; "
                f"scripted tools are {sorted(self.responses)}"
            )
        # The last scripted response repeats. A test that scripts one reply and
        # then asserts on retries should be asserting on `calls`, not fighting an
        # exhausted queue.
        nxt = queue.popleft() if len(queue) > 1 else queue[0]
        if isinstance(nxt, Exception):
            raise nxt
        return unwrap(tool, nxt)

    def calls_to(self, tool: str) -> list[dict[str, ToolArgument]]:
        return [arguments for name, arguments in self.calls if name == tool]
