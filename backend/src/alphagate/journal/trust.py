"""Which bytes came from outside the trust boundary — specs/06 D5.

> "The journal can then answer a question most agent projects cannot: *which
> bytes in this decision came from outside the trust boundary?*"

That sentence is a claim, and a claim in a spec is worth what the code can
demonstrate. `unwrap` (specs/04 D7) keeps the `_alpaca_mcp_security` envelope on
the `ToolResult`; `Submission` and `OutcomeRecord` carry it forward; the encoder
writes it into the line. This module is the last step: reading a journalled
record back and *saying* where its untrusted bytes are.

Two sources, and it matters that they are different:

* **`mcp`** — broker output, wrapped by the Alpaca MCP server, which marks its
  own tools `api_structured` or `external_text`. The second is the dangerous
  one: free text that a careless agent could splice into a prompt.
* **`model`** — the LLM's own response. Not envelope-marked, because nothing
  wraps it, but no more trusted for that. A rationale is a string a model wrote,
  and the only reason it is safe here is that `Choice.resolve` treats the index
  as a lookup into a menu we built rather than as an instruction.

Everything else in a record — the market read, the candidates, the verdict — was
computed by the pure layers from data we fetched. That is the point of the
answer: the untrusted set is small, enumerable, and none of it reaches the Gate.

Works on the **read** form (plain dicts off disk), not on live objects, so what
it reports is what a judge would find in the file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["UntrustedSource", "trust_report", "untrusted_sources"]

_ENVELOPE_KEY = "envelope"


@dataclass(frozen=True, slots=True)
class UntrustedSource:
    """One region of a record whose bytes came from outside."""

    path: str
    """Dotted path into the journal line, e.g. `submission.raw`."""
    kind: str
    """`mcp` or `model`."""
    trust: str
    """The envelope's own marker, or `unwrapped` where nothing wrapped it."""
    risk: str = ""
    """`api_structured` or `external_text`, from the MCP server's own
    classification. Empty for a model response, which nobody classifies."""

    @property
    def is_prompt_injection_surface(self) -> bool:
        """True where these bytes could carry instructions to a later prompt.

        `external_text` is the MCP server's own warning; a model response is one
        by construction. `api_structured` is not — it is fields we parse by name,
        and a rejection reason we never re-prompt with.
        """
        return self.kind == "model" or self.risk == "external_text"


def untrusted_sources(record: Mapping[str, Any]) -> tuple[UntrustedSource, ...]:
    """Enumerate the untrusted regions of one journalled cycle, in order.

    Deterministic: same line, same tuple, every time. It is rendered in the
    dashboard and read in the demo, and a list that reorders between runs is a
    list nobody can diff.
    """
    found: list[UntrustedSource] = []

    call = record.get("call")
    if isinstance(call, Mapping) and call.get("raw_response"):
        found.append(
            UntrustedSource(
                path="call.raw_response",
                kind="model",
                trust="unwrapped",
            )
        )

    for section in ("submission", "outcome"):
        block = record.get(section)
        if not isinstance(block, Mapping):
            continue
        envelope = _envelope_of(block)
        if envelope is None:
            continue
        found.append(
            UntrustedSource(
                path=f"{section}.raw" if "raw" in block else section,
                kind="mcp",
                trust=str(envelope.get("trust") or "unwrapped"),
                risk=str(envelope.get("risk") or ""),
            )
        )

    return tuple(found)


def trust_report(record: Mapping[str, Any]) -> str:
    """One line for the dashboard and the demo. Empty when nothing is untrusted.

    A cycle that never called a model and never reached the broker — `NO_SETUP`,
    the majority case — has no untrusted bytes at all, and saying so plainly is
    better than rendering an empty table under a heading.
    """
    sources = untrusted_sources(record)
    if not sources:
        return "no untrusted bytes: this decision was computed from fetched data alone"
    parts = [
        f"{source.path} ({source.kind}"
        + (", prompt-injection surface" if source.is_prompt_injection_surface else "")
        + ")"
        for source in sources
    ]
    return "untrusted: " + "; ".join(parts)


def _envelope_of(block: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = block.get(_ENVELOPE_KEY)
    return value if isinstance(value, Mapping) else None
