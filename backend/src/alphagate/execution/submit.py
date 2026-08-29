"""The only door — specs/04 D1 and D4.

```python
def submit(order: GatedOrder, mcp: McpSession) -> Submission
def cancel(order_id: OrderId, mcp: McpSession) -> None
```

`GatedOrder` is constructible only inside `risk.gate` (specs/03 D3), and
`submit` accepts nothing else. There is no REST fallback for orders, because a
fallback is a bypass, and the whole claim of this project is that no bypass
exists. The `isinstance` check below is not defensive programming against a type
error — it is the runtime half of specs/01 Rule 2, and it is why the type
annotation is worth believing at three in the morning on day five.

**Retry policy.** Three attempts, exponential backoff, on transport failure and
5xx only. A 4xx retried three times is a malformed request three times.

**A timeout is not a failure.** It is an unknown outcome: the order may be live.
It is resolved by reading back `get_order_by_client_id` with the derived
idempotency key, never by resubmitting. This is the single most important
control-flow decision in the module, and it is why `_read_back` exists and why
nothing in the retry loop treats `ToolTimeout` as retryable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Final

from alphagate.execution.errors import (
    ExecutionError,
    MalformedToolOutput,
    ToolTimeout,
    TransportFailure,
)
from alphagate.execution.idempotency import client_order_id as derive_client_order_id
from alphagate.execution.lifecycle import (
    OrderId,
    OrderStatus,
    Submission,
    guard_against_partial_fill,
    submission_from,
)
from alphagate.execution.mapping import PLACE_ORDER_TOOL, to_tool_arguments
from alphagate.execution.session import McpSession, ToolArgument, ToolResult
from alphagate.risk import GatedOrder

__all__ = [
    "CANCEL_TOOL",
    "MAX_ATTEMPTS",
    "READ_BACK_TOOL",
    "cancel",
    "read_back",
    "submit",
]

CANCEL_TOOL: Final = "cancel_order_by_id"
READ_BACK_TOOL: Final = "get_order_by_client_id"
MAX_ATTEMPTS: Final = 3
BACKOFF_BASE: Final = 0.5
"""Seconds. 0.5, 1.0 — two waits, because there are three attempts."""

type Sleeper = Callable[[float], None]


def submit(
    order: GatedOrder,
    mcp: McpSession,
    *,
    trading_day: date | None = None,
    sleep: Sleeper = time.sleep,
) -> Submission:
    """Send one gated order. The only way an order reaches Alpaca.

    `sleep` is injected so the retry path is testable without waiting; it is not
    a knob for production. `trading_day` defaults to the Eastern date of the
    order's approval, and only needs supplying for a manual resubmission on a
    later day (specs/04 D4).
    """
    if not isinstance(order, GatedOrder):
        raise TypeError(
            f"submit() accepts a GatedOrder, got {type(order).__name__}. "
            "specs/01 Rule 2: every order reaching Alpaca passed the Gate. There "
            "is no bypass — construct one with risk.gate.evaluate."
        )

    key = derive_client_order_id(order, trading_day)
    arguments: dict[str, ToolArgument] = {**to_tool_arguments(order), "client_order_id": key}

    last_failure: TransportFailure | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = mcp.call(PLACE_ORDER_TOOL, arguments)
        except ToolTimeout:
            # Unknown outcome, not a failure. Ask what happened; never resend.
            return read_back(key, mcp, submitted_at=order.approved_at, attempts=attempt)
        except TransportFailure as failure:
            last_failure = failure
            if attempt == MAX_ATTEMPTS:
                break
            sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
            continue

        _reject_error_payload(result)
        return guard_against_partial_fill(
            submission_from(
                result,
                client_order_id=key,
                submitted_at=order.approved_at,
                attempts=attempt,
            )
        )

    raise TransportFailure(
        f"{PLACE_ORDER_TOOL} failed {MAX_ATTEMPTS} times for {key}: {last_failure}"
    )


def read_back(
    key: str,
    mcp: McpSession,
    *,
    submitted_at: datetime,
    attempts: int = 1,
) -> Submission:
    """Resolve an unknown outcome by asking what the broker has.

    A `get` that itself fails is left to raise. Guessing here would mean deciding
    between "no order exists" and "an order exists that we cannot see", and those
    two guesses have opposite consequences — one loses a position, the other
    duplicates it.
    """
    result = mcp.call(READ_BACK_TOOL, {"client_order_id": key})
    if _looks_like_not_found(result):
        raise ExecutionError(
            f"order {key} timed out and cannot be read back; it may or may not "
            "exist. Reconcile by hand before submitting anything else — "
            "resubmitting blind is how one position becomes two (specs/04 D4)."
        )
    return guard_against_partial_fill(
        submission_from(
            result,
            client_order_id=key,
            submitted_at=submitted_at,
            attempts=attempts,
            resolved_by_readback=True,
        )
    )


def cancel(order_id: OrderId, mcp: McpSession) -> None:
    """Cancel a live order. Idempotent at the broker; not retried here.

    A cancel that fails is reported, not retried into the ground: the caller is
    usually a human at the kill switch who wants to know it did not work, not a
    loop that keeps trying while the market moves.
    """
    result = mcp.call(CANCEL_TOOL, {"order_id": str(order_id)})
    _reject_error_payload(result)


def _reject_error_payload(result: ToolResult) -> None:
    """Turn a tool-level error object into an exception.

    The MCP server answers some failures with `{"error": ...}` and HTTP 200. A
    response that says "error" and is read as an order is an order that does not
    exist being recorded as one that does.
    """
    data = result.data
    error = data.get("error") or data.get("detail")
    if error and "status" not in data:
        raise MalformedToolOutput(f"{result.tool} returned an error: {error}")


def _looks_like_not_found(result: ToolResult) -> bool:
    if "status" in result.data:
        return False
    blob = result.raw.lower()
    return "not found" in blob or "404" in blob


def status_of(submission: Submission) -> OrderStatus:
    """Convenience for callers that only want the state."""
    return submission.status


def is_queued_for_open(submission: Submission) -> bool:
    """An order placed outside market hours, waiting for the bell.

    Verified against the live paper account: a spread submitted at 07:24 UTC
    came back `accepted` with both legs `accepted` (specs/04 D5). This is a
    normal outcome and the journal records it as one — not an error, and not a
    fill.
    """
    return submission.status is OrderStatus.ACCEPTED


def arguments_for(order: GatedOrder, trading_day: date | None = None) -> Mapping[str, ToolArgument]:
    """The exact payload `submit` would send. For the journal and for tests.

    Exposed so a test can assert on what goes over the wire without a session,
    and so the journal can record the request next to the response.
    """
    return {
        **to_tool_arguments(order),
        "client_order_id": derive_client_order_id(order, trading_day),
    }
