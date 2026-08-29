"""Order status and the shape of a submission — specs/04 D5.

```
accepted ──▶ new ──▶ partially_filled ──▶ filled
    │         │              │
    └─────────┴──────────────┴──▶ canceled | expired | rejected
```

Orders placed while the market is closed sit at `accepted` and queue for the
next open. That is a normal outcome, not a failure, and the journal records it
as one.

`partially_filled` on a multi-leg order is the dangerous state, and it gets the
loudest treatment in this file. A spread half filled is a naked leg — the exact
position specs/02 D3 arranges to be unrepresentable and specs/03 exists to
prevent. Alpaca fills `mleg` atomically so it should not occur; that is why it
is an exception rather than a branch, and why nothing here tries to leg out of
it automatically.

An unrecognised status maps to `UNKNOWN` rather than raising. Alpaca has around
seventeen statuses and we act on seven; a status we have not seen before is not
a reason to crash mid-cycle, but it *is* a reason to treat the order as
unresolved rather than done — which is what `is_terminal` returning False on
`UNKNOWN` arranges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Final, NewType

from alphagate.execution.errors import MalformedToolOutput, PartialFillBreach
from alphagate.execution.session import SecurityEnvelope, ToolResult

__all__ = [
    "LegStatus",
    "OrderId",
    "OrderStatus",
    "Submission",
    "guard_against_partial_fill",
    "parse_status",
]

OrderId = NewType("OrderId", str)


class OrderStatus(Enum):
    """The Alpaca statuses we act on, plus a catch-all for the rest."""

    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    DONE_FOR_DAY = "done_for_day"
    REPLACED = "replaced"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """No further transition is expected without us doing something.

        `UNKNOWN` is deliberately not terminal: an unrecognised status is an
        unresolved order, and calling it done is how one goes missing.
        """
        return self in _TERMINAL

    @property
    def is_live(self) -> bool:
        """The order exists at the broker and may still fill."""
        return self in _LIVE

    @property
    def is_filled(self) -> bool:
        return self is OrderStatus.FILLED


_TERMINAL: Final = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
        OrderStatus.REPLACED,
    }
)
_LIVE: Final = frozenset(
    {
        OrderStatus.PENDING_NEW,
        OrderStatus.ACCEPTED,
        OrderStatus.NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.DONE_FOR_DAY,
    }
)


def parse_status(raw: object) -> OrderStatus:
    """Map Alpaca's status string onto the enum. Never raises."""
    try:
        return OrderStatus(str(raw).strip().lower())
    except ValueError:
        return OrderStatus.UNKNOWN


@dataclass(frozen=True, slots=True)
class LegStatus:
    """One leg of a multi-leg order, as the broker currently sees it."""

    symbol: str
    status: OrderStatus
    filled_qty: int
    filled_avg_price: Decimal | None
    """`None` until something fills. Never zero — an unfilled leg has no price,
    and zero is a price."""

    @property
    def is_filled(self) -> bool:
        return self.status.is_filled or self.filled_qty > 0


@dataclass(frozen=True, slots=True)
class Submission:
    """What came back from submitting one gated order.

    This is the value the journal records (specs/06 D2 `SubmissionRecord`) and
    the value the reconciler reads. It carries `raw` verbatim because a
    paraphrased rejection reason is not a rejection reason, and `envelope`
    because specs/06 D5 wants to be able to say which bytes came from outside
    the trust boundary.
    """

    client_order_id: str
    order_id: OrderId | None
    status: OrderStatus
    raw_status: str
    """Alpaca's own string, kept even when it maps to `UNKNOWN`."""
    submitted_at: datetime
    legs: tuple[LegStatus, ...]
    reason: str | None
    """Rejection reason, verbatim. `None` unless the broker gave one."""
    attempts: int
    resolved_by_readback: bool
    """True when a timeout was resolved by reading the order back rather than by
    a direct answer — specs/04 D4. Worth recording: it is the difference between
    "we know" and "we asked again"."""
    envelope: SecurityEnvelope | None
    raw: str

    @property
    def is_rejected(self) -> bool:
        return self.status is OrderStatus.REJECTED

    @property
    def is_partial(self) -> bool:
        """A multi-leg order with some legs filled and some not.

        Both readings are checked: the parent order reporting
        `partially_filled`, and the legs simply disagreeing with each other. The
        second catches a broker that fills legs without updating the parent —
        which is precisely the situation nobody would think to look for.
        """
        if len(self.legs) < 2:
            return self.status is OrderStatus.PARTIALLY_FILLED
        filled = sum(1 for leg in self.legs if leg.is_filled)
        return self.status is OrderStatus.PARTIALLY_FILLED or 0 < filled < len(self.legs)


def guard_against_partial_fill(submission: Submission) -> Submission:
    """Raise on a half-filled spread. Returns the submission otherwise.

    Written to be used as `return guard_against_partial_fill(submission)` so
    that no path can produce a `Submission` without passing through it.
    """
    if submission.is_partial:
        raise PartialFillBreach(submission)
    return submission


def submission_from(
    result: ToolResult,
    *,
    client_order_id: str,
    submitted_at: datetime,
    attempts: int,
    resolved_by_readback: bool = False,
) -> Submission:
    """Read an order object out of a tool response.

    `status` is required rather than defaulted. An order whose state we could not
    read is not an order in an unremarkable state; it is an order we have lost
    track of, and it must be loud now rather than surprising at 16:05.
    """
    data = result.data
    raw_status = str(result.require("status"))
    return Submission(
        client_order_id=str(data.get("client_order_id") or client_order_id),
        order_id=OrderId(str(data["id"])) if data.get("id") else None,
        status=parse_status(raw_status),
        raw_status=raw_status,
        submitted_at=submitted_at,
        legs=_legs_of(data),
        reason=_reason_of(data),
        attempts=attempts,
        resolved_by_readback=resolved_by_readback,
        envelope=result.envelope,
        raw=result.raw,
    )


def _legs_of(data: Mapping[str, Any]) -> tuple[LegStatus, ...]:
    raw_legs = data.get("legs")
    if not isinstance(raw_legs, Sequence) or isinstance(raw_legs, (str, bytes)):
        return ()
    return tuple(_leg(leg) for leg in raw_legs if isinstance(leg, Mapping))


def _leg(leg: Mapping[str, Any]) -> LegStatus:
    return LegStatus(
        symbol=str(leg.get("symbol", "")),
        status=parse_status(leg.get("status")),
        filled_qty=_int(leg.get("filled_qty")),
        filled_avg_price=_money(leg.get("filled_avg_price")),
    )


def _reason_of(data: Mapping[str, Any]) -> str | None:
    for key in ("reject_reason", "rejected_reason", "reason", "message", "error"):
        value = data.get(key)
        if value:
            return str(value)
    return None


def _int(value: object) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError) as exc:
        raise MalformedToolOutput(f"not a quantity: {value!r}") from exc


def _money(value: object) -> Decimal | None:
    """Prices arrive as strings and stay exact. `None` stays `None`, never 0."""
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise MalformedToolOutput(f"not a price: {value!r}") from exc
    if not parsed.is_finite():
        raise MalformedToolOutput(f"price must be finite, got {value!r}")
    return parsed
