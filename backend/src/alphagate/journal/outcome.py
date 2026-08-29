"""The later facts — specs/06 D2 `OutcomeRecord`, written as D3 amendments.

A cycle record is complete the moment the cycle ends. What it cannot contain is
what had not happened yet: the fill that lands forty minutes after submission,
the realised P&L on the close three days later. Those are the facts D3 insists
arrive as **separate lines keyed by `cycle_id`**, never as an edit.

So this module has exactly two jobs and deliberately no third:

**Turn a broker answer into a record.** `outcome_from` is pure — a `Submission`
in, an `OutcomeRecord` out — which means the amendment written after a live
read-back and the amendment written in a replay are produced by the same
function from the same bytes. Nothing here reads a clock; `observed_at` is a
parameter, same rule as everywhere else (specs/01 Rule 5).

**Carry the envelope with it** (D5). A fill is a fact that came from outside the
trust boundary, and the amendment says so in the same line, attached to the data
it described. An outcome whose provenance has been dropped one hop downstream is
exactly the hop where a journal stops being able to answer the question.

Money is `Decimal` end to end. `filled_avg_price` arrives from Alpaca as a
string and stays one all the way to the file; the only thing that would make it
a float is somebody deciding to be helpful.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from alphagate.execution import LegStatus, OrderId, OrderStatus, SecurityEnvelope, Submission

__all__ = ["Fill", "OutcomeRecord", "outcome_from", "realised"]


@dataclass(frozen=True, slots=True)
class Fill:
    """One leg, as the broker last reported it.

    `price` is `None` until something fills, never zero — an unfilled leg has no
    price and zero is a price. The distinction is inherited from `LegStatus` and
    it survives the journal for the same reason it exists there.
    """

    symbol: str
    quantity: int
    price: Decimal | None
    status: str

    @classmethod
    def of(cls, leg: LegStatus) -> Fill:
        return cls(
            symbol=leg.symbol,
            quantity=leg.filled_qty,
            price=leg.filled_avg_price,
            status=leg.status.value,
        )

    @property
    def notional(self) -> Decimal | None:
        """Per-contract price times contracts. `None` propagates rather than
        collapsing to zero: an unpriced leg makes the total unknown, not free."""
        if self.price is None:
            return None
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """What became of one cycle's order — specs/06 D2, written as an amendment.

    Written once per observation, not once per cycle. Two read-backs produce two
    amendment lines and the later one wins on read, which is what makes a
    reconciler safe to re-run: the file grows, the answer converges, and no
    earlier line is touched.
    """

    cycle_id: str
    observed_at: datetime
    status: OrderStatus
    raw_status: str
    """Alpaca's own string, kept even where it maps to `UNKNOWN`. A status we
    did not recognise is a fact about the broker, not a gap in the record."""
    client_order_id: str
    order_id: OrderId | None
    fills: tuple[Fill, ...]
    realised_pl: Decimal | None = None
    """`None` until the position is closed. Not zero — an open position has no
    realised P&L, and zero would read as a scratch."""
    closed_at: datetime | None = None
    reason: str | None = None
    """Rejection or cancellation reason, verbatim."""
    resolved_by_readback: bool = False
    envelope: SecurityEnvelope | None = None
    """specs/06 D5. Which bytes came from outside the trust boundary."""

    @property
    def is_terminal(self) -> bool:
        """No further amendment is expected without us doing something.

        `UNKNOWN` is not terminal. An unrecognised status is an unresolved order
        and calling it done is how one goes missing (specs/04 D4).
        """
        return self.status.is_terminal

    @property
    def filled_quantity(self) -> int:
        return sum(fill.quantity for fill in self.fills)

    def as_amendment(self) -> dict[str, object]:
        """The keys this outcome contributes to the cycle's final state.

        Nested under `outcome` rather than splattered across the top level, so a
        reader can always tell an observed fact from a decided one — which is
        the whole point of D3 and would be quietly undone by flattening.
        """
        return {"outcome": self}


def outcome_from(
    submission: Submission,
    *,
    cycle_id: str,
    observed_at: datetime,
    realised_pl: Decimal | None = None,
    closed_at: datetime | None = None,
) -> OutcomeRecord:
    """Read a broker answer into a journal amendment. Pure.

    `cycle_id` is passed rather than derived from `client_order_id`: the order id
    is a fingerprint of the *order* (specs/04 D5) and two cycles may legitimately
    produce the same one, whereas the cycle id is the identity of the decision
    and that is what the journal is keyed on.
    """
    return OutcomeRecord(
        cycle_id=cycle_id,
        observed_at=observed_at,
        status=submission.status,
        raw_status=submission.raw_status,
        client_order_id=submission.client_order_id,
        order_id=submission.order_id,
        fills=tuple(Fill.of(leg) for leg in submission.legs),
        realised_pl=realised_pl,
        closed_at=closed_at,
        reason=submission.reason,
        resolved_by_readback=submission.resolved_by_readback,
        envelope=submission.envelope,
    )


def realised(
    outcome: OutcomeRecord, profit_and_loss: Decimal, *, closed_at: datetime
) -> OutcomeRecord:
    """The same outcome with the close attached — a second amendment, later.

    Returns a new record rather than mutating one. The first amendment said what
    filled; this one says what it made. Both are on disk, in order, and neither
    overwrites the decision that produced them.
    """
    return OutcomeRecord(
        cycle_id=outcome.cycle_id,
        observed_at=closed_at,
        status=outcome.status,
        raw_status=outcome.raw_status,
        client_order_id=outcome.client_order_id,
        order_id=outcome.order_id,
        fills=outcome.fills,
        realised_pl=profit_and_loss,
        closed_at=closed_at,
        reason=outcome.reason,
        resolved_by_readback=outcome.resolved_by_readback,
        envelope=outcome.envelope,
    )
