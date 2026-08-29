"""Derived idempotency keys — specs/04 D4.

Every submission carries a `client_order_id`. The API rejects duplicates, which
is what makes retry-after-timeout safe — the one case where a naive retry
doubles a position.

**The id is derived, not random.** A UUID regenerated on retry is not an
idempotency key, it is a second order. So it is a hash of the facts that define
the order: the proposal it came from, the exact structure, the quantity, the
intent, and the trading day.

The trading day is in the key on purpose. The same proposal, legitimately
re-proposed tomorrow, is a *different* order and must be allowed through. The
same proposal re-submitted twenty seconds later is the same order and must not
be. A day boundary is the coarsest window that gets both right, and coarse is
the safe direction here.

That day is the **Eastern** date, not the UTC one. A 20:30 UTC order is the same
US trading day as a 14:00 UTC order, and hashing the UTC date would split them
across two keys every afternoon — handing back exactly the duplicate the key
exists to prevent.
"""

from __future__ import annotations

from datetime import date, datetime
from hashlib import blake2b
from typing import Final
from zoneinfo import ZoneInfo

from alphagate.options import format_occ
from alphagate.risk import GatedOrder

__all__ = ["MARKET_TZ", "client_order_id", "order_fingerprint", "trading_day_of"]

MARKET_TZ: Final = ZoneInfo("America/New_York")
PREFIX: Final = "alphagate"
_DIGEST_CHARS: Final = 24
"""Well under Alpaca's 128-character ceiling, and 96 bits of collision
resistance over a four-day competition placing tens of orders."""


def trading_day_of(moment: datetime) -> date:
    """The US market date a timestamp belongs to."""
    return moment.astimezone(MARKET_TZ).date()


def order_fingerprint(order: GatedOrder, trading_day: date) -> str:
    """The canonical string an id is derived from. Stable and human-readable.

    Returned rather than kept private so a mismatch is debuggable: when two
    orders that should share an id do not, you want to diff these, not two
    hex digests.

    Leg order is not part of the identity — `OptionStructure` normalises leg
    order at construction (specs/02 D3), so two orderings of the same spread
    produce the same fingerprint and therefore the same key.
    """
    legs = "|".join(
        f"{format_occ(leg.contract)}:{leg.side.value}:{leg.quantity}"
        for leg in order.structure.legs
    )
    return "/".join(
        (
            order.proposal_id,
            order.structure.kind.value,
            legs,
            str(order.quantity),
            order.intent.value,
            trading_day.isoformat(),
        )
    )


def client_order_id(order: GatedOrder, trading_day: date | None = None) -> str:
    """A stable, derived idempotency key for one order.

    `trading_day` defaults to the Eastern date of the approval timestamp, which
    is the right answer for every order the agent submits in the cycle that
    produced it. It stays a parameter for the case that is not that: a manual
    resubmission the next morning of an order that never filled.

    The limit price is deliberately **not** in the key. Repricing an unfilled
    order is a `replace_order_by_id`, not a new order, and putting the price in
    the key would let a one-cent adjustment open a second position.
    """
    day = trading_day if trading_day is not None else trading_day_of(order.approved_at)
    digest = blake2b(order_fingerprint(order, day).encode("utf-8"), digest_size=16).hexdigest()
    return f"{PREFIX}-{digest[:_DIGEST_CHARS]}"
