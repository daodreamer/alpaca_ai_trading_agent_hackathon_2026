"""The sign convention clash — specs/04 D2. **Read this twice.**

|                              | Credit received | Debit paid |
| ---------------------------- | --------------- | ---------- |
| `StructureRisk.net_premium`  | **positive**    | negative   |
| Alpaca `limit_price` (mleg)  | **negative**    | positive   |

The domain keeps the finance convention because that is what makes the
`max_profit` / `max_loss` arithmetic in specs/02 D4 read correctly. Alpaca
inverts it: its schema says *"positive = debit/cost, negative = credit/proceeds"*.

Getting this wrong sends a credit spread as a debit spread at the same absolute
price — an order that is wrong by twice the premium and **will happily fill**.
There is no validation error, no rejection, no alert. Just a fill at the worst
price in the room.

So the flip happens in exactly one function, with a name that says so, in a
module that does nothing else. A sign flip applied in two places is a sign flip
applied nowhere.

The second conversion here is quieter but just as capable of being wrong by a
factor of a hundred: `net_premium` is **total cash for the whole structure**,
and `limit_price` is **per share, for one unit of the strategy**. Alpaca
multiplies the price back out by `qty`, `ratio_qty` and the contract multiplier.
Sending the total where a per-share price belongs is a 100× error in the
direction of "why did that fill instantly".
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.risk import GatedOrder

__all__ = [
    "PRICE_TICK",
    "alpaca_limit_price",
    "alpaca_limit_price_inverse",
    "net_premium_per_unit",
]

PRICE_TICK: Final = Decimal("0.01")
"""Option prices are quoted in cents. Sub-penny limits are rejected by the API."""


def alpaca_limit_price(net_premium: Decimal) -> str:
    """Domain says credit-positive; Alpaca says debit-positive. Flip, once, here.

    Takes a **per-share** net premium — see `net_premium_per_unit`. Returns a
    string because every argument in the tool schema is a string, and formatting
    a `Decimal` at the call site is one more place to lose a digit.
    """
    if not isinstance(net_premium, Decimal):
        raise InvariantViolation(
            f"net_premium must be Decimal, got {type(net_premium).__name__}: "
            "a float here is a rounding error on a live order"
        )
    if not net_premium.is_finite():
        raise InvariantViolation(f"net_premium must be finite, got {net_premium}")
    return _plain(-net_premium)


def alpaca_limit_price_inverse(wire: str | Decimal) -> Decimal:
    """Read an Alpaca limit price back into the domain convention.

    Exists for the round-trip property test and for reconciliation, which reads
    prices off live orders and has to compare them with what the domain thinks it
    asked for. Same flip, stated once in the other direction.
    """
    try:
        value = wire if isinstance(wire, Decimal) else Decimal(str(wire).strip())
    except ArithmeticError as exc:
        raise InvariantViolation(f"not a price: {wire!r}") from exc
    if not value.is_finite():
        raise InvariantViolation(f"limit price must be finite, got {wire!r}")
    return -value


def net_premium_per_unit(order: GatedOrder) -> Decimal:
    """Total structure cash → price per share for one unit of the strategy.

    `StructureRisk.net_premium` is cash for the structure as constructed: it
    already carries the contract multiplier and the per-leg quantity. Alpaca
    wants the price of *one* 1:1 unit, per share, and rebuilds the cash itself
    from `qty` × `ratio_qty` × multiplier. So the multiplier and the structure's
    own quantity come back out here — and `GatedOrder.quantity` does **not**,
    because that is what `qty` on the wire is for.

    Quantised to a cent, half-even. The residue is at most half a cent per share
    and it is not this layer's job to be clever about it: how aggressively to
    price relative to the mid is a strategy decision (specs/07), and burying it
    in a rounding mode would put it somewhere nobody would think to look.
    """
    structure = order.structure
    divisor = Decimal(structure.multiplier * structure.quantity)
    if divisor <= 0:  # pragma: no cover - guaranteed by specs/02 D3
        raise InvariantViolation(f"structure has no notional: {structure}")
    return (order.limit_price / divisor).quantize(PRICE_TICK, rounding=ROUND_HALF_EVEN)


def _plain(value: Decimal) -> str:
    """Format without scientific notation and without a negative zero.

    `str(Decimal("0E-2"))` is `"0E-2"`, and `-0.00` is a price that means nothing
    but looks alarming in a journal. Both are formatting accidents; neither
    should reach the wire.
    """
    quantised = value.quantize(PRICE_TICK, rounding=ROUND_HALF_EVEN)
    if quantised == 0:
        quantised = abs(quantised)
    return f"{quantised:.2f}"
